"""Assemble any config from the search space into a trainable model.

One entry point, build_model(). The pieces come from point_clouds/blocks/, each
taken from a different paper and rewritten to share one interface, so the search
can try arrangements that appear in none of them.

GRAPHS ARE BUILT ONCE PER CUTOFF AND CACHED
-------------------------------------------
Measured 2026-08-20: rebuilding batches every epoch fragmented the heap badly
enough that a run took a day and produced nothing, with time per epoch climbing
from 2.0 s to 6.8 s and swap from 4.4 GB to 9.7 GB. Batches are therefore
assembled once and reused, and epochs shuffle the order they are visited.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from common.metrics import r2_score, seed_all
from point_clouds.blocks.edge_features import EdgeFeaturiser
from point_clouds.pointnet import TARGETS
from point_clouds.search_space import make_pooling, output_dim


class CloudBatch:
    """One batch of clouds glued into a single disconnected graph, on device."""

    def __init__(self, points, index, y, edges=None, edge_vector=None):
        self.points, self.index, self.y = points, index, y
        self.edges, self.edge_vector = edges, edge_vector
        self.n_nodes = int(points.shape[0])
        self.n_clouds = int(y.shape[0])


class CloudData:
    """Positions and labels for one split, batched once per (cutoff, batch_size)."""

    def __init__(self, clouds: List[np.ndarray], y: np.ndarray, box: float,
                 device: torch.device,
                 label_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.clouds = [np.mod(c, box).astype(np.float32) for c in clouds]
        self.box, self.device = box, device
        self.label_mean, self.label_spread = (
            label_stats if label_stats is not None else (y.mean(0), y.std(0)))
        self.y_scaled = ((y - self.label_mean) / self.label_spread).astype(np.float32)
        self.n_clouds = len(clouds)
        self.mean_size = float(np.mean([len(c) for c in self.clouds]))
        self._cache: Dict[Tuple, List[CloudBatch]] = {}

    def batches(self, batch_size: int, cutoff: Optional[float]) -> List[CloudBatch]:
        key = (batch_size, cutoff)
        if key in self._cache:
            return self._cache[key]
        out, radius = [], (cutoff * self.box if cutoff else None)
        for start in range(0, self.n_clouds, batch_size):
            ids = range(start, min(start + batch_size, self.n_clouds))
            pts, idx, edges, vecs, offset = [], [], [], [], 0
            for slot, cid in enumerate(ids):
                p = self.clouds[cid]
                pts.append(p)
                idx.append(np.full(len(p), slot))
                if radius:
                    pairs = cKDTree(p, boxsize=self.box).query_pairs(
                        radius, output_type="ndarray")
                    if len(pairs):
                        both = np.concatenate([pairs, pairs[:, ::-1]]).T
                        v = p[both[0]] - p[both[1]]
                        v -= self.box * np.round(v / self.box)
                        edges.append(both + offset)
                        vecs.append(v)
                offset += len(p)
            t = lambda a, dt=None: torch.as_tensor(a if dt is None else a.astype(dt)).to(self.device)
            out.append(CloudBatch(
                t(np.concatenate(pts) / self.box),
                t(np.concatenate(idx).astype(np.int64)),
                t(self.y_scaled[list(ids)]),
                t(np.concatenate(edges, axis=1).astype(np.int64)) if edges else None,
                t(np.concatenate(vecs).astype(np.float32) / (radius or 1.0)) if vecs else None))
        self._cache[key] = out
        return out


class SearchableModel(nn.Module):
    """A DeepSets or message-passing network assembled from a config."""

    def __init__(self, config: Dict, mean_size: float):
        super().__init__()
        h = config["hidden"]
        self.family = config["family"]
        self.pooling = make_pooling(config["pooling"], h, mean_size)
        pooled = output_dim(self.pooling, h)

        if self.family == "deepsets":
            self.phi = nn.Sequential(nn.Linear(3, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU())
        else:
            self.featuriser = EdgeFeaturiser(
                n_basis=config["n_basis"], use_local_density=False,
                angular_powers=(1, 2, 3) if config["angular"] else (),
                normalise_angular=True)
            probe = self.featuriser(torch.randn(8, 3), torch.tensor([[0,1,2,3,0,1,2,3],
                                                                     [1,0,3,2,2,3,0,1]]),
                                    4, cutoff=1.0)
            self.edge_dim, self.node_dim = int(probe.edge.shape[1]), int(probe.node.shape[1])
            # With angular features off the featuriser returns no node features at
            # all, so there is nothing to project. Start from a learned constant
            # instead, which is the right thing anyway: absolute position in a
            # periodic box carries no information, so every node should begin
            # identical and acquire its identity only through message passing.
            self.node_start = (nn.Linear(self.node_dim, h) if self.node_dim > 0
                               else None)
            if self.node_start is None:
                self.node_constant = nn.Parameter(torch.zeros(h))
            self.message = nn.ModuleList(
                nn.Sequential(nn.Linear(2 * h + self.edge_dim, h), nn.ReLU(),
                              nn.Linear(h, h)) for _ in range(config["layers"]))
            self.update = nn.ModuleList(
                nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())
                for _ in range(config["layers"]))

        self.readout = nn.Sequential(nn.Linear(pooled, h), nn.ReLU(),
                                     nn.Linear(h, len(TARGETS)))

    def _initial_nodes(self, n_nodes: int, device, node_features) -> torch.Tensor:
        if self.node_start is None or node_features is None:
            return self.node_constant.expand(n_nodes, -1)
        return self.node_start(node_features)

    def forward(self, batch: CloudBatch) -> torch.Tensor:
        if self.family == "deepsets":
            h = self.phi(batch.points)
        else:
            if batch.edges is None:
                h = self._initial_nodes(batch.n_nodes, batch.points.device, None)
            else:
                feats = self.featuriser(batch.edge_vector, batch.edges,
                                        batch.n_nodes, cutoff=1.0)
                h = self._initial_nodes(batch.n_nodes, batch.points.device, feats.node)
                src, dst = batch.edges[0], batch.edges[1]
                ones = torch.ones(len(dst), 1, device=h.device)
                for message, update in zip(self.message, self.update):
                    m = message(torch.cat([h[src], h[dst], feats.edge], dim=1))
                    gathered = torch.zeros_like(h).index_add_(0, dst, m)
                    degree = torch.zeros(batch.n_nodes, 1, device=h.device
                                         ).index_add_(0, dst, ones)
                    h = update(torch.cat([h, gathered / degree.clamp_min(1)], dim=1))
        return self.readout(self.pooling(h, batch.index, batch.n_clouds))


def build_model(config: Dict, mean_size: float) -> SearchableModel:
    return SearchableModel(config, mean_size)


def train_and_score(config: Dict, train: CloudData, evaluate_on: CloudData,
                    seed: int, epochs: int, batch_size: int,
                    device: torch.device) -> np.ndarray:
    """Train one configuration and return R2 per target on `evaluate_on`."""
    seed_all(seed)
    cutoff = config.get("cutoff") if config["family"] == "gnn" else None
    model = build_model(config, train.mean_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_function = nn.MSELoss()
    batches = train.batches(batch_size, cutoff)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        for i in rng.permutation(len(batches)):
            optimiser.zero_grad()
            loss_function(model(batches[i]), batches[i].y).backward()
            optimiser.step()
        if device.type == "mps":
            torch.mps.empty_cache()

    model.eval()
    with torch.no_grad():
        predicted = np.concatenate([model(b).cpu().numpy()
                                    for b in evaluate_on.batches(batch_size, cutoff)])
    physical = predicted.astype(np.float64) * evaluate_on.label_spread + evaluate_on.label_mean
    truth = evaluate_on.y_scaled.astype(np.float64) * evaluate_on.label_spread + evaluate_on.label_mean
    del model, optimiser
    if device.type == "mps":
        torch.mps.empty_cache()
    return r2_score(physical, truth)
