"""Radius-graph message passing over galaxy positions.

Connect every pair of galaxies closer than a cutoff Rc, then let each galaxy
repeatedly exchange messages with its neighbours. After L rounds a galaxy's
vector reflects everything within L steps of it, so the network sees local
structure that a permutation-invariant set model cannot.

WHAT THE MODEL IS ALLOWED TO SEE
--------------------------------
Absolute position inside a periodic box is meaningless: the origin is arbitrary
and the box wraps, so a universe shifted sideways is the same universe. Node
features therefore start as a constant, and every piece of information the model
receives arrives through edges as the separation d_ij / Rc, which is unchanged
by translation, rotation and reflection.

That is a deliberate deviation from CosmoBench Sec. 4.1, which feeds absolute
positions as node features and adds two dot products built from them. Recorded
here rather than left silent.

POOLING IS NOT A DETAIL
-----------------------
Measured 2026-08-18 on DeepSets: sum pooling scored 0.5233 on CAMELS Omega_m
against a count-only reference of 0.5058, meaning it learned nothing but the
galaxy count, while mean pooling scored -0.0006. The pooling choice decides
whether a model has access to the counting shortcut, so it is an explicit
argument here and defaults to mean.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from point_clouds.pointnet import TARGETS, POOLINGS, pool

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "graph_cache"


def radius_graph(positions: np.ndarray, box: float, cutoff: float) -> np.ndarray:
    """Edges between galaxies closer than cutoff, as a (2, n_edges) array.

    Each undirected pair is stored in both directions so messages flow both
    ways. The tree is told the box wraps, so a galaxy at the far edge is
    correctly a neighbour of one at the near edge.
    """
    wrapped = np.mod(positions, box)
    pairs = cKDTree(wrapped, boxsize=box).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    both = np.concatenate([pairs, pairs[:, ::-1]])
    return both.T.astype(np.int64)


def edge_lengths(positions: np.ndarray, edges: np.ndarray, box: float) -> np.ndarray:
    """Separation for each edge, using the wrap-around convention."""
    delta = positions[edges[0]] - positions[edges[1]]
    delta -= box * np.round(delta / box)
    return np.sqrt((delta ** 2).sum(axis=1))


class Batch:
    """A handful of clouds glued into one disconnected graph, on device."""

    def __init__(self, edges, edge_feature, index, y, n_nodes, n_clouds):
        self.edges, self.edge_feature = edges, edge_feature
        self.index, self.y = index, y
        self.n_nodes, self.n_clouds = n_nodes, n_clouds


class GraphSet:
    """Per-cloud graphs, assembled ONCE into fixed batches held on device.

    WHY BATCHES ARE BUILT ONCE AND REUSED
    -------------------------------------
    The obvious design rebuilds each batch every epoch from a fresh shuffle of
    the clouds. That is what this class used to do, and it is why a run took a
    day and produced nothing.

    Each rebuild concatenates NumPy arrays whose size differs batch to batch,
    because clouds differ in density, then ships them to the GPU. Over 200
    epochs times 12 models that is roughly 45,600 rebuilds of differently
    shaped buffers. The heap fragments, pages spill to swap, and the machine
    slows down as it goes.

    MEASURED 2026-08-20 with the rebuild-every-epoch design, one model:
        epoch  0   2.04 s   swap 4.4 GB
        epoch 99   6.77 s   swap 9.7 GB
    A 3.3x slowdown purely from allocation churn, still worsening at epoch 99.

    The whole dataset is only about 150 MB as graphs, so every batch fits on
    device simultaneously. Epochs then shuffle the ORDER batches are visited
    rather than re-partitioning the clouds. Randomness across epochs is slightly
    reduced, which is the standard trade every graph training loop makes.
    """

    def __init__(self, clouds: List[np.ndarray], y: np.ndarray, box: float,
                 cutoff_fraction: float, device: torch.device,
                 label_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        cutoff = cutoff_fraction * box
        self.device = device
        self.graphs = []
        for positions in clouds:
            e = radius_graph(positions, box, cutoff)
            self.graphs.append((len(positions), e,
                                (edge_lengths(positions, e, box) / cutoff
                                 ).astype(np.float32)))
        self.label_mean, self.label_spread = (
            label_stats if label_stats is not None else (y.mean(0), y.std(0)))
        self.y_scaled = ((y - self.label_mean) / self.label_spread).astype(np.float32)
        self.n_clouds = len(clouds)
        self.n_nodes = sum(g[0] for g in self.graphs)
        self._batches: Dict[int, List[Batch]] = {}

    def _assemble(self, cloud_ids: np.ndarray) -> Batch:
        edges, lengths, index, offset = [], [], [], 0
        for slot, cid in enumerate(cloud_ids):
            n, e, d = self.graphs[cid]
            edges.append(e + offset)
            lengths.append(d)
            index.append(np.full(n, slot))
            offset += n
        to = lambda a: torch.as_tensor(a).to(self.device)
        return Batch(to(np.concatenate(edges, axis=1)),
                     to(np.concatenate(lengths)[:, None]),
                     to(np.concatenate(index)),
                     to(self.y_scaled[cloud_ids]),
                     offset, len(cloud_ids))

    def batches(self, batch_size: int, order_seed: int = 0) -> List[Batch]:
        """Fixed batches, built on first request and cached on device.

        The partition is deterministic given batch_size, so repeated calls
        return the same objects rather than allocating new ones.
        """
        if batch_size not in self._batches:
            ids = np.arange(self.n_clouds)
            self._batches[batch_size] = [
                self._assemble(ids[s:s + batch_size])
                for s in range(0, self.n_clouds, batch_size)]
        return self._batches[batch_size]



class MessagePassingNet(nn.Module):
    """L rounds of neighbour exchange, then one vector per cloud."""

    def __init__(self, hidden: int = 64, layers: int = 3, pooling: str = "mean"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.pooling = pooling
        self.node_start = nn.Parameter(torch.zeros(hidden))
        self.message = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.update = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
            for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, len(TARGETS)))

    def forward(self, graph: 'Batch') -> torch.Tensor:
        source, target = graph.edges[0], graph.edges[1]
        h = self.node_start.expand(graph.n_nodes, -1)

        for message, update in zip(self.message, self.update):
            m = message(torch.cat([h[source], h[target], graph.edge_feature], dim=1))
            # Mean over each node's neighbours, so a galaxy in a dense region does
            # not simply shout louder than one in a void.
            gathered = torch.zeros_like(h).index_add_(0, target, m)
            degree = torch.zeros(graph.n_nodes, 1, device=h.device).index_add_(
                0, target, torch.ones(len(target), 1, device=h.device))
            h = update(torch.cat([h, gathered / degree.clamp_min(1)], dim=1))

        return self.readout(pool(h, graph.index, graph.n_clouds, self.pooling,
                                 count_scale=graph.n_nodes / graph.n_clouds))
