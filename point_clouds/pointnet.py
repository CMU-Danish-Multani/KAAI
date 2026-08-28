"""Permutation-invariant networks that read raw galaxy positions.

CosmoBench feeds these models positions only, no mass and no velocity, so the
input for one universe is an n-by-3 table and nothing else.

    DeepSets(X) = rho( POOL_i phi(x_i) )

phi looks at one galaxy at a time and rho reads the pooled summary. Because the
pooling step ignores order, the whole thing gives the same answer however the
galaxies are listed, which is the correct symmetry: a universe has no first
galaxy.

WHY THE POOLING CHOICE IS THE WHOLE EXPERIMENT
----------------------------------------------
Sum pooling adds N vectors together, so its output scales with N and the model
can read off the galaxy count. Mean pooling divides that out. Max pooling keeps
extremes and is close to count-blind.

In CAMELS the galaxy count is a documented shortcut to Omega_m, so the pooling
operation alone decides whether the model has access to a leak. Everything else
about the architecture is held fixed, which makes this a controlled comparison
rather than an observation.

BATCHING
--------
Clouds have different sizes (588 to 4511 in CAMELS), so padding to a common
length would waste most of the tensor. Instead every cloud in a batch is
concatenated into one long list of points, alongside an index saying which cloud
each point came from. Pooling is then a scatter-add over that index. This is the
same trick PyTorch Geometric uses for graphs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.metrics import seed_all
from point_clouds.load import BOX, open_suite, read_cloud, read_labels, sim_names

TARGETS = ("Omega_m", "sigma_8")
POOLINGS = ("sum", "mean", "max")


def load_positions(suite: str, split: str, limit: int = 0,
                   fixed_count: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Raw positions per cloud, scaled to the unit cube, plus the labels.

    Dividing by the box side puts every suite on the same numeric footing, so
    the same architecture and learning rate work for a 25 unit box and a 100
    unit one.
    """
    with open_suite(suite, split) as f:
        names = sim_names(f)[:limit] if limit else sim_names(f)
        labels = read_labels(f)
        clouds = []
        for sim in names:
            positions, _, extra = read_cloud(f, sim)
            if fixed_count:
                # Keep the most massive, so the cut is deterministic and matches
                # the trimming used for the correlation-function control.
                if len(positions) < fixed_count:
                    raise AssertionError(
                        f"{suite}/{split}/{sim}: {len(positions)} galaxies, fewer "
                        f"than the requested fixed count {fixed_count}")
                keep = np.argsort(extra["Mstar"])[::-1][:fixed_count]
                positions = positions[keep]
            clouds.append(np.mod(positions, BOX[suite]).astype(np.float32) / BOX[suite])
        y = np.stack([labels[t][:len(names)] for t in TARGETS], axis=1)
    return clouds, y.astype(np.float32)


class Batched:
    """All clouds of a split as one flat point list plus a cloud index.

    Targets are stored rescaled to mean 0 spread 1 on TRAIN statistics, the same
    convention as the correlation-function path. Omega_m lives in [0.1, 0.5] and
    sigma_8 in [0.6, 1.0], so an untrained network's raw outputs start orders of
    magnitude away and the first gradients are enormous. Sum pooling compounds
    that by scaling the summary with the galaxy count.
    """

    def __init__(self, clouds: List[np.ndarray], y: np.ndarray, device: torch.device,
                 label_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.label_mean, self.label_spread = (
            label_stats if label_stats is not None else (y.mean(0), y.std(0)))
        y = (y - self.label_mean) / self.label_spread
        self.points = torch.as_tensor(np.concatenate(clouds)).to(device)
        index = np.repeat(np.arange(len(clouds)), [len(c) for c in clouds])
        self.index = torch.as_tensor(index, dtype=torch.long).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.sizes = torch.as_tensor([len(c) for c in clouds]).to(device)
        self.n_clouds = len(clouds)

    def subset(self, cloud_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Points belonging to the given clouds, re-indexed from 0."""
        mask = torch.isin(self.index, cloud_ids)
        points, old = self.points[mask], self.index[mask]
        remap = torch.full((self.n_clouds,), -1, dtype=torch.long, device=points.device)
        remap[cloud_ids] = torch.arange(len(cloud_ids), device=points.device)
        return points, remap[old], self.y[cloud_ids]


def pool(values: torch.Tensor, index: torch.Tensor, n: int, how: str,
         count_scale: float = 1.0) -> torch.Tensor:
    """Combine every cloud's points into one vector per cloud.

    `sum` divides by count_scale, a FIXED constant, not by each cloud's own size.
    That distinction is the entire experiment. Dividing by a constant keeps the
    summary proportional to the galaxy count, so the count stays readable, while
    keeping magnitudes near 1 so the network can actually train. Dividing by each
    cloud's own size, which is what `mean` does, removes the count instead.

    Without the constant, adding roughly 2500 vectors leaves the summary about
    2500 times larger than the next layer was initialised for. Measured
    2026-08-18: raw sum pooling scored -13.7 on Omega_m, meaning it never
    trained at all, which would have made the comparison meaningless rather
    than informative.
    """
    out = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    if how == "sum":
        return out.index_add_(0, index, values) / count_scale
    if how == "mean":
        counts = torch.zeros(n, 1, device=values.device).index_add_(
            0, index, torch.ones(len(index), 1, device=values.device))
        return out.index_add_(0, index, values) / counts.clamp_min(1)
    if how == "max":
        return out.fill_(-torch.inf).index_reduce_(0, index, values, "amax",
                                                   include_self=True).clamp_min(-1e30)
    raise ValueError(f"unknown pooling {how!r}")


class DeepSets(nn.Module):
    """phi over each point, pool, then rho over the summary."""

    def __init__(self, hidden: int = 64, pooling: str = "mean", n_in: int = 3,
                 count_scale: float = 1.0):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.pooling = pooling
        self.count_scale = count_scale
        self.phi = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(TARGETS)))

    def forward(self, points: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        return self.rho(pool(self.phi(points), index, n, self.pooling,
                             self.count_scale))


def fit(train: Batched, model: nn.Module, seed: int, epochs: int = 100,
        batch_size: int = 32, learning_rate: float = 1e-3) -> nn.Module:
    seed_all(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    shuffle = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(train.n_clouds, generator=shuffle).to(train.points.device)
        for start in range(0, train.n_clouds, batch_size):
            ids = order[start:start + batch_size]
            points, index, y = train.subset(ids)
            optimiser.zero_grad()
            loss_function(model(points, index, len(ids)), y).backward()
            optimiser.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, data: Batched, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, data.n_clouds, batch_size):
        ids = torch.arange(start, min(start + batch_size, data.n_clouds),
                           device=data.points.device)
        points, index, _ = data.subset(ids)
        out.append(model(points, index, len(ids)).cpu().numpy())
    model.train()
    scaled = np.concatenate(out).astype(np.float64)
    return scaled * data.label_spread + data.label_mean      # back to physical units
