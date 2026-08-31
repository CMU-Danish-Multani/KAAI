"""Embedding networks that turn a galaxy point cloud into a fixed length vector.

An LtU-ILI embedding network is any nn.Module mapping the data to features. It is
passed to `load_nde_sbi(..., embedding_net=...)` and trained jointly with the density
estimator, so the compression is learned for the inference task rather than chosen in
advance.

WHY POOLING CHOICE IS NOT COSMETIC HERE
---------------------------------------
A point cloud has no order, so the network must be permutation invariant, which means
pooling over points at some stage. Measured earlier in this project: a held out probe
recovers log N from the pooled output at R2 +0.9138 for sum and +0.8968 for max, and
-0.6616 for mean. In CAMELS the galaxy count correlates with Omega_m at 0.73, so a
count sensitive pooling lets a network score well without learning any structure. We
measured that costing sum pooling a spurious +0.149 on Omega_m, which vanished to
+0.0003 once the count was held fixed.

The tasks here use a fixed number of points, which closes that channel by
construction. Mean is still the default, because a default that cannot leak is worth
more than one that merely does not leak today.
"""

from typing import Tuple

import torch
import torch.nn as nn


def _mlp(sizes: Tuple[int, ...]) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])          # no activation on the output


class DeepSets(nn.Module):
    """Permutation invariant set encoder: per point MLP, pool, then a second MLP.

    The standard architecture for catalogue data, and what Deistler et al. call the
    baseline for i.i.d. observations. Input is (batch, n_points, n_features), or the
    same thing flattened, which is what sbi hands over after its own reshaping.
    """

    def __init__(self, n_points: int, n_features: int = 3, hidden: int = 64,
                 out_features: int = 32, pooling: str = "mean"):
        super().__init__()
        self.n_points, self.n_features = n_points, n_features
        self.pooling = _check_pooling(pooling)
        self.per_point = _mlp((n_features, hidden, hidden))
        self.head = _mlp((hidden, hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_points, self.n_features)
        h = self.per_point(x)
        if self.pooling == "mean":
            h = h.mean(dim=1)
        elif self.pooling == "sum":
            # Divided by a FIXED constant, not the per cloud count. Dividing by the
            # count would make this a mean; not dividing at all made training diverge
            # to R2 -13.7 earlier in this project when clouds held ~2500 points.
            h = h.sum(dim=1) / self.n_points
        else:
            h = h.max(dim=1).values
        return self.head(h)


class PointNetLite(nn.Module):
    """DeepSets with a per point residual from the pooled context.

    One round of message passing through a global summary, which lets a point's
    representation depend on the cloud it sits in. Cheaper than a radius graph and it
    needs no cutoff hyperparameter, which the graph version does.
    """

    def __init__(self, n_points: int, n_features: int = 3, hidden: int = 64,
                 out_features: int = 32, pooling: str = "mean"):
        super().__init__()
        self.n_points, self.n_features = n_points, n_features
        self.pooling = _check_pooling(pooling)
        self.encode = _mlp((n_features, hidden, hidden))
        self.mix = _mlp((2 * hidden, hidden, hidden))
        self.head = _mlp((hidden, hidden, out_features))

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        if self.pooling == "sum":
            return h.sum(dim=1) / self.n_points
        return h.max(dim=1).values          # "max", the only value left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_points, self.n_features)
        h = self.encode(x)
        context = self._pool(h).unsqueeze(1).expand(-1, self.n_points, -1)
        h = self.mix(torch.cat([h, context], dim=-1))
        return self.head(self._pool(h))


class FlattenMlp(nn.Module):
    """A plain MLP over the flattened cloud. Deliberately NOT permutation invariant.

    This is the control. If a set encoder does not beat it, then permutation
    invariance is not buying anything on this task, and the zoo should say so rather
    than assume the inductive bias helps.
    """

    def __init__(self, n_points: int, n_features: int = 3, hidden: int = 128,
                 out_features: int = 32):
        super().__init__()
        self.net = _mlp((n_points * n_features, hidden, hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


EMBEDDINGS = {"deepSets": DeepSets, "pointNetLite": PointNetLite,
              "flattenMlp": FlattenMlp}


# Positions are stored scaled to [0, 1] by box side. Allow a little slack for float
# error in the cache, and nothing like enough to hide a z-scored input, which lands
# near +/- 3.5.
_UNIT_BOX_TOLERANCE = 1e-3


POOLINGS = ("mean", "sum", "max")


def _check_pooling(pooling: str) -> str:
    """Reject an unknown pooling instead of falling through to one.

    DeepSets validated and the other two did not, so a typo or an unsupported value
    silently trained a max pooled network while the catalogue recorded whatever string
    was asked for. That is the worst kind of defect for a zoo: the entry is not the
    thing its own config says it is. Mean is the default everywhere because sum and
    max were measured to recover log N at probe R2 +0.914 and +0.897 while mean gives
    -0.662, so they can leak the point count.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    return pooling


def _to_unit_box(x: torch.Tensor) -> torch.Tensor:
    """Check the cloud is already in the unit box, and pass it through unchanged.

    SUPERSEDED 2026-08-30. This used to rescale by the batch minimum and maximum,
    because sbi z-scored x before the embedding saw it and ltu-ili would not pass
    `z_score_x='none'` through. sweep.build now calls sbi's posterior_nn factory
    directly with `z_score_x="none"`, so positions arrive in [0, 1] as stored and
    there is nothing left to undo.

    The rescale had to go because it read the extremes over the batch as well as over
    the points, which made the geometry depend on what else was in the batch. Training
    pooled extremes over 32 clouds and evaluation saw one cloud at a time, so a cloud
    was scaled differently in the two, by a factor of 1.003 to 1.006 per coordinate.

    That sounds negligible and is not, because the neighbour graph is discrete.
    Measured on camelsCloud: the k=16 neighbour sets differ for 32 of 32 clouds
    between batch-of-32 and batch-of-1 scaling, 6.7 per cent of neighbour slots flip
    to a different galaxy, and the edge vectors move by a relative L2 of 0.47. A
    negligible continuous perturbation produced a large discrete one.

    Raising rather than clamping is deliberate. If the framework ever starts scaling x
    again, the wrap below is silently wrong and the failure is invisible in the
    numbers, so it should stop the cell instead.
    """
    lo, hi = float(x.amin()), float(x.amax())
    if lo < -_UNIT_BOX_TOLERANCE or hi > 1.0 + _UNIT_BOX_TOLERANCE:
        raise ValueError(
            f"positions must arrive scaled to [0, 1], got [{lo:.4f}, {hi:.4f}]. The "
            f"periodic wrap in _periodic_offsets is only correct for a box of side 1, "
            f"so something upstream is rescaling x. Check that the estimator is built "
            f"with z_score_x='none'.")
    return x


def _periodic_offsets(x: torch.Tensor, k: int) -> torch.Tensor:
    """Offsets from each point to its k nearest neighbours, through a periodic box.

    The box wraps, so the shortest separation between two points is not always the
    direct one. Subtracting the rounded difference picks the wrapped copy, which is
    the minimum image convention.

    Returns (batch, n_points, k, 3). Only differences appear, never a coordinate, so
    anything built from this is translation invariant by construction.
    """
    x = _to_unit_box(x)
    d = x.unsqueeze(2) - x.unsqueeze(1)          # (b, n, n, 3)
    d = d - torch.round(d)                       # wrap into [-0.5, 0.5]
    dist = d.pow(2).sum(-1)
    # k + 1 because the nearest point to any point is itself, at distance zero.
    idx = dist.topk(k + 1, dim=-1, largest=False).indices[..., 1:]
    return torch.gather(d, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))


class PairwiseGnn(nn.Module):
    """Message passing over a k nearest neighbour graph built from relative offsets.

    Measured 2026-08-29: set encoders reading absolute positions score R2 -0.04 on a
    probe for Omega_m, because pooling per point features is a first moment statistic
    and clustering is a second moment one. A crude histogram of pairwise separations
    scored +0.2152 on the same clouds. This gives the network those separations with
    learnable features instead of fixed bins.
    """

    def __init__(self, n_points: int, n_features: int = 3, hidden: int = 64,
                 out_features: int = 32, k: int = 16, pooling: str = "mean"):
        super().__init__()
        if n_features != 3:
            raise ValueError("PairwiseGnn reads 3D positions, so n_features must be 3")
        if not 1 <= k < n_points:
            raise ValueError(f"k must be between 1 and n_points-1, got {k}")
        self.n_points, self.n_features, self.k = n_points, 3, k
        # Without this, pooling="sum" silently trained a max pooled network while the
        # catalogue recorded "sum". Same defect DeepSets already guarded against.
        self.pooling = _check_pooling(pooling)
        # Edge input is the offset and its length, so the network gets the scalar
        # separation directly rather than having to learn a norm.
        self.edge = _mlp((4, hidden, hidden))
        self.node = _mlp((hidden, hidden, hidden))
        self.head = _mlp((hidden, hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.n_points, self.n_features)
        off = _periodic_offsets(x, self.k)                       # (b, n, k, 3)
        r = off.pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        h = self.edge(torch.cat([off, r], dim=-1)).mean(dim=2)   # aggregate neighbours
        h = self.node(h)
        if self.pooling == "mean":
            h = h.mean(dim=1)
        elif self.pooling == "sum":
            h = h.sum(dim=1) / self.n_points
        else:
            h = h.max(dim=1).values
        return self.head(h)


EMBEDDINGS["pairwiseGnn"] = PairwiseGnn
