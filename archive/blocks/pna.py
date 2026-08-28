"""Principal Neighbourhood Aggregation, from arXiv:2004.05718 (Corso et al. 2020).

WHY THIS BLOCK EXISTS
---------------------
A single aggregator throws information away. The paper's argument is a counting
one: over a neighbourhood of n values, one aggregator can only ever return one
number, so there are pairs of neighbourhoods it cannot tell apart. Mean loses
spread, max loses everything below the top, and so on. PNA runs several
aggregators side by side and concatenates them, so the pair that fools mean is
usually separated by max or by standard deviation.

The second half of PNA is the DEGREE SCALER. Aggregating a neighbourhood of 3
and a neighbourhood of 300 with the same mean returns the same vector, so the
size of the neighbourhood is erased. PNA multiplies each aggregated vector by

    S(d, alpha) = (log(d + 1) / delta) ** alpha,   alpha in {1, 0, -1}

where d is the degree (for a graph-level readout, the number of points in the
cloud) and delta is the average of log(d + 1) over the TRAINING set, a fixed
constant passed in by the caller. alpha = 1 amplifies with size, alpha = 0 is
the plain aggregate, alpha = -1 attenuates. Three scalers times four
aggregators gives twelve copies of the feature dimension.

DOES THE OUTPUT DEPEND ON THE NUMBER OF POINTS N?
-------------------------------------------------
SWITCHABLE, via the constructor flag `use_degree_scalers`.

    use_degree_scalers=True  (default)  -> YES, by construction.
    use_degree_scalers=False            -> count-blind, with one caveat below.

This is the load-bearing property of this block for KAAI, so it is stated
loudly rather than buried. In CAMELS the galaxy count per cloud correlates 0.73
with Omega_m because a halo is only recorded once it reaches about 20 particles
and the particle mass depends on Omega_m. Measured on this repo: a GNN with sum
pooling scores 0.8020 on CAMELS Omega_m against 0.6600 with mean pooling, and
the gap vanishes on CAMELS-SAM where the count is fixed at 5000. So a block that
reads N is not automatically a better block, it is a block with access to a
resolution artifact.

Degree scalers are exactly that access, written down explicitly. With
use_degree_scalers=True the readout multiplies every summary by a deterministic
function of N, which is a strictly richer channel than sum pooling: sum pooling
gives the network N times the mean, while PNA gives it log(N + 1) raised to
three different powers times four different summaries. Both variants are meant
to be measured, and the difference between them is the size of the leak this
block is exploiting.

HONEST CAVEAT ON THE COUNT-BLIND VARIANT
----------------------------------------
With scalers off, mean and standard deviation are unbiased with respect to N,
but max and min are not: the maximum of N draws from a fixed distribution grows
with N (extreme value statistics), so a residual, weak N-dependence survives.

Measured by the smoke test below, halving N from 2000 to 1000 on Gaussian
features, single run, one seed. The number is the factor each aggregator's block
mean is multiplied by:

    mean 0.9995, std 0.9996, max 0.9557, min 0.9237

So mean and standard deviation move by well under a tenth of a percent while max
and min move by 4 to 8 percent. Set aggregators=("mean", "std") for a strictly
count-blind readout. This is a statistical fact about order statistics, not a
property of this implementation, and the size of the residual on real CAMELS
clouds has not been measured.

MPS SUPPORT
-----------
Everything here is index_add_ and scatter_reduce, both of which run natively on
MPS in torch 2.12.1 and backpropagate correctly (verified against CPU on
2026-08-24). No fallback path is needed. index_reduce_ is avoided because it is
still flagged beta on MPS.

WIDTH, WHICH THE CALLER MUST HANDLE
-----------------------------------
pool returns (n_clouds, hidden). PNAReadout returns (n_clouds, out_features),
which for hidden 64 with four aggregators and three scalers is 768, or 256 with
the scalers off. The block itself holds no learnable parameters, so the cost is
entirely in the first layer of whatever MLP follows it. Size that layer from
readout.out_features rather than from hidden.

EMPTY SEGMENTS
--------------
A cloud with no points, or an isolated node with no incoming edges, returns
zeros from every aggregator rather than an infinity, and its degree is clamped
to 1 before the scaler so that alpha = -1 cannot divide by zero. The clamp
matches the reference PyTorch Geometric implementation of PNAConv.
"""

import math
import time
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

AGGREGATORS: Tuple[str, ...] = ("mean", "max", "min", "std")

# The subset that carries no recoverable channel for N. Measured R2(N) of
# -2.5890 against +0.7006 for the full set. Also about 7x cheaper.
COUNT_BLIND_AGGREGATORS: Tuple[str, ...] = ("mean", "std")
SCALER_ALPHAS: Tuple[float, ...] = (1.0, 0.0, -1.0)

# Variance from E[x^2] - E[x]^2 can land a hair below zero in float32 when the
# spread is tiny relative to the mean, which is the usual case for pooled
# hidden units. Clamping before the square root keeps the gradient finite.
_VARIANCE_FLOOR = 1e-12


def average_log_degree(degrees: torch.Tensor) -> float:
    """delta of PNA equation (5): the mean of log(d + 1) over the training set.

    Pass the per-node degrees for message passing, or the per-cloud point counts
    for a graph-level readout. It must be computed on TRAIN only and then held
    fixed, exactly like a normalisation constant, otherwise the scaler leaks
    test set statistics into the model.
    """
    d = torch.as_tensor(degrees, dtype=torch.float32).clamp_min(1.0)
    return float(torch.log(d + 1.0).mean())


def _segment_counts(index: torch.Tensor, n: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    ones = torch.ones(index.shape[0], 1, device=device, dtype=dtype)
    return torch.zeros(n, 1, device=device, dtype=dtype).index_add_(0, index, ones)


def _segment_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    return out.index_add_(0, index, values)


def _segment_extreme(values: torch.Tensor, index: torch.Tensor, n: int,
                     reduce: str) -> torch.Tensor:
    """amax or amin per segment, with empty segments returning zero.

    include_self=False leaves untouched rows at their initial value, so seeding
    with zeros avoids the infinities that a -inf seed would leave behind.
    """
    out = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    target = index.unsqueeze(1).expand(-1, values.shape[1])
    return out.scatter_reduce(0, target, values, reduce=reduce, include_self=False)


def aggregate_segments(values: torch.Tensor, index: torch.Tensor, n: int,
                       aggregators: Sequence[str] = AGGREGATORS) -> torch.Tensor:
    """Several aggregators over the same segments, concatenated.

    values is (total, feature_dim), index is (total,) saying which of the n
    segments each row belongs to. Returns (n, feature_dim * len(aggregators)),
    blocks laid out in the order the aggregators were given.
    """
    if values.dim() != 2:
        raise ValueError(f"values must be (total, feature_dim), got {tuple(values.shape)}")
    unknown = [a for a in aggregators if a not in AGGREGATORS]
    if unknown:
        raise ValueError(f"unknown aggregators {unknown}, expected from {AGGREGATORS}")
    if len(aggregators) == 0:
        raise ValueError("at least one aggregator is required")

    counts = _segment_counts(index, n, values.device, values.dtype).clamp_min(1.0)
    mean = _segment_sum(values, index, n) / counts

    parts = []
    for how in aggregators:
        if how == "mean":
            parts.append(mean)
        elif how == "max":
            parts.append(_segment_extreme(values, index, n, "amax"))
        elif how == "min":
            parts.append(_segment_extreme(values, index, n, "amin"))
        elif how == "std":
            mean_square = _segment_sum(values * values, index, n) / counts
            variance = (mean_square - mean * mean).clamp_min(_VARIANCE_FLOOR)
            parts.append(torch.sqrt(variance))
    return torch.cat(parts, dim=1)


def degree_scalers(degree: torch.Tensor, delta: Union[float, torch.Tensor],
                   alphas: Sequence[float] = SCALER_ALPHAS) -> torch.Tensor:
    """S(d, alpha) for each alpha, as (n, len(alphas)).

    degree is (n,) or (n, 1). Degrees below 1 are clamped so that the
    attenuating scaler alpha = -1 stays finite for isolated nodes.
    """
    d = degree.reshape(-1).to(torch.float32).clamp_min(1.0)
    base = torch.log(d + 1.0) / delta
    return torch.stack([base ** float(a) for a in alphas], dim=1)


def apply_degree_scalers(aggregated: torch.Tensor, degree: torch.Tensor,
                         delta: Union[float, torch.Tensor],
                         alphas: Sequence[float] = SCALER_ALPHAS) -> torch.Tensor:
    """Multiply every aggregated column by every scaler.

    (n, C) times (n, A) gives (n, C * A), with the A scaler variants of one
    column sitting next to each other.
    """
    scaled = aggregated.unsqueeze(2) * degree_scalers(degree, delta, alphas).unsqueeze(1)
    return scaled.flatten(1)


def pna_neighbour_aggregate(messages: torch.Tensor, target: torch.Tensor,
                            n_nodes: int, delta: Union[float, torch.Tensor],
                            aggregators: Sequence[str] = AGGREGATORS,
                            use_degree_scalers: bool = True,
                            alphas: Sequence[float] = SCALER_ALPHAS) -> torch.Tensor:
    """PNA over the edges of a graph, for use inside a message passing layer.

    messages is (n_edges, feature_dim), one message per directed edge, and
    target is edges[1] from the repo's Batch, so target[e] is the node that
    receives message e. Returns (n_nodes, feature_dim * len(aggregators) *
    len(alphas)) when scalers are on, and one factor of len(alphas) smaller when
    they are off.

    Here d is the node's in-degree, which is a local density, not the cloud's
    galaxy count. It is still not innocent: mean degree inside a fixed cutoff
    radius is proportional to number density, so a per-node degree scaler leaks
    the same count information the readout does, only spread across nodes. Turn
    it off here as well when running the count-blind arm.
    """
    aggregated = aggregate_segments(messages, target, n_nodes, aggregators)
    if not use_degree_scalers:
        return aggregated
    degree = _segment_counts(target, n_nodes, messages.device, messages.dtype)
    return apply_degree_scalers(aggregated, degree, delta, alphas)


class PNAReadout(nn.Module):
    """Graph-level PNA pooling, a drop-in replacement for pointnet.pool.

    Call it as readout(values, index, n) with the same arguments pool takes,
    minus the `how` and `count_scale` that PNA replaces. The one difference the
    caller must handle is width: pool returns (n, feature_dim) while this
    returns (n, out_features), so size the following MLP from `out_features`
    rather than from the hidden dimension.

    delta is stored as a buffer so it travels with .to(device) and lands in the
    state dict, which makes a saved model carry the training set constant it was
    fitted with instead of silently taking whatever the next caller passes.

    use_degree_scalers=False is the COUNT-BLIND variant. See the module
    docstring: this flag is the experiment, not a tuning knob.
    """

    def __init__(self, feature_dim: int, delta: float,
                 aggregators: Sequence[str] = AGGREGATORS,
                 use_degree_scalers: bool = True,
                 alphas: Sequence[float] = SCALER_ALPHAS):
        super().__init__()
        if delta <= 0.0:
            raise ValueError(
                f"delta must be positive, got {delta}. It is the training set mean of "
                "log(d + 1), so compute it with average_log_degree.")
        if use_degree_scalers and len(alphas) == 0:
            raise ValueError("use_degree_scalers=True needs at least one alpha")
        unknown = [a for a in aggregators if a not in AGGREGATORS]
        if unknown:
            raise ValueError(f"unknown aggregators {unknown}, expected from {AGGREGATORS}")

        self.feature_dim = feature_dim
        self.aggregators = tuple(aggregators)
        self.alphas = tuple(float(a) for a in alphas)
        self.use_degree_scalers = use_degree_scalers
        self.register_buffer("delta", torch.tensor(float(delta)))
        width = feature_dim * len(self.aggregators)
        self.out_features = width * len(self.alphas) if use_degree_scalers else width

    def depends_on_point_count(self) -> bool:
        """True when the output carries a recoverable channel for N.

        Degree scalers are the obvious route, but NOT the only one. The `max`
        and `min` aggregators leak N on their own, because the extreme of a
        sample grows with how many draws you take. That is invisible to a
        duplication test, since duplicating every point cannot move an extreme,
        which is how it survived the first review.

        MEASURED 2026-08-24 with blocks/count_screen.py, held-out probe
        recovering log N from the readout, 3 seeds:
            scalers ON,  aggregators (mean, max, min, std)   R2(N) = +0.9969
            scalers OFF, aggregators (mean, max, min, std)   R2(N) = +0.7006
            scalers OFF, aggregators (mean, std)             R2(N) = -2.5890
        The middle row is the one this method used to report as count-blind. An
        R2 of 0.70 for recovering N is a stronger channel than the 0.73
        correlation between count and Omega_m that this project exists to guard
        against.
        """
        return self.use_degree_scalers or bool(
            {"max", "min"} & set(self.aggregators))

    def extra_repr(self) -> str:
        return (f"feature_dim={self.feature_dim}, out_features={self.out_features}, "
                f"aggregators={self.aggregators}, "
                f"use_degree_scalers={self.use_degree_scalers}, "
                f"alphas={self.alphas}, delta={float(self.delta):.4f}")

    def forward(self, values: torch.Tensor, index: torch.Tensor,
                n: int) -> torch.Tensor:
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim {self.feature_dim}, got {values.shape[1]}")
        aggregated = aggregate_segments(values, index, n, self.aggregators)
        if not self.use_degree_scalers:
            return aggregated
        counts = _segment_counts(index, n, values.device, values.dtype)
        return apply_degree_scalers(aggregated, counts, self.delta, self.alphas)


def _smoke_test(n_clouds: int = 32, points_per_cloud: int = 2000,
                feature_dim: int = 64, seed: int = 0) -> None:
    """Shapes, finiteness, device, the N-dependence claim, and every guard.

    Run from the repo root with: python -m point_clouds.blocks.pna
    """
    from common.metrics import resolve_device, seed_all

    seed_all(seed)
    device = resolve_device("auto")
    total = n_clouds * points_per_cloud
    # Offset to mean 1 so that a relative change is a meaningful quantity. On
    # zero-centred features the per-cloud mean is itself near zero and any ratio
    # against it is dominated by its own sampling noise rather than by N.
    values = torch.randn(total, feature_dim, device=device) + 1.0
    index = torch.arange(n_clouds, device=device).repeat_interleave(points_per_cloud)
    counts = torch.full((n_clouds,), float(points_per_cloud))
    delta = average_log_degree(counts)

    print(f"device                 {device}")
    print(f"input                  {tuple(values.shape)} over {n_clouds} clouds")
    print(f"delta                  {delta:.4f}")

    scaled = PNAReadout(feature_dim, delta).to(device)
    blind = PNAReadout(feature_dim, delta, use_degree_scalers=False).to(device)
    out_scaled, out_blind = scaled(values, index, n_clouds), blind(values, index, n_clouds)
    for name, module, out in (("scalers on", scaled, out_scaled),
                              ("scalers off", blind, out_blind)):
        print(f"readout {name:<12}  {tuple(out.shape)}  out_features="
              f"{module.out_features}  finite={bool(torch.isfinite(out).all())}  "
              f"depends_on_N={module.depends_on_point_count()}  "
              f"range=[{float(out.min()):.4f}, {float(out.max()):.4f}]")

    # The scaler ratio has a closed form, so it is checked against arithmetic
    # rather than against another run of the same code. Halving N must multiply
    # the alpha block by (log(N/2 + 1) / log(N + 1)) ** alpha.
    half = points_per_cloud // 2
    ratio = math.log(half + 1.0) / math.log(points_per_cloud + 1.0)
    factors = degree_scalers(torch.tensor([float(half)]), delta)[0] / \
        degree_scalers(torch.tensor([float(points_per_cloud)]), delta)[0]
    for alpha, measured in zip(scaled.alphas, factors.tolist()):
        print(f"scaler alpha={alpha:+.0f}       halving N multiplies by {measured:.5f}, "
              f"closed form {ratio ** alpha:.5f}")

    # With scalers off, each aggregator is checked separately, because mean and
    # standard deviation are unbiased in N while max and min are not. The signed
    # ratio of block means is the right statistic: sampling noise cancels in it
    # and a systematic N-dependence does not.
    keep = (torch.arange(total, device=device) % points_per_cloud) < half
    half_blind = blind(values[keep], index[keep], n_clouds)
    for slot, name in enumerate(blind.aggregators):
        columns = slice(slot * feature_dim, (slot + 1) * feature_dim)
        before, after = out_blind[:, columns].mean(), half_blind[:, columns].mean()
        print(f"scalers off, {name:<4}      halving N multiplies the block mean by "
              f"{float(after / before):.4f}")

    # Message passing path, on a random graph with an uneven degree distribution.
    n_nodes, n_edges = 20_000, 400_000
    generator = torch.Generator(device="cpu").manual_seed(seed)
    edges = torch.randint(0, n_nodes, (2, n_edges), generator=generator).to(device)
    messages = torch.randn(n_edges, feature_dim, device=device)
    node_delta = average_log_degree(torch.bincount(edges[1].cpu(), minlength=n_nodes))
    neighbour = pna_neighbour_aggregate(messages, edges[1], n_nodes, torch.tensor(node_delta))
    print(f"neighbour aggregate    {tuple(neighbour.shape)}  "
          f"finite={bool(torch.isfinite(neighbour).all())}  node delta={node_delta:.4f}")

    # Gradients must survive the max and min aggregators on MPS.
    leaf = values.detach().clone().requires_grad_(True)
    scaled(leaf, index, n_clouds).sum().backward()
    print(f"backward               grad finite={bool(torch.isfinite(leaf.grad).all())}  "
          f"nonzero fraction={float((leaf.grad != 0).float().mean()):.4f}")

    # MPS against CPU, because a silently wrong kernel is the failure this misses.
    cpu_out = PNAReadout(feature_dim, delta)(values.cpu(), index.cpu(), n_clouds)
    print(f"mps vs cpu             max abs difference "
          f"{float((out_scaled.cpu() - cpu_out).abs().max()):.3e}")

    # Cost of the pooling step alone, against the mean pooling it replaces.
    def timed(fn, repeats: int = 20) -> float:
        synchronise = torch.mps.synchronize if device.type == "mps" else (lambda: None)
        fn()
        synchronise()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        synchronise()
        return 1e3 * (time.perf_counter() - start) / repeats

    plain_mean = lambda: _segment_sum(values, index, n_clouds) / _segment_counts(
        index, n_clouds, device, values.dtype).clamp_min(1.0)
    print(f"forward cost           mean pooling {timed(plain_mean):.2f} ms, "
          f"PNA readout {timed(lambda: scaled(values, index, n_clouds)):.2f} ms")

    for description, thunk in (
            ("delta <= 0", lambda: PNAReadout(feature_dim, 0.0)),
            ("unknown aggregator", lambda: PNAReadout(feature_dim, delta,
                                                      aggregators=("median",))),
            ("no aggregators", lambda: aggregate_segments(values, index, n_clouds, ())),
            ("no alphas", lambda: PNAReadout(feature_dim, delta, alphas=())),
            ("1-D values", lambda: aggregate_segments(values[:, 0], index, n_clouds)),
            ("wrong feature dim", lambda: scaled(values[:, :8], index, n_clouds))):
        try:
            thunk()
            print(f"GUARD NOT TRIPPED      {description}")
        except ValueError as error:
            print(f"guard tripped          {description}: {error}")


if __name__ == "__main__":
    _smoke_test()
