"""Learnable quasi-arithmetic pooling, the power mean member of the family.

Source: arXiv 2602.04941, Tokar and Sanner, "Improving Set Function
Approximation with Quasi-Arithmetic Neural Networks". The paper's central idea
is the Kolmogorov mean, also called the quasi-arithmetic mean,

    M_psi(X) = psi_inverse( (1/n) * sum_i psi(x_i) )

where psi is a learnable invertible scalar function. Fixed pooling forces the
encoder phi to produce whatever shape that one pooling operation happens to
need. Making psi learnable moves part of the approximation burden into the
pooling step itself.

WHY THE POWER MEAN AND NOT THE PAPER'S FULL psi
-----------------------------------------------
The paper implements psi as an invertible neural network (a RevNet block). This
file implements the single-parameter member psi(x) = x**p instead, which gives

    M_p(x) = ( (1/n) * sum_i x_i**p ) ** (1/p)

That is a deliberate reduction and it should be reported as one. arXiv 2602.04941
lists the power mean as a BASELINE ("Holder's Power DeepSets"), not as its
contribution, so a win here is not a reproduction of that paper's result. Three
reasons for starting at the reduced version anyway:

  1. It adds one parameter per feature channel, not a whole invertible network.
     On CAMELS, where a 49-parameter linear fit currently beats a 671k-parameter
     graph network, adding capacity is the thing that has repeatedly failed.
  2. p is directly readable. A trained p tells you which pooling the data wanted,
     which is a measurement, not just a score.
  3. It is the same operation as GeM pooling in image retrieval, arXiv 1711.02512
     section 3.2, which is a well-tested recipe with a learnable per-channel p.

The full invertible psi is the obvious follow-up if p alone moves the number.

DOES THE OUTPUT DEPEND ON THE NUMBER OF POINTS N
------------------------------------------------
SWITCHABLE. Count-blind by default.

  normalise="count" (default): divide by each cloud's own N. The result is then
    invariant to replicating the cloud, exactly as mean pooling is. There is
    no algebraic N in the result. Duplicating every galaxy moved the pooled
    vector by at most 6.4e-06 in the smoke test, which is float32 rounding on
    values of order 0.4 and not a residual count signal. This is the honest
    setting.

  normalise="fixed": divide by count_scale, a FIXED constant, the same
    convention as pool(..., "sum", count_scale) in point_clouds/pointnet.py.
    The output then carries a factor (N / count_scale) ** (1/p), so the galaxy
    count is readable straight off the pooled vector.

That switch is the leak channel. In CAMELS the galaxy count correlates 0.73 with
Omega_m, because a halo is only recorded once it reaches about 20 particles and
the particle mass depends on Omega_m, so a count-aware model can score well
without learning any clustering. Measured on this repo: GNN sum 0.8020 against
mean 0.6600 on CAMELS, and 0.5170 against 0.5196 on CAMELS-SAM where the count
is held fixed at 5000. The switch is exposed as the `depends_on_count` property
so an experiment has to name which side it is on.

At p = 1 the two settings collapse onto the existing operations exactly:
count-blind is mean pooling, count-aware is sum pooling divided by count_scale.
Both equalities are asserted in the smoke test against pointnet.pool.

One caveat that the algebra hides. Count-blind means there is no explicit N
factor: the output is a function of the empirical distribution of the features
and nothing else. It does not mean the output is statistically independent of N.
As p grows the power mean approaches the maximum, and the maximum of a larger
sample from a fixed distribution drifts upward. Plain max pooling has the same
property. So a large learned p is a partial route back to the count, and p is
worth reading off a trained model for that reason alone.

WHICH POOLINGS ARE REACHABLE
----------------------------
  p -> min_exponent (0.1)  geometric mean
  p = 1                    arithmetic mean, or sum under normalise="fixed"
  p large                  maximum

Minimum pooling, which needs p -> minus infinity, is NOT reachable and that is
on purpose. The power mean is singular at p = 0, so any parameterisation that
lets p change sign has to cross a point where 1/p blows up, and the optimiser
would have to traverse it. Worse, negative p is dominated by the smallest value
in the cloud, which under positive_map="clamp" is usually the clamp floor eps
rather than anything the data said. Both failure modes are avoided by keeping p
strictly positive.

Reaching a near-exact maximum needs p much larger than log(N), because the 1/N
factor contributes N ** (-1/p): at N = 2000 and p = 32 that factor is still
0.79. arXiv 2602.04941 states the same thing as an O(1 / log n) approximation
error for max-decomposable functions. max_exponent defaults to 32, which is a
strong soft maximum rather than a hard one. Raise it if a trained p saturates.

NUMERICAL STABILITY
-------------------
Computing sum(z ** p) directly overflows float32 for p of any size. Everything
here happens in log space:

    log M_p = mu + ( LSE_i( p * (log z_i - mu) ) - log D ) / p

with mu the per-cloud per-channel mean of log z, LSE the log-sum-exp, and D the
divisor chosen by the normalise switch. Two shifts do two different jobs. The
per-cloud maximum inside LSE stops exp from overflowing at large p. Centring by
mu stops the catastrophic cancellation at small p, where LSE(p*t) and log N
agree to more digits than float32 carries. The sum is accumulated with expm1 and
read back with log1p so that the small quantity is never buried inside a large
one. Both shifts are exact shifts, so detaching them leaves the value and every
gradient unchanged, including the gradient with respect to p.

The exponent itself is bounded by construction, p = min + (max - min) *
sigmoid(raw), so no clamp can be skipped and p can never reach 0.

MPS
---
Verified on torch 2.12.1, device mps: index_add_, scatter_reduce_ with amax,
expm1, log1p, and backward through the whole block. scatter_reduce_ is used
rather than index_reduce_ because the latter is flagged beta by PyTorch and
prints a warning. No operation here needs a CPU fallback. float64 is
unsupported on MPS and is not used.
"""

import argparse
import time
from typing import Union

import torch
import torch.nn as nn

NORMALISERS = ("count", "fixed")
POSITIVE_MAPS = ("clamp", "softplus")

Exponent = Union[torch.Tensor, float]


def _counts(index: torch.Tensor, n: int) -> torch.Tensor:
    """Number of points in each cloud, as an (n, 1) float tensor."""
    ones = torch.ones(len(index), 1, device=index.device)
    return torch.zeros(n, 1, device=index.device).index_add_(0, index, ones)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    return out.index_add_(0, index, values)


def _scatter_max(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Per-cloud maximum. Clouds with no points come back as 0, not -inf."""
    out = torch.full((n, values.shape[1]), -torch.inf,
                     device=values.device, dtype=values.dtype)
    out.scatter_reduce_(0, index[:, None].expand(-1, values.shape[1]), values,
                        reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def to_positive(values: torch.Tensor, how: str, eps: float) -> torch.Tensor:
    """Map features into the strictly positive domain the power mean needs.

    `clamp` is the identity wherever the input is already non-negative, which
    covers both models in this repo: DeepSets.phi and MessagePassingNet.update
    both end in a ReLU. It flattens every negative value onto eps, so use it
    only when negatives are absent or meaningless.

    `softplus` is monotone and invertible over the whole real line, so it keeps
    the quasi-arithmetic structure intact for a pre-pool activation that can go
    negative. The eps floor matters: softplus underflows to exactly 0 in float32
    below about -104, and log(0) would poison the whole cloud.
    """
    if how == "clamp":
        return values.clamp_min(eps)
    if how == "softplus":
        return nn.functional.softplus(values) + eps
    raise ValueError(f"positive_map must be one of {POSITIVE_MAPS}, got {how!r}")


def from_positive(values: torch.Tensor, how: str, eps: float) -> torch.Tensor:
    """Undo to_positive, so the block returns a mean in the original units.

    This is the outer psi_inverse of the quasi-arithmetic form. Inverse softplus
    is written as y + log1p(-exp(-y)) rather than log(expm1(y)) because the
    latter overflows for y above about 88 in float32.
    """
    if how == "clamp":
        return values
    if how == "softplus":
        y = (values - eps).clamp_min(eps)
        return y + torch.log1p(-torch.exp(-y))
    raise ValueError(f"positive_map must be one of {POSITIVE_MAPS}, got {how!r}")


def quasi_arithmetic_pool(values: torch.Tensor, index: torch.Tensor, n: int,
                          exponent: Exponent, normalise: str = "count",
                          count_scale: float = 1.0,
                          positive_map: str = "clamp",
                          eps: float = 1e-6) -> torch.Tensor:
    """Power mean of each cloud's points, in log space. Returns (n, feature_dim).

    Argument order matches pool() in point_clouds/pointnet.py so the two can be
    swapped at a call site. `exponent` broadcasts against the feature axis, so
    it may be a scalar, a (1,) tensor, or a (feature_dim,) tensor.
    """
    if normalise not in NORMALISERS:
        raise ValueError(f"normalise must be one of {NORMALISERS}, got {normalise!r}")
    if normalise == "fixed" and count_scale <= 0:
        raise ValueError("count_scale must be positive when normalise='fixed'")

    z = to_positive(values, positive_map, eps)
    log_z = torch.log(z)

    counts = _counts(index, n)
    safe_counts = counts.clamp_min(1.0)

    # Both shifts are exact, so detaching them changes no value and no gradient.
    centre = (_scatter_sum(log_z, index, n) / safe_counts).detach()
    shifted = exponent * (log_z - centre[index])
    top = _scatter_max(shifted.detach(), index, n)

    spread = _scatter_sum(torch.expm1(shifted - top[index]), index, n)
    log_mean = top + torch.log1p(spread / safe_counts)
    if normalise == "fixed":
        log_mean = log_mean + torch.log(safe_counts / count_scale)

    pooled = from_positive(torch.exp(centre + log_mean / exponent), positive_map, eps)
    return torch.where(counts > 0, pooled, torch.zeros_like(pooled))


class QuasiArithmeticPool(nn.Module):
    """Power mean pooling with a learnable exponent per feature channel.

    Drop-in for pool(values, index, n, how, count_scale). Costs feature_dim
    parameters when per_channel is True and 1 otherwise. arXiv 1711.02512
    section 5.2 measured a single shared p as the better of the two for image
    retrieval, so per_channel=False is worth running as a control rather than
    assumed to be worse.
    """

    def __init__(self, feature_dim: int, normalise: str = "count",
                 count_scale: float = 1.0, init_exponent: float = 1.0,
                 min_exponent: float = 0.1, max_exponent: float = 32.0,
                 per_channel: bool = True, positive_map: str = "clamp",
                 eps: float = 1e-6):
        super().__init__()
        if normalise not in NORMALISERS:
            raise ValueError(f"normalise must be one of {NORMALISERS}, got {normalise!r}")
        if positive_map not in POSITIVE_MAPS:
            raise ValueError(f"positive_map must be one of {POSITIVE_MAPS}, got {positive_map!r}")
        if not 0.0 < min_exponent < max_exponent:
            raise ValueError("need 0 < min_exponent < max_exponent")
        if not min_exponent < init_exponent < max_exponent:
            raise ValueError(
                f"init_exponent must lie strictly inside "
                f"({min_exponent}, {max_exponent}), got {init_exponent}")

        self.normalise = normalise
        self.count_scale = count_scale
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.positive_map = positive_map
        self.eps = eps

        fraction = (init_exponent - min_exponent) / (max_exponent - min_exponent)
        raw = torch.logit(torch.tensor(fraction))
        self.raw_exponent = nn.Parameter(
            torch.full((feature_dim if per_channel else 1,), float(raw)))

    @property
    def exponent(self) -> torch.Tensor:
        """p per channel, bounded in (min_exponent, max_exponent) by construction."""
        span = self.max_exponent - self.min_exponent
        return self.min_exponent + span * torch.sigmoid(self.raw_exponent)

    @property
    def depends_on_count(self) -> bool:
        """Whether the pooled vector carries the galaxy count. See module docstring."""
        return self.normalise == "fixed"

    @depends_on_count.setter
    def depends_on_count(self, value: bool) -> None:
        self.normalise = "fixed" if value else "count"

    def forward(self, values: torch.Tensor, index: torch.Tensor,
                n: int) -> torch.Tensor:
        return quasi_arithmetic_pool(values, index, n, self.exponent,
                                     self.normalise, self.count_scale,
                                     self.positive_map, self.eps)

    def extra_repr(self) -> str:
        return (f"normalise={self.normalise}, count_scale={self.count_scale}, "
                f"depends_on_count={self.depends_on_count}, "
                f"exponent_range=({self.min_exponent}, {self.max_exponent}), "
                f"positive_map={self.positive_map}")


# ----------------------------------------------------------------------------
# Smoke test. Run with: python -m point_clouds.blocks.quasi_arithmetic
# ----------------------------------------------------------------------------

def make_batch(n_clouds: int, points_per_cloud: int, feature_dim: int,
               device: torch.device):
    """Post-ReLU-looking features, so about half the entries are exactly zero.

    That is the hard case for a log-space pooling, because every zero becomes
    log(eps) and sits about 14 below the rest of the cloud.
    """
    total = n_clouds * points_per_cloud
    values = torch.relu(torch.randn(total, feature_dim, device=device))
    index = torch.repeat_interleave(
        torch.arange(n_clouds, device=device), points_per_cloud)
    return values, index


def check_shape_and_finiteness(values, index, n_clouds, feature_dim, device):
    block = QuasiArithmeticPool(feature_dim).to(device)
    out = block(values, index, n_clouds)
    print(f"  output shape        {tuple(out.shape)}  expected ({n_clouds}, {feature_dim})")
    print(f"  output device       {out.device}")
    print(f"  all finite          {bool(torch.isfinite(out).all())}")
    print(f"  output min/mean/max {out.min():.6f} / {out.mean():.6f} / {out.max():.6f}")
    print(f"  parameters          {sum(p.numel() for p in block.parameters())}")
    print(f"  depends_on_count    {block.depends_on_count}")
    print(f"  repr                {block}")
    assert out.shape == (n_clouds, feature_dim)
    assert torch.isfinite(out).all()
    assert out.device.type == device.type
    return out


def check_special_cases(values, index, n_clouds, count_scale):
    """p = 1 must reproduce the existing mean and sum pooling exactly."""
    from point_clouds.pointnet import pool

    got_mean = quasi_arithmetic_pool(values, index, n_clouds, 1.0, "count")
    want_mean = pool(values, index, n_clouds, "mean")
    gap_mean = (got_mean - want_mean).abs().max().item()

    got_sum = quasi_arithmetic_pool(values, index, n_clouds, 1.0, "fixed",
                                    count_scale=count_scale)
    want_sum = pool(values, index, n_clouds, "sum", count_scale=count_scale)
    gap_sum = (got_sum - want_sum).abs().max().item()

    print(f"  p=1 count-blind vs pointnet mean, max abs gap  {gap_mean:.3e}")
    print(f"  p=1 count-aware vs pointnet sum,  max abs gap  {gap_sum:.3e}")
    assert gap_mean < 1e-5 and gap_sum < 1e-4


def check_monotone_in_p(values, index, n_clouds):
    """mean <= M_2 <= M_8 <= M_32 <= max, elementwise. True for any data."""
    from point_clouds.pointnet import pool

    ladder = {p: quasi_arithmetic_pool(values, index, n_clouds, float(p), "count")
              for p in (1, 2, 8, 32)}
    top = pool(values, index, n_clouds, "max")
    steps = [ladder[1], ladder[2], ladder[8], ladder[32], top]
    names = ["M_1 (mean)", "M_2", "M_8", "M_32", "max"]
    for name, step in zip(names, steps):
        print(f"  {name:<11} mean over clouds and channels  {step.mean():.6f}")
    for lower, upper, a, b in zip(steps, steps[1:], names, names[1:]):
        worst = (lower - upper).max().item()
        assert worst < 1e-5, f"{a} exceeded {b} by {worst}"
    print(f"  M_32 / max ratio    {(ladder[32] / top.clamp_min(1e-9)).mean():.4f}"
          f"  (a soft maximum, not a hard one)")


def check_count_dependence(values, index, n_clouds, count_scale):
    """The leak test. Replicating every point must not move a count-blind pool."""
    doubled_values = torch.cat([values, values])
    doubled_index = torch.cat([index, index])
    p = 2.0

    blind = quasi_arithmetic_pool(values, index, n_clouds, p, "count")
    blind_doubled = quasi_arithmetic_pool(doubled_values, doubled_index, n_clouds,
                                          p, "count")
    drift = (blind - blind_doubled).abs().max().item()

    aware = quasi_arithmetic_pool(values, index, n_clouds, p, "fixed",
                                  count_scale=count_scale)
    aware_doubled = quasi_arithmetic_pool(doubled_values, doubled_index, n_clouds,
                                          p, "fixed", count_scale=count_scale)
    ratio = (aware_doubled / aware.clamp_min(1e-9)).mean().item()

    print(f"  count-blind drift on replication   {drift:.3e}  expected 0")
    print(f"  count-aware ratio on replication   {ratio:.6f}  expected "
          f"2**(1/p) = {2 ** (1 / p):.6f}")
    assert drift < 1e-5
    assert abs(ratio - 2 ** (1 / p)) < 1e-3


def check_gradients(values, index, n_clouds, feature_dim, device):
    block = QuasiArithmeticPool(feature_dim).to(device)
    leaf = values.clone().requires_grad_(True)
    block(leaf, index, n_clouds).square().mean().backward()
    input_grad = leaf.grad
    exponent_grad = block.raw_exponent.grad
    print(f"  input grad finite   {bool(torch.isfinite(input_grad).all())}"
          f"  abs mean {input_grad.abs().mean():.3e}")
    print(f"  exponent grad finite {bool(torch.isfinite(exponent_grad).all())}"
          f"  abs mean {exponent_grad.abs().mean():.3e}")
    assert torch.isfinite(input_grad).all() and torch.isfinite(exponent_grad).all()
    assert exponent_grad.abs().max() > 0, "p would never move"


def check_softplus_path(values, index, n_clouds):
    """The mode for features that can go negative."""
    signed = values - 0.5
    out = quasi_arithmetic_pool(signed, index, n_clouds, 2.0, "count",
                                positive_map="softplus")
    print(f"  softplus mode finite {bool(torch.isfinite(out).all())}"
          f"  min {out.min():.6f}  max {out.max():.6f}")
    round_trip = from_positive(to_positive(signed, "softplus", 1e-6),
                               "softplus", 1e-6)
    print(f"  softplus round trip max abs gap  "
          f"{(round_trip - signed).abs().max():.3e}")
    assert torch.isfinite(out).all()
    assert (round_trip - signed).abs().max() < 1e-3


def check_guards(feature_dim, device):
    """Every guard in this file, deliberately tripped."""
    tripped = []

    def expect_error(label, call):
        try:
            call()
        except ValueError as error:
            tripped.append(f"{label}: {error}")
        else:
            raise AssertionError(f"guard {label} did not fire")

    expect_error("bad normalise",
                 lambda: QuasiArithmeticPool(feature_dim, normalise="average"))
    expect_error("bad positive_map",
                 lambda: QuasiArithmeticPool(feature_dim, positive_map="abs"))
    expect_error("init outside range",
                 lambda: QuasiArithmeticPool(feature_dim, init_exponent=99.0))
    expect_error("bad exponent range",
                 lambda: QuasiArithmeticPool(feature_dim, min_exponent=0.0))
    expect_error("bad count_scale", lambda: quasi_arithmetic_pool(
        torch.ones(4, feature_dim, device=device),
        torch.zeros(4, dtype=torch.long, device=device), 1, 1.0, "fixed",
        count_scale=0.0))
    expect_error("bad positive_map, functional",
                 lambda: to_positive(torch.ones(1, 1, device=device), "abs", 1e-6))
    for line in tripped:
        print(f"  tripped {line}")

    # An empty cloud must pool to zeros rather than to 1.
    values = torch.rand(6, feature_dim, device=device)
    index = torch.tensor([0, 0, 0, 2, 2, 2], dtype=torch.long, device=device)
    out = quasi_arithmetic_pool(values, index, 3, 2.0, "fixed", count_scale=3.0)
    print(f"  empty cloud row     {out[1].abs().max():.6f}  expected 0.000000")
    print(f"  neighbours nonzero  {bool((out[0].abs().sum() > 0) and (out[2].abs().sum() > 0))}")
    assert out[1].abs().max() == 0


def check_speed(values, index, n_clouds, feature_dim, device, repeats: int = 50):
    from point_clouds.pointnet import pool

    block = QuasiArithmeticPool(feature_dim).to(device)

    def timed(call) -> float:
        for _ in range(5):
            call()
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        if device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - start) / repeats * 1e3

    mean_ms = timed(lambda: pool(values, index, n_clouds, "mean"))
    ours_ms = timed(lambda: block(values, index, n_clouds))
    print(f"  mean pooling        {mean_ms:.3f} ms per call")
    print(f"  quasi-arithmetic    {ours_ms:.3f} ms per call"
          f"  ({ours_ms / mean_ms:.1f}x, forward only, n={repeats})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clouds", type=int, default=32)
    parser.add_argument("--points", type=int, default=2000)
    parser.add_argument("--features", type=int, default=64)
    args = parser.parse_args()
    if min(args.clouds, args.points, args.features) < 1:
        raise SystemExit("clouds, points and features must all be at least 1")

    from common.metrics import resolve_device, seed_all

    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"torch {torch.__version__}  device {device}  seed {args.seed}")
    print(f"batch {args.clouds} clouds x {args.points} points x "
          f"{args.features} features")

    values, index = make_batch(args.clouds, args.points, args.features, device)
    count_scale = float(args.points)

    print("\n[1] shape, device, finiteness")
    check_shape_and_finiteness(values, index, args.clouds, args.features, device)
    print("\n[2] p = 1 reproduces the existing pooling exactly")
    check_special_cases(values, index, args.clouds, count_scale)
    print("\n[3] monotone in p, between mean and max")
    check_monotone_in_p(values, index, args.clouds)
    print("\n[4] count dependence, the leak switch")
    check_count_dependence(values, index, args.clouds, count_scale)
    print("\n[5] gradients")
    check_gradients(values, index, args.clouds, args.features, device)
    print("\n[6] softplus positivity mode")
    check_softplus_path(values, index, args.clouds)
    print("\n[7] guards")
    check_guards(args.features, device)
    print("\n[8] cost against mean pooling")
    check_speed(values, index, args.clouds, args.features, device)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
