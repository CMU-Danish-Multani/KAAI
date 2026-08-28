"""FISHNETS aggregation, from arXiv:2310.03812 (Makinen, Alsing and Wandelt 2024).

WHY THIS BLOCK EXISTS
---------------------
Mean pooling gives every galaxy exactly the same vote. That is the right thing
to do when every galaxy is equally informative, and the wrong thing to do when
some are informative and some are noise. A galaxy sitting in a dense filament
and a galaxy alone in a void both push the summary by the same amount, so the
useful one gets diluted by the useless ones.

FISHNETS lets each galaxy cast a vote AND state how strongly it believes it,
then takes a confidence weighted average. The confidence is not invented: it is
Fisher information, the standard statistical measure of how sharply a single
observation pins down a parameter.

THE STATISTICS, PLAINLY
-----------------------
For independent observations the log likelihood adds up, so its first
derivative (the score, which says which way the parameter should move) and its
second derivative (the Fisher information, which says how confident that move
is) also add up. The paper's equations (7), (8) and (6):

    t_total = sum_i t_i          per element scores add
    F_total = sum_i F_i          per element Fisher information adds
    theta   = F_total^-1 t_total + c

The last line is one step of Fisher scoring, and for a Gaussian model it is the
exact maximum likelihood estimate in a single step. When the likelihood is not
known, paper equations (9) and (10) replace t_i and F_i with twin neural
networks, and paper equation (15) uses the same ratio as a drop in aggregator
inside a graph network.

THE REPARAMETERISATION USED HERE
--------------------------------
The paper's score head outputs t_i directly. This module has the score head
output a per element LOCATION m_i instead, and forms t_i = F_i m_i internally,
so the aggregate reads

    out = (sum_i F_i)^-1 (sum_i F_i m_i)

which is algebraically the paper's equation (11) with t_i := F_i m_i, and is
the same object the paper's own linear regression score in equation (27) has:
there t_i is a residual divided by sigma_i squared, that is, precision times a
location. Writing it this way makes the output a genuine precision weighted
mean, so it is bounded by the smallest and largest m_i and cannot blow up when
the total precision happens to be small. Set score_form="raw" for the literal
paper parameterisation, where the score head output is summed unweighted.

F is DIAGONAL here, not a full Cholesky factor. The paper's Appendix B builds
a full matrix as F = L L^T with softplus on the diagonal of L. That costs
n_p (n_p + 1) / 2 outputs and a linear solve per cloud. A diagonal F costs n_p
outputs and an elementwise divide, needs no matrix factorisation at all, and
therefore has no MPS coverage problem (see MPS below). The cost is that the
score components cannot trade information with each other inside the
aggregation step; the rho network downstream can still mix them.

DOES THE OUTPUT DEPEND ON THE NUMBER OF POINTS N?
-------------------------------------------------
SWITCHABLE, and readable at runtime as the property `depends_on_count`.

    expose_total_fisher=False, prior_mode="per_element"  (defaults) -> NO.
    expose_total_fisher=True                                        -> YES.
    prior_mode="total"                                              -> YES, weakly.

This is the load bearing property of this block for KAAI, so it is stated
loudly rather than buried. In CAMELS the galaxy count per cloud correlates 0.73
with Omega_m, because a halo is only recorded once it reaches about 20
particles and the particle mass depends on Omega_m. Measured on this repo: a
GNN with sum pooling scores 0.8020 on CAMELS Omega_m against 0.6600 with mean
pooling, and that gap collapses to 0.5170 against 0.5196 on CAMELS-SAM where
the count is fixed at 5000. A block that reads N is therefore not a better
block, it is a block with access to a resolution artifact.

Why the default is count blind. Numerator and denominator are both sums over
the same N elements, so the ratio is homogeneous of degree zero: duplicate
every point in a cloud and both sums double and the answer does not move. This
is checked numerically in the smoke test below rather than asserted.

Why exposing F_total is NOT count blind. F_total = sum_i F_i grows linearly
with N, so handing it downstream hands the network log N in clean, low noise
form. The module concatenates log1p(F_total) rather than F_total itself,
because F_total reaches about 2000 for a CAMELS cloud and would swamp a linear
layer initialised for inputs near 1. log1p is monotone, so this compresses the
magnitude without removing any of the count information.

Why prior_mode="total" is weakly count dependent. Paper equation (28) adds a
prior precision C_p^-1 once to the SUMMED Fisher. That makes the estimate
(Fbar + prior / N)^-1 tbar, which shrinks toward zero for small N and stops
shrinking for large N, so N is faintly readable. The default here instead adds
the prior to EACH element, which keeps the ratio homogeneous and makes it a
floor on each element's weight rather than a Bayesian prior. Use "total" only
when reproducing the paper exactly.

MPS
---
Everything here runs on MPS. The operations used are linear layers, softplus,
elementwise multiply and divide, index_add_ and log1p, all of which have MPS
kernels; index_add_ is already the pooling primitive in point_clouds/pointnet.py
and point_clouds/gnn.py. The full Cholesky variant of the paper would need
torch.linalg.cholesky_solve or torch.linalg.solve, whose MPS coverage has been
patchy, which is a second reason the diagonal form is the default here. If a
full matrix version is ever wanted, run that single op on CPU and move back.

Two numerical notes. MPS is float32 only, so there is no float64 fallback for
the divide, and index_add_ accumulation order is not guaranteed, so repeated
calls can differ in the last few bits. Both matter only at the 1e-6 level,
which is why the count blindness check below uses a tolerance rather than
exact equality.
"""

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.metrics import resolve_device, seed_all

SCORE_FORMS = ("weighted", "raw")
PRIOR_MODES = ("per_element", "total")

# Denominator floor. Only ever binds for a cloud with no points at all, where
# the numerator is zero too, so the output is a clean zero rather than a NaN.
PRECISION_FLOOR = 1e-6


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum rows of `values` into `n` groups given by `index`."""
    out = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    return out.index_add_(0, index, values)


def _mean_pool(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """The mean branch of point_clouds.pointnet.pool, replicated as the speed baseline.

    Replicated rather than imported because importing pointnet pulls in the HDF5
    loading stack, and this file is meant to stand alone.
    """
    ones = torch.ones(len(index), 1, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n, 1, device=values.device, dtype=values.dtype)
    counts = counts.index_add_(0, index, ones)
    return _scatter_sum(values, index, n) / counts.clamp_min(1)


def _head(n_in: int, n_out: int, hidden: int) -> nn.Module:
    """One linear layer, or a small MLP when hidden > 0.

    The paper uses a single linear layer before aggregation in the graph setting
    (Section 5) and three hidden layers in the standalone inference setting
    (Appendix C.2), so both are reachable. SiLU matches the paper's swish.
    """
    if hidden <= 0:
        return nn.Linear(n_in, n_out)
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(),
                         nn.Linear(hidden, n_out))


class FishnetsAggregation(nn.Module):
    """Precision weighted pooling. Drop in replacement for pointnet.pool.

    The call signature matches `point_clouds.pointnet.pool`: values of shape
    (total_points, n_in), a long index of shape (total_points,) saying which
    group each row belongs to, and the number of groups n. Groups are clouds
    for a readout, or target nodes for message aggregation inside a graph.
    """

    def __init__(self, n_in: int, n_score: int = 8, hidden: int = 0,
                 expose_total_fisher: bool = False,
                 prior_precision: float = 1.0,
                 prior_mode: str = "per_element",
                 score_form: str = "weighted") -> None:
        super().__init__()
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}")
        if score_form not in SCORE_FORMS:
            raise ValueError(f"score_form must be one of {SCORE_FORMS}")
        if prior_precision < 0.0:
            raise ValueError("prior_precision must be non negative")
        if n_score < 1:
            raise ValueError("n_score must be at least 1")

        self.n_score = n_score
        self.expose_total_fisher = expose_total_fisher
        self.prior_precision = prior_precision
        self.prior_mode = prior_mode
        self.score_form = score_form
        self.score_head = _head(n_in, n_score, hidden)
        self.fisher_head = _head(n_in, n_score, hidden)

    @property
    def out_dim(self) -> int:
        """Width of the returned tensor, for sizing whatever reads it."""
        return 2 * self.n_score if self.expose_total_fisher else self.n_score

    @property
    def depends_on_count(self) -> bool:
        """Whether the output can change when only the number of points changes."""
        return self.expose_total_fisher or self.prior_mode == "total"

    def forward(self, values: torch.Tensor, index: torch.Tensor,
                n: int) -> torch.Tensor:
        if values.dim() != 2:
            raise ValueError(f"values must be 2D, got shape {tuple(values.shape)}")
        if index.dtype != torch.long:
            raise ValueError(f"index must be long, got {index.dtype}")
        if index.shape[0] != values.shape[0]:
            raise ValueError(
                f"index has {index.shape[0]} entries for {values.shape[0]} rows")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")

        precision = F.softplus(self.fisher_head(values))
        if self.prior_mode == "per_element":
            precision = precision + self.prior_precision
        location = self.score_head(values)
        numerator = precision * location if self.score_form == "weighted" else location

        fisher_total = _scatter_sum(precision, index, n)
        score_total = _scatter_sum(numerator, index, n)
        if self.prior_mode == "total":
            # Paper eq. (28). The matching prior term on the score, eq. (27), is
            # zero because the fiducial point and the prior mean are both zero.
            fisher_total = fisher_total + self.prior_precision

        estimate = score_total / fisher_total.clamp_min(PRECISION_FLOOR)
        if not self.expose_total_fisher:
            return estimate
        return torch.cat([estimate, torch.log1p(fisher_total)], dim=1)


class FishnetsDeepSets(nn.Module):
    """phi over each point, FISHNETS aggregation, then rho over the summary.

    Signature deliberately matches `point_clouds.pointnet.DeepSets.forward`, so
    the existing fit and predict loops accept this model without any change.
    `n_out` defaults to 2 to match len(pointnet.TARGETS); pointnet is not
    imported here because that would pull in the HDF5 loading stack and this
    file is meant to be importable on its own.
    """

    def __init__(self, hidden: int = 64, n_in: int = 3, n_score: int = 8,
                 expose_total_fisher: bool = False, n_out: int = 2) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.aggregate = FishnetsAggregation(
            hidden, n_score=n_score, expose_total_fisher=expose_total_fisher)
        self.rho = nn.Sequential(nn.Linear(self.aggregate.out_dim, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_out))

    @property
    def depends_on_count(self) -> bool:
        return self.aggregate.depends_on_count

    def forward(self, points: torch.Tensor, index: torch.Tensor,
                n: int) -> torch.Tensor:
        return self.rho(self.aggregate(self.phi(points), index, n))


def _fake_clouds(n_clouds: int, n_points: int, n_in: int,
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flat point list plus cloud index, in the layout pointnet.Batched uses."""
    values = torch.randn(n_clouds * n_points, n_in, device=device)
    index = torch.arange(n_clouds, device=device).repeat_interleave(n_points)
    return values, index


def _report_shape(name: str, out: torch.Tensor) -> None:
    print(f"{name:<28} shape={tuple(out.shape)} finite={bool(out.isfinite().all())} "
          f"min={out.min().item():+.4f} max={out.max().item():+.4f} "
          f"mean={out.mean().item():+.4f} std={out.std().item():.4f}")


def check_count_blindness(block: FishnetsAggregation, values: torch.Tensor,
                          index: torch.Tensor, n: int) -> Tuple[float, float]:
    """Duplicate every point, then measure how far the output moved.

    Expected before running: the score half must not move at all, because the
    numerator and the denominator both double. The exposed log1p(F_total) half
    must move by ln 2 = 0.6931, because F_total doubles and is far above 1.
    """
    with torch.no_grad():
        base = block(values, index, n)
        doubled = block(torch.cat([values, values]),
                        torch.cat([index, index]), n)
    score_shift = (doubled[:, :block.n_score] - base[:, :block.n_score]).abs().max()
    if not block.expose_total_fisher:
        return score_shift.item(), float("nan")
    fisher_shift = (doubled[:, block.n_score:] - base[:, block.n_score:]).abs().mean()
    return score_shift.item(), fisher_shift.item()


def check_guards(device: torch.device) -> None:
    """Trip every guard on purpose. A guard never tripped is a guard only hoped for."""
    block = FishnetsAggregation(4, n_score=2).to(device)
    values = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 0, 1, 1, 1, 1], device=device, dtype=torch.long)
    for name, call in (
            ("values not 2D", lambda: block(values[0], index, 2)),
            ("index not long", lambda: block(values, index.float(), 2)),
            ("length mismatch", lambda: block(values, index[:3], 2)),
            ("n below one", lambda: block(values, index, 0)),
            ("bad prior_mode", lambda: FishnetsAggregation(4, prior_mode="oops")),
            ("bad score_form", lambda: FishnetsAggregation(4, score_form="oops")),
            ("negative prior", lambda: FishnetsAggregation(4, prior_precision=-1.0)),
            ("n_score below one", lambda: FishnetsAggregation(4, n_score=0))):
        try:
            call()
        except ValueError as caught:
            print(f"  guard tripped   {name:<20} -> {caught}")
        else:
            raise AssertionError(f"guard {name!r} did not fire")

    # Empty group: cloud 2 gets no points, so the denominator floor is what
    # stands between this and a NaN.
    out = block(values, index, 3)
    print(f"  empty cloud row finite={bool(out[2].isfinite().all())} "
          f"values={out[2].tolist()}")


def time_forward(block: nn.Module, values: torch.Tensor, index: torch.Tensor,
                 n: int, repeats: int = 50) -> float:
    """Milliseconds per forward pass, with the MPS queue drained before timing."""
    for _ in range(5):
        block(values, index, n)
    if values.device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        block(values, index, n)
    if values.device.type == "mps":
        torch.mps.synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def smoke_test(n_clouds: int = 32, n_points: int = 2000, n_in: int = 64,
               seed: int = 0) -> None:
    """Shapes, finiteness, count blindness, guards, gradients and speed."""
    seed_all(seed)
    device = resolve_device("auto")
    print(f"device={device}  torch={torch.__version__}  seed={seed}")
    print(f"clouds={n_clouds}  points_per_cloud={n_points}  n_in={n_in}\n")

    values, index = _fake_clouds(n_clouds, n_points, n_in, device)

    blind = FishnetsAggregation(n_in, n_score=8).to(device)
    leaky = FishnetsAggregation(n_in, n_score=8, expose_total_fisher=True).to(device)
    paper = FishnetsAggregation(n_in, n_score=8, prior_mode="total",
                                score_form="raw").to(device)

    print("FORWARD")
    for name, block in (("count blind (default)", blind),
                        ("expose_total_fisher", leaky),
                        ("paper prior + raw score", paper)):
        with torch.no_grad():
            _report_shape(name, block(values, index, n_clouds))
        print(f"{'':<28} out_dim={block.out_dim} "
              f"depends_on_count={block.depends_on_count} "
              f"params={sum(p.numel() for p in block.parameters())}")

    print("\nCOUNT BLINDNESS, every point duplicated so N goes 2000 -> 4000")
    score_shift, _ = check_count_blindness(blind, values, index, n_clouds)
    print(f"  default            max |score shift| = {score_shift:.3e}   "
          f"(expected 0, float32 index_add_ noise only)")
    score_shift, fisher_shift = check_count_blindness(leaky, values, index, n_clouds)
    print(f"  expose_total_fisher max |score shift| = {score_shift:.3e}   "
          f"mean log1p(F) shift = {fisher_shift:.4f}  (expected ln 2 = 0.6931)")
    score_shift, _ = check_count_blindness(paper, values, index, n_clouds)
    print(f"  prior_mode=total   max |score shift| = {score_shift:.3e}   "
          f"(expected non zero, this is the weak N leak)")

    print("\nGUARDS")
    check_guards(device)

    print("\nGRADIENTS")
    model = FishnetsDeepSets(hidden=64, n_in=3, n_score=8).to(device)
    points, cloud_index = _fake_clouds(n_clouds, n_points, 3, device)
    target = torch.randn(n_clouds, 2, device=device)
    loss = ((model(points, cloud_index, n_clouds) - target) ** 2).mean()
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in model.parameters()])
    print(f"  FishnetsDeepSets loss={loss.item():.4f} "
          f"params={sum(p.numel() for p in model.parameters())} "
          f"grad_finite={bool(grads.isfinite().all())} "
          f"grad_absmax={grads.abs().max().item():.4e} "
          f"depends_on_count={model.depends_on_count}")

    print("\nSPEED, one forward over the same tensors")
    with torch.no_grad():
        base = time_forward(_mean_pool, values, index, n_clouds)
        fish = time_forward(blind, values, index, n_clouds)
    print(f"  mean pooling {base:.3f} ms   fishnets {fish:.3f} ms   "
          f"ratio {fish / base:.2f}x")


if __name__ == "__main__":
    smoke_test()
