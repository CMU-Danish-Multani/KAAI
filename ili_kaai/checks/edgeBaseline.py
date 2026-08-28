"""What edge deficit does a PROVABLY CORRECT posterior show under the same prior?

    conda run -n ltuili python -u -m ili_kaai.checks.edgeBaseline

edgeCoverage.py measured that coverage collapses near the prior walls, worst for the
parameter that is least constrained. That alone does not prove the models are failing.
A correct posterior contracts toward the prior when data is uninformative, so a truth
near a wall gets missed by a posterior sitting in the middle. Marginal coverage must
equal nominal; conditional coverage at a given point in parameter space need not.

So this builds the baseline. With a uniform prior and a Gaussian likelihood the exact
posterior is a truncated normal, which can be sampled directly. Its coverage is correct
by construction. Split it the same way and the difference between that deficit and ours
is the part the architectures are responsible for.

Predictions were written into runLog.md before this was first run.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import truncnorm

from ili_kaai.checks.edgeCoverage import near_wall, split_coverage

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[2] / "ili_kaai" / "results"
OUT = RESULTS / "edgeBaseline.json"

# The CAMELS prior box, and the R2 each parameter actually achieved in the sweep.
BOX = {"Omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0)}
MEASURED_R2 = {"Omega_m": 0.865, "sigma_8": 0.365}


def likelihood_width(lo: float, hi: float, r2: float) -> float:
    """Noise level that reproduces a given R2 under a uniform prior.

    R2 = 1 - Var(posterior) / Var(prior), and for a likelihood much narrower than the
    prior the posterior variance is the likelihood variance. A uniform prior on
    [lo, hi] has standard deviation (hi - lo) / sqrt(12).
    """
    return (hi - lo) / np.sqrt(12.0) * np.sqrt(max(1.0 - r2, 1e-6))


def exact_posterior(n_points: int, n_draws: int, seed: int) -> tuple:
    """Truths from the prior, and samples from the exactly correct posterior.

    theta ~ Uniform(lo, hi);  x | theta ~ N(theta, s);  so the posterior is
    N(x, s) truncated to [lo, hi]. No approximation anywhere.
    """
    rng = np.random.default_rng(seed)
    names = list(BOX)
    theta = np.empty((n_points, len(names)))
    samples = np.empty((n_draws, n_points, len(names)))

    for j, name in enumerate(names):
        lo, hi = BOX[name]
        s = likelihood_width(lo, hi, MEASURED_R2[name])
        theta[:, j] = rng.uniform(lo, hi, n_points)
        x = theta[:, j] + rng.normal(0.0, s, n_points)
        a, b = (lo - x) / s, (hi - x) / s          # truncnorm works in standard units
        samples[:, :, j] = truncnorm.rvs(
            a, b, loc=x, scale=s, size=(n_draws, n_points),
            random_state=rng.integers(1 << 31))
    return theta, samples, names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-points", type=int, default=200, help="matches the sweep")
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--edge-fraction", type=float, default=0.10)
    args = p.parse_args()

    if not 0 < args.edge_fraction < 0.5:
        raise SystemExit("--edge-fraction must be between 0 and 0.5")

    lo = [BOX[n][0] for n in BOX]
    hi = [BOX[n][1] for n in BOX]
    rows: List[Dict] = []
    for seed in args.seeds:
        theta, samples, names = exact_posterior(args.n_points, args.n_draws, seed)
        r = split_coverage(samples, theta, near_wall(theta, lo, hi, args.edge_fraction))
        overall = float(((theta >= np.percentile(samples, 16, axis=0))
                         & (theta <= np.percentile(samples, 84, axis=0))).mean())
        r.update({"seed": seed, "labels": names, "overall": round(overall, 4)})
        rows.append(r)

    print(f"  exactly correct posterior, same prior box, matched to measured R2")
    print(f"  edge = within {args.edge_fraction:.0%} of a wall, "
          f"{args.n_points} points, {len(args.seeds)} seeds\n")
    print(f"  {'param':10s}{'edge':>9s}{'interior':>10s}{'deficit':>9s}"
          f"{'ours':>9s}{'ours minus correct':>20s}")

    ours = {"Omega_m": -0.178, "sigma_8": -0.581}   # mean over the three NPE entries
    summary = {}
    for i, name in enumerate(rows[0]["labels"]):
        e = float(np.mean([r["edge"][i] for r in rows]))
        it = float(np.mean([r["interior"][i] for r in rows]))
        deficit = e - it
        summary[name] = {"edge": round(e, 4), "interior": round(it, 4),
                         "correctDeficit": round(deficit, 4),
                         "ourDeficit": ours[name],
                         "excess": round(ours[name] - deficit, 4)}
        print(f"  {name:10s}{e:9.3f}{it:10.3f}{deficit:+9.3f}"
              f"{ours[name]:+9.3f}{ours[name] - deficit:+20.3f}")

    print(f"\n  overall coverage of the correct posterior: "
          f"{np.mean([r['overall'] for r in rows]):.4f}  (must be near 0.680)")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"edgeFraction": args.edge_fraction, "nPoints": args.n_points,
                   "nDraws": args.n_draws, "box": BOX, "matchedR2": MEASURED_R2,
                   "perSeed": rows, "summary": summary}, fh, indent=2)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
