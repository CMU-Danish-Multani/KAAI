"""Is our TARP call trustworthy, or is it the reason TARP and marginal coverage disagree?

    conda run -n ltuili python -m ili_kaai.checks.tarpCalibration

The sweep measured marginal coverage at 0.614 and TARP at 0.713 on the same posteriors,
against a nominal 0.680. One of the two is misreporting. This settles it without
retraining anything, by building posteriors whose true coverage is known by construction
and asking each metric to recover it.

Construction: theta is drawn from the prior, a noisy estimate theta_hat = theta + eps is
formed with eps ~ N(0, S), and posterior samples are drawn as N(theta_hat, f^2 S). At
f = 1 the posterior is exactly calibrated, so both metrics must return the nominal
level. f < 1 is overconfident and f > 1 is underconfident.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
from common.metrics import credible_coverage
from tarp import get_tarp_coverage

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "ili_kaai" / "results" / "tarpCalibration.json"

# Correlated, because Omega_m and sigma_8 are, and TARP is a multivariate test.
COV = np.array([[1.0, 0.6], [0.6, 1.0]])


def make(f: float, n_points: int, n_draws: int, seed: int):
    """Posteriors that are exactly Bayes-calibrated when f == 1.

    A conjugate Gaussian model is used rather than a uniform prior. With a uniform
    prior the analytic posterior is truncated at the prior edges, so a plain Gaussian
    posterior puts mass outside the prior and is not Bayes-calibrated even though its
    frequentist interval coverage is correct. TARP tests the Bayesian property, so the
    test model has to satisfy it exactly or the metric gets blamed for the test.

        prior       theta      ~ N(0, P)
        likelihood  theta_hat  ~ N(theta, S)
        posterior   theta|hat  ~ N(Sigma S^-1 theta_hat, Sigma),  Sigma = (P^-1+S^-1)^-1
    """
    rng = np.random.default_rng(seed)
    P, S = 2.0 * COV, COV
    Pi, Si = np.linalg.inv(P), np.linalg.inv(S)
    Sigma = np.linalg.inv(Pi + Si)

    theta = rng.multivariate_normal(np.zeros(2), P, size=n_points)
    theta_hat = theta + rng.multivariate_normal(np.zeros(2), S, size=n_points)
    mu = theta_hat @ Si.T @ Sigma.T
    noise = rng.multivariate_normal(np.zeros(2), Sigma, size=(n_draws, n_points))
    return mu[None, :, :] + f * noise, theta


def marginal_coverage(samples: np.ndarray, theta: np.ndarray, level: float) -> float:
    """Averaged over parameters, which is what this check compares against."""
    return float(credible_coverage(samples, theta, level).mean())


def tarp_at(samples: np.ndarray, theta: np.ndarray, level: float, norm: bool) -> float:
    ecp, alpha = get_tarp_coverage(samples, theta, norm=norm, seed=0)
    return float(np.interp(level, alpha, ecp))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-points", type=int, nargs="+", default=[100, 500, 2000],
                   help="100 is what the sweep used")
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--level", type=float, default=0.68)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-band", action="store_true",
                   help="measure the band an exactly calibrated posterior wanders in")
    args = p.parse_args()

    if args.noise_band:
        noise_band(level=args.level, n_draws=args.n_draws)
        return

    if not 0 < args.level < 1:
        raise SystemExit("--level must be strictly between 0 and 1")

    rows = []
    print(f"  nominal level {args.level}, {args.n_draws} draws\n")
    print(f"  {'width':>6} {'points':>7} {'marginal':>9} {'tarp norm=T':>12} "
          f"{'tarp norm=F':>12}   verdict")
    for f, label in ((1.0, "calibrated"), (0.7, "overconfident"),
                     (1.4, "underconfident")):
        for n in args.n_points:
            s, t = make(f, n, args.n_draws, args.seed)
            row = {"widthFactor": f, "truth": label, "nPoints": n,
                   "marginal": round(marginal_coverage(s, t, args.level), 4),
                   "tarpNormTrue": round(tarp_at(s, t, args.level, True), 4),
                   "tarpNormFalse": round(tarp_at(s, t, args.level, False), 4)}
            rows.append(row)
            print(f"  {f:6.1f} {n:7d} {row['marginal']:9.4f} "
                  f"{row['tarpNormTrue']:12.4f} {row['tarpNormFalse']:12.4f}   {label}")

    ref = [r for r in rows if r["widthFactor"] == 1.0]
    payload = {"level": args.level, "nDraws": args.n_draws, "rows": rows,
               "biasAtCalibrated": {
                   "marginal": round(float(np.mean([r["marginal"] for r in ref])
                                           - args.level), 4),
                   "tarpNormTrue": round(float(np.mean([r["tarpNormTrue"] for r in ref])
                                               - args.level), 4),
                   "tarpNormFalse": round(float(np.mean([r["tarpNormFalse"] for r in ref])
                                                - args.level), 4)}}
    print("\n  bias on the exactly calibrated case (should be 0):")
    for k, v in payload["biasAtCalibrated"].items():
        print(f"    {k:14s} {v:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()


def noise_band(n_points: int = 200, n_draws: int = 1000, n_seeds: int = 3,
               repeats: int = 30, level: float = 0.68) -> Dict:
    """How far an EXACTLY calibrated posterior wanders at the sweep's settings.

    The zoo's admission rule needs a threshold for calling an entry overconfident.
    Choosing one by eye makes the rule arbitrary. This measures it instead: build
    posteriors that are calibrated by construction, read them the same way the
    sweep does, and see how much the reading moves on noise alone.
    """
    singles = [marginal_coverage(*make(1.0, n_points, n_draws, s), level)
               for s in range(repeats * n_seeds)]
    arr = np.array(singles)
    means = np.array([arr[i:i + n_seeds].mean()
                      for i in range(0, len(arr), n_seeds)])
    out = {"nPoints": n_points, "nDraws": n_draws, "nSeeds": n_seeds,
           "level": level, "repeats": repeats,
           "singleSeedMean": round(float(arr.mean()), 4),
           "singleSeedStd": round(float(arr.std()), 4),
           "bias": round(float(arr.mean() - level), 4),
           "seedMeanStd": round(float(means.std()), 4),
           "twoSigma": round(float(2 * means.std()), 4)}
    path = OUT.parent / "calibrationNoiseBand.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  exactly calibrated at {n_points} points, {n_seeds}-seed mean:")
    print(f"    reads {out['singleSeedMean']:.4f} against nominal {level}, "
          f"bias {out['bias']:+.4f}")
    print(f"    seed-mean std {out['seedMeanStd']:.4f}, so 2 sigma is "
          f"{out['twoSigma']:.4f}")
    print(f"  wrote {path}")
    return out


if __name__ == "__main__":
    main()
