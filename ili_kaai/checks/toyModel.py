"""STAGE A3. Reproduce LtU-ILI Section 4.1 (Equation 14) to prove we drive the framework.

    conda run -n ltuili python -m ili_kaai.checks.toyModel

The point is not the science. The point is that this problem is analytic, so a wiring
mistake shows up as a violated prediction rather than as a plausible-looking number.

Deviation from the paper, recorded deliberately: Ho et al. use SNPE over 10 rounds of
2000 simulations. Sequential inference is a stated non-goal in notes/plans.md Section 6
because our real tasks are fixed catalogues with no on-the-fly simulator. This runs
amortized NPE at the same total budget of 20,000 simulations.

Predictions were written into runLog.md before this was first run.
"""

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "ili_kaai" / "results"
N_PARAMS = 3
N_DATA = 10

# Equation 14: k_i = (2i/3) - 3 for i in 0..9
K = (2.0 * np.arange(N_DATA) / 3.0) - 3.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simulate(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equation 14. theta has shape (n, 3), returns shape (n, 10)."""
    phi0 = theta[:, 0] + theta[:, 1]
    phi1 = theta[:, 1] - 3.0 * theta[:, 2] ** 2
    signal = 3.0 * np.sin(K[None, :] + phi0[:, None]) + phi1[:, None] * K[None, :] ** 2
    return signal + rng.normal(size=signal.shape)


def draw(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Standard normal prior on each parameter, as in the paper."""
    theta = rng.normal(size=(n, N_PARAMS))
    return theta, simulate(theta, rng)


def build_runner(device: str, out_dir: Path, epochs: int):
    prior = ili.utils.IndependentNormal(
        loc=[0.0] * N_PARAMS, scale=[1.0] * N_PARAMS, device=device)
    nets = [
        ili.utils.load_nde_sbi(engine="NPE", model="maf",
                               hidden_features=50, num_transforms=5),
        ili.utils.load_nde_sbi(engine="NPE", model="maf",
                               hidden_features=50, num_transforms=5),
    ]
    return InferenceRunner.load(
        backend="sbi", engine="NPE", prior=prior, nets=nets, device=device,
        train_args={"training_batch_size": 128, "learning_rate": 1e-3,
                    "max_num_epochs": epochs, "stop_after_epochs": 20},
        out_dir=out_dir)


def posterior_stats(posterior, x: np.ndarray, theta: np.ndarray,
                    n_draws: int, device: str) -> Dict:
    """Recovery R2 per parameter, plus the two degeneracies the equations imply."""
    means, corrs, in68, in95 = [], [], [], []
    for i in range(len(x)):
        s = posterior.sample((n_draws,),
                             x=torch.tensor(x[i], dtype=torch.float32, device=device),
                             show_progress_bars=False).cpu().numpy()
        means.append(s.mean(0))
        corrs.append(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
        lo68, hi68 = np.percentile(s, [16, 84], axis=0)
        lo95, hi95 = np.percentile(s, [2.5, 97.5], axis=0)
        in68.append((theta[i] >= lo68) & (theta[i] <= hi68))
        in95.append((theta[i] >= lo95) & (theta[i] <= hi95))

    means = np.stack(means)
    ss_res = ((theta - means) ** 2).sum(0)
    ss_tot = ((theta - theta.mean(0)) ** 2).sum(0)
    return {
        "r2_per_parameter": [float(v) for v in 1.0 - ss_res / ss_tot],
        "t0_t1_posterior_correlation_mean": float(np.mean(corrs)),
        "empirical_coverage_68": [float(v) for v in np.stack(in68).mean(0)],
        "empirical_coverage_95": [float(v) for v in np.stack(in95).mean(0)],
        "n_test_points": int(len(x)),
        "n_posterior_draws": n_draws,
    }


def check(stats: Dict) -> Dict[str, bool]:
    """The four predictions registered in runLog.md before this ran."""
    r2 = stats["r2_per_parameter"]
    c68 = float(np.mean(stats["empirical_coverage_68"]))
    c95 = float(np.mean(stats["empirical_coverage_95"]))
    return {
        "P1_t2_unidentifiable_r2_below_0.1": bool(r2[2] < 0.1),
        "P2_t0_t1_anticorrelated_below_-0.7":
            bool(stats["t0_t1_posterior_correlation_mean"] < -0.7),
        "P3_coverage68_in_0.60_0.75": bool(0.60 <= c68 <= 0.75),
        "P3_coverage95_in_0.90_0.98": bool(0.90 <= c95 <= 0.98),
        "P4_t0_and_t1_r2_above_0.5": bool(r2[0] > 0.5 and r2[1] > 0.5),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-sims", type=int, default=20000,
                   help="matches the paper's 10 rounds x 2000")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-draws", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"],
                   help="cpu by default: sbi 0.22 predates stable MPS support")
    args = p.parse_args()

    if args.n_sims < 100 or args.n_test < 10:
        raise SystemExit("--n-sims must be at least 100 and --n-test at least 10")

    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    theta, x = draw(args.n_sims, rng)
    theta_t, x_t = draw(args.n_test, rng)
    print(f"  simulated {args.n_sims} train, {args.n_test} test, "
          f"x shape {x.shape}, theta shape {theta.shape}", flush=True)

    runner = build_runner(args.device, OUT_DIR / "toy", args.epochs)
    posterior, summaries = runner(loader=NumpyLoader(x=x, theta=theta))
    print("  training done", flush=True)

    stats = posterior_stats(posterior, x_t, theta_t, args.n_draws, args.device)
    verdict = check(stats)

    payload = {"stage": "A3 toy model, LtU-ILI Section 4.1",
               "deviation_from_paper": "amortized NPE, not SNPE; same 20k budget",
               "args": vars(args), "stats": stats, "predictions": verdict,
               "all_predictions_held": all(verdict.values()),
               "final_validation_loss": [float(s["validation_log_probs"][-1])
                                         for s in summaries],
               "versions": {"torch": torch.__version__, "numpy": np.__version__,
                            "python": platform.python_version()}}

    print(f"\n  R2 per parameter        t0 {stats['r2_per_parameter'][0]:+.4f}   "
          f"t1 {stats['r2_per_parameter'][1]:+.4f}   "
          f"t2 {stats['r2_per_parameter'][2]:+.4f}")
    print(f"  t0-t1 posterior corr    {stats['t0_t1_posterior_correlation_mean']:+.4f}")
    print(f"  empirical coverage 68   "
          f"{np.mean(stats['empirical_coverage_68']):.4f}  (nominal 0.68)")
    print(f"  empirical coverage 95   "
          f"{np.mean(stats['empirical_coverage_95']):.4f}  (nominal 0.95)")
    print("\n  registered predictions:")
    for k, v in verdict.items():
        print(f"    {'HELD  ' if v else 'FAILED'} {k}")

    out = OUT_DIR / "toyModel.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
