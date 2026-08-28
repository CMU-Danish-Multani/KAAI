"""STAGE 1 GATE -- reproduce CosmoBench Table 2's 2PCF baseline.

Nothing else in this project counts until this passes. The point is not to build
a good model; it is to prove the scoring, splitting and normalisation are
trustworthy before anything is built on top of them.

    python -m point_clouds.training.step1_gate_2pcf

Targets, from CosmoBench Table 2 (R2 on the test split, +/- 1 bootstrap std):

    CAMELS-SAM   Omega_m 0.73 +/- 0.03    sigma_8 0.82 +/- 0.02
    CAMELS       Omega_m 0.84 +/- 0.02    sigma_8 0.30 +/- 0.06

Acceptance bands are the published value +/- 2 std, fixed in
notes/spec_stage1_gate.md BEFORE this was first run. Widened from 1 std to 2
because our hyperparameter search is not identical to theirs. Failing the band
means stop and debug. It does not mean widen the band.

Model and training follow CosmoBench Sec. B.1: a 4-layer MLP on log-scaled
correlation features, 300 epochs, 100 Optuna trials with the TPE sampler,
selected on validation and reported on test.
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn

from common.metrics import bootstrap_r2, r2_score, resolve_device, seed_all
from point_clouds.tpcf import load_or_build, to_features

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "point_clouds" / "results"

# Published value and bootstrap std, CosmoBench Table 2.
PUBLISHED: Dict[str, Dict[str, Tuple[float, float]]] = {
    "CAMELS-SAM": {"Omega_m": (0.73, 0.03), "sigma_8": (0.82, 0.02)},
    "CAMELS":     {"Omega_m": (0.84, 0.02), "sigma_8": (0.30, 0.06)},
}
BAND_STDS = 2.0
TARGETS = ("Omega_m", "sigma_8")
EPOCHS = 300


class MLP(nn.Module):
    """Four weight layers, per Sec. B.1: input -> h1 -> h2 -> h3 -> (Omega_m, sigma_8)."""

    def __init__(self, n_in: int, h1: int, h2: int, h3: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, len(TARGETS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


CONST_TOL = 1e-6


def standardise(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray], label: str = ""):
    """Normalise on TRAIN statistics, dropping features that never vary in train.

    Measuring statistics across all splits would leak test-set facts into
    training and silently inflate every score.

    Dropping constant columns is not cosmetic. A column with train std of zero
    divided by a clipped epsilon turns any tiny val or test variation into a
    value of order 1e8, which destroys the network. Caught 2026-08-17 when
    trimming clouds to 588 galaxies emptied the innermost correlation bins and
    produced an R2 of -1.7e11.
    """
    keep = x["train"].std(0) > CONST_TOL
    if not keep.all():
        print(f"    {label}dropped {int((~keep).sum())} of {len(keep)} features "
              f"as constant in train", flush=True)
    if not keep.any():
        raise AssertionError("every feature is constant in train")

    x = {s: v[:, keep] for s, v in x.items()}
    x_mu, x_sd = x["train"].mean(0), x["train"].std(0)
    y_mu, y_sd = y["train"].mean(0), y["train"].std(0)

    return {s: {"x": ((x[s] - x_mu) / x_sd).astype(np.float32),
                "y_std": ((y[s] - y_mu) / y_sd).astype(np.float32),
                "y_raw": y[s].astype(np.float64)} for s in x}, (y_mu, y_sd)


def load_suite(suite: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Features and labels for all three splits, normalised with TRAIN statistics."""
    raw = {s: load_or_build(suite, s) for s in ("train", "val", "test")}
    return standardise({s: to_features(raw[s]["xi"]) for s in raw},
                       {s: raw[s]["y"] for s in raw}, f"{suite}: ")


def train_once(data, params: dict, seed: int, device: torch.device) -> Tuple[nn.Module, np.ndarray]:
    """Train one model. Returns the model and its validation R2 per target."""
    seed_all(seed)
    tensors = {s: (torch.as_tensor(data[s]["x"]).to(device),
                   torch.as_tensor(data[s]["y_std"]).to(device)) for s in data}
    xtr, ytr = tensors["train"]

    model = MLP(xtr.shape[1], params["h1"], params["h2"], params["h3"],
                params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(xtr), params["batch_size"]

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()

    return model, evaluate(model, data, "val", device)[0]


@torch.no_grad()
def evaluate(model: nn.Module, data, split: str, device: torch.device,
             y_stats=None) -> Tuple[np.ndarray, np.ndarray]:
    """R2 per target on one split, plus the raw predictions."""
    model.eval()
    x = torch.as_tensor(data[split]["x"]).to(device)
    pred_std = model(x).cpu().numpy().astype(np.float64)
    model.train()
    if y_stats is None:
        return r2_score(pred_std, data[split]["y_std"].astype(np.float64)), pred_std
    y_mu, y_sd = y_stats
    pred = pred_std * y_sd + y_mu
    return r2_score(pred, data[split]["y_raw"]), pred


def search(data, n_trials: int, device: torch.device, seed: int) -> dict:
    """Optuna TPE over the Sec. B.1 hyperparameter ranges. Selected on validation."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "h1": trial.suggest_int("h1", 64, 128),
            "h2": trial.suggest_int("h2", 64, 128),
            "h3": trial.suggest_int("h3", 16, 64),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [4, 16, 64]),
        }
        _, val_r2 = train_once(data, params, seed, device)
        return float(np.mean(val_r2))

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n = trial.number + 1
        if n % 10 == 0 or n == n_trials:
            print(f"    trial {n:4d}/{n_trials}  best mean val R2 so far {study.best_value:.4f}",
                  flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[progress])
    return study.best_params


def run_suite(suite: str, n_trials: int, seeds: list, device: torch.device) -> dict:
    print("\n" + "=" * 74)
    print(f"{suite}")
    print("=" * 74)
    data, y_stats = load_suite(suite)
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {data[s]['x'].shape[0]:5d} clouds x {data[s]['x'].shape[1]} features")

    t0 = time.time()
    best = search(data, n_trials, device, seed=seeds[0])
    t_search = time.time() - t0
    print(f"\n  best config after {n_trials} trials ({t_search / 60:.1f} min): {best}")

    # Retrain the winner across seeds. A single run cannot support a comparative
    # claim, and the spread is what tells us whether a gap is real.
    per_seed, preds = [], None
    for s in seeds:
        model, _ = train_once(data, best, s, device)
        test_r2, preds = evaluate(model, data, "test", device, y_stats)
        per_seed.append(test_r2)
        print(f"    seed {s}: test R2  Omega_m {test_r2[0]:.4f}   sigma_8 {test_r2[1]:.4f}")
    per_seed = np.stack(per_seed)

    boot_mean, boot_std = bootstrap_r2(preds, data["test"]["y_raw"])

    print(f"\n  {'target':10s} {'ours (seed mean+/-sd)':>26s} {'published':>16s} "
          f"{'band':>16s}  verdict")
    result, passed = {}, True
    for i, name in enumerate(TARGETS):
        mu, sd = PUBLISHED[suite][name]
        lo, hi = mu - BAND_STDS * sd, mu + BAND_STDS * sd
        ours = per_seed[:, i].mean()
        # One seed has no spread, which is not the same as a spread of zero.
        # Printing +/- 0.0000 would read as an extremely tight result.
        ours_sd = per_seed[:, i].std() if len(seeds) > 1 else None
        spread = f"+/- {ours_sd:<8.4f}" if ours_sd is not None else "(single run) "
        ok = lo <= ours <= hi
        passed &= ok
        print(f"  {name:10s} {ours:>14.4f} {spread} {mu:>10.2f}+/-{sd:<4.2f} "
              f"[{lo:.2f}, {hi:.2f}]  {'PASS' if ok else 'FAIL'}")
        result[name] = {
            "ours_mean": float(ours),
            "ours_std_across_seeds": None if ours_sd is None else float(ours_sd),
            "ours_per_seed": [float(v) for v in per_seed[:, i]],
            "bootstrap_mean": float(boot_mean[i]), "bootstrap_std": float(boot_std[i]),
            "published_mean": mu, "published_std": sd,
            "band": [float(lo), float(hi)], "pass": bool(ok),
        }
    return {"suite": suite, "best_params": best, "n_trials": n_trials,
            "seeds": seeds, "search_minutes": round(t_search / 60, 2),
            "targets": result, "pass": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suites", type=str, nargs="+", default=["CAMELS-SAM", "CAMELS"],
                        choices=["CAMELS", "CAMELS-SAM"])
    parser.add_argument("--trials", type=int, default=100,
                        help="Optuna trials; CosmoBench Sec. B.1 used 100")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="seeds for retraining the winning config")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="mps by default, per standing instruction. NOTE: for the "
                             "tiny 2PCF tensors CPU measured 3.1x faster")
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    device = resolve_device(args.device)
    print("=" * 74)
    print("STAGE 1 GATE -- 2PCF baseline reproduction")
    print("=" * 74)
    print(f"  device {device}   torch {torch.__version__}   optuna {optuna.__version__}")
    print(f"  {args.trials} trials, seeds {args.seeds}, {EPOCHS} epochs per model")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "step1_gate_2pcf.json"

    def write(results: list, complete: bool) -> None:
        """Persist after every suite, so a killed run still leaves what it finished."""
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({
                "stage": "1 gate: 2PCF baseline reproduction",
                "complete": complete,
                "suites_requested": args.suites, "trials": args.trials,
                "device": str(device), "epochs": EPOCHS, "band_stds": BAND_STDS,
                "versions": {"torch": torch.__version__, "optuna": optuna.__version__,
                             "numpy": np.__version__, "python": platform.python_version()},
                "results": results,
            }, fh, indent=2)

    write([], complete=False)          # stamp immediately, so a stale file cannot masquerade
    results = []
    for suite in args.suites:
        results.append(run_suite(suite, args.trials, args.seeds, device))
        write(results, complete=len(results) == len(args.suites))
        print(f"  checkpointed {suite} to {out.name}", flush=True)

    print("\n" + "=" * 74)
    verdict = all(r["pass"] for r in results)
    print(f"  STAGE 1 GATE: {'PASS' if verdict else 'FAIL'}")
    print(f"  wrote {out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
