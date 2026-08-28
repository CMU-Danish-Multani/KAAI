"""STAGE 2 DIAGNOSTICS -- three questions that decide the search space.

    python -u -m point_clouds.training.step2_diagnostics

Predictions for all three were written into runLog.md BEFORE this was first run.

A. SEARCH VARIANCE. Every one of the four Stage 1 targets landed above its
   published value. The paper's reported uncertainty is a bootstrap over the
   test set only, so it excludes variance from the hyperparameter search and
   from training. Rerunning the search under different Optuna seeds measures
   what that missing variance actually is, and therefore whether the offset is
   real or is search noise.

B. LEAK EXPLOITABILITY. Counting galaxies correlates 0.73 with Omega_m in
   CAMELS. That is a correlation, not a demonstration that a model uses it.
   Appending the count as an extra input feature measures what explicit access
   buys. CAMELS-SAM, where the count is constant, is the built-in control.

   Note the subtlety this is testing: xi(r) is already normalised by N(N-1), so
   the count is largely divided out of the Stage 1 features. The passing gate
   may already be mostly leak free, which would be worth knowing.

C. FIXED-COUNT CONTROL. Trimming every CAMELS cloud to its suite minimum of 588
   most massive galaxies closes the leak by construction. This is the matched
   control Stage 4.2 needs. It confounds two effects, closing the leak and
   simply having fewer galaxies, so the same trim is applied to CAMELS-SAM
   where the leak is already closed and any drop is pure information loss.
"""

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna
import torch

from common.metrics import r2_score, resolve_device
from point_clouds.tpcf import load_or_build, to_features
from point_clouds.training.step1_gate_2pcf import (RESULTS_DIR, TARGETS, standardise,
                                                   evaluate, search, train_once)

FIXED_COUNT = 588        # the CAMELS suite minimum, measured across all splits


def assemble(suite: str, fixed_count: int = 0, with_ngal: bool = False):
    """Features and labels for all splits, normalised on TRAIN statistics."""
    raw = {s: load_or_build(suite, s, fixed_count=fixed_count)
           for s in ("train", "val", "test")}
    x = {s: to_features(raw[s]["xi"]) for s in raw}
    if with_ngal:
        for s in raw:
            x[s] = np.concatenate([x[s], raw[s]["ngal"][:, None].astype(float)], axis=1)

    label = f"{suite}{'/top' + str(fixed_count) if fixed_count else ''}"
    label += "+ngal: " if with_ngal else ": "
    data, y_stats = standardise(x, {s: raw[s]["y"] for s in raw}, label)
    return data, y_stats, raw


def measure(data, y_stats, trials: int, search_seed: int, train_seeds: List[int],
            device: torch.device) -> Dict:
    """One search followed by retraining the winner across seeds. Returns test R2."""
    best = search(data, trials, device, seed=search_seed)
    per_seed = []
    for s in train_seeds:
        model, _ = train_once(data, best, s, device)
        per_seed.append(evaluate(model, data, "test", device, y_stats)[0])
    per_seed = np.stack(per_seed)
    return {"best_params": best,
            "test_r2_mean": [float(v) for v in per_seed.mean(0)],
            "test_r2_seed_std": [float(v) for v in per_seed.std(0)]}


def diag_a(suites, trials, search_seeds, train_seeds, device) -> Dict:
    print("\n" + "=" * 74)
    print("A. SEARCH VARIANCE -- is the Stage 1 offset real, or search noise?")
    print("=" * 74)
    out = {}
    for suite in suites:
        data, y_stats, _ = assemble(suite)
        runs = []
        for ss in search_seeds:
            r = measure(data, y_stats, trials, ss, train_seeds, device)
            runs.append(r)
            print(f"  {suite:11s} search seed {ss}: Omega_m {r['test_r2_mean'][0]:.4f}   "
                  f"sigma_8 {r['test_r2_mean'][1]:.4f}   (batch {r['best_params']['batch_size']})",
                  flush=True)
        arr = np.array([r["test_r2_mean"] for r in runs])
        out[suite] = {"runs": runs,
                      "across_search_seeds_mean": [float(v) for v in arr.mean(0)],
                      "across_search_seeds_std": [float(v) for v in arr.std(0)],
                      "range": [[float(arr[:, i].min()), float(arr[:, i].max())]
                                for i in range(arr.shape[1])]}
        for i, t in enumerate(TARGETS):
            print(f"    {suite} {t}: mean {arr[:, i].mean():.4f}  "
                  f"std across searches {arr[:, i].std():.4f}  "
                  f"range [{arr[:, i].min():.4f}, {arr[:, i].max():.4f}]")
    return out


def diag_b(suites, trials, search_seed, train_seeds, device) -> Dict:
    print("\n" + "=" * 74)
    print("B. LEAK EXPLOITABILITY -- what does explicit access to the count buy?")
    print("=" * 74)
    out = {}
    for suite in suites:
        row = {}
        for tag, with_ngal in (("xi_only", False), ("xi_plus_ngal", True)):
            data, y_stats, raw = assemble(suite, with_ngal=with_ngal)
            r = measure(data, y_stats, trials, search_seed, train_seeds, device)
            row[tag] = r
            print(f"  {suite:11s} {tag:13s} Omega_m {r['test_r2_mean'][0]:.4f}   "
                  f"sigma_8 {r['test_r2_mean'][1]:.4f}", flush=True)
        delta = np.array(row["xi_plus_ngal"]["test_r2_mean"]) - \
            np.array(row["xi_only"]["test_r2_mean"])
        row["delta"] = [float(v) for v in delta]
        print(f"  {suite:11s} {'DELTA':13s} Omega_m {delta[0]:+.4f}   sigma_8 {delta[1]:+.4f}")
        out[suite] = row
    return out


def diag_c(suites, trials, search_seed, train_seeds, device) -> Dict:
    print("\n" + "=" * 74)
    print(f"C. FIXED-COUNT CONTROL -- trim every cloud to {FIXED_COUNT} most massive")
    print("=" * 74)
    out = {}
    for suite in suites:
        row = {}
        for tag, fc in (("full", 0), (f"top{FIXED_COUNT}", FIXED_COUNT)):
            data, y_stats, raw = assemble(suite, fixed_count=fc)
            n = raw["train"]["ngal"]
            r = measure(data, y_stats, trials, search_seed, train_seeds, device)
            r["ngal_train_min_max"] = [int(n.min()), int(n.max())]
            row[tag] = r
            print(f"  {suite:11s} {tag:10s} ngal {n.min():5d}-{n.max():5d}   "
                  f"Omega_m {r['test_r2_mean'][0]:.4f}   sigma_8 {r['test_r2_mean'][1]:.4f}",
                  flush=True)
        delta = np.array(row[f"top{FIXED_COUNT}"]["test_r2_mean"]) - \
            np.array(row["full"]["test_r2_mean"])
        row["delta"] = [float(v) for v in delta]
        print(f"  {suite:11s} {'DELTA':10s} {'':17s} Omega_m {delta[0]:+.4f}   "
              f"sigma_8 {delta[1]:+.4f}")
        out[suite] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trials", type=int, default=20,
                   help="Stage 1 measured that ~20 finds what 100 finds")
    p.add_argument("--search-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", type=str, default="cpu",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--only", type=str, nargs="+", default=["a", "b", "c"],
                   choices=["a", "b", "c"])
    args = p.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    device = resolve_device(args.device)
    out = RESULTS_DIR / "step2_diagnostics.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"stage": "2 diagnostics (point cloud track)", "complete": False,
               "trials": args.trials, "search_seeds": args.search_seeds,
               "train_seeds": args.train_seeds, "device": str(device),
               "fixed_count": FIXED_COUNT,
               "versions": {"torch": torch.__version__, "optuna": optuna.__version__,
                            "numpy": np.__version__, "python": platform.python_version()},
               "diagnostics": {}}

    def save(complete: bool = False) -> None:
        payload["complete"] = complete
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    save()
    print("=" * 74)
    print("STAGE 2 DIAGNOSTICS -- point cloud track")
    print("=" * 74)
    print(f"  device {device}   {args.trials} trials per search   "
          f"search seeds {args.search_seeds}   train seeds {args.train_seeds}")

    both = ["CAMELS-SAM", "CAMELS"]
    if "a" in args.only:
        payload["diagnostics"]["A_search_variance"] = diag_a(
            both, args.trials, args.search_seeds, args.train_seeds, device)
        save()
    if "b" in args.only:
        payload["diagnostics"]["B_leak_exploitability"] = diag_b(
            both, args.trials, args.search_seeds[0], args.train_seeds, device)
        save()
    if "c" in args.only:
        payload["diagnostics"]["C_fixed_count"] = diag_c(
            both, args.trials, args.search_seeds[0], args.train_seeds, device)
        save()

    save(complete=True)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
