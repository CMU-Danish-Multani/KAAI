"""STAGE 4 -- the architecture search, run as two arms on equal budgets.

    python -u -m point_clouds.training.architecture_search

THE EXPERIMENT
--------------
Two searches over the same space with the same budget, differing in one thing:

    SCREENED   only pooling operations that provably cannot see the point count
    OPEN       count-aware options are on the menu too

Neither search is told anything about shortcuts. Both are told only to maximise
validation R2. The question is what the OPEN arm reaches for, and how much of
its score survives when the shortcut is removed.

Count status for every option was measured with blocks/count_screen.py, which
varies the number of points while holding their distribution fixed and asks
whether a held-out probe can recover the count.

WHY THE TEST SPLIT IS TOUCHED ONCE
----------------------------------
Selecting on test is Kapoor and Narayanan's L1.3 leakage one level up
(arXiv 2207.07048). The search sees validation only. The winner of each arm is
retrained across seeds and scored on test exactly once, and the number of test
evaluations is printed, which is a cheap credibility signal that the NAS
literature does not generally report.
"""

import argparse
import json
import platform
import time
from typing import Dict, List

import numpy as np
import optuna
import torch

from common.metrics import bootstrap_r2, resolve_device
from point_clouds.load import BOX
from point_clouds.pointnet import TARGETS, load_positions
from point_clouds.search_space import (COUNT_AWARE_POOLINGS, COUNT_BLIND_POOLINGS,
                                       describe, pooling_is_count_blind, sample_config)
from point_clouds.searchable import CloudData, train_and_score
from point_clouds.training.step1_gate_2pcf import RESULTS_DIR

# Our own measured numbers on CAMELS Omega_m, for the search to be judged against.
OUR_BASELINES = {"2PCF + MLP": 0.8597, "LLS 49 params": 0.8034,
                 "GNN mean pooling": 0.6600, "GNN sum pooling": 0.8020,
                 "counting galaxies alone": 0.5058}

TEST_EVALUATIONS = {"count": 0}


def load_suite(suite: str, device: torch.device, limit: int):
    box = BOX[suite]
    splits, stats = {}, None
    for name in ("train", "val", "test"):
        clouds, y = load_positions(suite, name, limit)
        splits[name] = CloudData([c * box for c in clouds], y, box, device, stats)
        if stats is None:
            stats = (splits["train"].label_mean, splits["train"].label_spread)
    return splits


def run_arm(arm: str, splits, trials: int, epochs: int, batch_size: int,
            seeds: List[int], device: torch.device) -> Dict:
    print("\n" + "=" * 78)
    print(f"ARM: {arm.upper()}   pooling options: "
          f"{', '.join(COUNT_BLIND_POOLINGS if arm == 'screened' else COUNT_BLIND_POOLINGS + COUNT_AWARE_POOLINGS)}")
    print("=" * 78, flush=True)

    history = []

    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial, arm)
        started = time.time()
        try:
            val = train_and_score(config, splits["train"], splits["val"],
                                  seed=0, epochs=epochs, batch_size=batch_size,
                                  device=device)
        except RuntimeError as exc:
            print(f"    trial {trial.number:3d} FAILED {describe(config)}: "
                  f"{str(exc)[:60]}", flush=True)
            return -10.0
        score = float(np.mean(val))
        history.append({"config": config, "val_r2": [float(v) for v in val],
                        "mean": score})
        print(f"    trial {trial.number:3d}  {score:+.4f}  "
              f"[{time.time() - started:5.1f}s]  {describe(config)}", flush=True)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials)

    best = max(history, key=lambda h: h["mean"])["config"]
    print(f"\n  winner: {describe(best)}", flush=True)
    print(f"  count-blind pooling: {pooling_is_count_blind(best['pooling'])}")

    per_seed = []
    for seed in seeds:
        per_seed.append(train_and_score(best, splits["train"], splits["test"],
                                        seed=seed, epochs=epochs,
                                        batch_size=batch_size, device=device))
        TEST_EVALUATIONS["count"] += 1
    per_seed = np.stack(per_seed)
    mean, spread = per_seed.mean(0), per_seed.std(0)

    for i, target in enumerate(TARGETS):
        spread_text = (f"+/- {spread[i]:.4f}" if len(seeds) > 1 else "(single run)")
        print(f"  TEST {target:9s} {mean[i]:+.4f} {spread_text}")

    return {"arm": arm, "winner": best,
            "winner_pooling_count_blind": pooling_is_count_blind(best["pooling"]),
            "test_r2_mean": [float(v) for v in mean],
            "test_r2_seed_std": [float(v) for v in spread] if len(seeds) > 1 else None,
            "history": history}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suite", type=str, default="CAMELS", choices=["CAMELS", "CAMELS-SAM"])
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--arms", type=str, nargs="+", default=["screened", "open"],
                   choices=["screened", "open"])
    p.add_argument("--device", type=str, default="mps",
                   choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()
    if args.trials < 1 or args.epochs < 1:
        raise SystemExit("--trials and --epochs must be at least 1")

    device = resolve_device(args.device)
    print("=" * 78)
    print("STAGE 4 -- ARCHITECTURE SEARCH, TWO ARMS")
    print("=" * 78)
    print(f"  suite {args.suite}   device {device}   {args.trials} trials per arm   "
          f"{args.epochs} epochs   seeds {args.seeds}")
    print("  our baselines on CAMELS Omega_m: "
          + ", ".join(f"{k} {v:.4f}" for k, v in OUR_BASELINES.items()), flush=True)

    splits = load_suite(args.suite, device, args.limit)
    print(f"  train {splits['train'].n_clouds} clouds, "
          f"mean size {splits['train'].mean_size:.0f}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"architecture_search_{args.suite}.json"
    results = []

    def save(complete: bool) -> None:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"stage": "4: architecture search", "complete": complete,
                       "config": vars(args), "device": str(device),
                       "our_baselines_camels_omega_m": OUR_BASELINES,
                       "test_set_evaluations": TEST_EVALUATIONS["count"],
                       "versions": {"torch": torch.__version__,
                                    "optuna": optuna.__version__,
                                    "numpy": np.__version__,
                                    "python": platform.python_version()},
                       "results": results}, fh, indent=2)

    save(False)
    for arm in args.arms:
        results.append(run_arm(arm, splits, args.trials, args.epochs,
                               args.batch_size, args.seeds, device))
        save(len(results) == len(args.arms))

    print("\n" + "=" * 78)
    for r in results:
        print(f"  {r['arm']:9s} {r['test_r2_mean'][0]:+.4f} / {r['test_r2_mean'][1]:+.4f}   "
              f"pooling={r['winner']['pooling']}  "
              f"count-blind={r['winner_pooling_count_blind']}")
    if len(results) == 2:
        gap = results[1]["test_r2_mean"][0] - results[0]["test_r2_mean"][0]
        print(f"\n  OPEN minus SCREENED on Omega_m: {gap:+.4f}")
        print("  INTERPRETED: that gap is the part of the best score that the "
              "shortcut explains.")
    print(f"\n  test set evaluations: {TEST_EVALUATIONS['count']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
