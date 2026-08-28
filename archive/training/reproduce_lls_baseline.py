"""STAGE 3, model 1 of 3 -- reproduce CosmoBench's 49-parameter LLS baseline.

    python -u -m point_clouds.training.reproduce_lls_baseline

Targets, CosmoBench Table 2 (test split, +/- 1 bootstrap std):

    CAMELS-SAM   Omega_m 0.77 +/- 0.03    sigma_8 0.82 +/- 0.02
    CAMELS       Omega_m 0.78 +/- 0.03    sigma_8 0.28 +/- 0.06

Acceptance bands are published value +/- 2 std, fixed here BEFORE the first run,
matching the convention set in notes/spec_stage1_gate.md. Landing outside a band
in either direction means stop and debug, not widen the band.

NO SEEDS HERE, AND THAT IS NOT AN OVERSIGHT. Least squares has a closed-form
solution and the greedy radius selection is deterministic, so this model has no
random component to vary. Reporting a spread across seeds would be reporting
zero variance that was never at risk. Bootstrap error bars over the test set are
the only meaningful uncertainty, and they are what the paper reports too.
"""

import argparse
import json
import platform
from typing import Dict

import numpy as np

from common.metrics import bootstrap_r2, r2_score
from point_clouds.lls import TARGETS, load_or_build, fit_and_predict
from point_clouds.training.step1_gate_2pcf import RESULTS_DIR

PUBLISHED: Dict[str, Dict[str, tuple]] = {
    "CAMELS-SAM": {"Omega_m": (0.77, 0.03), "sigma_8": (0.82, 0.02)},
    "CAMELS":     {"Omega_m": (0.78, 0.03), "sigma_8": (0.28, 0.06)},
}
BAND_STDS = 2.0


def run_suite(suite: str, rebuild: bool) -> dict:
    print("\n" + "=" * 74)
    print(suite)
    print("=" * 74)
    splits = {s: load_or_build(suite, s, rebuild) for s in ("train", "val", "test")}
    for s, d in splits.items():
        print(f"  {s:5s} {d['x'].shape[0]:5d} clouds x {d['x'].shape[1]} candidate features")

    predictions, detail = fit_and_predict(splits["train"], splits["val"], splits["test"])
    truth = splits["test"]["y"]
    scores = r2_score(predictions, truth)
    boot_mean, boot_std = bootstrap_r2(predictions, truth)

    print(f"\n  {'target':10s} {'ours':>10s} {'bootstrap':>14s} {'published':>14s} "
          f"{'band':>14s}  verdict")
    result, passed = {}, True
    for i, name in enumerate(TARGETS):
        mu, sd = PUBLISHED[suite][name]
        lo, hi = mu - BAND_STDS * sd, mu + BAND_STDS * sd
        ok = lo <= scores[i] <= hi
        passed &= ok
        print(f"  {name:10s} {scores[i]:10.4f} {boot_mean[i]:8.4f}+/-{boot_std[i]:<5.4f} "
              f"{mu:8.2f}+/-{sd:<5.2f} [{lo:.2f}, {hi:.2f}]  {'PASS' if ok else 'FAIL'}")
        result[name] = {
            "ours": float(scores[i]),
            "bootstrap_mean": float(boot_mean[i]), "bootstrap_std": float(boot_std[i]),
            "seed_spread": None,          # deterministic model, see module docstring
            "published_mean": mu, "published_std": sd,
            "band": [float(lo), float(hi)], "pass": bool(ok),
            **detail[name],
        }
    n_params = detail[TARGETS[0]]["n_parameters"]
    print(f"\n  parameters per target: {n_params}   (paper reports 49)")
    return {"suite": suite, "targets": result, "n_parameters": n_params,
            "pass": bool(passed)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suites", type=str, nargs="+", default=["CAMELS-SAM", "CAMELS"],
                   choices=["CAMELS", "CAMELS-SAM"])
    p.add_argument("--rebuild", action="store_true",
                   help="recompute pairwise statistics instead of using the cache")
    args = p.parse_args()

    print("=" * 74)
    print("STAGE 3 -- LLS baseline (49 parameters, no neural network)")
    print("=" * 74)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "lls_baseline.json"
    results = []

    def save(complete: bool) -> None:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"stage": "3: LLS baseline", "complete": complete,
                       "band_stds": BAND_STDS,
                       "versions": {"numpy": np.__version__,
                                    "python": platform.python_version()},
                       "results": results}, fh, indent=2)

    save(False)
    for suite in args.suites:
        results.append(run_suite(suite, args.rebuild))
        save(len(results) == len(args.suites))

    print("\n" + "=" * 74)
    print(f"  LLS BASELINE: {'PASS' if all(r['pass'] for r in results) else 'FAIL'}")
    print(f"  wrote {out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
