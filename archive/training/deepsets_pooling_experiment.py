"""STAGE 3, model 2 of 3 -- DeepSets on raw positions, and what pooling decides.

    python -u -m point_clouds.training.deepsets_pooling_experiment

Predictions were written into runLog.md BEFORE this was built.

THE EXPERIMENT
--------------
One architecture, one training recipe, one thing changed: how the per-galaxy
vectors are combined into a per-universe summary.

    sum    output scales with the number of galaxies -> the count is readable
    mean   the count is divided out
    max    keeps extremes, close to count-blind

In CAMELS the galaxy count is a documented shortcut to Omega_m. In CAMELS-SAM it
is fixed at 5000, so the shortcut does not exist. Crossing those two facts gives
a two-by-two: the same one-line change should matter enormously in one suite and
not at all in the other.

If that lands, then an architecture search can acquire a shortcut through a
choice nobody would think of as a shortcut. That is the paper's spine, shown
rather than argued.

REFERENCE LINE
--------------
A linear fit on the galaxy count alone, and nothing else, is reported alongside.
It is what a model gains from pure counting, so it says how much of any
sum-pooling score is the leak rather than structure.
"""

import argparse
import json
import platform
from typing import Dict, List

import numpy as np
import torch

from common.metrics import bootstrap_r2, r2_score, resolve_device
from point_clouds.pointnet import (POOLINGS, TARGETS, Batched, DeepSets, fit,
                                   load_positions, predict)
from point_clouds.training.step1_gate_2pcf import RESULTS_DIR


def count_only_reference(splits: Dict[str, tuple]) -> Dict:
    """R2 from a linear fit on the galaxy count alone. The pure-leak baseline."""
    def design(clouds):
        return np.hstack([np.array([[len(c)] for c in clouds], dtype=np.float64),
                          np.ones((len(clouds), 1))])

    train_clouds, train_y = splits["train"]
    test_clouds, test_y = splits["test"]
    weights = np.linalg.lstsq(design(train_clouds), train_y.astype(np.float64),
                              rcond=None)[0]
    predicted = design(test_clouds) @ weights
    scores = r2_score(predicted, test_y.astype(np.float64))
    return {t: float(scores[i]) for i, t in enumerate(TARGETS)}


def run_suite(suite: str, seeds: List[int], epochs: int, hidden: int,
              device: torch.device, limit: int) -> Dict:
    print("\n" + "=" * 76)
    print(suite)
    print("=" * 76)

    splits = {s: load_positions(suite, s, limit) for s in ("train", "val", "test")}
    # Train statistics measured once and handed to the other splits. Letting val
    # or test standardise on their own labels would leak them into scoring.
    train_batch = Batched(*splits["train"], device)
    stats = (train_batch.label_mean, train_batch.label_spread)
    batched = {"train": train_batch,
               **{s: Batched(*splits[s], device, label_stats=stats)
                  for s in ("val", "test")}}
    sizes = [len(c) for c in splits["train"][0]]
    print(f"  train {batched['train'].n_clouds} clouds, "
          f"{min(sizes)}-{max(sizes)} galaxies each, "
          f"{batched['train'].points.shape[0]:,} points total")

    reference = count_only_reference(splits)
    print(f"  count-only linear reference: "
          + "   ".join(f"{t} {reference[t]:+.4f}" for t in TARGETS))

    results = {"count_only_reference": reference, "pooling": {}}
    for pooling in POOLINGS:
        per_seed = []
        for seed in seeds:
            torch.manual_seed(seed)
            # A fixed constant taken from train, so it is identical for every
            # cloud and cannot itself carry per-cloud count information.
            model = DeepSets(hidden=hidden, pooling=pooling,
                             count_scale=float(np.mean(sizes)))
            fit(batched["train"], model, seed=seed, epochs=epochs)
            predicted = predict(model, batched["test"])
            per_seed.append(r2_score(predicted, splits["test"][1].astype(np.float64)))
        per_seed = np.stack(per_seed)
        mean, spread = per_seed.mean(0), per_seed.std(0)
        results["pooling"][pooling] = {
            "test_r2_mean": [float(v) for v in mean],
            "test_r2_seed_std": [float(v) for v in spread] if len(seeds) > 1 else None,
            "test_r2_per_seed": [[float(v) for v in row] for row in per_seed],
            "n_parameters": sum(p.numel() for p in model.parameters()),
        }
        spread_text = ("+/- " + " / ".join(f"{v:.4f}" for v in spread)
                       if len(seeds) > 1 else "(single run)")
        print(f"  {pooling:5s} pooling   "
              + "   ".join(f"{t} {mean[i]:+.4f}" for i, t in enumerate(TARGETS))
              + f"   {spread_text}", flush=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suites", type=str, nargs="+", default=["CAMELS", "CAMELS-SAM"],
                   choices=["CAMELS", "CAMELS-SAM"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="clouds per split, 0 means all")
    p.add_argument("--device", type=str, default="mps",
                   choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")

    device = resolve_device(args.device)
    print("=" * 76)
    print("STAGE 3 -- DeepSets on raw positions: does pooling decide the leak?")
    print("=" * 76)
    print(f"  device {device}   seeds {args.seeds}   {args.epochs} epochs   "
          f"hidden {args.hidden}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "deepsets_pooling.json"
    results = {}

    def save(complete: bool) -> None:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"stage": "3: DeepSets pooling experiment", "complete": complete,
                       "seeds": args.seeds, "epochs": args.epochs, "hidden": args.hidden,
                       "device": str(device),
                       "versions": {"torch": torch.__version__,
                                    "numpy": np.__version__,
                                    "python": platform.python_version()},
                       "results": results}, fh, indent=2)

    save(False)
    for suite in args.suites:
        results[suite] = run_suite(suite, args.seeds, args.epochs, args.hidden,
                                   device, args.limit)
        save(len(results) == len(args.suites))

    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
