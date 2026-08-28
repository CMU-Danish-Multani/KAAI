"""STAGE 3, model 3 of 3 -- radius-graph message passing, and does pooling still decide?

    python -u -m point_clouds.training.gnn_experiment

TWO QUESTIONS AT ONCE
---------------------
1. Put a graph network in our own pipeline, so the comparison against a simple
   summary becomes our measurement rather than the paper's.
2. Test whether the DeepSets pooling result generalises. Measured 2026-08-18: a
   set model reached 0.5233 on CAMELS Omega_m with sum pooling against a
   count-only reference of 0.5058, and -0.0006 with mean pooling. If the same
   one-word change moves a graph network the same way, the finding is about
   architectures in general rather than about one weak model.

HOW THIS DIFFERS FROM COSMOBENCH, STATED PLAINLY
------------------------------------------------
This is not a strict reproduction and should not be reported as one.

    theirs                            ours
    absolute positions as node        constant node features
      features, plus two dot
      products built from them
    E(3)-invariant feature variants   separation d_ij / Rc only
    1.0M to 1.2M parameters           67k parameters
    searched over 100 configs         one fixed config

The node-feature choice is deliberate. Absolute position inside a periodic box
is meaningless, since the origin is arbitrary and the box wraps, so a universe
shifted sideways is the same universe. Feeding positions directly lets a model
key on something that cannot generalise.

Published numbers are therefore a REFERENCE POINT, not a pass or fail gate.
Landing near them means our graph network is in the right regime. Landing far
off is information, not necessarily a bug.
"""

import argparse
import json
import platform
import re
import subprocess
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from common.metrics import r2_score, resolve_device, seed_all
from point_clouds.gnn import GraphSet, MessagePassingNet
from point_clouds.load import BOX
from point_clouds.pointnet import TARGETS, load_positions
from point_clouds.training.step1_gate_2pcf import RESULTS_DIR

# CosmoBench Table 2, GNN row. Reference only, see the module docstring.
PUBLISHED_GNN = {
    "CAMELS-SAM": {"Omega_m": (0.75, 0.03), "sigma_8": (0.83, 0.02)},
    "CAMELS":     {"Omega_m": (0.78, 0.03), "sigma_8": (0.24, 0.06)},
}


def release_cache(device: torch.device) -> None:
    """Hand pooled GPU memory back to the system.

    MEASURED 2026-08-20, and this is not a micro-optimisation. Every batch has a
    different number of edges, because clouds differ in density, so the caching
    allocator sees thousands of distinct tensor shapes and cannot reuse its
    pools. Left alone it accumulated 14.2 GB of cache while only 0.02 GB was
    live. One model survives that; three seeds followed by a larger suite do not,
    and a 24 GB machine starts swapping. A run left overnight in that state
    produced nothing in 13.7 hours at a 20 percent CPU duty cycle.

    Clearing each epoch costs nothing measurable: 1.72 against 1.74 s per epoch,
    with the cache held down from 14.21 GB to 1.33 GB.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def swap_gb() -> float:
    """System swap in use. On a unified-memory Mac this is the honest signal.

    Resident memory does not capture GPU allocations or pages that have spilled
    to swap. A memory alarm watching resident memory sat silent through 80
    minutes of thrashing on 2026-08-20 while swap climbed to 24 GB.
    """
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    found = re.search(r"used = ([\d.]+)M", out)
    return float(found.group(1)) / 1024 if found else float("nan")


def train_and_score(train: GraphSet, test: GraphSet, pooling: str, seed: int,
                    epochs: int, batch_size: int, hidden: int, layers: int,
                    learning_rate: float, device: torch.device,
                    progress: bool = False) -> np.ndarray:
    t_start = time.time()
    seed_all(seed)
    model = MessagePassingNet(hidden=hidden, layers=layers, pooling=pooling).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    rng = np.random.default_rng(seed)

    # Built once and reused. See GraphSet.batches for why this matters.
    batches = train.batches(batch_size)
    model.train()
    for epoch in range(epochs):
        for i in rng.permutation(len(batches)):
            batch = batches[i]
            optimiser.zero_grad()
            loss_function(model(batch), batch.y).backward()
            optimiser.step()
        if progress and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:4d}/{epochs}  {time.time() - t_start:6.1f}s "
                  f"elapsed  swap {swap_gb():5.2f} GB", flush=True)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test.batches(batch_size):
            predictions.append(model(batch).cpu().numpy())
    scaled = np.concatenate(predictions).astype(np.float64)
    physical = scaled * test.label_spread + test.label_mean
    truth = test.y_scaled.astype(np.float64) * test.label_spread + test.label_mean
    del model, optimiser
    release_cache(device)
    return r2_score(physical, truth)


def run_suite(suite: str, args, device: torch.device) -> Dict:
    print("\n" + "=" * 78)
    print(suite)
    print("=" * 78)
    box = BOX[suite]
    splits = {}
    for name in ("train", "test"):
        clouds, y = load_positions(suite, name)
        stats = None if name == "train" else (splits["train"].label_mean,
                                              splits["train"].label_spread)
        splits[name] = GraphSet([c * box for c in clouds], y, box,
                                args.cutoff, device, label_stats=stats)
    print(f"  train {splits['train'].n_clouds} clouds, "
          f"{splits['train'].n_nodes:,} galaxies, cutoff {args.cutoff:g} x box")

    out = {}
    for pooling in args.poolings:
        per_seed, t0 = [], time.time()
        for seed in args.seeds:
            per_seed.append(train_and_score(
                splits["train"], splits["test"], pooling, seed, args.epochs,
                args.batch_size, args.hidden, args.layers, args.lr, device,
                progress=args.progress))
        per_seed = np.stack(per_seed)
        mean, spread = per_seed.mean(0), per_seed.std(0)
        minutes = (time.time() - t0) / 60
        out[pooling] = {
            "test_r2_mean": [float(v) for v in mean],
            "test_r2_seed_std": [float(v) for v in spread] if len(args.seeds) > 1 else None,
            "test_r2_per_seed": [[float(v) for v in row] for row in per_seed],
            "minutes": round(minutes, 1),
        }
        # One seed has no spread. Printing +/- 0.0000 would read as an extremely
        # tight result rather than as no result at all.
        line = "   ".join(
            f"{t} {mean[i]:+.4f}" + (f" +/- {spread[i]:.4f}" if len(args.seeds) > 1
                                     else " (single run)")
            for i, t in enumerate(TARGETS))
        print(f"  {pooling:5s} pooling   {line}   [{minutes:.1f} min]", flush=True)

    for i, t in enumerate(TARGETS):
        mu, sd = PUBLISHED_GNN[suite][t]
        print(f"    reference, published GNN {t}: {mu:.2f} +/- {sd:.2f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suites", type=str, nargs="+", default=["CAMELS", "CAMELS-SAM"],
                   choices=["CAMELS", "CAMELS-SAM"])
    p.add_argument("--poolings", type=str, nargs="+", default=["mean", "sum"],
                   choices=["mean", "sum", "max"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32,
                   help="32 measured 2x faster per epoch than 8 on MPS")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cutoff", type=float, default=0.015,
                   help="radius as a fraction of the box side; paper searched 0.01 to 0.02")
    p.add_argument("--progress", action="store_true",
                   help="print per-epoch timing and swap, so slow is never "
                        "mistaken for stuck")
    p.add_argument("--device", type=str, default="mps",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="mps measured 6.4x faster than cpu on this workload")
    args = p.parse_args()

    for name, value in (("--epochs", args.epochs), ("--batch-size", args.batch_size),
                        ("--hidden", args.hidden), ("--layers", args.layers)):
        if value < 1:
            raise SystemExit(f"{name} must be at least 1")

    device = resolve_device(args.device)
    print("=" * 78)
    print("STAGE 3 -- graph neural network on raw positions")
    print("=" * 78)
    print(f"  device {device}   seeds {args.seeds}   {args.epochs} epochs   "
          f"batch {args.batch_size}   hidden {args.hidden}   layers {args.layers}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "gnn_experiment.json"
    results = {}

    def save(complete: bool) -> None:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"stage": "3: GNN", "complete": complete,
                       "config": vars(args), "device": str(device),
                       "published_reference": PUBLISHED_GNN,
                       "versions": {"torch": torch.__version__,
                                    "numpy": np.__version__,
                                    "python": platform.python_version()},
                       "results": results}, fh, indent=2)

    save(False)
    for suite in args.suites:
        results[suite] = run_suite(suite, args, device)
        save(len(results) == len(args.suites))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
