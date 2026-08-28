"""Does the hard-edged uniform prior distort the coverage result?

    conda run -n ltuili python -u -m ili_kaai.checks.edgeCoverage

The prior is a box. Its walls sit exactly where the CAMELS design stops, so a true
parameter value near a wall has a posterior that cannot place mass beyond it. That
mass piles up against the wall and the credible interval shifts away from the truth,
which would depress coverage for those points specifically.

About a tenth of test points sit near a wall. If those points carry a large coverage
deficit, part of the measured overconfidence is an artefact of the prior rather than a
property of the architectures. This splits coverage by distance to the wall and
measures it, per parameter, otherwise identically to the sweep.

Predictions were written into runLog.md before this was first run.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.metrics import credible_coverage
from ili_kaai.architectures import ZOO
from ili_kaai.sweep import build, draw, seed_all
from ili_kaai.tasks import TASKS, load, prior_bounds

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[2] / "ili_kaai" / "results"
OUT = RESULTS / "edgeCoverage.json"


def near_wall(truth: np.ndarray, lo: List[float], hi: List[float],
              frac: float) -> np.ndarray:
    """Boolean per point per parameter: is this value within frac of a prior wall."""
    lo_a, hi_a = np.asarray(lo), np.asarray(hi)
    span = hi_a - lo_a
    return ((truth - lo_a) < frac * span) | ((hi_a - truth) < frac * span)


def split_coverage(samples: np.ndarray, truth: np.ndarray, edge: np.ndarray,
                   level: float = 0.68) -> Dict:
    """Coverage at edge points and interior points, computed per parameter."""
    q = [50.0 * (1.0 - level), 50.0 * (1.0 + level)]
    lo, hi = np.percentile(samples, q, axis=0)
    inside = (truth >= lo) & (truth <= hi)

    out = {"edge": [], "interior": [], "nEdge": [], "nInterior": []}
    for i in range(truth.shape[1]):
        m = edge[:, i]
        out["nEdge"].append(int(m.sum()))
        out["nInterior"].append(int((~m).sum()))
        out["edge"].append(round(float(inside[m, i].mean()), 4) if m.any() else None)
        out["interior"].append(
            round(float(inside[~m, i].mean()), 4) if (~m).any() else None)
    return out


def run(arch_key: str, task_key: str, seed: int, n_draws: int, frac: float,
        device: str) -> Dict:
    arch, task = ZOO[arch_key], TASKS[task_key]
    seed_all(seed)
    data = load(task)
    xtr = np.concatenate([data["train"][0], data["val"][0]])
    ttr = np.concatenate([data["train"][1], data["val"][1]])

    from ili.dataloaders import NumpyLoader
    runner, _ = build(arch, task, device, RESULTS / "runs" / f"edge_{arch_key}")
    posterior, _ = runner(loader=NumpyLoader(x=xtr, theta=ttr))

    xte, tte = data["test"]
    samples = draw(posterior, xte, n_draws, arch.sample_method, device)
    lo, hi = prior_bounds(task)
    res = split_coverage(samples, tte, near_wall(tte, lo, hi, frac))
    res.update({"architecture": arch_key, "task": task_key, "seed": seed,
                "labels": task.labels})
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--architectures", type=str, nargs="+",
                   default=["npeMaf", "npeMdn", "npeNsf"])
    p.add_argument("--slow-control", type=str, default="nreMlp",
                   help="one seed of the only entry inside the noise band")
    p.add_argument("--task", type=str, default="camelsJoint", choices=list(TASKS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--edge-fraction", type=float, default=0.10,
                   help="a point is 'edge' if within this fraction of the prior span")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    if not 0 < args.edge_fraction < 0.5:
        raise SystemExit("--edge-fraction must be between 0 and 0.5")

    rows: List[Dict] = []
    jobs = [(a, s) for a in args.architectures for s in args.seeds]
    if args.slow_control:
        jobs.append((args.slow_control, args.seeds[0]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"  task {args.task}, edge = within {args.edge_fraction:.0%} of a prior wall\n")
    print(f"  {'entry':16s}{'seed':>5s}{'param':>10s}{'edge cov':>10s}"
          f"{'interior':>10s}{'gap':>8s}{'n edge':>8s}")
    for arch_key, seed in jobs:
        r = run(arch_key, args.task, seed, args.n_draws, args.edge_fraction,
                args.device)
        rows.append(r)
        for i, name in enumerate(r["labels"]):
            gap = r["edge"][i] - r["interior"][i]
            print(f"  {arch_key:16s}{seed:5d}{name:>10s}{r['edge'][i]:10.3f}"
                  f"{r['interior'][i]:10.3f}{gap:+8.3f}{r['nEdge'][i]:8d}", flush=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump({"edgeFraction": args.edge_fraction, "task": args.task,
                       "nDraws": args.n_draws, "rows": rows}, fh, indent=2)

    gaps = [r["edge"][i] - r["interior"][i] for r in rows
            for i in range(len(r["labels"]))]
    print(f"\n  mean edge minus interior across all cells: {np.mean(gaps):+.4f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
