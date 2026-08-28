"""Trainable parameter count per zoo entry.

The sweep recorded zero for every entry: sbi wraps the density estimator so that
neither `posterior.parameters()` nor `posterior.posteriors` exists on the returned
object. Counting is deterministic given the architecture and the task shapes, so it is
measured here by building each net once rather than by re-running 72 trainings.
"""

import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

import ili
from ili_kaai.architectures import ZOO
from ili_kaai.tasks import TASKS, load, prior_bounds

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "ili_kaai" / "results" / "paramCount.json"


def count(arch, task) -> int:
    """Build the estimator on real shapes and count. sbi builds lazily, so the net
    only materialises once it has seen a batch."""
    x, theta = load(task)["train"]
    lo, hi = prior_bounds(task)
    prior = ili.utils.Uniform(low=lo, high=hi, device="cpu")
    build = ili.utils.load_nde_sbi(engine=arch.engine, model=arch.model,
                                   **arch.model_args)[0]   # returns a list of builders
    xt = torch.tensor(x[:64], dtype=torch.float32)
    tt = torch.tensor(theta[:64], dtype=torch.float32)
    net = build(tt, xt)
    per_member = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return int(per_member * arch.repeats)


def main() -> None:
    out: Dict[str, Dict[str, int]] = {}
    for tk, task in TASKS.items():
        for ak, arch in ZOO.items():
            try:
                out.setdefault(ak, {})[tk] = count(arch, task)
            except Exception as exc:
                out.setdefault(ak, {})[tk] = -1
                print(f"  {ak} {tk} FAILED {type(exc).__name__}: {exc}")
    for ak, row in out.items():
        print(f"  {ak:17s} " + "  ".join(f"{tk} {v:>8,}" for tk, v in row.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
