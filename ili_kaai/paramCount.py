"""Trainable parameter count per zoo entry, per task.

    conda run -n ltuili python -m ili_kaai.paramCount

The sweep records zero for every entry: sbi wraps the density estimator so that neither
`posterior.parameters()` nor `posterior.posteriors` exists on the returned object.
Counting is deterministic given the architecture and the task shapes, so it is measured
here by building each net once rather than by re-running every training.

WHY THIS GOES THROUGH sweep.build RATHER THAN REBUILDING THE NETS
-----------------------------------------------------------------
The first version called ili.utils.load_nde_sbi directly. That knows one of the three
build paths the sweep actually uses, so it wrote -1 for all eight lampe entries and all
seven embedding entries: 22 of 30 entries had no parameter count in a field the
catalogue advertises. Worse, a count from a net built differently from the one that was
trained is not a measurement of anything.

So this calls the same build() the sweep calls, with pretraining skipped because
counting weights does not need a trained embedding. What is counted is what was
trained.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ili_kaai.architectures import ZOO, Architecture
from ili_kaai.sweep import build
from ili_kaai.tasks import TASKS, Task, load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ili_kaai" / "results" / "paramCount.json"


def materialise(net, theta: torch.Tensor, x: torch.Tensor, prior,
                backend: str) -> nn.Module:
    """Turn whatever the backend returned into a module whose weights can be counted.

    Neither backend hands back a module. sbi returns a builder called as (theta, x).
    lampe returns a _Lampe_Net_Constructor called as (x, theta, prior), the opposite
    argument order plus the prior. Dispatch on the entry's declared backend rather
    than probing, so a silent signature change fails loudly instead of miscounting.
    """
    if isinstance(net, nn.Module):
        return net
    if backend == "lampe":
        return net(x, theta, prior)
    return net(theta, x)


def trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count(arch: Architecture, task: Task, device: str = "cpu") -> Optional[int]:
    """Total trainable weights across every member the entry trains.

    An embedding shared across ensemble members would be counted once per member. No
    current entry both embeds and ensembles, and this returns None rather than a wrong
    number if that ever changes.
    """
    if arch.embedding and arch.repeats > 1:
        return None
    data = load(task)
    x, theta = data["train"]
    xt = torch.tensor(np.asarray(x[:64]), dtype=torch.float32, device=device)
    tt = torch.tensor(np.asarray(theta[:64]), dtype=torch.float32, device=device)
    runner, prior, _pretrain = build(arch, task, device, ROOT / "ili_kaai" / "results",
                                     pretrain=False)
    return sum(trainable(materialise(net, tt, xt, prior, arch.backend))
               for net in runner.nets)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    out: Dict[str, Dict[str, Optional[int]]] = {}
    failures = 0
    for ak, arch in ZOO.items():
        for tk, task in TASKS.items():
            # Skip pairs the entry could never run. A summary vector estimator on a
            # point cloud is a type error, not a failed measurement.
            if bool(arch.embedding) != (task.modality == "point_cloud"):
                continue
            try:
                out.setdefault(ak, {})[tk] = count(arch, task, args.device)
            except Exception as exc:
                # null, never -1 and never 0. A count that was not obtained is absent,
                # and a number here would be averaged into totals downstream.
                out.setdefault(ak, {})[tk] = None
                failures += 1
                print(f"  {ak} {tk} FAILED {type(exc).__name__}: {exc}")

    for ak, row in out.items():
        cells = "  ".join(f"{tk} {'null' if v is None else format(v, ',')}"
                          for tk, v in row.items())
        print(f"  {ak:29s} {cells}")

    counted = sum(1 for row in out.values() for v in row.values() if v is not None)
    total = sum(len(row) for row in out.values())
    print(f"\n  {counted}/{total} entry-task pairs counted, {failures} failed")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
