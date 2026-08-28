"""Does a block leak the number of points? The project's standard leak screen.

WHY THE OBVIOUS TEST IS WRONG
-----------------------------
The intuitive check is duplication: feed a cloud, then feed the same cloud with
every point repeated twice, and see whether the output moves. It is wrong, and
it fails in the most dangerous direction.

Duplicating every point cannot move a maximum or a minimum. So `max` and `min`
aggregation return bit-identical output under duplication and appear perfectly
count-blind, while in fact tracking log N at Pearson r = +0.87. Measured
2026-08-24 during review of pna.py. A screen built on duplication would have
certified the leakiest aggregators as clean.

THE TEST THAT WORKS
-------------------
Vary N while holding the point DISTRIBUTION fixed, so the count is the only
thing that changes, then ask whether a held-out linear probe can recover N from
the block's output.

Recoverability is the statistic that matters. A four percent systematic shift
with a small spread is a near-perfect regressor for N, not a residual, which is
exactly how the pna.py caveat understated its own leak.

Reading R2(N):
    near 0 or negative   the block cannot see the count
    above ~0.3           a count channel exists
    above ~0.5           stronger than the 0.73 count-to-Omega_m correlation
                         this project guards against, once squared
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# The CAMELS galaxy-count range, measured across all three splits.
CAMELS_COUNT_RANGE = (588, 4511)
LEAK_THRESHOLD = 0.30


def _probe_r2(features: np.ndarray, counts: np.ndarray, seed: int) -> float:
    """Held-out R2 of a linear probe recovering the count from block output.

    Train and test are split in half. A negative value is the no-signal
    signature: the probe overfits noise and generalises worse than the mean.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(counts))
    half = len(counts) // 2
    train, test = order[:half], order[half:]

    target = np.log(counts.astype(np.float64))
    design = np.hstack([features, np.ones((len(features), 1))])
    weights, *_ = np.linalg.lstsq(design[train], target[train], rcond=None)
    predicted = design[test] @ weights

    residual = ((predicted - target[test]) ** 2).sum()
    total = ((target[test] - target[test].mean()) ** 2).sum()
    return float(1.0 - residual / total) if total > 0 else float("nan")


def screen(block: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
           feature_dim: int = 64, n_clouds: int = 400,
           count_range: Tuple[int, int] = CAMELS_COUNT_RANGE,
           seeds: Optional[List[int]] = None,
           device: Optional[torch.device] = None) -> Dict[str, float]:
    """Measure whether `block` leaks the point count.

    `block` takes (points, index, n_clouds) and returns (n_clouds, out_dim),
    matching the signature of point_clouds.pointnet.pool consumers.

    Every cloud is drawn from ONE fixed distribution, so the count is the only
    thing that differs between clouds. Any recoverable signal is a count channel.
    """
    seeds = seeds or [0, 1, 2]
    device = device or torch.device("cpu")
    scores = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        counts = rng.integers(count_range[0], count_range[1] + 1, size=n_clouds)

        points = torch.randn(int(counts.sum()), feature_dim,
                             generator=torch.Generator().manual_seed(seed)) + 1.0
        index = torch.as_tensor(np.repeat(np.arange(n_clouds), counts), dtype=torch.long)

        with torch.no_grad():
            out = block(points.to(device), index.to(device), n_clouds)
        scores.append(_probe_r2(out.detach().cpu().numpy().astype(np.float64),
                                counts, seed))

    scores = np.array(scores)
    return {"r2_recovering_count": float(scores.mean()),
            "spread": float(scores.std()) if len(seeds) > 1 else None,
            "leaks": bool(scores.mean() > LEAK_THRESHOLD),
            "n_seeds": len(seeds)}


def report(name: str, result: Dict[str, float]) -> str:
    spread = (f" +/- {result['spread']:.4f}" if result["spread"] is not None
              else " (single run)")
    verdict = "LEAKS THE COUNT" if result["leaks"] else "count-blind"
    return f"  {name:34s} R2(N) {result['r2_recovering_count']:+.4f}{spread}   {verdict}"
