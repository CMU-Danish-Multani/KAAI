"""Turn cached correlation functions into model-ready feature tables.

One entry point, `assemble()`. Everything else is called by it.

    assemble(suite, ...)
      |
      +-- load_or_build()   <- tpcf.py, reads or computes xi(r) per cloud
      +-- to_features()     <- tpcf.py, absolute value then log10
      +-- standardise()     <- rescale on TRAIN statistics only

WHY TRAIN-ONLY STATISTICS
-------------------------
Measuring the mean and spread across all three splits would let facts about the
test set influence training. Scores come out higher and are wrong, and nothing
about the run looks unusual while it happens.

WHY CONSTANT FEATURES ARE DROPPED RATHER THAN CLIPPED
-----------------------------------------------------
The obvious way to avoid dividing by a zero spread is to clip it to a small
number. That is worse than it looks. A feature that never varies in training
still varies slightly in validation, and dividing that variation by 1e-8 produces
inputs of order 1e8 which destroy the network. Measured 2026-08-17: trimming
clouds to 588 galaxies empties the innermost correlation bin, and the clipped
version returned an R2 of -1.7e11. Dropping the column instead is both correct
and visible, since the count of dropped columns is printed and recorded.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from point_clouds.tpcf import load_or_build, to_features

# A feature whose training spread falls below this is treated as carrying no
# information at all, and is removed rather than rescaled.
CONSTANT_TOLERANCE = 1e-6

SPLITS = ("train", "val", "test")

FeatureTable = Dict[str, Dict[str, np.ndarray]]
LabelStats = Tuple[np.ndarray, np.ndarray]


def standardise(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray],
                label: str = "") -> Tuple[FeatureTable, LabelStats]:
    """Rescale features and labels to mean 0, spread 1, using TRAIN statistics.

    Returns the rescaled table plus the label mean and spread, which callers
    need in order to convert predictions back into physical units before
    scoring.
    """
    keep = x["train"].std(0) > CONSTANT_TOLERANCE
    if not keep.all():
        print(f"    {label}dropped {int((~keep).sum())} of {len(keep)} features "
              f"as constant in train", flush=True)
    if not keep.any():
        raise AssertionError(f"{label}every feature is constant in train")

    x = {split: values[:, keep] for split, values in x.items()}
    x_mean, x_spread = x["train"].mean(0), x["train"].std(0)
    y_mean, y_spread = y["train"].mean(0), y["train"].std(0)

    table = {split: {"x": ((x[split] - x_mean) / x_spread).astype(np.float32),
                     "y_scaled": ((y[split] - y_mean) / y_spread).astype(np.float32),
                     "y_physical": y[split].astype(np.float64)}
             for split in x}
    return table, (y_mean, y_spread)


def assemble(suite: str, fixed_count: int = 0, with_galaxy_count: bool = False,
             splits: Optional[List[str]] = None) -> Tuple[FeatureTable, LabelStats, dict]:
    """Feature table, label statistics, and the raw cached arrays, for one suite.

    fixed_count      keep only the N most massive galaxies per cloud, which
                     closes the counting leak by construction.
    with_galaxy_count append the galaxy count as an extra input column, which
                     hands the model the leak explicitly.
    """
    splits = list(splits or SPLITS)
    raw = {split: load_or_build(suite, split, fixed_count=fixed_count)
           for split in splits}

    x = {split: to_features(raw[split]["xi"]) for split in splits}
    if with_galaxy_count:
        for split in splits:
            counts = raw[split]["ngal"][:, None].astype(float)
            x[split] = np.concatenate([x[split], counts], axis=1)

    label = suite + (f"/top{fixed_count}" if fixed_count else "")
    label += "+count: " if with_galaxy_count else ": "

    table, label_stats = standardise(x, {s: raw[s]["y"] for s in splits}, label)
    return table, label_stats, raw
