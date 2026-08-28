"""Three CAMELS inference tasks, and honest scoring for posteriors.

WHY POSTERIORS NEED A SECOND ADMISSION CHECK
--------------------------------------------
Everything measured in this project so far is R2, which says how close a point
prediction lands. It says nothing about whether a model's stated uncertainty is
honest. A model that is confidently wrong scores well on R2 and is useless to a
cosmologist, because a measurement of the universe without a trustworthy error
bar is not a measurement.

So a zoo entry that outputs a posterior has to clear a second bar. We use
empirical coverage: when the model says it is ninety percent confident, is the
truth inside that interval ninety percent of the time. A model that claims more
certainty than it has is OVERCONFIDENT, which is the dangerous direction,
because it produces tight error bars around wrong answers.

THE THREE TASKS
---------------
All three read the same input, the two-point correlation function computed in
point_clouds/tpcf.py, so the comparison isolates the inference head rather than
confounding it with a different summary. Matched compute per the brief.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from point_clouds.tpcf import load_or_build, to_features

# The ranges the simulations actually sampled. These are the prior.
PARAM_LIMITS = {"Omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0)}
NOMINAL_LEVELS = (0.50, 0.68, 0.90, 0.95)


@dataclass
class Task:
    key: str
    targets: Tuple[str, ...]
    description: str


TASKS: Tuple[Task, ...] = (
    Task("camels_omega_m_posterior", ("Omega_m",),
         "Posterior over the matter density from a galaxy correlation function."),
    Task("camels_sigma_8_posterior", ("sigma_8",),
         "Posterior over the clustering amplitude from a galaxy correlation function."),
    Task("camels_joint_posterior", ("Omega_m", "sigma_8"),
         "Joint posterior over both cosmological parameters."),
)

ALL_TARGETS = ("Omega_m", "sigma_8")


def load_task(task: Task, suite: str = "CAMELS") -> Dict[str, np.ndarray]:
    """Features and parameters for one task, standardised on TRAIN statistics."""
    raw = {s: load_or_build(suite, s) for s in ("train", "val", "test")}
    x = {s: to_features(raw[s]["xi"]) for s in raw}
    keep = x["train"].std(0) > 1e-6
    x = {s: v[:, keep] for s, v in x.items()}
    mean, spread = x["train"].mean(0), x["train"].std(0)
    x = {s: ((v - mean) / spread).astype(np.float32) for s, v in x.items()}

    idx = [ALL_TARGETS.index(t) for t in task.targets]
    y = {s: raw[s]["y"][:, idx].astype(np.float32) for s in raw}
    return {"x": x, "theta": y, "n_features": int(keep.sum())}


def coverage(samples: np.ndarray, truth: np.ndarray,
             levels: Tuple[float, ...] = NOMINAL_LEVELS) -> Dict[str, float]:
    """Fraction of test cases where the truth falls inside each central interval.

    samples is (n_test, n_draws, n_params); truth is (n_test, n_params).

    A well calibrated posterior returns roughly the nominal level. Below nominal
    means overconfident, which is the failure that matters: tight error bars
    around wrong answers. Above nominal means needlessly cautious, which wastes
    information but does not mislead.
    """
    out: Dict[str, float] = {}
    errors: List[float] = []
    for level in levels:
        lo = np.quantile(samples, (1 - level) / 2, axis=1)
        hi = np.quantile(samples, 1 - (1 - level) / 2, axis=1)
        inside = float(np.mean(np.all((truth >= lo) & (truth <= hi), axis=1)))
        out[f"coverage_{int(level * 100)}"] = round(inside, 4)
        errors.append(abs(inside - level))
    out["calibration_error"] = round(float(np.mean(errors)), 4)
    out["overconfident"] = bool(np.mean([out[f"coverage_{int(l*100)}"] - l
                                         for l in levels]) < -0.05)
    return out


def posterior_r2(samples: np.ndarray, truth: np.ndarray) -> List[float]:
    """R2 of the posterior mean, so posterior entries stay comparable to the rest."""
    predicted = samples.mean(axis=1)
    out = []
    for i in range(truth.shape[1]):
        residual = ((predicted[:, i] - truth[:, i]) ** 2).sum()
        total = ((truth[:, i] - truth[:, i].mean()) ** 2).sum()
        out.append(round(float(1 - residual / total), 4))
    return out
