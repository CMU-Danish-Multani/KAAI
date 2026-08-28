"""The three CAMELS inference tasks the zoo is benchmarked on.

Every entry in the zoo is evaluated on all three at matched compute, which is what
makes the standardised evaluation results comparable across architectures.

The tasks vary on the two axes the Claude skill takes as input, so that a
recommendation has something to discriminate on:

    task                 modality          dim(x)  dim(theta)  data regime
    camelsJoint          summary vector    25      2           N varies, 588 to 4511
    camelsOmega          summary vector    25      1           N varies
    camelsSamJoint       summary vector    25      2           N fixed at 5000

The data vector is the two-point correlation function in 25 log spaced bins, the
binning CosmoBench Table 7 used, recomputed from positions by point_clouds/tpcf.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "tpcf_cache"
QUIJOTE = ROOT / "data" / "Quijote"
LOG_FLOOR = 1e-6
PARAM_NAMES = ("Omega_m", "sigma_8")
QUIJOTE_PARAMS = ("Omega_b", "Omega_m", "h", "n_s", "sigma_8")


@dataclass(frozen=True)
class Task:
    key: str
    suite: str
    params: Tuple[int, ...]          # column indices into the label array
    modality: str
    description: str

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def labels(self) -> List[str]:
        names = QUIJOTE_PARAMS if self.suite == "Quijote" else PARAM_NAMES
        return [names[i] for i in self.params]


TASKS: Dict[str, Task] = {
    "camelsJoint": Task(
        key="camelsJoint", suite="CAMELS", params=(0, 1), modality="summary_vector",
        description="Joint Omega_m and sigma_8 from the galaxy two-point correlation "
                    "function in a 25 Mpc/h CAMELS box. Galaxy counts vary between "
                    "588 and 4511 across simulations."),
    "camelsOmega": Task(
        key="camelsOmega", suite="CAMELS", params=(0,), modality="summary_vector",
        description="Omega_m alone from the same CAMELS correlation function. A one "
                    "dimensional posterior, so nothing tests joint degeneracy here."),
    "camelsSamJoint": Task(
        key="camelsSamJoint", suite="CAMELS-SAM", params=(0, 1),
        modality="summary_vector",
        description="Joint Omega_m and sigma_8 from the correlation function of a "
                    "100 Mpc/h CAMELS-SAM box, where every catalogue is cut to exactly "
                    "5000 galaxies so the count carries no information."),
    "quijoteAll": Task(
        key="quijoteAll", suite="Quijote", params=(0, 1, 2, 3, 4),
        modality="summary_vector",
        description="Five cosmological parameters from the correlation function of a "
                    "1000 Mpc/h Quijote box, 19651 training simulations. Every "
                    "catalogue holds exactly 5000 halos. Clustering alone is known to "
                    "be insensitive to the expansion rate, so h should recover poorly."),
    "quijoteJoint": Task(
        key="quijoteJoint", suite="Quijote", params=(1, 4),
        modality="summary_vector",
        description="Omega_m and sigma_8 only, from the same Quijote correlation "
                    "function. Matched to the CAMELS tasks in parameter count so the "
                    "only thing that changes is the data regime."),
}


def _standardise(x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalise on TRAIN statistics only. Constant columns are dropped, not divided
    by an epsilon, which once produced an R2 of -1.7e11."""
    keep = x["train"].std(0) > 1e-8
    mu, sd = x["train"][:, keep].mean(0), x["train"][:, keep].std(0)
    return {s: ((v[:, keep] - mu) / sd).astype(np.float32) for s, v in x.items()}


def _load_quijote(task: Task) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Quijote ships its own correlation function, so nothing is recomputed here.

    The shipped binning is 24 bins from 2 to 80 Mpc/h, which is NOT the binning
    CosmoBench Table 2 used. Numbers from this task therefore do not compare to
    published Quijote results. That is acceptable because the zoo compares
    architectures against each other, where only consistency across entries matters.
    """
    raw = {}
    for split in ("train", "val", "test"):
        path = QUIJOTE / f"tpcf_top5000_{split}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing")
        with h5py.File(path, "r") as f:
            xi = f["tpcf"][:]
            y = np.stack([f["params"][QUIJOTE_PARAMS[i]][:] for i in task.params],
                         axis=1)
        raw[split] = (np.log10(np.abs(xi) + LOG_FLOOR), y)

    x = _standardise({s: v[0] for s, v in raw.items()})
    return {s: (x[s], raw[s][1].astype(np.float32)) for s in raw}


def load(task: Task) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Returns {split: (x, theta)} with x standardised on train statistics.

    theta is left in physical units. LtU-ILI handles parameter scaling through the
    prior, and leaving theta physical keeps the reported R2 comparable to published
    numbers.
    """
    if task.suite == "Quijote":
        return _load_quijote(task)

    raw = {}
    for split in ("train", "val", "test"):
        path = CACHE / f"{task.suite}_{split}_25bins_0.0125_12.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Build it with: python -m point_clouds.tpcf")
        with np.load(path) as d:
            raw[split] = (np.log10(np.abs(d["xi"]) + LOG_FLOOR), d["y"])

    x = _standardise({s: v[0] for s, v in raw.items()})
    return {s: (x[s], raw[s][1][:, list(task.params)].astype(np.float32))
            for s in raw}


def prior_bounds(task: Task) -> Tuple[List[float], List[float]]:
    """Uniform prior over the CAMELS Latin hypercube ranges, per parameter.

    Taken from the simulation suite design, not fitted to the training labels, so the
    prior does not quietly encode the test set.
    """
    if task.suite == "Quijote":
        lo_all = [0.02, 0.10, 0.50, 0.80, 0.60]      # Quijote latin hypercube design
        hi_all = [0.08, 0.50, 0.90, 1.20, 1.00]
    else:
        lo_all, hi_all = [0.1, 0.6], [0.5, 1.0]
    return ([lo_all[i] for i in task.params], [hi_all[i] for i in task.params])


def summary() -> None:
    for t in TASKS.values():
        d = load(t)
        print(f"  {t.key:16s} {t.suite:11s} x {d['train'][0].shape[1]:2d} dims   "
              f"theta {t.n_params}   "
              f"train {len(d['train'][0])}  val {len(d['val'][0])}  "
              f"test {len(d['test'][0])}   {t.labels}")


if __name__ == "__main__":
    summary()
