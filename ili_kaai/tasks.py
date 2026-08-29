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

from dataclasses import dataclass, field
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

# The CAMELS latin hypercube design box. Taken from the suite design, and checked
# against the labels: every observed range matches to four decimals, so these are the
# design bounds rather than bounds fitted to the data we train on.
CAMELS_BOX = {"Omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0),
              "A_SN1": (0.25, 4.0), "A_AGN1": (0.25, 4.0),
              "A_SN2": (0.5, 2.0), "A_AGN2": (0.5, 2.0)}
CLOUD_POINTS = 512


@dataclass(frozen=True)
class Task:
    key: str
    suite: str
    params: Tuple[int, ...]          # column indices into the label array
    modality: str
    description: str
    # Point cloud tasks select parameters by name, because the cached label order is
    # written by the loader rather than fixed here.
    paramNames: Tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        return len(self.paramNames) if self.paramNames else len(self.params)

    @property
    def labels(self) -> List[str]:
        if self.paramNames:
            return list(self.paramNames)
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
    "camelsCloud": Task(
        key="camelsCloud", suite="CAMELS", params=(), modality="point_cloud",
        paramNames=("Omega_m", "sigma_8"),
        description="Omega_m and sigma_8 from the raw 3D positions of the 512 most "
                    "massive galaxies in a 25 Mpc/h CAMELS box. Same parameters and "
                    "same simulations as camelsJoint, so the only thing that differs "
                    "is the modality: a set of 512 points instead of a 25 number "
                    "summary. Requires a permutation invariant embedding."),
    "camelsSamCloud": Task(
        key="camelsSamCloud", suite="CAMELS-SAM", params=(), modality="point_cloud",
        paramNames=("Omega_m", "sigma_8"),
        description="The same point cloud task on a 100 Mpc/h CAMELS-SAM box, where "
                    "every catalogue holds exactly 5000 galaxies before trimming."),
    "camelsCloudAll": Task(
        key="camelsCloudAll", suite="CAMELS", params=(), modality="point_cloud",
        paramNames=("Omega_m", "sigma_8", "A_SN1", "A_SN2", "A_AGN1", "A_AGN2"),
        description="All six CAMELS parameters from the same point cloud: two "
                    "cosmological and four supernova and AGN feedback parameters. "
                    "Raises dim(theta) from 2 to 6 with no new data, which is the "
                    "axis both Thiele Section 2.7 and Deistler Table 1 say should "
                    "shift the balance from NPE toward NLE."),
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


def _load_cloud(task: Task) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Fixed size point clouds from the cache built by point_clouds.cloudCache.

    x is (n_sims, n_points, 3) with positions already scaled to [0, 1] by box side, so
    the same embedding network reads either suite without rescaling. No standardisation
    is applied: a position is already in natural units and centring it would destroy
    the periodic box structure the geometry lives in.
    """
    from point_clouds.cloudCache import load_or_build
    out = {}
    for split in ("train", "val", "test"):
        d = load_or_build(task.suite, split, CLOUD_POINTS)
        names = [str(n) for n in d["labelNames"]]
        missing = set(task.paramNames) - set(names)
        if missing:
            raise KeyError(f"{task.suite} has no labels {sorted(missing)}; "
                           f"available: {names}")
        cols = [names.index(n) for n in task.paramNames]
        out[split] = (d["clouds"].astype(np.float32),
                      d["y"][:, cols].astype(np.float32))
    return out


def load(task: Task) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Returns {split: (x, theta)} with x standardised on train statistics.

    theta is left in physical units. LtU-ILI handles parameter scaling through the
    prior, and leaving theta physical keeps the reported R2 comparable to published
    numbers.
    """
    if task.modality == "point_cloud":
        return _load_cloud(task)
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
    if task.paramNames:
        return ([CAMELS_BOX[n][0] for n in task.paramNames],
                [CAMELS_BOX[n][1] for n in task.paramNames])
    if task.suite == "Quijote":
        lo_all = [0.02, 0.10, 0.50, 0.80, 0.60]      # Quijote latin hypercube design
        hi_all = [0.08, 0.50, 0.90, 1.20, 1.00]
    else:
        lo_all, hi_all = [0.1, 0.6], [0.5, 1.0]
    return ([lo_all[i] for i in task.params], [hi_all[i] for i in task.params])


def summary() -> None:
    for t in TASKS.values():
        d = load(t)
        shape = "x".join(str(v) for v in d["train"][0].shape[1:])
        print(f"  {t.key:16s} {t.suite:11s} {t.modality:14s} x {shape:9s} "
              f"theta {t.n_params}   train {len(d['train'][0])}  "
              f"test {len(d['test'][0])}   {t.labels}")


if __name__ == "__main__":
    summary()
