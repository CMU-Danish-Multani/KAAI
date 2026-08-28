"""Linear least squares on pairwise-distance statistics: CosmoBench's 49-parameter model.

This is the model that matters most to the project, and not because it is good.
On Quijote it beats a 671,000-parameter graph network that trains for a day.
Any architecture search that cannot rediscover something this simple is
searching the wrong space, so it belongs inside the search rather than beside
it as a baseline.

WHAT IT MEASURES
----------------
Take every pair of galaxies closer together than some cutoff distance Rc, and
look at the spread of their squared separations. Four numbers summarise that
spread: the mean, the standard deviation, and the one-third and two-thirds
quantiles. Repeat for 12 different cutoffs and you have 48 numbers per universe.
Fit a straight line through them, add a bias term, and that is 49 parameters.

CosmoBench Sec. 4.1: the 12 cutoffs are chosen greedily on the validation split,
separately for each target parameter, and predictions are clipped to the range
the simulations actually sampled.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from point_clouds.load import BOX, open_suite, read_cloud, read_labels, sim_names

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "pairstat_cache"

# The paper says 12 cutoffs are selected greedily from a candidate set but does
# not say what the candidate set was. 20 log-spaced radii, as a fraction of the
# box side, recorded here rather than left implicit.
N_CANDIDATES = 20
CANDIDATE_RANGE = (1 / 200, 1 / 2.5)
N_SELECTED = 12
N_STATISTICS = 4                      # mean, std, 1/3-quantile, 2/3-quantile

# Sec. 4.1: predictions are clipped to the limits the simulations sampled.
PARAMETER_LIMITS = {"Omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0)}
TARGETS = ("Omega_m", "sigma_8")


def candidate_radii(box: float) -> np.ndarray:
    lo, hi = CANDIDATE_RANGE
    return np.logspace(np.log10(lo * box), np.log10(hi * box), N_CANDIDATES)


def statistics_of_cloud(positions: np.ndarray, box: float,
                        radii: np.ndarray) -> np.ndarray:
    """The 4 statistics at every candidate radius, flattened. Shape (4 * n_radii,).

    Pairs are found once at the largest radius and then filtered down, since
    every smaller cutoff is a subset. The box wraps, so separations use the
    minimum-image convention rather than raw coordinate differences.
    """
    wrapped = np.mod(positions, box)
    tree = cKDTree(wrapped, boxsize=box)
    pairs = tree.query_pairs(radii.max(), output_type="ndarray")

    if len(pairs) == 0:
        return np.zeros(N_STATISTICS * len(radii))

    separation = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    separation -= box * np.round(separation / box)
    squared = (separation ** 2).sum(axis=1)

    features = []
    for radius in radii:
        inside = squared[squared <= radius ** 2]
        if len(inside) < 2:
            features += [0.0] * N_STATISTICS
        else:
            features += [inside.mean(), inside.std(),
                         float(np.quantile(inside, 1 / 3)),
                         float(np.quantile(inside, 2 / 3))]
    return np.array(features)


def build_split(suite: str, split: str) -> Dict[str, np.ndarray]:
    radii = candidate_radii(BOX[suite])
    with open_suite(suite, split) as f:
        names = sim_names(f)
        labels = read_labels(f)
        rows = []
        for i, sim in enumerate(names):
            positions, _, _ = read_cloud(f, sim)
            rows.append(statistics_of_cloud(positions, BOX[suite], radii))
            if (i + 1) % 100 == 0:
                print(f"    {suite}/{split}: {i + 1}/{len(names)}", flush=True)
        y = np.stack([labels[t] for t in TARGETS], axis=1)
    return {"x": np.stack(rows), "y": y, "radii": radii}


def load_or_build(suite: str, split: str, rebuild: bool = False) -> Dict[str, np.ndarray]:
    path = CACHE_DIR / f"{suite}_{split}_{N_CANDIDATES}radii.npz"
    if path.exists() and not rebuild:
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    data = build_split(suite, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"    wrote {path}", flush=True)
    return data


def _columns_for(radius_indices: List[int]) -> np.ndarray:
    """Feature column indices belonging to a set of radii."""
    return np.array([i * N_STATISTICS + s
                     for i in radius_indices for s in range(N_STATISTICS)])


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with a bias term."""
    design = np.hstack([x, np.ones((len(x), 1))])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _predict(weights: np.ndarray, x: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    design = np.hstack([x, np.ones((len(x), 1))])
    return np.clip(design @ weights, *limits)


def select_radii_greedily(train: Dict, val: Dict, target: int) -> List[int]:
    """Add one radius at a time, keeping whichever most improves validation R2.

    Selection uses validation only. Using test here would be the exact leak the
    whole project is about.
    """
    from common.metrics import r2_score

    limits = PARAMETER_LIMITS[TARGETS[target]]
    chosen: List[int] = []
    remaining = list(range(N_CANDIDATES))

    for _ in range(N_SELECTED):
        best_score, best_radius = -np.inf, None
        for candidate in remaining:
            columns = _columns_for(chosen + [candidate])
            weights = _fit(train["x"][:, columns], train["y"][:, target])
            predicted = _predict(weights, val["x"][:, columns], limits)
            score = float(r2_score(predicted[:, None], val["y"][:, target][:, None])[0])
            if score > best_score:
                best_score, best_radius = score, candidate
        chosen.append(best_radius)
        remaining.remove(best_radius)
    return chosen


def fit_and_predict(train: Dict, val: Dict, test: Dict) -> Tuple[np.ndarray, Dict]:
    """Fit one linear model per target and predict on test. Returns (predictions, detail)."""
    predictions, detail = [], {}
    for target, name in enumerate(TARGETS):
        chosen = select_radii_greedily(train, val, target)
        columns = _columns_for(chosen)
        weights = _fit(train["x"][:, columns], train["y"][:, target])
        predictions.append(_predict(weights, test["x"][:, columns],
                                    PARAMETER_LIMITS[name]))
        detail[name] = {"selected_radius_indices": chosen,
                        "selected_radii": [float(train["radii"][i]) for i in chosen],
                        "n_parameters": int(weights.size)}
    return np.stack(predictions, axis=1), detail
