"""Build fixed size point clouds from the raw catalogues, cached as .npz.

    python -m point_clouds.cloudCache

Reading the HDF5 catalogues is slow and the zoo's sweep reads them once per cell, so
they are turned into arrays first, exactly as tpcf.py does for the correlation
function.

WHY A FIXED NUMBER OF POINTS
----------------------------
CAMELS clouds hold between 588 and 4293 galaxies, and that count correlates with
Omega_m at 0.73. Measured earlier in this project: a network free to see the count
scores +0.149 higher on Omega_m, and that entire advantage vanishes to +0.0003 once
the count is held fixed. Taking the N most massive galaxies closes the channel by
construction rather than by hoping the architecture ignores it.

Positions are divided by the box side, so every suite lands in [0, 1] and the same
embedding network can read either without rescaling.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from point_clouds.load import BOX, open_suite, read_cloud, read_labels, sim_names

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cloud_cache"

# CAMELS' smallest cloud holds 588 galaxies, so anything at or below that is available
# in every cloud of both suites without padding.
DEFAULT_POINTS = 512


def cache_path(suite: str, split: str, n_points: int) -> Path:
    return CACHE_DIR / f"{suite}_{split}_top{n_points}.npz"


def build_split(suite: str, split: str, n_points: int) -> Dict[str, np.ndarray]:
    """The n_points most massive galaxies of every cloud, positions in [0, 1]."""
    box = BOX[suite]
    with open_suite(suite, split) as f:
        names = sim_names(f)
        labels = read_labels(f)
        clouds = np.empty((len(names), n_points, 3), dtype=np.float32)
        for i, sim in enumerate(names):
            positions, _, extra = read_cloud(f, sim)
            if len(positions) < n_points:
                raise AssertionError(
                    f"{suite}/{split}/{sim} holds {len(positions)} galaxies, "
                    f"fewer than the {n_points} requested")
            keep = np.argsort(extra["Mstar"])[::-1][:n_points]
            clouds[i] = (np.mod(positions[keep], box) / box).astype(np.float32)
            if (i + 1) % 200 == 0:
                print(f"    {suite}/{split}: {i + 1}/{len(names)}", flush=True)
        # `seed` and `LH` are simulation identifiers, not physical parameters.
        # Leaving them in would hand the network a target it can only memorise.
        names_kept = [k for k in labels if k not in ("seed", "LH")]
        y = np.stack([labels[k][:len(names)] for k in names_kept], axis=1)
    return {"clouds": clouds, "y": y.astype(np.float32),
            "labelNames": np.array(names_kept)}


def load_or_build(suite: str, split: str, n_points: int = DEFAULT_POINTS,
                  rebuild: bool = False) -> Dict[str, np.ndarray]:
    path = cache_path(suite, split, n_points)
    if path.exists() and not rebuild:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    data = build_split(suite, split, n_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"    wrote {path}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--suites", type=str, nargs="+", default=["CAMELS", "CAMELS-SAM"],
                   choices=["CAMELS", "CAMELS-SAM"])
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--n-points", type=int, default=DEFAULT_POINTS)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    if args.n_points < 8:
        raise SystemExit("--n-points must be at least 8 to be a meaningful cloud")

    for suite in args.suites:
        print(f"\n{suite}: box {BOX[suite]:g} cMpc/h, keeping the "
              f"{args.n_points} most massive galaxies")
        for split in args.splits:
            d = load_or_build(suite, split, args.n_points, args.rebuild)
            print(f"  {split:5s} clouds {d['clouds'].shape}  labels {d['y'].shape}  "
                  f"{list(d['labelNames'])}")
            print(f"        positions in [{d['clouds'].min():.3f}, "
                  f"{d['clouds'].max():.3f}]  -> {cache_path(suite, split, args.n_points).name}")


if __name__ == "__main__":
    main()
