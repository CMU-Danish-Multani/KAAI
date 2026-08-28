"""Two-point correlation function, recomputed to CosmoBench's own binning.

WHY THIS FILE EXISTS
--------------------
The `tpcf_*.hdf5` files shipped with the dataset do NOT use the binning that
produced Table 2 of the paper. Measured 2026-08-17:

    suite        shipped file              paper, Table 2
    CAMELS       19 bins, 0.1   to 12      25 bins, 0.0125 to 12
    CAMELS-SAM   19 bins, 1.0   to 40      25 bins, 0.0125 to 12
    Quijote      24 bins, 2.0   to 80      25 bins, 0.5    to 480

A correlation function measured over different distance ranges is a different
input. So the shipped files cannot reproduce the published numbers, and this
module recomputes xi(r) from the positions instead.

WHAT THE CORRELATION FUNCTION IS
--------------------------------
Count every pair of galaxies and record how far apart each pair is. Compare that
against how many pairs you would get by scattering the same number of galaxies
at random in the same box. A clumpy universe shows an excess of close pairs.
xi(r) is that excess, as a fraction: xi = 0 means "same as random".

DEVIATION FROM THE PAPER, RECORDED DELIBERATELY
-----------------------------------------------
The paper uses the Landy-Szalay estimator with 100x as many random points as
data. For a periodic cube the random-random term has a closed form, so we use
    xi_i = DD_i / RR_i - 1,   RR_i = N(N-1) * V_shell_i / V_box
which is exact rather than sampled, and far cheaper. `calibrate()` verifies the
normalisation against uniform random points, where xi must come out at zero.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from Corrfunc.theory.DD import DD

from point_clouds.load import BOX, open_suite, read_cloud, read_labels, sim_names

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "tpcf_cache"

# CosmoBench Sec. B.1. "Base" is Rmin = B/2000, Rmax = 3B/25 for box side B.
# Table 7 records which variant produced each column of Table 2.
N_BINS = 25
BIN_RANGE: Dict[str, Tuple[float, float]] = {
    "Quijote":    (0.5,    480.0),   # Base x 4Rmax
    "CAMELS-SAM": (0.0125,  12.0),   # Base x Rmin/4
    "CAMELS":     (0.0125,  12.0),   # Base x 4Rmax
}

# Applied to xi before it reaches the model, per Sec. B.1: absolute value first,
# because shot noise makes xi slightly negative at large radii, then log to
# compress a range that reaches thousands in the innermost CAMELS bins.
LOG_FLOOR = 1e-6


def bin_edges(suite: str) -> np.ndarray:
    lo, hi = BIN_RANGE[suite]
    return np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)


def xi_of_cloud(positions: np.ndarray, box: float, edges: np.ndarray,
                nthreads: int = 4) -> np.ndarray:
    """xi(r) for one cloud, using the analytic random-random term."""
    p = np.mod(positions, box)                      # the box wraps at its edges
    n = len(p)
    counts = DD(autocorr=1, nthreads=nthreads, binfile=edges,
                X1=p[:, 0].copy(), Y1=p[:, 1].copy(), Z1=p[:, 2].copy(),
                periodic=True, boxsize=box)["npairs"].astype(float)

    # Corrfunc counts ordered pairs, verified against calibrate() below.
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rr = n * (n - 1) * shell / box ** 3
    return counts / rr - 1.0


def to_features(xi: np.ndarray) -> np.ndarray:
    """Turn raw xi into the model's input, following Sec. B.1."""
    return np.log10(np.abs(xi) + LOG_FLOOR)


def calibrate(box: float = 100.0, n: int = 20000, seed: int = 0,
              tol: float = 0.05) -> None:
    """Uniform random points must give xi = 0 in every bin.

    This is the guard on the pair-count normalisation. If Corrfunc's counting
    convention were wrong by a factor of two, xi would sit near -0.5 or +1.0
    instead, and every downstream number would be silently wrong.
    """
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, box, size=(n, 3))
    edges = np.logspace(np.log10(box / 50), np.log10(box / 5), 8)
    xi = xi_of_cloud(p, box, edges)
    worst = float(np.abs(xi).max())
    print(f"  calibration on {n:,} uniform random points, {len(xi)} bins")
    print(f"    xi range [{xi.min():+.4f}, {xi.max():+.4f}]   largest |xi| = {worst:.4f}")
    if worst > tol:
        raise AssertionError(
            f"xi should be ~0 for random points, got |xi| up to {worst:.4f}. "
            "The pair-count normalisation is wrong.")
    print(f"    PASS (tolerance {tol})")


def build_split(suite: str, split: str, nthreads: int = 4,
                limit: int = 0, fixed_count: int = 0) -> Dict[str, np.ndarray]:
    """xi(r) for every cloud in one split, with its labels attached.

    fixed_count > 0 keeps only the N most massive galaxies in every cloud. That
    closes the counting leak by construction: the count stops varying, so it can
    no longer carry information about Omega_m. Used for the Stage 4 control.
    """
    edges = bin_edges(suite)
    with open_suite(suite, split) as f:
        names = sim_names(f)
        labels = read_labels(f)
        if limit:
            names = names[:limit]
        n_expected = len(labels["Omega_m"])
        if not limit and len(names) != n_expected:
            raise AssertionError(
                f"{suite}/{split}: {len(names)} clouds but {n_expected} label rows")
        xis, ngal = [], []
        for i, sim in enumerate(names):
            positions, _, extra = read_cloud(f, sim)
            if fixed_count:
                if len(positions) < fixed_count:
                    raise AssertionError(
                        f"{suite}/{split}/{sim}: {len(positions)} galaxies, "
                        f"fewer than the requested fixed count {fixed_count}")
                keep = np.argsort(extra["Mstar"])[::-1][:fixed_count]
                positions = positions[keep]
            xis.append(xi_of_cloud(positions, BOX[suite], edges, nthreads))
            ngal.append(len(positions))
            if (i + 1) % 100 == 0:
                print(f"    {suite}/{split}: {i + 1}/{len(names)}", flush=True)
        y = np.stack([labels["Omega_m"][:len(names)],
                      labels["sigma_8"][:len(names)]], axis=1)

    ngal = np.array(ngal)
    if fixed_count and len(np.unique(ngal)) != 1:
        raise AssertionError("fixed_count requested but the count still varies")
    return {"xi": np.stack(xis), "y": y, "ngal": ngal, "edges": edges}


def cache_path(suite: str, split: str, fixed_count: int = 0) -> Path:
    lo, hi = BIN_RANGE[suite]
    tag = f"_top{fixed_count}" if fixed_count else ""
    return CACHE_DIR / f"{suite}_{split}_{N_BINS}bins_{lo:g}_{hi:g}{tag}.npz"


def load_or_build(suite: str, split: str, nthreads: int = 4,
                  rebuild: bool = False, fixed_count: int = 0) -> Dict[str, np.ndarray]:
    """Read the cached xi for one split, computing it first if needed."""
    path = cache_path(suite, split, fixed_count)
    if path.exists() and not rebuild:
        with np.load(path) as d:
            return {k: d[k] for k in d.files}
    data = build_split(suite, split, nthreads, fixed_count=fixed_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"    wrote {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suites", type=str, nargs="+", default=["CAMELS-SAM", "CAMELS"],
                        choices=["CAMELS", "CAMELS-SAM"],
                        help="Quijote is excluded: its positions are not downloaded")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRATION -- does the pair-count normalisation give xi = 0 for noise?")
    print("=" * 70)
    calibrate()

    for suite in args.suites:
        lo, hi = BIN_RANGE[suite]
        print("\n" + "=" * 70)
        print(f"{suite}: {N_BINS} log bins, {lo:g} to {hi:g} cMpc/h, box {BOX[suite]:g}")
        print("=" * 70)
        for split in args.splits:
            data = load_or_build(suite, split, args.nthreads, args.rebuild)
            xi = data["xi"]
            print(f"  {split:5s} {xi.shape[0]:5d} clouds x {xi.shape[1]} bins   "
                  f"xi range [{xi.min():.3g}, {xi.max():.3g}]   "
                  f"negatives {100 * (xi < 0).mean():.1f}%   -> {cache_path(suite, split).name}")


if __name__ == "__main__":
    main()
