"""Read the point cloud data -- boxes of galaxies from simulated universes.

WHAT A POINT CLOUD IS
---------------------
One simulation = one cube of space = one "cloud". Inside it are thousands of
galaxies, and all we really have is where each one sits: an x, y and z
position. The whole cube has ONE answer attached: the two dials (Omega_m and
sigma_8) that were used to create that universe.

HOW THE FILE IS ORGANISED ON DISK
---------------------------------
The .hdf5 files are like folders inside a folder. Each simulation gets its own
group, and inside it every property is a SEPARATE list:

    ALL_galaxies_val.hdf5
      |
      +-- LH/
      |     +-- LH_0/            <- simulation 0 = one cloud
      |     |     +-- X          [2109 numbers]  each galaxy's x position
      |     |     +-- Y          [2109 numbers]
      |     |     +-- Z          [2109 numbers]
      |     |     +-- VX, VY, VZ how fast each galaxy moves
      |     |     +-- Mstar      how heavy each galaxy is
      |     +-- LH_1/            <- simulation 1, maybe 2940 galaxies
      |     ...
      +-- params/                <- the answers
            +-- Omega_m          [200 numbers]  one per simulation
            +-- sigma_8          [200 numbers]

Notice X, Y and Z are three separate lists, not one table. Gluing them into
positions is our job -- that is what read_cloud() does.

HOW THE CODE FLOWS
------------------
    summary_table("CAMELS")            <- the usual entry point
      |
      +-- open_suite()                 opens the .hdf5 file
      +-- sim_names()                  lists the simulations inside it
      +-- read_labels()                grabs the answers for all of them
      +-- for each simulation:
            +-- read_cloud()           glue X,Y,Z into positions
            +-- summarise_cloud()      boil the cloud down to ~9 numbers
                  +-- pair_counts()    measure how clumped it is
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

# How wide each simulation cube is, in cMpc/h (1 unit is about 3 million
# light years). The cubes wrap around at their edges like a Pac-Man screen.
BOX = {"CAMELS": 25.0, "CAMELS-SAM": 100.0, "Quijote": 1000.0}

ROOT = Path(__file__).resolve().parents[1]   # project root
PATHS = {
    "CAMELS": "data/CAMELS/ALL_galaxies_{split}.hdf5",
    "CAMELS-SAM": "data/CAMELS-SAM/top5000_galaxies_{split}.hdf5",
}


# ===========================================================================
# PART 1 -- READING THE FILE
# ===========================================================================

def open_suite(suite, split="val"):
    """Open one .hdf5 file, e.g. open_suite("CAMELS", "val")."""
    return h5py.File(ROOT / PATHS[suite].format(split=split), "r")


def _numbers(group, name):
    """Pull one list of numbers out of the file as a plain numpy array."""
    return np.asarray(group[name][...])


def sim_names(f):
    """Names of every simulation in the file, in order: LH_0, LH_1, LH_2...

    Sorting matters. Plain alphabetical order would give LH_0, LH_1, LH_10,
    LH_100 -- so we sort by the NUMBER after the underscore instead. If this
    order did not match the order of the answers, every cloud would get the
    wrong label.
    """
    return sorted(f["LH"].keys(), key=lambda name: int(name.split("_")[1]))


def read_labels(f):
    """The answers for every simulation, as {name: list of numbers}.

    Omega_m and sigma_8 are what we want to predict. The A_SN / A_AGN entries
    are also stored here, but they describe how violently supernovae and black
    holes blow gas around -- we are NOT predicting those, they are just noise.
    """
    labels = {}
    for name in f["params"].keys():
        labels[name] = _numbers(f["params"], name).ravel()
    return labels


def read_cloud(f, sim):
    """Read one cloud. Returns (positions, velocities, everything else).

    The file keeps X, Y and Z as three separate lists. We stack them into one
    table so that row i is galaxy i's position:

        X = [1.2, 5.4, ...]
        Y = [3.3, 0.8, ...]   ->   positions = [[1.2, 3.3, 9.1],
        Z = [9.1, 2.7, ...]                     [5.4, 0.8, 2.7], ...]
    """
    group = f["LH"][sim]

    positions = np.stack([_numbers(group, axis) for axis in ("X", "Y", "Z")], axis=1)
    velocities = np.stack([_numbers(group, axis) for axis in ("VX", "VY", "VZ")], axis=1)

    # Whatever else this suite happens to store -- Mstar, Mgas, mHI and so on.
    # The suites do not all carry the same fields, so we keep them in a dict
    # rather than assuming particular names exist.
    used = {"X", "Y", "Z", "VX", "VY", "VZ"}
    extra = {name: _numbers(group, name) for name in group.keys() if name not in used}

    return positions, velocities, extra


# ===========================================================================
# PART 2 -- TURNING A CLOUD INTO A FEW NUMBERS
# ===========================================================================

def pair_counts(positions, box, radii):
    """What fraction of galaxy PAIRS are closer together than each radius?

    This is how cosmologists measure clumpiness. Pick every possible pair of
    galaxies, measure the gap, and count how many are closer than 5 units,
    than 10 units, and so on. A clumpy universe has lots of close pairs; a
    smooth one has few.

    Checking every pair by hand would be slow (5000 galaxies = 12.5 million
    pairs), so we build a search index that can answer "how many neighbours
    within distance r" quickly. `boxsize` tells it the cube wraps around, so
    a galaxy at 99.9 counts as next to one at 0.1.
    """
    index = cKDTree(np.mod(positions, box), boxsize=box)
    n = len(positions)

    # count_neighbors counts each galaxy against itself too, so subtract n.
    # Then divide by the total number of possible pairs to get a fraction --
    # otherwise clouds with more galaxies would always look clumpier.
    counts = index.count_neighbors(index, radii)
    return (counts - n) / (n * (n - 1))


def summarise_cloud(positions, velocities, extra, box):
    """Boil one whole cloud down to about nine numbers.

    Two kinds: how the galaxies are SPREAD OUT, and what the galaxies
    THEMSELVES are like.
    """
    # Measure clumpiness at 2%, 5% and 10% of the box width. Using fractions
    # rather than fixed distances keeps small and large suites comparable.
    radii = np.array([0.02, 0.05, 0.10]) * box
    close, mid, far = pair_counts(positions, box, radii)

    speed = np.linalg.norm(velocities, axis=1)          # length of each velocity
    mass = np.log10(np.clip(extra["Mstar"], 1.0, None))  # raw mass -> powers of 10

    return {
        "n_galaxies": len(positions),
        # how spread out they are
        "pairs_close": close,
        "pairs_mid": mid,
        "pairs_far": far,
        "clustering_ratio": close / mid if mid > 0 else np.nan,
        # what the galaxies themselves are like
        "mean_logMstar": mass.mean(),
        "std_logMstar": mass.std(),
        "mean_speed": speed.mean(),
        "std_speed": speed.std(),
    }


def summary_table(suite, split="val", limit=None):
    """Summarise EVERY cloud in a split, and attach each one's answers.

    Returns {column name: array of one value per cloud} -- a table with one
    row per simulation, ready to correlate against Omega_m and sigma_8.
    """
    rows = []

    with open_suite(suite, split) as f:
        names = sim_names(f)[:limit]
        labels = read_labels(f)

        for i, sim in enumerate(names):
            positions, velocities, extra = read_cloud(f, sim)
            row = summarise_cloud(positions, velocities, extra, BOX[suite])
            # labels are stored in the same order as sim_names(), so index i
            # of the answers belongs to simulation i
            row["Omega_m"] = float(labels["Omega_m"][i])
            row["sigma_8"] = float(labels["sigma_8"][i])
            rows.append(row)

    # flip a list of dicts into a dict of arrays (one array per column)
    return {column: np.array([row[column] for row in rows]) for column in rows[0]}


# ===========================================================================
# RUN THIS FILE DIRECTLY TO SEE THE DATA
#
#     python data_load/clouds.py
#
# Everything below only runs when you launch this file yourself. Importing it
# from somewhere else skips it entirely.
# ===========================================================================

if __name__ == "__main__":
    SUITE = "CAMELS-SAM"        # try "CAMELS" too -- it has different fields
    SPLIT = "val"

    print("=" * 70)
    print(f"OPENING  {SUITE}  ({SPLIT} split)")
    print("=" * 70)
    print(f"  file: {PATHS[SUITE].format(split=SPLIT)}")
    print(f"  cube: {BOX[SUITE]} cMpc/h on each side")

    with open_suite(SUITE, SPLIT) as f:

        # ---- what simulations are in here? --------------------------------
        names = sim_names(f)
        print(f"\n  {len(names)} clouds inside. First eight are named:")
        print(f"    {', '.join(names[:8])} ...")

        # ---- the answers ---------------------------------------------------
        labels = read_labels(f)
        print(f"\n{'=' * 70}\nTHE ANSWERS  (one row per cloud)\n{'=' * 70}")
        print(f"  stored under params/: {', '.join(sorted(labels))}")
        print(f"\n  {'cloud':<8} {'Omega_m':>9} {'sigma_8':>9}   <- what we predict")
        for i in range(5):
            print(f"  {names[i]:<8} {labels['Omega_m'][i]:9.4f} "
                  f"{labels['sigma_8'][i]:9.4f}")
        print(f"  {'...':<8} ({len(names) - 5} more)")

        # ---- one actual cloud ---------------------------------------------
        sim = names[0]
        positions, velocities, extra = read_cloud(f, sim)

        print(f"\n{'=' * 70}\nONE CLOUD IN DETAIL:  {sim}\n{'=' * 70}")
        print(f"  {len(positions):,} galaxies")
        print(f"  positions  {positions.shape}   <- x, y, z glued together")
        print(f"  velocities {velocities.shape}")
        print(f"  also stored: {', '.join(sorted(extra))}")

        print("\n  where the first six galaxies sit (units of cMpc/h):")
        print(f"    {'galaxy':>7} {'x':>9} {'y':>9} {'z':>9}")
        for i in range(6):
            x, y, z = positions[i]
            print(f"    {i:>7} {x:9.3f} {y:9.3f} {z:9.3f}")

        print("\n  the cube spans:")
        for axis, name in enumerate("xyz"):
            column = positions[:, axis]
            print(f"    {name}: {column.min():7.3f} to {column.max():7.3f}  "
                  f"(box is 0 to {BOX[SUITE]})")

        print("\n  how fast they move (km/s):")
        speed = np.linalg.norm(velocities, axis=1)
        print(f"    slowest {speed.min():7.1f}   average {speed.mean():7.1f}   "
              f"fastest {speed.max():7.1f}")

        # ---- what we boil it down to --------------------------------------
        print(f"\n{'=' * 70}\nTHAT CLOUD, BOILED DOWN TO NUMBERS\n{'=' * 70}")
        summary = summarise_cloud(positions, velocities, extra, BOX[SUITE])
        for key, value in summary.items():
            print(f"  {key:<18} {value:12.6f}")
        print(f"\n  {'-> its answer:':<18} Omega_m {labels['Omega_m'][0]:.4f}   "
              f"sigma_8 {labels['sigma_8'][0]:.4f}")

    print(f"\n{'=' * 70}")
    print("  Those ~9 numbers are ALL we keep. The 5,000 galaxy positions")
    print("  are read, summarised, and thrown away.")
    print("=" * 70)
