"""Explore the CosmoBench point clouds. Run: python explore_clouds.py

Each section answers one question. Findings are written up in DATA.md.
"""

import sys
from pathlib import Path

# This script lives in exploration/, but imports viz.py and data_load/ from the
# project root, so put the root on the import path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Plots belong to THIS domain, not a shared folder.
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

import numpy as np

from point_clouds.load import (BOX, open_suite, pair_counts, read_cloud,
                              read_labels, sim_names, summary_table)
from common.viz import (BLUE, ORANGE, constant_features, correlation_chart,
                 correlations, header, plt, print_correlation_table)

SUITES = ["CAMELS", "CAMELS-SAM"]

# ==========================================================================
# 1. WHAT IS IN EACH FILE?
#
# Unlike the trees (one Python list), clouds are nested HDF5 groups: one
# group per simulation, and every property in its own separate array.
# ==========================================================================
header("1. FILE CONTENTS")

for suite in SUITES:
    with open_suite(suite, "val") as f:
        names = sim_names(f)
        labels = read_labels(f)
        sizes = [f["LH"][s]["X"].shape[0] for s in names]
        fields = sorted(f["LH"][names[0]].keys())
        print(f"\n  {suite}   box = {BOX[suite]} cMpc/h per side")
        print(f"    {len(names)} clouds | galaxies per cloud: "
              f"min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")
        print(f"    per-galaxy fields: {fields}")
        print(f"    params stored     : {sorted(labels)}")

# ==========================================================================
# 2. WHAT ARE WE PREDICTING, AND WHAT IS NOISE?
#
# Only Omega_m and sigma_8 are the target. The A_SN / A_AGN values describe
# how violently supernovae and black holes blow gas around. They were varied
# too, so they act as nuisance -- the same cosmology can look different
# depending on them.
# ==========================================================================
header("2. THE LABELS")

for suite in SUITES:
    with open_suite(suite, "val") as f:
        labels = read_labels(f)
    print(f"\n  {suite}")
    for k, v in sorted(labels.items()):
        role = ("TARGET" if k in ("Omega_m", "sigma_8")
                else "id" if k in ("seed", "LH") else "nuisance")
        print(f"    {k:14s} [{v.min():10.4f}, {v.max():10.4f}]   {role}")

# ==========================================================================
# 3. WHAT DOES A POINT CLOUD ACTUALLY LOOK LIKE?
#
# The single most useful picture here. Two universes with very different
# amounts of matter, drawn as a thin slice through the cube. If the task is
# possible at all, the difference should be visible by eye.
# ==========================================================================
header("3. WHAT A CLOUD LOOKS LIKE")

suite = "CAMELS-SAM"
box = BOX[suite]

# Read the two extreme clouds once and reuse them for both figures below.
with open_suite(suite, "val") as f:
    names, labels = sim_names(f), read_labels(f)
    lo_i = int(np.argmin(labels["Omega_m"]))
    hi_i = int(np.argmax(labels["Omega_m"]))
    extremes = [(read_cloud(f, names[i])[0], labels["Omega_m"][i]) for i in (lo_i, hi_i)]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (pos, omega), which in zip(axes, extremes, ("lowest", "highest")):
    slab = pos[pos[:, 2] < box * 0.15]              # thin slice, so structure shows
    ax.scatter(slab[:, 0], slab[:, 1], s=3, alpha=0.55, color=BLUE, edgecolors="none")
    ax.set_title(f"{which} $\\Omega_m$ = {omega:.3f}\n{len(slab)} galaxies in slice",
                 fontsize=10)
    ax.set_xlabel("x  [cMpc/h]"); ax.set_aspect("equal")
    ax.set_xlim(0, box); ax.set_ylim(0, box)
axes[0].set_ylabel("y  [cMpc/h]")
fig.suptitle(f"{suite}: a slice through two simulated universes", fontsize=11)
fig.tight_layout()
fig.savefig(PLOTS / "07_cloud_slices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  drew Omega_m {extremes[0][1]:.3f} vs {extremes[1][1]:.3f} "
      f"-> plots/07_cloud_slices.png")

# ==========================================================================
# 4. HOW CLUSTERED IS EACH UNIVERSE?
#
# The classic cosmology measurement: count how many galaxy pairs sit closer
# than a given distance. More close pairs = a clumpier universe. This is the
# idea behind the two-point correlation function baseline.
# ==========================================================================
header("4. CLUSTERING CURVES")

fig, ax = plt.subplots(figsize=(6.6, 4))
radii = np.linspace(0.01, 0.25, 22) * box
for (pos, omega), which, colour in zip(extremes,
                                       ("low $\\Omega_m$", "high $\\Omega_m$"),
                                       (BLUE, ORANGE)):
    ax.plot(radii, pair_counts(pos, box, radii), lw=2, color=colour,
            label=f"{which} = {omega:.3f}")
ax.set_xlabel("separation  [cMpc/h]")
ax.set_ylabel("fraction of galaxy pairs closer than this")
# Backwards from the naive expectation, and the effect is real (r = -0.71 over
# all 204 clouds). Cause: these files keep the 5000 MOST MASSIVE galaxies
# whatever the cosmology. In a matter-poor universe such galaxies are rare, so
# the top 5000 sit in the most extreme density peaks -- and rare peaks cluster
# hard. Matter-rich universes make massive galaxies common, so the same cut
# picks ordinary, less-clustered objects. This is a SELECTION effect, not
# structure growth.
ax.set_title("At fixed galaxy count, matter-POOR universes look more clustered",
             fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / "08_clustering.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  -> plots/08_clustering.png")

# ==========================================================================
# 5. WHICH SUMMARIES CARRY THE ANSWER?
#
# Same experiment as the trees: reduce each cloud to a handful of numbers,
# then see which move with the two dials.
#
# TWO TRAPS, both of which show up strongly below:
#
#  - mean_speed / std_speed use VELOCITIES. The benchmark task feeds the
#    model positions only; velocities are the answer to a different task.
#    So a high correlation here is not a usable result.
#
#  - n_galaxies is a documented shortcut. Halos only get counted once they
#    contain ~20 particles, and particle mass depends on Omega_m, so simply
#    counting galaxies leaks the answer without using structure at all.
#    This is why CAMELS-SAM and Quijote ship fixed-size "top 5000" files.
# ==========================================================================
header("5. WHAT PREDICTS THE TWO DIALS")

for suite in SUITES:
    print(f"\n  {suite}  (summarising every cloud...)")
    T = summary_table(suite, "val")
    fixed = constant_features(T)
    if fixed:
        print(f"    constant, so excluded: {fixed}")
    corr_om = correlations(T, "Omega_m")
    corr_s8 = correlations(T, "sigma_8")
    print_correlation_table(corr_om, corr_s8)

    tag = suite.lower().replace("-", "")
    correlation_chart(corr_om, "$\\Omega_m$",
                      f"{suite}: what knows about the matter density?",
                      PLOTS / f"09_{tag}_omega.png")
    correlation_chart(corr_s8, "$\\sigma_8$",
                      f"{suite}: what knows about the lumpiness?",
                      PLOTS / f"10_{tag}_sigma.png")

print("\ndone -- wrote plots/07..10")
