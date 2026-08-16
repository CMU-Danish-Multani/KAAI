"""Everything we know about the CS-Trees data, as one runnable script.

Each section answers one question. Run it top to bottom:

    python explore_trees.py

Findings are written up in DATA.md; this is the code that produced them.
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
import torch

from merger_trees.load import HALF_DEX, load, summary_table
from common.viz import (BLUE, ORANGE, MUTED, correlation_chart, correlations, header,
                 plt, print_correlation_table)

# ==========================================================================
# 1. CAN WE TRUST THE SPLITS?
#
# 25 trees come from each simulation and all 25 carry the SAME answer. If a
# simulation appeared in both train and test, a model could recognise it and
# copy the answer -- a great score that means nothing. So check before
# anything else, because every later number depends on this being clean.
# ==========================================================================
header("1. SPLIT SIZES AND LEAKAGE")

# Load one split at a time and keep only the summary numbers. Holding all
# three at once costs several GB, and only `val` is needed after this section.
sim_ids, val = {}, None
for name in ("train", "val", "test"):
    trees = load(name)
    sizes = np.array([t.num_nodes for t in trees])
    sim_ids[name] = {int(t.lh_id) for t in trees}
    print(f"  {name:6s} {len(trees):6,d} trees | {len(sim_ids[name]):4d} simulations | "
          f"nodes: min {sizes.min():5d}  median {int(np.median(sizes)):5d}  "
          f"max {sizes.max():6d}  total {sizes.sum():,d}")
    if name == "val":
        val = trees
    else:
        del trees

print()
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    shared = sim_ids[a] & sim_ids[b]
    print(f"  {a}/{b:5s} {'CLEAN' if not shared else f'LEAK - {len(shared)} shared'}")

counts = np.bincount([int(t.lh_id) for t in val])
counts = counts[counts > 0]
print(f"\n  trees per simulation: {counts.min()}-{counts.max()} (median {int(np.median(counts))})")

# ==========================================================================
# 2. HOW IS ONE TREE ACTUALLY STORED?
#
# Not as nested objects -- as flat arrays. The tree's shape is never stored
# anywhere; it is implied entirely by the list of arrows in edge_index.
# ==========================================================================
header("2. HOW ONE TREE IS STORED")

t = min(val, key=lambda g: g.num_nodes)          # smallest = easiest to read
print(f"  {t}\n")

print("  x -- one row per blob, one column per property:")
print(f"  {'blob':>5}  {'mass':>8} {'concentr':>9} {'v_max':>8} {'time a':>8}")
for i in range(5):
    m, c, v, a = t.x[i]
    print(f"  {i:5d}  {m:8.3f} {c:9.3f} {v:8.3f} {a:8.3f}")

src, dst = t.edge_index.numpy()
print(f"\n  edge_index {tuple(t.edge_index.shape)} -- read COLUMN by column:")
for i in range(4):
    print(f"    blob {src[i]:3d}  ->  blob {dst[i]:3d}   (older merges into newer)")

# A real merger = two arrows landing on the same blob. Everything else is
# just one halo growing, which is most of the tree.
n_progenitors = np.bincount(dst, minlength=t.num_nodes)
mergers = np.where(n_progenitors >= 2)[0]
print(f"\n  real mergers in this tree: {len(mergers)} of {t.num_nodes} blobs")
node = mergers[0]
print(f"  example -- blob {node} was formed by:")
for p in src[dst == node]:
    print(f"    blob {p:3d}  mass {t.x[p,0]:6.3f}  at a={t.x[p,3]:.3f}")
print(f"    ---> blob {node:3d}  mass {t.x[node,0]:6.3f}  at a={t.x[node,3]:.3f}")

print(f"\n  y {tuple(t.y.shape)} -- ONE answer for the whole tree, not per blob:")
print(f"    Omega_m = {t.y[0,0]:.4f}   sigma_8 = {t.y[0,1]:.4f}")

# n-1 edges is the defining property of a tree. Confirms no loops.
is_tree = all(g.edge_index.shape[1] == g.num_nodes - 1 for g in val[:200])
print(f"\n  edges == nodes - 1 for first 200 trees: {is_tree}")

# mask_main is undocumented -- work out what it refers to.
main, halo_ids = np.asarray(t.mask_main), np.asarray(t.node_halo_id).ravel()
match = np.isin(main, halo_ids).mean()
print(f"  mask_main: {match:.0%} of its values are node_halo_ids "
      f"-> {'main-branch halo IDs' if match > 0.9 else 'something else'}")

# ==========================================================================
# 3. WHAT DO THE FOUR FEATURES LOOK LIKE?
# ==========================================================================
header("3. THE FOUR NODE FEATURES")

X = torch.cat([g.x for g in val]).numpy()
mass, conc, vmax, when = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
names = ["log M", "concentration", "log v_max", "scale factor a"]

for i, name in enumerate(names):
    col = X[:, i]
    print(f"  {name:16s} min {col.min():9.4f}  max {col.max():9.4f}  "
          f"mean {col.mean():8.4f}  std {col.std():7.4f}")
print("\n  NOTE: mass and v_max are ALREADY log10. Do not log them again.")
print("  NOTE: the scales differ ~30x, so normalise before training.")

fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
for ax, name, col in zip(axes, names, X.T):
    ax.hist(col, bins=60, color=BLUE, edgecolor="none")
    ax.set_xlabel(name)
    ax.set_yticks([])
axes[0].set_ylabel("nodes")
fig.suptitle("Distribution of the four node features", y=1.04, fontsize=11)
fig.tight_layout()
fig.savefig(PLOTS / "01_features.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 4. THREE ODDITIES IN THOSE DISTRIBUTIONS
#
# Each one is a trap: something that looks like physics but isn't, or looks
# like a bug but isn't.
# ==========================================================================
header("4. DATA QUIRKS")

# (a) The sharp cliff in the mass histogram is an EDITING decision, not
#     physics: blobs below 3e10 solar masses were deleted to shrink the files.
#     Check by looking for a jump in the counts right at that threshold.
cut = np.log10(3e10)
print(f"  (a) mass cliff -- paper says blobs below 3e10 (log10 {cut:.4f}) were pruned")
reference = ((mass >= cut) & (mass < cut + 0.05)).sum()
for lo in np.arange(cut - 0.15, cut + 0.16, 0.05):
    n = ((mass >= lo) & (mass < lo + 0.05)).sum()
    bar = "#" * int(40 * n / max(reference, 1))
    mark = "  <-- stated cut" if abs(lo - cut) < 0.026 else ""
    print(f"      {lo:6.3f}  {n:9,d}  {bar}{mark}")
print(f"      only {(mass < cut).mean():.3%} of blobs survive below it")
print("      -> the cliff is the pruning threshold, NOT a feature of the universe")

# (b) The spike near zero concentration. Is it one sentinel "missing" value,
#     or a genuine population of low-concentration halos? Count DISTINCT
#     values: a sentinel would be a handful, a real spread is thousands.
low = conc[conc < 0.05]
print(f"\n  (b) concentration spike at zero -- {len(low):,d} blobs below 0.05 "
      f"({len(low)/len(conc):.2%})")
print(f"      distinct values among them: {len(np.unique(low)):,d}")
print("      -> a GENUINE spread of low values, not a 'missing data' placeholder")

# (c) Scale factor only takes a limited set of values -- these are simulation
#     snapshots. It may work better as an embedding than as a float.
unique_a = np.unique(when)
print(f"\n  (c) scale factor is DISCRETE: {len(unique_a)} distinct values")
print(f"      earliest {unique_a[:3].round(4)}  latest {unique_a[-3:].round(4)}")

# ==========================================================================
# 5. WHICH SUMMARIES CARRY THE ANSWER?
#
# The key experiment. Boil each tree down to simple statistics, then measure
# which ones move with each dial. This tells us what a model will have to
# use -- and proves there is signal here at all -- before we build one.
# ==========================================================================
header("5. WHAT PREDICTS THE TWO DIALS")

print("  summarising trees (slow part)...")
V = summary_table(val)
corr_om = correlations(V, "Omega_m")
corr_s8 = correlations(V, "sigma_8")

print_correlation_table(corr_om, corr_s8)

print("\n  Omega_m is read from CONCENTRATION (how squished the blobs are).")
print("  sigma_8 is read from TIME (how much of the tree sits early).")
print("  Mass is nearly useless for both -- the counter-intuitive result.")

correlation_chart(corr_om, "$\\Omega_m$",
                  "Which tree summaries know about the matter density?",
                  PLOTS / "03_correlations.png")
correlation_chart(corr_s8, "$\\sigma_8$",
                  "Time-based summaries carry $\\sigma_8$; mass carries nothing",
                  PLOTS / "05_sigma8.png")

# ==========================================================================
# 6. DOES THE GROWTH HISTORY HELP?
#
# Physically it should: matter-rich universes build halos earlier. But the
# early history was deleted by the pruning, so we expect it to be weak.
# Two checks -- the median growth curve, and the standard formation-time
# statistic a_50.
# ==========================================================================
header("6. GROWTH HISTORY (mostly destroyed by pruning)")

fig, ax = plt.subplots(figsize=(6.4, 4))
grid = np.linspace(0.1, 1.0, 28)
q_lo, q_hi = np.quantile(V["Omega_m"], [0.25, 0.75])
for keep, label, colour in (
        (V["Omega_m"] > q_hi, "high $\\Omega_m$ (matter-rich)", ORANGE),
        (V["Omega_m"] < q_lo, "low $\\Omega_m$ (matter-poor)", BLUE)):
    curves = []
    for g in [tr for tr, k in zip(val, keep) if k][:600]:
        a, m = g.x[:, 3].numpy(), g.x[:, 0].numpy()
        curves.append([m[a <= t].max() - m.max() if (a <= t).any() else np.nan for t in grid])
    ax.plot(grid, np.nanmedian(curves, axis=0), lw=2, color=colour, label=label)
ax.axhline(-HALF_DEX, color="#b8b6b0", lw=1, ls="--")
ax.text(0.12, -HALF_DEX + 0.03, "half of final mass", color=MUTED, fontsize=8)
ax.set_xlabel("scale factor a   (0.1 = early universe, 1.0 = today)")
ax.set_ylabel("mass relative to final  [dex]")
ax.set_title("Matter-rich halos assemble slightly earlier — but the gap is small", fontsize=11)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(PLOTS / "02_growth.png", dpi=150, bbox_inches="tight")
plt.close(fig)

ok = np.isfinite(V["a_50"])
print(f"  a_50 (standard formation-time statistic) vs Omega_m: "
      f"r = {corr_om['a_50']:+.3f}")
print("  -> essentially useless, because the early history was deleted.")
print("     Concentration works instead: it is measured per-blob, so it survived.")

fig, ax = plt.subplots(figsize=(6.4, 4))
edges = np.quantile(V["Omega_m"][ok], np.linspace(0, 1, 13))
mids = 0.5 * (edges[1:] + edges[:-1])
meds = [np.median(V["a_50"][ok][(V["Omega_m"][ok] >= lo) & (V["Omega_m"][ok] < hi)])
        for lo, hi in zip(edges[:-1], edges[1:])]
ax.scatter(V["Omega_m"][ok], V["a_50"][ok], s=4, alpha=0.12, color=BLUE, edgecolors="none")
ax.plot(mids, meds, color=ORANGE, lw=2.5, label="median")
ax.set_xlabel("$\\Omega_m$  (matter density)")
ax.set_ylabel("$a_{50}$  — when half the mass was in place")
ax.set_title(f"Half-mass time carries almost no $\\Omega_m$ signal "
             f"(r = {corr_om['a_50']:+.2f})", fontsize=11)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / "04_formation_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# 7. HOW MUCH DO TREE SIZES VARY?
#
# Matters for batching: wildly different sizes in one batch is awkward.
# ==========================================================================
header("7. TREE SIZE SPREAD")

sizes = V["n_nodes"]
print(f"  {sizes.min()} to {sizes.max()} blobs, median {int(np.median(sizes))} "
      f"-- a {sizes.max() // sizes.min()}x spread")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.hist(sizes, bins=np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 50),
        color=BLUE, edgecolor="none")
ax.set_xscale("log")
ax.set_xlabel("blobs per tree  (log scale)")
ax.set_ylabel("number of trees")
ax.set_title(f"Trees vary {sizes.max() // sizes.min()}x in size — "
             f"median {int(np.median(sizes))} blobs", fontsize=11)
fig.tight_layout()
fig.savefig(PLOTS / "06_tree_sizes.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nwrote plots/01..06")
