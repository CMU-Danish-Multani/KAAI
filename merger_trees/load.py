"""Shared helpers for the CS-Trees merger tree data.

Everything that more than one script needs lives here: loading, finding the
root of a tree, boiling a tree down to summary numbers, and the plot style.
"""

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/CAMELS-SAM/trees"

# Masses are stored as log10, so "half the mass" is a subtraction of 0.301.
HALF_DEX = np.log10(2)


def load(split):
    """Load one split. Returns a plain Python list of PyG Data objects."""
    return torch.load(DATA_DIR / f"CS_tree_{split}.pt", weights_only=False)


def root_index(tree):
    """Index of today's halo.

    Edges point from the older blob to the newer one, so the root is the only
    blob that never appears as a source -- nothing comes after it.
    """
    src = tree.edge_index[0].numpy()
    return int(np.argmax(~np.isin(np.arange(tree.num_nodes), src)))


def summarise(tree):
    """Boil one tree down to a row of numbers we can correlate against labels.

    The point is to find out which simple statistics already carry information
    about the two dials, before building any model.
    """
    x = tree.x.numpy()
    mass, conc, vmax, when = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    dst = tree.edge_index[1].numpy()
    root = root_index(tree)

    # A blob with 2+ arrows pointing in is a real merger. Most blobs have 1
    # (a halo simply growing), so this counts the genuinely interesting events.
    n_progenitors = np.bincount(dst, minlength=tree.num_nodes)
    n_mergers = int((n_progenitors >= 2).sum())

    return {
        # --- shape of the tree
        "n_nodes": tree.num_nodes,
        "n_mergers": n_mergers,
        "merger_rate": n_mergers / tree.num_nodes,
        # --- today's halo
        "root_M": mass[root], "root_c": conc[root], "root_vmax": vmax[root],
        # --- averages over every blob in the tree
        "mean_M": mass.mean(), "max_M": mass.max(),
        "mean_c": conc.mean(), "std_c": conc.std(),
        "mean_vmax": vmax.mean(),
        # --- timing: when the tree's blobs existed
        "mean_a": when.mean(), "min_a": when.min(),
        "frac_early": (when < 0.3).mean(),      # share of blobs from early times
        "a_50": half_mass_time(mass, when, mass[root]),
        # --- the answer
        "Omega_m": float(tree.y[0, 0]), "sigma_8": float(tree.y[0, 1]),
    }


def half_mass_time(mass, when, root_mass):
    """Scale factor at which the halo first held half its present-day mass.

    A standard formation-time measure in astronomy. NOTE: it turns out to be
    nearly useless here (r = +0.05 with Omega_m) because small blobs were
    deleted from this dataset, so the early mass history is incomplete.
    Kept because that negative result is worth being able to reproduce.
    """
    order = np.argsort(when)
    a_sorted, m_sorted = when[order], mass[order]
    uniq_a, starts = np.unique(a_sorted, return_index=True)
    ends = list(starts[1:]) + [len(a_sorted)]
    # heaviest blob alive at each time, made non-decreasing
    peak = np.maximum.accumulate([m_sorted[i:j].max() for i, j in zip(starts, ends)])
    reached = np.where(peak >= root_mass - HALF_DEX)[0]
    return uniq_a[reached[0]] if len(reached) else np.nan


def summary_table(trees):
    """summarise() every tree, returned as {column_name: array}."""
    rows = [summarise(t) for t in trees]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
