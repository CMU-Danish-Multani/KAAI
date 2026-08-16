"""Shared chart style and helpers.

Used by both point_clouds/ and merger_trees/. Each domain writes into its
OWN plots/ folder -- this module only supplies style and maths.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"      # series colours, used in this order
POS, NEG = "#2a78d6", "#e34948"          # positive / negative correlations
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#eceae5"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": "#d8d7d2",
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def header(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def correlations(table, target):
    """How strongly each summary moves with a target. -1 to +1; 0 = no link.

    Constant columns are skipped: a value that never changes cannot correlate
    with anything, and would otherwise come out as NaN (0/0).
    """
    out = {}
    for f in table:
        if f in TARGETS:
            continue
        if np.std(table[f]) == 0:
            continue
        out[f] = float(np.corrcoef(table[f], table[target])[0, 1])
    return out


def constant_features(table):
    """Summaries that never vary -- usually a deliberate choice by the dataset."""
    return [f for f in table if f not in TARGETS and np.std(table[f]) == 0]


TARGETS = ("Omega_m", "sigma_8")


def correlation_chart(corr, target_label, title, path):
    """Horizontal bar chart of correlations. Blue = positive, red = negative."""
    order = sorted(corr, key=lambda f: corr[f])
    values = [corr[f] for f in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(order) + 1.6))
    ax.barh(y, values, color=[POS if v > 0 else NEG for v in values], height=0.62)
    ax.axvline(0, color="#b8b6b0", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(f"correlation with {target_label}")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v + (0.02 if v > 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=8, color=MUTED)
    ax.set_xlim(min(values) - 0.18, max(values) + 0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_correlation_table(corr_a, corr_b, label_a="r(Omega_m)", label_b="r(sigma_8)"):
    print(f"\n  {'summary':16s} {label_a:>12s} {label_b:>12s}")
    for f in sorted(corr_a, key=lambda k: -abs(corr_a[k])):
        print(f"  {f:16s} {corr_a[f]:12.3f} {corr_b[f]:12.3f}")
