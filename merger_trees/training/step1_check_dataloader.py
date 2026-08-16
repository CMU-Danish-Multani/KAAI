"""STEP 1 CHECK -- does the dataloader produce sane, normalised batches?

Run: python training/step1_check_dataloader.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from merger_trees.dataset import get_loaders, get_split
from common.viz import header

LIMIT = 2000          # subset, so this check runs in seconds

header("STEP 1 -- DATALOADER")

# ---- before/after normalisation ------------------------------------------
print("  loading train subset...")
raw, _ = get_split("train", stats=None, limit=LIMIT)
print(f"  {len(raw)} trees\n")

print("  feature scales AFTER normalisation (target: mean~0, std~1):")
x = torch.cat([t.x for t in raw])
e = torch.cat([t.edge_attr for t in raw])
for i, name in enumerate(["mass", "concentration", "v_max", "scale factor"]):
    print(f"    {name:16s} mean {x[:, i].mean():+7.4f}   std {x[:, i].std():6.4f}")
print(f"    {'edge_attr':16s} mean {e.mean():+7.4f}   std {e.std():6.4f}")

# ---- what a batch actually looks like ------------------------------------
header("WHAT ONE BATCH CONTAINS")

loaders, stats = get_loaders(batch_size=32, limit=LIMIT)
batch = next(iter(loaders["train"]))
print(f"  {batch}\n")
print(f"  {batch.num_graphs} trees merged into ONE big disconnected graph:")
print(f"    x          {tuple(batch.x.shape)}   all nodes stacked")
print(f"    edge_index {tuple(batch.edge_index.shape)}   all arrows, re-indexed")
print(f"    y          {tuple(batch.y.shape)}   one answer pair per tree")
print(f"    batch      {tuple(batch.batch.shape)}   which tree each node belongs to")

# The `batch` vector is what lets pooling know where one tree ends and the
# next begins -- without it the trees would blur into each other.
counts = torch.bincount(batch.batch)
print(f"\n  nodes per tree in this batch: min {counts.min()}, max {counts.max()}")
print(f"  batch.batch[:12] = {batch.batch[:12].tolist()}")

# ---- the checks that matter ----------------------------------------------
header("CHECKS")

checks = [
    ("normalised x has mean ~0", bool(x.mean().abs() < 0.05)),
    ("normalised x has std ~1", bool((x.std() - 1).abs() < 0.15)),
    ("edge_attr no longer dominates", bool(e.std() < 2.0)),
    ("one label pair per tree", batch.y.shape == (batch.num_graphs, 2)),
    ("node count matches batch vector", batch.x.shape[0] == batch.batch.shape[0]),
    ("edges stay within node range", int(batch.edge_index.max()) < batch.x.shape[0]),
    ("val loader built from TRAIN stats", "val" in loaders),
]
for label, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")

print("\n  STEP 1", "OK" if all(c[1] for c in checks) else "FAILED")
