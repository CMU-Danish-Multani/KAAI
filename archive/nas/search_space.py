"""The architecture search space: interchangeable parts, assembled from a config.

WHY A SEARCH SPACE RATHER THAN FIVE MODELS
------------------------------------------
No published paper contains the combination we are looking for. Each of the
blocks in point_clouds/blocks/ was taken from a different paper and rewritten to
plug into the same socket, so the search can try arrangements nobody assembled.
That is BioArc's thesis (arXiv 2512.00283) applied to geometric data: its
measured result was that hybrids beat the best single-module architecture found
by restricting the search to one block type.

THE TWO ARMS
------------
Every pooling option is labelled with whether a held-out probe can recover the
number of points from its output, measured by blocks/count_screen.py.

    SCREENED   only count-blind options are offered
    OPEN       count-aware options are offered too

The point is not that one scores higher. It is to measure how much of the best
score found in the OPEN arm is the galaxy-count artifact rather than structure.

Measured R2 for recovering log N, 3 seeds, from count_screen.py:
    mean pooling                       -0.66    blind
    quasi-arithmetic, count normalise  -0.78    blind
    attention, count_aware off         -3.94    blind
    fishnets, fisher not exposed       -0.10    blind
    PNA, no scalers, mean+std          -2.59    blind
    sum pooling                        +0.91    AWARE
    max pooling                        +0.90    AWARE
    PNA with degree scalers            +1.00    AWARE
    attention, count_aware on          +1.00    AWARE
    fishnets exposing total Fisher     +1.00    AWARE
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from point_clouds.blocks.attention_readout import AttentionReadout
from point_clouds.blocks.fishnets import FishnetsAggregation
from point_clouds.blocks.pna import COUNT_BLIND_AGGREGATORS, PNAReadout
from point_clouds.blocks.quasi_arithmetic import QuasiArithmeticPool
from point_clouds.pointnet import TARGETS, pool

COUNT_BLIND_POOLINGS = ("mean", "quasi", "attention", "fishnets", "pna_blind")
COUNT_AWARE_POOLINGS = ("sum", "max", "pna_scaled")
ALL_POOLINGS = COUNT_BLIND_POOLINGS + COUNT_AWARE_POOLINGS


class _Plain(nn.Module):
    """Wraps the parameter-free poolings behind the same interface as the rest."""

    def __init__(self, how: str, count_scale: float):
        super().__init__()
        self.how, self.count_scale = how, count_scale

    def forward(self, values, index, n):
        return pool(values, index, n, self.how, self.count_scale)


def output_dim(module: nn.Module, dim: int) -> int:
    """Measure what a pooling module actually returns, rather than trusting a claim.

    Declared widening factors were wrong for two of the eight options when this
    was checked on 2026-08-24: attention returns seeds times dim, and fishnets
    returns its score dimension rather than the input dimension. Measuring costs
    one forward pass on four dummy clouds and cannot drift out of date.
    """
    probe = torch.randn(40, dim)
    index = torch.repeat_interleave(torch.arange(4), 10)
    was_training = module.training
    module.eval()
    with torch.no_grad():
        out = module(probe, index, 4)
    module.train(was_training)
    return int(out.shape[1])


def make_pooling(name: str, dim: int, count_scale: float) -> nn.Module:
    """Build a pooling module by name. Its output width is measured, not declared."""
    if name in ("sum", "mean", "max"):
        return _Plain(name, count_scale)
    if name == "quasi":
        return QuasiArithmeticPool(dim, normalise="count")
    if name == "attention":
        return AttentionReadout(dim, seeds=4, heads=4, count_aware=False)
    if name == "fishnets":
        return FishnetsAggregation(dim, expose_total_fisher=False)
    if name.startswith("pna"):
        scaled = name == "pna_scaled"
        aggs = None if scaled else COUNT_BLIND_AGGREGATORS
        return PNAReadout(dim, float(np.log(count_scale + 1.0)),
                          **({} if scaled else {"aggregators": aggs}),
                          use_degree_scalers=scaled)
    raise ValueError(f"unknown pooling {name!r}")


def pooling_is_count_blind(name: str) -> bool:
    return name in COUNT_BLIND_POOLINGS


def sample_config(trial, arm: str) -> Dict:
    """Draw one architecture from the space. `arm` is 'screened' or 'open'."""
    poolings = (COUNT_BLIND_POOLINGS if arm == "screened" else ALL_POOLINGS)
    family = trial.suggest_categorical("family", ["deepsets", "gnn"])
    config = {
        "family": family,
        "pooling": trial.suggest_categorical("pooling", list(poolings)),
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    }
    if family == "gnn":
        config["layers"] = trial.suggest_int("layers", 1, 5)
        config["cutoff"] = trial.suggest_categorical("cutoff", [0.010, 0.015, 0.020, 0.030])
        config["angular"] = trial.suggest_categorical("angular", [False, True])
        config["n_basis"] = trial.suggest_categorical("n_basis", [8, 16])
    return config


def describe(config: Dict) -> str:
    parts = [config["family"], f"pool={config['pooling']}", f"h={config['hidden']}"]
    if config["family"] == "gnn":
        parts += [f"L={config['layers']}", f"Rc={config['cutoff']:g}",
                  f"ang={'on' if config['angular'] else 'off'}",
                  f"basis={config['n_basis']}"]
    parts.append(f"lr={config['lr']:.1e}")
    return " ".join(parts)
