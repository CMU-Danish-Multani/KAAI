"""Scoring, seeding and device selection. Shared by the cloud and tree tracks.

R2 follows CosmoBench equation (1). Uncertainty follows the paper's own protocol:
bootstrap resampling of the test set, reported as one standard deviation.
"""

import random
from typing import Tuple

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed every generator that can change a run's result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_generator(seed: int) -> torch.Generator:
    """A seeded generator for any DataLoader that shuffles.

    Accepting a --seed and then leaving the loader unseeded is the usual way a
    run quietly stops being reproducible.
    """
    return torch.Generator().manual_seed(seed)


def resolve_device(choice: str = "auto") -> torch.device:
    """CUDA, then MPS on Apple Silicon, then CPU."""
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def r2_score(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """R2 for each column. 1 is perfect, 0 is no better than predicting the mean.

    Both arrays are (n_samples, n_targets).
    """
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def bootstrap_r2(pred: np.ndarray, true: np.ndarray, n_boot: int = 1000,
                 seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of R2 across bootstrap resamples of the test set.

    This is what CosmoBench reports as its '+/- 1 std'.
    """
    rng = np.random.default_rng(seed)
    n = len(true)
    scores = np.stack([r2_score(pred[i], true[i])
                       for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return scores.mean(axis=0), scores.std(axis=0)
