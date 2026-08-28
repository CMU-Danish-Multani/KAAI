"""The multilayer perceptron baseline, and the machinery for tuning it.

This is CosmoBench's own 2PCF model, from its Appendix B.1: a four-layer network
reading the correlation function and predicting the two cosmological parameters.
It lives here rather than inside an experiment script because three separate
experiments train it, and an experiment that imports from another experiment is
a structure that stops making sense the moment there is a third one.

    tune()                     <- pick hyperparameters on the validation split
      |
      +-- fit()                <- train one model to completion
      +-- score()              <- R2 on a split

The hyperparameter ranges are the paper's, not ours. Deviating from them would
make the reproduction a different experiment.
"""

from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn

from common.metrics import r2_score, seed_all

# CosmoBench Appendix B.1. Every range here is quoted from the paper.
SEARCH_SPACE = {
    "layer_1": (64, 128),
    "layer_2": (64, 128),
    "layer_3": (16, 64),
    "learning_rate": (1e-5, 1e-2),
    "dropout": (0.0, 0.5),
    "batch_sizes": [4, 16, 64],
}
EPOCHS = 300
N_OUTPUTS = 2                       # Omega_m and sigma_8


class CorrelationMLP(nn.Module):
    """Four weight layers: features -> h1 -> h2 -> h3 -> (Omega_m, sigma_8)."""

    def __init__(self, n_features: int, layer_1: int, layer_2: int, layer_3: int,
                 dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, layer_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layer_1, layer_2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layer_2, layer_3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(layer_3, N_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit(table: dict, hyperparameters: dict, seed: int,
        device: torch.device) -> CorrelationMLP:
    """Train one model to completion on the training split.

    The whole split is moved onto the device once and indexed by a shuffled
    permutation, rather than going through a DataLoader. With 600 rows of 25
    numbers the loader's per-batch overhead would dominate the arithmetic.
    """
    seed_all(seed)
    x_train = torch.as_tensor(table["train"]["x"]).to(device)
    y_train = torch.as_tensor(table["train"]["y_scaled"]).to(device)

    model = CorrelationMLP(x_train.shape[1], hyperparameters["layer_1"],
                           hyperparameters["layer_2"], hyperparameters["layer_3"],
                           hyperparameters["dropout"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    loss_function = nn.MSELoss()

    # Seeded separately from the weights so the shuffle order is reproducible.
    shuffle = torch.Generator().manual_seed(seed)
    n_rows, batch_size = len(x_train), hyperparameters["batch_size"]

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n_rows, generator=shuffle).to(device)
        for start in range(0, n_rows, batch_size):
            rows = order[start:start + batch_size]
            optimiser.zero_grad()
            loss_function(model(x_train[rows]), y_train[rows]).backward()
            optimiser.step()
    return model


@torch.no_grad()
def score(model: CorrelationMLP, table: dict, split: str, device: torch.device,
          label_stats=None) -> Tuple[np.ndarray, np.ndarray]:
    """R2 per target on one split, plus the predictions.

    Passing label_stats converts predictions back to physical units first. R2 is
    invariant to that rescaling, so it is done for legibility of the predictions
    rather than to change the score.
    """
    model.eval()
    x = torch.as_tensor(table[split]["x"]).to(device)
    predictions = model(x).cpu().numpy().astype(np.float64)
    model.train()

    if label_stats is None:
        truth = table[split]["y_scaled"].astype(np.float64)
        return r2_score(predictions, truth), predictions

    mean, spread = label_stats
    predictions = predictions * spread + mean
    return r2_score(predictions, table[split]["y_physical"]), predictions


def tune(table: dict, n_trials: int, device: torch.device, seed: int,
         quiet: bool = False) -> dict:
    """Search the paper's hyperparameter ranges, selecting on validation R2.

    Returns the winning hyperparameters. Stage 1 measured that roughly 20 trials
    find what the paper's 100 find, so callers may reasonably pass far fewer.
    """
    def objective(trial: optuna.Trial) -> float:
        hyperparameters = {
            "layer_1": trial.suggest_int("layer_1", *SEARCH_SPACE["layer_1"]),
            "layer_2": trial.suggest_int("layer_2", *SEARCH_SPACE["layer_2"]),
            "layer_3": trial.suggest_int("layer_3", *SEARCH_SPACE["layer_3"]),
            "learning_rate": trial.suggest_float("learning_rate",
                                                 *SEARCH_SPACE["learning_rate"], log=True),
            "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
            "batch_size": trial.suggest_categorical("batch_size",
                                                    SEARCH_SPACE["batch_sizes"]),
        }
        model = fit(table, hyperparameters, seed, device)
        return float(np.mean(score(model, table, "val", device)[0]))

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        if not quiet and (done % 10 == 0 or done == n_trials):
            print(f"    trial {done:4d}/{n_trials}  best mean val R2 {study.best_value:.4f}",
                  flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, callbacks=[report])
    return study.best_params


def tune_then_retrain(table: dict, label_stats, n_trials: int, tuning_seed: int,
                      training_seeds: List[int], device: torch.device,
                      quiet: bool = False) -> Dict:
    """Tune once, then retrain the winner under several seeds and score on test.

    Retraining across seeds is not optional. A single run cannot support a
    comparative claim, and the spread across seeds is what says whether a gap
    between two configurations is real.
    """
    best = tune(table, n_trials, device, tuning_seed, quiet)
    per_seed = np.stack([score(fit(table, best, seed, device), table, "test",
                               device, label_stats)[0]
                         for seed in training_seeds])
    return {
        "hyperparameters": best,
        "test_r2_mean": [float(v) for v in per_seed.mean(0)],
        # One seed has no spread, which is not the same as a spread of zero.
        "test_r2_seed_std": ([float(v) for v in per_seed.std(0)]
                             if len(training_seeds) > 1 else None),
        "test_r2_per_seed": [[float(v) for v in row] for row in per_seed],
    }
