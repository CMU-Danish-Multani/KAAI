"""The three inference heads named in the brief, at matched compute.

    NPE   neural posterior estimation with a masked autoregressive flow
    NRE   neural ratio estimation, a classifier between matched and mismatched pairs
    MDN   mixture density network, a weighted sum of Gaussians

All three consume the same input, the two-point correlation function, so the
comparison isolates the inference head. All three get the same training budget,
which is what "matched compute" in the brief requires: without it a comparison
measures patience rather than method.

WHY THESE PRODUCE SOMETHING THE REST OF THE ZOO CANNOT
------------------------------------------------------
Every other entry returns a number. These return a distribution, which is what a
cosmologist needs, because the likelihood cannot be written down: you can
simulate a universe from parameters but you cannot compute the probability of a
universe given parameters. So the relationship is learned backwards from
simulations.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from sbi.inference import NPE, NRE
from sbi.utils import BoxUniform

from zoo.inference.tasks import PARAM_LIMITS, Task, coverage, load_task, posterior_r2

HEADS = ("npe_maf", "npe_nsf", "npe_mdn", "nre")
MATCHED_EPOCHS = 200
N_POSTERIOR_DRAWS = 500


def _prior(task: Task, device: str) -> BoxUniform:
    lo = torch.tensor([PARAM_LIMITS[t][0] for t in task.targets], dtype=torch.float32)
    hi = torch.tensor([PARAM_LIMITS[t][1] for t in task.targets], dtype=torch.float32)
    return BoxUniform(low=lo.to(device), high=hi.to(device), device=device)


def fit(head: str, task: Task, data: Dict, seed: int = 0,
        device: str = "cpu", epochs: int = MATCHED_EPOCHS) -> Dict:
    """Train one inference head and score it on the test split.

    Returns posterior-mean R2 so it stays comparable with the rest of the zoo,
    plus coverage, which is the check the rest of the zoo cannot provide.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = _prior(task, device)
    theta = torch.as_tensor(data["theta"]["train"]).to(device)
    x = torch.as_tensor(data["x"]["train"]).to(device)

    started = time.time()
    if head.startswith("npe"):
        estimator = {"npe_maf": "maf", "npe_nsf": "nsf", "npe_mdn": "mdn"}[head]
        engine = NPE(prior=prior, density_estimator=estimator, device=device,
                     show_progress_bars=False)
    elif head == "nre":
        engine = NRE(prior=prior, device=device, show_progress_bars=False)
    else:
        raise ValueError(f"unknown head {head!r}")

    engine.append_simulations(theta, x)
    trained = engine.train(max_num_epochs=epochs, show_train_summary=False)
    posterior = engine.build_posterior(trained)
    train_minutes = (time.time() - started) / 60

    x_test = torch.as_tensor(data["x"]["test"]).to(device)
    truth = data["theta"]["test"]
    draws = []
    for row in x_test:
        s = posterior.sample((N_POSTERIOR_DRAWS,), x=row, show_progress_bars=False)
        draws.append(s.detach().cpu().numpy())
    samples = np.stack(draws)

    result = {"head": head, "task": task.key, "seed": seed,
              "train_minutes": round(train_minutes, 2),
              "epochs": epochs,
              "n_parameters": sum(p.numel() for p in trained.parameters()),
              "posterior_mean_r2": posterior_r2(samples, truth)}
    result.update(coverage(samples, truth))
    return result


def run_all(device: str = "cpu", seeds: Tuple[int, ...] = (0, 1, 2),
            heads: Tuple[str, ...] = HEADS,
            tasks: Optional[Tuple[Task, ...]] = None) -> list:
    from zoo.inference.tasks import TASKS
    tasks = tasks or TASKS
    out = []
    for task in tasks:
        data = load_task(task)
        for head in heads:
            per_seed = []
            for seed in seeds:
                try:
                    per_seed.append(fit(head, task, data, seed, device))
                except Exception as exc:            # a head that cannot fit is a finding
                    print(f"    {head} / {task.key} seed {seed} FAILED: "
                          f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
            if not per_seed:
                continue
            agg = {"head": head, "task": task.key, "seeds": len(per_seed),
                   "train_minutes": round(float(np.mean([r["train_minutes"] for r in per_seed])), 2),
                   "n_parameters": per_seed[0]["n_parameters"]}
            r2 = np.array([r["posterior_mean_r2"] for r in per_seed])
            agg["posterior_mean_r2"] = [round(float(v), 4) for v in r2.mean(0)]
            agg["posterior_mean_r2_std"] = ([round(float(v), 4) for v in r2.std(0)]
                                            if len(per_seed) > 1 else None)
            for k in ("coverage_50", "coverage_68", "coverage_90", "coverage_95",
                      "calibration_error"):
                agg[k] = round(float(np.mean([r[k] for r in per_seed])), 4)
            agg["overconfident"] = bool(np.mean([r["overconfident"] for r in per_seed]) > 0.5)
            out.append(agg)
            print(f"  {task.key:28s} {head:8s} R2 {agg['posterior_mean_r2']}  "
                  f"cov90 {agg['coverage_90']:.3f}  calib_err {agg['calibration_error']:.3f}  "
                  f"[{agg['train_minutes']:.1f} min]", flush=True)
    return out
