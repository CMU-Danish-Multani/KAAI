"""Benchmark every zoo architecture on every CAMELS task at matched compute.

    conda run -n ltuili python -u -m ili_kaai.sweep --smoke      # one cell, wiring check
    conda run -n ltuili python -u -m ili_kaai.sweep              # the full sweep

Writes ili_kaai/results/sweep.json incrementally, with "complete": false stamped at
launch, so a run killed part way leaves a readable partial rather than a file that
claims to be finished.

Recorded per cell, and this set is the zoo's standardised evaluation:
    r2              point recovery from the posterior mean, per parameter
    logProbTruth    mean log density the posterior assigns the true parameters
    coverage68/95   fraction of test points whose truth falls in the credible interval
    tarpAtNominal   TARP expected coverage at the 68 and 95 credibility levels
    trainSeconds    wall clock, so compute is measured rather than assumed matched
    nParameters     trainable weights
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner

from common.metrics import credible_coverage, seed_all
from ili_kaai.architectures import TRAIN_ARGS, ZOO, Architecture
from ili_kaai.embeddings import EMBEDDINGS
from ili_kaai.tasks import TASKS, Task, load, prior_bounds

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ili_kaai" / "results"



def pretrain_embedding(emb: nn.Module, x: np.ndarray, theta: np.ndarray,
                       epochs: int, device: str, lr: float = 1e-3,
                       batch_size: int = 32) -> float:
    """Train the embedding alone on plain regression. Returns seconds spent.

    Measured: joint training from scratch collapses. Validation log probability rises
    while R2 stays at zero, because the flow models the marginal distribution of theta
    and ignores x. The embedding starts as noise, so conditioning on it hurts, so the
    flow learns to ignore the context, so the embedding never receives a gradient.

    Giving the embedding a head start removes the thing that triggers the collapse.
    Targets are standardised here because a plain squared error on raw parameters
    would weight sigma_8 and Omega_m by their arbitrary units.
    """
    with torch.no_grad():
        out_dim = emb(torch.tensor(x[:2], device=device)).shape[1]
    head = nn.Linear(out_dim, theta.shape[1]).to(device)
    model = nn.Sequential(emb, head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.tensor(x, device=device)
    T = torch.tensor(theta, device=device)
    mean, spread = T.mean(0), T.std(0).clamp_min(1e-8)
    target = (T - mean) / spread

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[b]), target[b]).backward()
            opt.step()
    return time.time() - t0


def build(arch: Architecture, task: Task, device: str, out_dir: Path):
    """Both backends go through the same runner, so entries stay comparable.

    lampe takes its device at net construction time; sbi takes it at runner
    construction. That asymmetry is the only thing this function has to know.
    """
    lo, hi = prior_bounds(task)
    prior = ili.utils.Uniform(low=lo, high=hi, device=device)
    embed, pretrain_seconds = {}, 0.0
    if arch.embedding:
        # The embedding must see exactly what the density estimator sees, no more and
        # no less. run_cell hands the runner train and val concatenated, because
        # ltu-ili carves its own validation split out of whatever it is given, so
        # "val" here is simply more training data. Pretraining on train alone left the
        # embedding 200 simulations short and cost R2 0.235 -> 0.125 at one seed.
        d = load(task)
        x_train = np.concatenate([d["train"][0], d["val"][0]])
        theta_train = np.concatenate([d["train"][1], d["val"][1]])
        if x_train.ndim != 3:
            raise ValueError(f"{arch.key} needs a point cloud task, but {task.key} "
                             f"gives x with shape {x_train.shape[1:]}")
        n_points, n_features = x_train.shape[1], x_train.shape[2]
        net = EMBEDDINGS[arch.embedding](n_points=n_points, n_features=n_features,
                                         **arch.embedding_args).to(device)
        if arch.pretrainEpochs:
            pretrain_seconds = pretrain_embedding(
                net, x_train, theta_train, arch.pretrainEpochs, device)
        embed = {"embedding_net": net}

    members = arch.mixture or ((arch.model, arch.model_args),) * arch.repeats
    if arch.backend == "lampe":
        nets = [n for model, args in members
                for n in ili.utils.load_nde_lampe(
                    model=model, engine=arch.engine, device=device, **embed, **args)]
    elif arch.embedding:
        # sbi z-scores every dimension of x independently before the embedding sees
        # it, and ltu-ili's argument validation will not pass `z_score_x` through. For
        # a point cloud that scaling is destructive rather than merely unnecessary: x
        # is (n_points, 3), so each galaxy slot gets its own affine map and the
        # relative geometry the embedding exists to read is scrambled. Measured: a
        # held out probe for Omega_m falls from +0.2326 to +0.0598 under it.
        # Positions are already in [0, 1] by construction, so scaling is switched off
        # by calling sbi's factory directly with the same arguments ltu-ili would use.
        try:                                     # sbi moved this between versions
            from sbi.neural_nets import posterior_nn
        except ImportError:
            from sbi.utils import posterior_nn
        nets = [posterior_nn(model=model, z_score_x="none", **embed, **args)
                for model, args in members]
    else:
        nets = [ili.utils.load_nde_sbi(engine=arch.engine, model=model,
                                       **embed, **args)
                for model, args in members]
    runner = InferenceRunner.load(
        backend=arch.backend, engine=arch.engine, prior=prior, nets=nets,
        device=device, train_args=dict(TRAIN_ARGS), out_dir=out_dir)
    return runner, prior, pretrain_seconds


def draw(posterior, x: np.ndarray, n_draws: int, method: str,
         device: str) -> np.ndarray:
    """Posterior samples for every evaluation point. Shape (n_draws, n_points, n_par).

    NPE samples directly. NLE and NRE sample the learned proxy with emcee, because the
    variational sampler raises RecursionError on this sbi version.
    """
    out = []
    for row in x:
        xt = torch.tensor(row, dtype=torch.float32, device=device)
        if method == "direct":
            s = posterior.sample((n_draws,), x=xt, show_progress_bars=False)
        else:
            s = posterior.sample((n_draws,), x=xt, show_progress_bars=False,
                                 num_chains=10, warmup_steps=50, thin=1)
        out.append(np.asarray(s.cpu() if torch.is_tensor(s) else s))
    return np.stack(out, axis=1)


def score(samples: np.ndarray, truth: np.ndarray) -> Dict:
    """Everything the zoo records, computed from posterior samples alone so that NPE,
    NLE and NRE are measured the same way."""
    mean = samples.mean(0)
    ss_res = ((truth - mean) ** 2).sum(0)
    ss_tot = ((truth - truth.mean(0)) ** 2).sum(0)

    cov68 = credible_coverage(samples, truth, 0.68)
    cov95 = credible_coverage(samples, truth, 0.95)

    logp = []
    for i in range(truth.shape[0]):
        try:
            kde = gaussian_kde(samples[:, i, :].T)
            logp.append(float(np.atleast_1d(kde.logpdf(truth[i]))[0]))
        except np.linalg.LinAlgError:
            logp.append(float("nan"))       # degenerate posterior, recorded not hidden

    out = {"r2": [float(v) for v in 1.0 - ss_res / ss_tot],
           "coverage68": [float(v) for v in cov68],
           "coverage95": [float(v) for v in cov95],
           "logProbTruth": float(np.nanmean(logp)),
           "logProbFailures": int(np.isnan(logp).sum())}

    if truth.shape[1] > 1:                  # TARP needs a multivariate parameter space
        from tarp import get_tarp_coverage
        ecp, alpha = get_tarp_coverage(samples, truth, norm=True, seed=0)
        out["tarpAt68"] = float(np.interp(0.68, alpha, ecp))
        out["tarpAt95"] = float(np.interp(0.95, alpha, ecp))
    else:
        out["tarpAt68"] = None              # null, not zero: not measurable at dim 1
        out["tarpAt95"] = None
    return out


def run_cell(arch: Architecture, task: Task, seed: int, n_eval: int, n_draws: int,
             device: str) -> Dict:
    seed_all(seed)
    data = load(task)
    xtr = np.concatenate([data["train"][0], data["val"][0]])
    ttr = np.concatenate([data["train"][1], data["val"][1]])

    runner, _, pretrain_seconds = build(
        arch, task, device, RESULTS / "runs" / f"{arch.key}_{task.key}")
    t0 = time.time()
    posterior, summaries = runner(loader=NumpyLoader(x=xtr, theta=ttr))
    train_seconds = time.time() - t0
    
    xte, tte = data["test"]
    if n_eval and n_eval < len(xte):
        idx = np.random.default_rng(seed).choice(len(xte), n_eval, replace=False)
        xte, tte = xte[idx], tte[idx]

    t1 = time.time()
    samples = draw(posterior, xte, n_draws, arch.sample_method, device)
    eval_seconds = time.time() - t1

    n_par = sum(sum(p.numel() for p in n.parameters())
                for n in getattr(posterior, "posteriors", [posterior])
                if hasattr(n, "parameters"))
    return {"architecture": arch.key, "task": task.key, "seed": seed,
            "trainSeconds": round(train_seconds, 1),
            "pretrainSeconds": round(pretrain_seconds, 1),
            "evalSeconds": round(eval_seconds, 1),
            "nParameters": int(n_par),
            "nEvalPoints": int(len(xte)), "nDraws": n_draws,
            "sampleMethod": arch.sample_method,
            "finalValLogProb": [float(s["validation_log_probs"][-1])
                                for s in summaries],
            **score(samples, tte)}


def aggregate(cells: List[Dict]) -> List[Dict]:
    """Mean and spread across seeds. A single seed reports null spread, never zero."""
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for c in cells:
        if "error" in c:
            continue
        grouped.setdefault((c["architecture"], c["task"]), []).append(c)

    out = []
    for (arch, task), rows in grouped.items():
        agg = {"architecture": arch, "task": task, "nSeeds": len(rows)}
        for field in ("r2", "coverage68", "coverage95"):
            stack = np.array([r[field] for r in rows], dtype=float)
            agg[field] = [round(float(v), 4) for v in stack.mean(0)]
            agg[f"{field}Std"] = ([round(float(v), 4) for v in stack.std(0)]
                                  if len(rows) > 1 else None)
        for field in ("logProbTruth", "trainSeconds", "tarpAt68", "tarpAt95"):
            vals = [r[field] for r in rows if r[field] is not None]
            agg[field] = round(float(np.mean(vals)), 4) if vals else None
            agg[f"{field}Std"] = (round(float(np.std(vals)), 4)
                                  if vals and len(rows) > 1 else None)
        agg["nParameters"] = rows[0]["nParameters"]
        out.append(agg)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--architectures", type=str, nargs="+", default=list(ZOO))
    p.add_argument("--tasks", type=str, nargs="+", default=list(TASKS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-eval", type=int, default=100,
                   help="test points used for coverage; 0 means all")
    p.add_argument("--n-draws", type=int, default=1000)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="one cheap cell, to check wiring before committing compute")
    p.add_argument("--out", type=str, default="sweep.json")
    args = p.parse_args()

    if args.smoke:
        args.architectures, args.tasks, args.seeds = ["npeMaf"], ["camelsJoint"], [0]
        args.n_eval, args.n_draws = 20, 200

    unknown = set(args.architectures) - set(ZOO) or set(args.tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown architecture or task: {sorted(unknown)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out
    payload = {"complete": False, "args": vars(args),
               "trainArgs": TRAIN_ARGS,
               "versions": {"torch": torch.__version__, "ili": "0.1.5",
                            "numpy": np.__version__,
                            "python": platform.python_version()},
               "cells": [], "aggregate": []}

    def save(done: bool = False) -> None:
        payload["complete"] = done
        payload["aggregate"] = aggregate(payload["cells"])
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    save()
    total = len(args.architectures) * len(args.tasks) * len(args.seeds)
    print(f"  {total} cells, device {args.device}, matched compute {TRAIN_ARGS}\n",
          flush=True)

    i = 0
    for task_key in args.tasks:
        for arch_key in args.architectures:
            for seed in args.seeds:
                i += 1
                head = f"[{i}/{total}] {arch_key:16s} {task_key:15s} seed {seed}"
                try:
                    cell = run_cell(ZOO[arch_key], TASKS[task_key], seed,
                                    args.n_eval, args.n_draws, args.device)
                    payload["cells"].append(cell)
                    print(f"  {head}  R2 {cell['r2']}  cov68 "
                          f"{[round(v, 2) for v in cell['coverage68']]}  "
                          f"{cell['trainSeconds']:.0f}s", flush=True)
                except Exception as exc:                  # a dead cell is a result
                    payload["cells"].append({"architecture": arch_key,
                                             "task": task_key, "seed": seed,
                                             "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  {head}  FAILED {type(exc).__name__}: {exc}", flush=True)
                save()

    save(done=True)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
