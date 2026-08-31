"""Rank zoo entries for a stated inference problem, and emit a runnable ltu-ili config.

    python -m skill.query --modality summary_vector --n-params 2 \
        --n-observations 1000 --compute-seconds 7200 --downstream

This is the structured retrieval arm of the skill. It reads zoo.json and nothing else,
so every number it quotes is traceable to a sweep by path, and it can be evaluated
offline without a model in the loop. The few shot arm lives in SKILL.md, where Claude
reads the same catalogue as prose. Both are scored on the same held out problems by
evaluate.py, because the brief poses which retrieval strategy wins as an open question.

TWO KINDS OF KNOWLEDGE, KEPT APART
----------------------------------
The ranking comes from measurements in this zoo. The engine advice comes from the
field's published decision tables, which cover cases this zoo has never measured. They
are reported separately and labelled, because blending them would let an uncited rule
of thumb hide inside a number that looks measured.

WHY COMPUTE IS A CONSTRAINT AND NOT A SCORE TERM
------------------------------------------------
Accuracy barely separates these entries. On camelsJoint, twenty of twenty one land
between R2 0.800 and 0.873, a spread of 0.073, while cost spans 0.9 to 3294 seconds
per cell, a factor of 3700. So ranking on accuracy alone ranks on noise.

Adding a cost penalty to the score was tried and rejected. In log10 seconds that term
has magnitude 3 while accuracy spans 0.073 and the calibration penalty about 0.7, so
cost swamps calibration in every query and the recommender always returns the fast
overconfident entry. That contradicts the thing this zoo measured most clearly.

So compute is a filter: trainSeconds + nObservations * inferenceSeconds <= budget. If
nothing fits, the cheapest options are returned anyway and marked over budget, because
"nothing fits, here is the closest and by how much" is useful and silence is not.

WHY AN UNINFORMATIVE ENTRY IS REMOVED BEFORE CALIBRATION IS CONSIDERED
----------------------------------------------------------------------
Found by running the held out set. The recommender picked npeMafFlatten for a point
cloud problem, an entry whose R2 is about zero, because a posterior that predicts
nothing is wide, and a wide posterior scores well on coverage. Calibration is only
meaningful conditional on the posterior being informative. An entry whose measured R2
is at or below zero does no better than predicting the prior mean, so it is removed
before scoring however well calibrated it looks.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
ZOO_PATH = ROOT / "ili_kaai" / "results" / "zoo.json"

MODALITIES = ("summary_vector", "point_cloud")

# The brief's own success criterion: "recommended architectures reach MCMC-equivalent
# posterior quality in under two GPU hours".
DEFAULT_BUDGET_SECONDS = 7200.0

# An entry must beat this mean R2 to be recommended at all. Zero is the score of
# predicting the prior mean, so anything at or below it carries no information about
# theta and its calibration is an artifact of being wide.
MIN_USEFUL_R2 = 0.05


@dataclass
class Query:
    modality: str
    nParams: int
    nObservations: int = 1
    computeSeconds: Optional[float] = DEFAULT_BUDGET_SECONDS
    downstream: bool = False
    # Are the observations exchangeable trials from one subject or system? Deistler
    # Table 1 lists this as a primary axis and this zoo has never measured it, so it
    # only ever produces literature advice, never a ranking.
    iidTrials: bool = False


@dataclass
class Recommendation:
    rank: int
    key: str
    task: str
    r2: List[float]
    calibrationVerdict: str
    calibrationSigma: Optional[float]
    trainSeconds: float
    inferenceSeconds: float
    totalSeconds: float
    overBudget: bool
    score: float
    why: List[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class Result:
    recommendations: List[Recommendation]
    excluded: List[Dict]
    uninformative: List[Dict]
    advisedEngine: Optional[str]
    advice: List[str]
    outOfScope: bool


def load_zoo(path: Path = ZOO_PATH) -> Dict:
    if not path.exists():
        raise SystemExit(f"{path} missing. Build it with: python -m ili_kaai.zoo")
    return json.loads(path.read_text(encoding="utf-8"))


def matching_measurement(entry: Dict, query: Query) -> Optional[Dict]:
    """The measured task that best resembles the user's problem.

    Closest parameter count wins, because dim(theta) is the axis both Thiele Section
    2.7 and Deistler Table 1 say changes which family leads. Ties go to the task with
    more seeds, since a tie broken by sample size is at least broken by evidence.
    """
    candidates = [m for m in entry["measurements"] if m["coverage68"] is not None]
    if not candidates:
        return None
    return min(candidates,
               key=lambda m: (abs(len(m["params"]) - query.nParams), -m["nSeeds"]))


def reliable_accuracy(measurement: Dict) -> tuple:
    """Mean R2 minus one seed standard deviation, and whether a spread was available.

    Ranking on the mean rewards an entry that got lucky on one seed. Measured here,
    npeMdnPairwiseGnn reaches R2 0.140 with a seed spread of 0.106, so its spread is
    three quarters of its mean, while npeMafPairwiseGnnPretrained reaches 0.250 with a
    spread of 0.020. The sweep's own conclusion was that from scratch training does not
    reliably fail, it fails unpredictably, and that this is worse for a practitioner
    than reliable failure. A lower confidence bound is how that belief becomes a rank.

    A single seed has no spread. It is reported as unavailable rather than as zero, so
    the caller can warn instead of silently treating it as perfectly reproducible.
    """
    mean = sum(measurement["r2"]) / len(measurement["r2"])
    std = measurement.get("r2Std")
    if not std:
        return mean, False
    spread = sum(std) / len(std)
    return mean - spread, True


def total_cost(measurement: Dict, query: Query) -> float:
    """Train once, then pay inference for every observation. Pretraining counts,
    because it is real compute the user spends."""
    pretrain = measurement.get("pretrainSeconds") or 0.0
    return (measurement["trainSeconds"] + pretrain
            + query.nObservations * measurement["inferenceSeconds"])


def literature_advice(query: Query, measured_max_dim: int) -> tuple:
    """Published decision rules for cases this zoo has not measured.

    Returned separately from the ranking and always labelled, so an uncited rule of
    thumb cannot hide inside a number that looks measured.
    """
    advised, notes = None, []

    if query.iidTrials:
        advised = "NLE"
        notes.append(
            "LITERATURE, not measured here. Deistler et al. (arXiv 2508.12939) Table 1 "
            "lists handling of i.i.d. observations as a primary axis, and NLE is the "
            "engine that takes it natively: it learns a per trial likelihood and "
            "multiplies across exchangeable trials. NPE for i.i.d. data needs as many "
            "simulations per parameter set as the largest trial count, plus a "
            "permutation invariant architecture. This zoo has never measured an "
            "i.i.d. trial setting, so the ranking below cannot speak to it.")
        notes.append(
            "The cost is that NLE runs a fresh MCMC chain per subject. At "
            f"{query.nObservations} subjects that multiplies the measured per "
            "observation inference cost by the same factor.")

    if query.nParams > measured_max_dim:
        notes.append(
            f"LITERATURE, not measured here. dim(theta) {query.nParams} is above the "
            f"largest this zoo measured, which is {measured_max_dim}. Miller et al. "
            "(2021) and LtU-ILI Section 2.3 both record that ratio estimators degrade "
            "at high parameter dimensionality without truncation, so NRE entries are "
            "penalised in the ranking below rather than trusted at face value.")

    if query.nObservations > 100:
        notes.append(
            "MEASURED here. Amortization decides at this observation count. NPE "
            "answers a new observation in a forward pass; NLE and NRE run a fresh "
            "MCMC chain each, measured at 493 to 3294 seconds per observation.")
    return advised, notes


def score(entry: Dict, measurement: Dict, query: Query,
          measured_max_dim: int) -> tuple:
    """Return (score, reasons). Higher is better. Every term cites its source."""
    reasons = []

    mean = sum(measurement["r2"]) / len(measurement["r2"])
    accuracy, has_spread = reliable_accuracy(measurement)
    if has_spread:
        reasons.append(
            f"Mean R2 {mean:+.3f} on {measurement['task']} over "
            f"{measurement['nSeeds']} seeds, ranked at {accuracy:+.3f} after "
            f"subtracting one seed standard deviation.")
    else:
        reasons.append(
            f"Mean R2 {mean:+.3f} on {measurement['task']}, single seed, no spread "
            f"available.")

    cost = total_cost(measurement, query)
    reasons.append(
        f"{cost:,.0f} s for {query.nObservations} observation(s): "
        f"{measurement['trainSeconds']:.1f} s training plus "
        f"{measurement['inferenceSeconds']:.1f} s per observation.")
    if not entry["amortized"] and query.nObservations > 1:
        reasons.append(
            f"{entry['engine']} is not amortized, so each of the "
            f"{query.nObservations} observations needs its own MCMC run.")

    # Calibration. Overconfidence is the dangerous direction, so it is penalised and
    # underconfidence is not. Weighted up when the posterior feeds something else,
    # because a downstream analysis inherits the error bars whole.
    sigma = measurement["calibrationSigma"] or 0.0
    weight = 0.30 if query.downstream else 0.10
    calibration_penalty = weight * abs(min(sigma, 0.0))
    reasons.append(
        f"Calibration {measurement['calibrationVerdict']} at {sigma:+.1f} sigma from "
        f"nominal 0.680, coverage {measurement['coverage68']:.3f}"
        + (" (weighted heavily: this feeds a downstream analysis)."
           if query.downstream else "."))

    # Extrapolation in dim(theta), applied only to the engines the literature says
    # degrade there. Log2 so the penalty grows with the gap without exploding.
    dim_gap = query.nParams - len(measurement["params"])
    dim_penalty = 0.0
    if dim_gap > 0 and entry["engine"] in ("NRE", "NLE"):
        dim_penalty = 0.15 * math.log2(1 + dim_gap)
        reasons.append(
            f"Penalised {dim_penalty:.2f} for extrapolating a {entry['engine']} "
            f"from dim(theta) {len(measurement['params'])} to {query.nParams}, which "
            f"Miller et al. (2021) and LtU-ILI Section 2.3 say degrades without "
            f"truncation.")

    return accuracy - calibration_penalty - dim_penalty, reasons


def warnings_for(entry: Dict, measurement: Dict, query: Query,
                 over_budget: bool) -> List[str]:
    """Everything the catalogue knows that should stop a user, stated before they run."""
    out = []
    if over_budget:
        cost = total_cost(measurement, query)
        out.append(
            f"OVER BUDGET: needs {cost:,.0f} s against {query.computeSeconds:,.0f} s "
            f"allowed, {cost / query.computeSeconds:,.1f} times over. Nothing in the "
            f"zoo fits this budget, so the cheapest options are shown anyway.")
    if measurement["hidesParameterDisagreement"]:
        detail = ", ".join(
            f"{p} {c:.3f} {w}" for p, c, w in zip(
                measurement["params"], measurement["coverage68ByParam"],
                measurement["verdictByParam"]))
        out.append(
            f"The task level verdict '{measurement['calibrationVerdict']}' is an "
            f"average over parameters that disagree: {detail}. Read the parameter you "
            f"are actually inferring.")
    if len(measurement["params"]) != query.nParams:
        out.append(
            f"Measured at dim(theta) {len(measurement['params'])}, you asked for "
            f"{query.nParams}. This is extrapolation, and the further apart they are "
            f"the less it transfers.")
    if measurement["nSeeds"] < 2:
        out.append(
            "Measured on a single seed, so there is no spread and no way to tell a "
            "real result from a lucky one.")
    gain = measurement.get("infoGainNats")
    if gain is not None and gain < 0:
        out.append(
            f"WORSE THAN THE PRIOR at the truth: log density "
            f"{measurement['logProbTruth']:.3f} against "
            f"{measurement['priorLogDensity']:.3f} for returning the prior, "
            f"{abs(gain):.2f} nats worse than doing nothing. The posterior mean is "
            f"fine and the density is not, so trust the point estimate and not the "
            f"error bars. Caveat: this is a KDE averaged over evaluation points and it "
            f"is biased against MCMC sampled entries, whose draws are autocorrelated.")
    if measurement["calibrationVerdict"] == "overconfident":
        out.append(
            "Error bars from this entry are too small. Almost every entry in the zoo "
            "is overconfident, so this is a property of the setting, not of this pick.")
    if entry["config"].get("pretrainEpochs"):
        out.append(
            f"Needs {entry['config']['pretrainEpochs']} epochs of embedding "
            f"pretraining. Trained jointly from scratch this collapses unpredictably, "
            f"about two runs in three.")
    if entry["modality"] == "point_cloud":
        out.append(
            "A hand designed correlation function still beats every learned embedding "
            "here on CAMELS, R2 0.870 against 0.250. The learned route is only "
            "competitive on CAMELS-SAM, 0.655 against 0.791.")
    return out


def recommend(zoo: Dict, query: Query, top: int = 5) -> Result:
    """Ranked recommendations, what the budget removed, and what was uninformative."""
    measured_max_dim = max(
        (len(m["params"]) for e in zoo["entries"] for m in e["measurements"]),
        default=1)
    advised, advice = literature_advice(query, measured_max_dim)

    if query.modality not in MODALITIES:
        advice.insert(0, (
            f"This zoo has never measured the modality '{query.modality}'. Every "
            f"entry was measured on a compressed summary vector or a point cloud, so "
            f"there is no ranking to give. A time series or an image needs an "
            f"embedding network, and the choice of embedding is the architectural "
            f"decision, which this catalogue cannot inform."))
        return Result([], [], [], advised, advice, outOfScope=True)

    candidates, excluded, uninformative = [], [], []
    for entry in zoo["entries"]:
        # Admission is the whole point of the catalogue: an entry with no measured
        # calibration cannot be recommended, however good it might be.
        if not entry["admitted"] or entry["modality"] != query.modality:
            continue
        measurement = matching_measurement(entry, query)
        if measurement is None:
            continue

        accuracy, _ = reliable_accuracy(measurement)
        if accuracy <= MIN_USEFUL_R2:
            uninformative.append({
                "key": entry["key"], "r2": round(accuracy, 4),
                "mean": round(sum(measurement["r2"]) / len(measurement["r2"]), 4)})
            continue

        cost = total_cost(measurement, query)
        over = (query.computeSeconds is not None and cost > query.computeSeconds)
        if over:
            excluded.append({"key": entry["key"], "seconds": round(cost, 1),
                             "over": round(cost / query.computeSeconds, 1)})
        candidates.append((entry, measurement, cost, over))

    # Prefer what fits. Fall back to the cheapest over budget options rather than
    # returning nothing, because a user with no affordable choice still needs to know
    # what the cheapest one costs.
    fitting = [c for c in candidates if not c[3]]
    pool = fitting if fitting else sorted(candidates, key=lambda c: c[2])[:top]
    if candidates and not fitting:
        advice.append(
            f"MEASURED. Nothing in the zoo fits {query.computeSeconds:,.0f} s at "
            f"{query.nObservations} observations. The cheapest is "
            f"{pool[0][0]['key']} at {pool[0][2]:,.0f} s. Either raise the budget or "
            f"reduce the observation count.")

    scored = []
    for entry, measurement, cost, over in pool:
        value, reasons = score(entry, measurement, query, measured_max_dim)
        scored.append((value, entry, measurement, cost, over, reasons))
    scored.sort(key=lambda row: -row[0])

    out = []
    for i, (value, entry, measurement, cost, over, reasons) in enumerate(
            scored[:top], start=1):
        out.append(Recommendation(
            rank=i, key=entry["key"], task=measurement["task"],
            r2=measurement["r2"],
            calibrationVerdict=measurement["calibrationVerdict"],
            calibrationSigma=measurement["calibrationSigma"],
            trainSeconds=measurement["trainSeconds"],
            inferenceSeconds=measurement["inferenceSeconds"],
            totalSeconds=round(cost, 1), overBudget=over,
            score=round(value, 4), why=reasons,
            warnings=warnings_for(entry, measurement, query, over)))
    excluded.sort(key=lambda row: -row["seconds"])
    return Result(out, excluded, uninformative, advised, advice, outOfScope=False)


def emit_config(zoo: Dict, key: str, prior_low: List[float],
                prior_high: List[float], device: str = "cpu",
                out_dir: str = "./ili_out", n_points: int = 512) -> str:
    """A runnable ltu-ili yaml for one entry, so a recommendation is executable.

    The prior is the caller's, not the zoo's. Every measurement here used the CAMELS
    or CAMELS-SAM Latin hypercube ranges, and silently reusing those for someone
    else's problem would encode our simulation design into their inference.
    """
    entry = next((e for e in zoo["entries"] if e["key"] == key), None)
    if entry is None:
        raise SystemExit(f"unknown entry {key}")
    config = entry["config"]

    # A heterogeneous ensemble is one net per member, not `repeats` copies of one
    # model. Emitting it as a single net promised three members and built one.
    if config.get("mixture"):
        nets = []
        for model_name, model_args in config["mixture"]:
            member = {"model": model_name}
            member.update(dict(model_args))
            member["signature"] = f"{key}_{model_name}"
            nets.append(member)
    else:
        net = {"model": config["model"]}
        net.update(dict(config["model_args"]))
        if config["repeats"] > 1:
            net["repeats"] = config["repeats"]
        net["signature"] = key
        nets = [net]

    doc = {
        "device": device,
        "out_dir": out_dir,
        "prior": {"module": "ili.utils", "class": "Uniform",
                  "args": {"low": prior_low, "high": prior_high}},
        # backend is required: InferenceRunner.from_config dispatches on it and raises
        # KeyError without it. Caught by loading an emitted config rather than by
        # reading the schema, which is why it is tested in checks/emittedConfig.py.
        "model": {"backend": config["backend"], "name": key,
                  "engine": config["engine"], "nets": nets},
        "train_args": config["train_args"],
    }
    if config["embedding"]:
        # config["embedding"] is the EMBEDDINGS registry key, not the class name, and
        # ili's load_from_config does getattr(module, config["class"]). Emitting the
        # key produced "module has no attribute 'pairwiseGnn'". Resolved through the
        # registry so the two can never drift.
        from ili_kaai.embeddings import EMBEDDINGS
        # n_points has no default and cannot be inferred from the catalogue, because
        # it is a property of the caller's cloud. Emitted at the value the entry was
        # measured on, flagged loudly in the header as something to replace.
        args = {"n_points": n_points, "n_features": 3}
        args.update(dict(config["embedding_args"]))
        doc["embedding_net"] = {
            "module": "ili_kaai.embeddings",
            "class": EMBEDDINGS[config["embedding"]].__name__,
            "args": args}

    first = entry["measurements"][0]
    header = (
        f"# ltu-ili configuration for zoo entry '{key}'.\n"
        f"# Emitted by skill/query.py from ili_kaai/results/zoo.json.\n"
        f"# Calibration on {first['task']}: {first['calibrationVerdict']} "
        f"({first['calibrationSigma']:+.1f} sigma from nominal 0.680).\n"
        f"# REPLACE the prior bounds below with your own. They default to the CAMELS\n"
        f"# Latin hypercube ranges this entry was measured on, which are almost\n"
        f"# certainly not your problem's prior.\n")
    if config["embedding"]:
        header += (f"# This entry needs an embedding, so x must be a point cloud of "
                   f"shape (nPoints, 3).\n"
                   f"# REPLACE embedding_net.args.n_points: it is set to {n_points}, "
                   f"the count this entry was\n"
                   f"# measured on, and it MUST equal the number of points in your "
                   f"clouds or the net will\n"
                   f"# not accept your data.\n")
    if config["pretrainEpochs"]:
        header += (f"# NOTE: measured to need {config['pretrainEpochs']} epochs of "
                   f"embedding pretraining. Joint training from scratch collapses\n"
                   f"# unpredictably, about two runs in three. ltu-ili has no config "
                   f"key for this; see ili_kaai/sweep.py pretrain_embedding.\n")
    return header + yaml.safe_dump(doc, sort_keys=False)


def render(result: Result, query: Query) -> str:
    lines = [f"Problem: {query.modality}, dim(theta) {query.nParams}, "
             f"{query.nObservations} observation(s)"
             + (f", budget {query.computeSeconds:,.0f} s" if query.computeSeconds
                else "")
             + (", feeds a downstream analysis" if query.downstream else "")
             + (", i.i.d. trials" if query.iidTrials else "")]

    if result.advisedEngine:
        lines.append(f"\nEngine advised by the literature: {result.advisedEngine}")
    for note in result.advice:
        lines.append(f"   * {note}")

    if not result.recommendations:
        lines.append("\nNo ranking available.")
        return "\n".join(lines)

    lines.append("\nRanked by measured accuracy and calibration:")
    for r in result.recommendations:
        flag = "  [OVER BUDGET]" if r.overBudget else ""
        lines.append(
            f"\n{r.rank}. {r.key}   score {r.score:+.3f}{flag}\n"
            f"   measured on {r.task}, R2 {[round(v, 3) for v in r.r2]}\n"
            f"   calibration {r.calibrationVerdict} ({r.calibrationSigma:+.1f} sigma)\n"
            f"   compute {r.totalSeconds:,.0f} s total")
        for w in r.why:
            lines.append(f"     - {w}")
        for w in r.warnings:
            lines.append(f"     WARNING {w}")

    if result.uninformative:
        lines.append(f"\nRemoved as uninformative, mean R2 at or below "
                     f"{MIN_USEFUL_R2} ({len(result.uninformative)} entries). A "
                     f"posterior that predicts nothing is wide, so it would otherwise "
                     f"score well on calibration:")
        for u in result.uninformative:
            lines.append(f"   {u['key']:30} mean R2 {u['mean']:+.4f} "
                         f"-> {u['r2']:+.4f} after seed spread")
    if result.excluded:
        lines.append(f"\nOver the {query.computeSeconds:,.0f} s budget "
                     f"({len(result.excluded)} entries), not hidden:")
        for x in result.excluded:
            lines.append(f"   {x['key']:30} {x['seconds']:>14,.0f} s  "
                         f"{x['over']:,.1f}x over")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modality", type=str, default="summary_vector",
                   choices=list(MODALITIES))
    p.add_argument("--n-params", type=int, default=2)
    p.add_argument("--n-observations", type=int, default=1)
    p.add_argument("--compute-seconds", type=float, default=DEFAULT_BUDGET_SECONDS,
                   help="training plus inference budget in seconds; default 7200, "
                        "the two GPU hours the brief names as its success criterion")
    p.add_argument("--downstream", action="store_true",
                   help="the posterior feeds a downstream analysis, so calibration "
                        "matters more than accuracy")
    p.add_argument("--iid-trials", action="store_true",
                   help="the observations are exchangeable trials from one subject")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--emit-config", type=str, default=None,
                   help="entry key to emit a runnable ltu-ili yaml for")
    p.add_argument("--prior-low", type=float, nargs="+", default=None)
    p.add_argument("--prior-high", type=float, nargs="+", default=None)
    p.add_argument("--out", type=str, default=None, help="write the yaml here")
    args = p.parse_args()

    if args.n_params < 1:
        raise SystemExit("--n-params must be at least 1")
    if args.n_observations < 1:
        raise SystemExit("--n-observations must be at least 1")

    zoo = load_zoo()
    query = Query(modality=args.modality, nParams=args.n_params,
                  nObservations=args.n_observations,
                  computeSeconds=args.compute_seconds, downstream=args.downstream,
                  iidTrials=args.iid_trials)
    result = recommend(zoo, query, top=args.top)
    print(render(result, query))

    if args.emit_config or result.recommendations:
        key = args.emit_config or result.recommendations[0].key
        low = args.prior_low or [0.1, 0.6][:args.n_params]
        high = args.prior_high or [0.5, 1.0][:args.n_params]
        if len(low) != args.n_params or len(high) != args.n_params:
            raise SystemExit("--prior-low and --prior-high need one value per parameter")
        text = emit_config(zoo, key, low, high)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\n  wrote {args.out}")
        else:
            print("\n" + "-" * 70 + f"\nrunnable config for {key}\n" + "-" * 70)
            print(text)


if __name__ == "__main__":
    main()
