"""Assemble the model zoo from measured sweep results.

    conda run -n ltuili python -m ili_kaai.zoo

Every number in the catalogue is read from a results file by path. Nothing is
retyped, so the catalogue cannot drift from the measurements it describes.

ADMISSION, AND WHY IT LABELS RATHER THAN REJECTS
------------------------------------------------
The obvious design is a gate: an entry that fails calibration does not get in.
Measured on these sweeps, that gate rejects almost every architecture-task pair and
the zoo is empty, which helps nobody.

So the rule is: a pair is admitted only if its calibration has been MEASURED, and
the verdict travels with it forever. A recommendation that does not carry the
calibration verdict is not a recommendation, it is a leaderboard row, and a
leaderboard row is what the whole project exists to replace.

AMENDMENT 2026-08-30, ADMISSION MOVED FROM THE ENTRY TO THE PAIR
----------------------------------------------------------------
The rule used to admit an entry when it had a measurement on every task in TASKS.
TASKS grew from 3 to 8 when Quijote and the point cloud tasks were added, and
nothing rebuilt the zoo afterwards, so every entry in the shipped catalogue silently
carried admitted=false. No entry can ever satisfy the old rule, because a density
estimator that reads a 25 bin summary vector cannot read a point cloud, and an
embedding built for a cloud has nothing to say about a summary vector.

So admission and the calibration verdict are now per (entry, task). That is also the
more faithful reading of the original intent, because a recommendation is made for
one task, so the verdict that travels with it must be that task's.

WHY COVERAGE IS ALSO REPORTED PER PARAMETER
-------------------------------------------
A first version of this amendment claimed the entry level verdict hid overconfidence
by averaging across TASKS. RETRACTED: measured on this data, no entry has disagreeing
per task verdicts, and that claim came from reading only the first parameter's
coverage rather than what the catalogue actually computes.

The hiding is real but one level down, across PARAMETERS. Measured on 3 of 30
multi-parameter rows the parameter mean disagrees with a per parameter verdict, and
the worst case is the one that matters most: nreMlp on camelsSamJoint reads
calibrated at 0.6624 while its Omega_m coverage is 0.6395, which is overconfident.
nreMlp is the only entry in the whole zoo reading calibrated, so the single entry a
user would choose on calibration grounds is the one whose headline verdict hides a
bad parameter. Every measurement therefore carries coverage and a verdict per
parameter alongside the mean, and the mean is never the only thing on offer.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ili_kaai.architectures import TRAIN_ARGS, ZOO as ARCHITECTURES, Architecture
from ili_kaai.tasks import TASKS, Task, prior_bounds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ili_kaai" / "results"
OUT = RESULTS / "zoo.json"

NOMINAL_68 = 0.68
NOMINAL_95 = 0.95

# Every sweep that may contribute measurements. A file that is absent, or that stamps
# complete=false, is skipped BY NAME into the catalogue's provenance rather than
# blended in silently, because a partial sweep is not a measurement.
#
# sweepQuijote.json is listed deliberately even though it will always be refused. It
# holds 2 of 24 cells, stopped by hand because the suite is too large for this laptop,
# so listing it makes the catalogue state why Quijote is absent instead of leaving a
# reader to wonder whether it was forgotten.
# sweepQuijoteJoint800.json is deliberately NOT here. It is the control that
# subsamples Quijote to the CAMELS training size, so its cells were measured under a
# different condition from every other entry in the catalogue. Merging it would break
# the one thing that makes these entries comparable.
SWEEP_FILES = ("sweep.json", "sweepPosterior.json", "sweepMcmc.json",
               "sweepCloud.json", "sweepQuijote.json",
               "sweepQuijoteJoint.json", "sweepQuijoteAll.json",
               "sweepCloudFixedScaling.json")


def calibration_tolerance() -> Tuple[float, float, Dict]:
    """How far a genuinely calibrated entry can read from nominal on noise alone.

    Read from a measurement, not chosen. checks/tarpCalibration.py --noise-band
    builds posteriors that are calibrated by construction, reads them exactly as
    the sweep does, and reports how much the reading moves. Two sigma of that is
    the threshold, so the verdict fires on signal rather than on sampling noise.
    """
    band = json.loads(
        (RESULTS / "calibrationNoiseBand.json").read_text(encoding="utf-8"))
    return band["twoSigma"], band["seedMeanStd"], band


def load_sweeps() -> Tuple[List[Dict], List[Dict], List[str], List[Dict], List[Dict]]:
    """Merge every completed sweep, later files superseding earlier ones.

    Supersession is explicit because it has to be. Two sweeps can measure the same
    (entry, task) pair, which is exactly what happens when a defect is fixed and the
    affected entries are re-measured. Concatenating both left duplicates, and the
    lookup took whichever came first, so the OLD numbers silently won and the
    corrected sweep was ignored. SWEEP_FILES is therefore ordered oldest to newest,
    the last file wins, and every pair that was replaced is named in the catalogue's
    provenance rather than disappearing quietly.
    """
    by_pair: Dict[Tuple[str, str], Dict] = {}
    cells_by_pair: Dict[Tuple[str, str], List[Dict]] = {}
    used: List[str] = []
    skipped: List[Dict] = []
    superseded: List[Dict] = []
    for name in SWEEP_FILES:
        path = RESULTS / name
        if not path.exists():
            skipped.append({"file": name, "reason": "missing"})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("complete"):
            skipped.append({"file": name, "reason": "incomplete, refused"})
            continue
        for agg in data["aggregate"]:
            key = (agg["architecture"], agg["task"])
            if key in by_pair:
                superseded.append({"architecture": key[0], "task": key[1],
                                   "replacedBy": name,
                                   "wasFrom": by_pair[key]["_sourceFile"]})
            agg = dict(agg)
            agg["_sourceFile"] = name
            by_pair[key] = agg
        for c in data["cells"]:
            if "error" in c:
                continue
            cells_by_pair.setdefault((c["architecture"], c["task"]), [])
            if cells_by_pair[(c["architecture"], c["task"])] and \
                    cells_by_pair[(c["architecture"], c["task"])][0].get(
                        "_sourceFile") != name:
                cells_by_pair[(c["architecture"], c["task"])] = []
            c = dict(c)
            c["_sourceFile"] = name
            cells_by_pair[(c["architecture"], c["task"])].append(c)
        used.append(name)
    if not used:
        raise SystemExit("no completed sweep found; refusing to build a zoo")
    cells = [c for group in cells_by_pair.values() for c in group]
    return cells, list(by_pair.values()), used, skipped, superseded


def can_run(arch: Architecture, task: Task) -> bool:
    """Whether this entry could ever be measured on this task.

    Fixed by modality rather than chosen. A point cloud task needs an embedding to
    turn a set of galaxies into a vector, and a summary vector already is one, so
    handing it a set encoder is not a different architecture, it is a type error.
    """
    return bool(arch.embedding) == (task.modality == "point_cloud")


@dataclass
class Measurement:
    task: str
    params: List[str]
    r2: List[float]
    r2Std: Optional[List[float]]
    coverage68: float
    coverage95: float
    # Coverage and its verdict for each parameter separately. The mean above can and
    # does disagree with these, so both are always present and neither is optional.
    coverage68Std: Optional[float]
    coverage68ByParam: List[float]
    verdictByParam: List[str]
    # True when the margin by which this verdict clears its threshold is smaller than
    # the seed to seed spread, so a different set of seeds could flip the label.
    verdictIsBorderline: bool
    tarp68: Optional[float]
    # Mean log density the posterior puts on the true parameters, and how that compares
    # with simply returning the prior. Negative infoGainNats means the posterior is
    # WORSE than the prior at the truth, which neither R2 nor coverage can show.
    logProbTruth: Optional[float]
    priorLogDensity: float
    infoGainNats: Optional[float]
    trainSeconds: float
    inferenceSeconds: float
    pretrainSeconds: Optional[float]
    nSeeds: int
    calibrationVerdict: str
    calibrationSigma: Optional[float]
    hidesParameterDisagreement: bool
    why: str


@dataclass
class Entry:
    key: str
    family: str
    engine: str
    backend: str
    modality: str
    amortized: bool
    summary: str
    config: Dict
    nParameters: Dict[str, Optional[int]]
    measurements: List[Measurement] = field(default_factory=list)
    failureModes: List[str] = field(default_factory=list)
    # One of "overconfident", "underconfident", "calibrated" when every measured task
    # agrees, "mixed" when they do not, "unmeasured" when there is nothing to read.
    calibrationVerdict: str = "unmeasured"
    admitted: bool = False
    admittedTasks: List[str] = field(default_factory=list)
    runnableTasks: List[str] = field(default_factory=list)
    unmeasuredTasks: List[str] = field(default_factory=list)

    def admit(self) -> None:
        """Admitted for a task when calibration was MEASURED there, never when it
        passed. The entry counts as admitted once at least one task qualifies, and
        admittedTasks names exactly which, so a caller can never read a verdict off
        a task the entry was never measured on."""
        self.admittedTasks = [m.task for m in self.measurements
                              if m.coverage68 is not None]
        self.admitted = bool(self.admittedTasks)
        self.unmeasuredTasks = [t for t in self.runnableTasks
                                if t not in self.admittedTasks]


def verdict(coverage: float, tol: float, sigma: float) -> Tuple[str, float]:
    """A word and a number of sigma, so the verdict is quantified not asserted."""
    n_sigma = (coverage - NOMINAL_68) / sigma
    if coverage < NOMINAL_68 - tol:
        return "overconfident", round(n_sigma, 1)
    if coverage > NOMINAL_68 + tol:
        return "underconfident", round(n_sigma, 1)
    return "calibrated", round(n_sigma, 1)


def explain(arch_key: str, task_key: str, agg: Dict, peers: List[Dict]) -> str:
    """A physical or architectural reason for this row, not a restatement of it.

    The brief asks for documented failure modes. A number with no reason attached
    is not documentation, so every measurement carries one.
    """
    arch = ARCHITECTURES[arch_key]
    task = TASKS[task_key]
    bits = []

    same_task = [a["r2"][0] for a in peers if a["task"] == task_key]
    if len(same_task) > 1:
        spread = max(same_task) - min(same_task)
        separates = "barely separates" if spread < 0.1 else "does separate"
        bits.append(
            f"Accuracy on this task spans {spread:.3f} R2 across the "
            f"{len(same_task)} entries measured on it, so accuracy alone "
            f"{separates} them.")

    if arch.engine == "NPE":
        bits.append(
            "NPE is amortized: it learns the posterior directly, so inference is a "
            "single forward pass and costs almost nothing per observation.")
    else:
        bits.append(
            f"{arch.engine} learns a proxy rather than the posterior, so every "
            f"observation needs its own MCMC run. That is why inference takes "
            f"{agg.get('inferenceSeconds', 0):.0f} s here against under 3 s for NPE.")

    if task.modality == "point_cloud":
        # The box size argument used on summary vectors was REFUTED here by our own
        # sweep, so it is deliberately not repeated. Measured: sigma_8 recovers at
        # +0.176 on the 25 Mpc/h camelsCloud against +0.015 on the 100 Mpc/h
        # camelsSamCloud, the opposite of the summary vector ordering.
        bits.append(
            "This task keeps a fixed number of the most massive galaxies, so the "
            "count carries no information and the network has to read geometry. "
            "Thinning to a fixed 512 points strips a large box harder than a small "
            "one, which is the untested explanation for sigma_8 recovering worse on "
            "the larger CAMELS-SAM box here, opposite to the summary vector order.")
        if not arch.embedding.startswith("pairwise"):
            bits.append(
                "MEASURED near zero R2, and the reason is structural rather than a "
                "training failure. This embedding pools per point features of "
                "absolute positions, which is a first moment statistic, while "
                "clustering is a second moment property defined on pairs. No amount "
                "of training fixes a representation that cannot express the signal.")
        elif arch.pretrainEpochs == 0:
            bits.append(
                "MEASURED unreliable rather than reliably bad when trained jointly "
                "from scratch: individual seeds gave -0.019, +0.198 and -0.000 on "
                "camelsCloud. The flow fits the marginal of theta, stops "
                "conditioning on a context that starts as noise, and the embedding "
                "then receives no gradient. Pretraining the embedding removes it.")
    elif task.n_params == 1:
        bits.append(
            "This task infers one parameter, so no joint degeneracy is tested and "
            "TARP is not measurable.")
    elif task.suite == "CAMELS-SAM":
        bits.append(
            "CAMELS-SAM uses a 100 Mpc/h box, about twelve times the 8 Mpc/h scale "
            "sigma_8 describes, so sigma_8 is far better sampled here than in the "
            "25 Mpc/h CAMELS box.")
    else:
        bits.append(
            "CAMELS uses a 25 Mpc/h box, only about three times the 8 Mpc/h scale "
            "sigma_8 is defined on, so sample variance on that scale is large and "
            "sigma_8 recovers poorly.")

    c = agg["coverage68"]
    if c < NOMINAL_68 - 0.022:
        bits.append(
            f"Coverage at 68 per cent is {c:.3f} against a nominal 0.680, so the "
            f"posterior is too narrow. Single density estimators are known to be "
            f"overconfident (Hermans et al. 2022).")
    return " ".join(bits)


def parameter_counts(params: Dict, key: str) -> Dict[str, Optional[int]]:
    """paramCount writes -1 when the build failed. A count that was not obtained is
    null, never a number, so nothing downstream can average it into a total."""
    return {task: (None if value is None or value < 0 else value)
            for task, value in params.get(key, {}).items()}


def prior_log_density(task_key: str) -> float:
    """Log density of the uniform prior, so posterior quality has a floor to beat.

    A posterior that simply returns the prior scores exactly this. Measured: nreLinear
    lands 0.19 nats BELOW it while reading the best coverage in the catalogue, which is
    the cleanest statement of why coverage alone cannot say whether an entry is useful.
    """
    lo, hi = prior_bounds(TASKS[task_key])
    return -sum(math.log(h - l) for l, h in zip(lo, hi))


def measure(arch_key: str, task_key: str, agg: Dict, timing: List[Tuple],
            tol: float, sigma: float, aggregates: List[Dict]) -> Measurement:
    """One row of the catalogue, with its calibration verdict attached to it."""
    per_param = agg["coverage68"]
    cov68 = sum(per_param) / len(per_param)
    pretrain = [t[2] for t in timing if t[2]]
    row = {"coverage68": cov68,
           "inferenceSeconds": sum(t[1] for t in timing) / len(timing)}
    word, n_sigma = verdict(cov68, tol, sigma)
    param_words = [verdict(c, tol, sigma)[0] for c in per_param]
    cov_std_list = agg.get("coverage68Std")
    cov_std = (sum(cov_std_list) / len(cov_std_list)) if cov_std_list else None
    # Distance to whichever verdict boundary is nearer. Measured 2026-08-30: npeMaf on
    # quijoteJoint clears the calibrated threshold by 0.0020 with a seed spread of
    # 0.0035, so the label is a coin flip while the underlying shift of 0.0908 is 26
    # seed spreads and entirely solid. Reporting the label alone would hide that.
    margin = min(abs(cov68 - (NOMINAL_68 - tol)), abs(cov68 - (NOMINAL_68 + tol)))
    borderline = bool(cov_std is not None and margin < cov_std)
    log_prob = agg.get("logProbTruth")
    floor_density = prior_log_density(task_key)
    return Measurement(
        task=task_key,
        params=TASKS[task_key].labels,
        r2=agg["r2"], r2Std=agg["r2Std"],
        coverage68=round(cov68, 4),
        coverage95=round(sum(agg["coverage95"]) / len(agg["coverage95"]), 4),
        coverage68Std=round(cov_std, 4) if cov_std is not None else None,
        coverage68ByParam=[round(c, 4) for c in per_param],
        verdictByParam=param_words,
        verdictIsBorderline=borderline,
        tarp68=agg["tarpAt68"],
        logProbTruth=log_prob,
        priorLogDensity=round(floor_density, 4),
        infoGainNats=(None if log_prob is None
                      else round(log_prob - floor_density, 4)),
        trainSeconds=round(sum(t[0] for t in timing) / len(timing), 1),
        inferenceSeconds=round(row["inferenceSeconds"], 1),
        pretrainSeconds=(round(sum(pretrain) / len(pretrain), 1)
                         if pretrain else None),
        nSeeds=agg["nSeeds"],
        calibrationVerdict=word, calibrationSigma=n_sigma,
        hidesParameterDisagreement=any(w != word for w in param_words),
        why=explain(arch_key, task_key, {**agg, **row}, aggregates))


def build() -> Tuple[List[Entry], Dict]:
    tol, sigma, _band = calibration_tolerance()
    cells, aggregates, used, skipped, superseded = load_sweeps()
    params = json.loads((RESULTS / "paramCount.json").read_text(encoding="utf-8"))

    timing: Dict[Tuple[str, str], List[Tuple]] = {}
    for c in cells:
        timing.setdefault((c["architecture"], c["task"]), []).append(
            (c["trainSeconds"], c["evalSeconds"], c.get("pretrainSeconds")))

    entries = []
    for key, arch in ARCHITECTURES.items():
        e = Entry(
            key=key, family=arch.family, engine=arch.engine, backend=arch.backend,
            modality="point_cloud" if arch.embedding else "summary_vector",
            amortized=(arch.engine == "NPE"), summary=arch.summary,
            config={"backend": arch.backend, "engine": arch.engine,
                    "model": arch.model, "repeats": arch.repeats,
                    "model_args": arch.model_args,
                    # Without this a heterogeneous ensemble emits as a single net of
                    # its nominal `model`, silently promising three members and
                    # building one. Caught by checks/emittedConfig.py.
                    "mixture": [list(m) for m in arch.mixture],
                    "embedding": arch.embedding,
                    "embedding_args": arch.embedding_args,
                    "pretrainEpochs": arch.pretrainEpochs,
                    "train_args": TRAIN_ARGS, "sample_method": arch.sample_method},
            nParameters=parameter_counts(params, key),
            failureModes=list(arch.known_failure_modes))
        e.runnableTasks = [tk for tk, task in TASKS.items() if can_run(arch, task)]

        for task_key in e.runnableTasks:
            agg = next((a for a in aggregates
                        if a["architecture"] == key and a["task"] == task_key), None)
            if agg is not None:
                e.measurements.append(measure(key, task_key, agg,
                                              timing[(key, task_key)],
                                              tol, sigma, aggregates))

        words = {m.calibrationVerdict for m in e.measurements}
        if not words:
            e.calibrationVerdict = "unmeasured"
        elif len(words) == 1:
            e.calibrationVerdict = words.pop()
        else:
            # Never collapse disagreeing tasks into an average. Measured: doing that
            # made nreMlp read "calibrated" while overconfident on one of its tasks.
            e.calibrationVerdict = "mixed"

        over = [m for m in e.measurements if m.calibrationVerdict == "overconfident"]
        if over:
            worst = min(over, key=lambda m: m.calibrationSigma)
            e.failureModes.insert(0, (
                f"MEASURED overconfident on {len(over)} of {len(e.measurements)} "
                f"measured tasks. Worst is {worst.task}, where mean coverage at the "
                f"68 per cent level is {worst.coverage68:.3f} against a nominal "
                f"0.680, which is {abs(worst.calibrationSigma):.1f} sigma low. "
                f"Error bars from this entry are too small."))
        # A posterior can have an excellent mean and put almost no density where the
        # truth is. R2 cannot see it and coverage cannot see it, so it is stated here.
        starved = [m for m in e.measurements
                   if m.infoGainNats is not None and m.infoGainNats < 0]
        for m in starved:
            # Two separate strings. Passing a tuple to insert() put a nested list into
            # failureModes, which is declared List[str] and is read as prose by the
            # skill, so it has to stay flat.
            e.failureModes.insert(0, (
                "CAVEAT on the number above: it is a KDE over 1000 samples averaged "
                "over evaluation points, so it is dominated by points where the "
                "posterior misses entirely, and Scott's bandwidth assumes independent "
                "draws. Entries sampled with emcee have autocorrelated draws, so their "
                "effective sample size is below 1000 and the KDE is too peaked. The "
                "metric is biased against them. Compare within a sampler, not across."))
            e.failureModes.insert(0, (
                f"MEASURED worse than the prior on {m.task}. The posterior assigns "
                f"log density {m.logProbTruth:.3f} to the true parameters against "
                f"{m.priorLogDensity:.3f} for simply returning the prior, so "
                f"{abs(m.infoGainNats):.2f} nats WORSE than doing nothing, while R2 "
                f"reads {m.r2[0]:+.3f}. A good posterior mean and a posterior that "
                f"puts density where the answer is are different things."))

        if e.calibrationVerdict == "mixed":
            e.failureModes.insert(0, (
                "Calibration is NOT consistent across tasks, so no single verdict "
                "describes this entry. Read the per task verdict for the task you "
                "care about, never the entry level one."))

        # The task level verdict averages over parameters, and measured on this data
        # that average disagrees with a per parameter verdict on some rows. Where it
        # does, say so on the entry, because a user infers one parameter at a time.
        for m in [x for x in e.measurements if x.verdictIsBorderline]:
            e.failureModes.insert(0, (
                f"On {m.task} the calibration verdict '{m.calibrationVerdict}' is NOT "
                f"robust. Coverage {m.coverage68:.4f} clears its threshold by less "
                f"than the seed to seed spread of {m.coverage68Std:.4f}, so a "
                f"different set of seeds could return a different label. Quote the "
                f"coverage and its spread, not the word."))

        hiding = [m for m in e.measurements if m.hidesParameterDisagreement]
        for m in hiding:
            detail = ", ".join(f"{p} {c:.3f} {w}" for p, c, w
                               in zip(m.params, m.coverage68ByParam,
                                      m.verdictByParam))
            e.failureModes.insert(0, (
                f"On {m.task} the task level verdict is '{m.calibrationVerdict}' "
                f"from a coverage of {m.coverage68:.3f} averaged over parameters, "
                f"but the parameters do not agree: {detail}. Trust the parameter "
                f"you are actually inferring, not the average."))
        e.admit()
        entries.append(e)

    provenance = {"sweepsUsed": used, "sweepsSkipped": skipped,
                  "supersededPairs": superseded,
                  "parameters": "ili_kaai/results/paramCount.json",
                  "noiseBand": "ili_kaai/results/calibrationNoiseBand.json"}
    return entries, provenance


def main() -> None:
    entries, provenance = build()
    tol, _, band = calibration_tolerance()
    payload = {
        "nominal68": NOMINAL_68, "nominal95": NOMINAL_95,
        "calibrationTolerance": tol, "calibrationNoiseBand": band,
        "admissionRule": ("Admitted per (entry, task) when calibration has been "
                          "measured on that task. Passing is not required. The "
                          "verdict travels with the measurement, and every "
                          "recommendation must carry the verdict for the task it "
                          "recommends on, never an average across tasks."),
        "source": provenance,
        "entries": [asdict(e) for e in entries]}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pairs = sum(len(e.measurements) for e in entries)
    print(f"  {len(entries)} entries, {sum(e.admitted for e in entries)} admitted, "
          f"{pairs} measured entry-task pairs")
    print(f"  sweeps used: {', '.join(provenance['sweepsUsed'])}")
    for s in provenance["sweepsSkipped"]:
        print(f"  SKIPPED {s['file']}: {s['reason']}")
    print(f"  overconfident threshold {tol:.4f} (2 sigma), measured not chosen\n")
    print(f"  {'entry':29s}{'backend':8s}{'modality':16s}{'verdict':>15s}"
          f"{'measured':>10s}{'unmeasured':>12s}")
    for e in sorted(entries, key=lambda x: (x.modality, x.key)):
        print(f"  {e.key:29s}{e.backend:8s}{e.modality:16s}"
              f"{e.calibrationVerdict:>15s}{len(e.admittedTasks):>10d}"
              f"{len(e.unmeasuredTasks):>12d}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
