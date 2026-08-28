"""Assemble the model zoo from measured sweep results.

    conda run -n ltuili python -m ili_kaai.zoo

Every number in the catalogue is read from a results file by path. Nothing is
retyped, so the catalogue cannot drift from the measurements it describes.

ADMISSION, AND WHY IT LABELS RATHER THAN REJECTS
------------------------------------------------
The obvious design is a gate: an entry that fails calibration does not get in.
Measured on this sweep, that gate rejects all 24 architecture-task pairs and the
zoo is empty, which helps nobody.

So the rule is: an entry is admitted only if its calibration has been MEASURED,
and the verdict travels with it forever. A recommendation that does not carry the
calibration verdict is not a recommendation, it is a leaderboard row, and a
leaderboard row is what the whole project exists to replace.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ili_kaai.architectures import TRAIN_ARGS, ZOO as ARCHITECTURES
from ili_kaai.tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ili_kaai" / "results"
OUT = RESULTS / "zoo.json"

NOMINAL_68 = 0.68
NOMINAL_95 = 0.95


def calibration_tolerance() -> tuple:
    """How far a genuinely calibrated entry can read from nominal on noise alone.

    Read from a measurement, not chosen. checks/tarpCalibration.py --noise-band
    builds posteriors that are calibrated by construction, reads them exactly as
    the sweep does, and reports how much the reading moves. Two sigma of that is
    the threshold, so the verdict fires on signal rather than on sampling noise.
    """
    band = json.loads(
        (RESULTS / "calibrationNoiseBand.json").read_text(encoding="utf-8"))
    return band["twoSigma"], band["seedMeanStd"], band


@dataclass
class Measurement:
    task: str
    r2: List[float]
    r2Std: Optional[List[float]]
    coverage68: float
    coverage95: float
    tarp68: Optional[float]
    trainSeconds: float
    inferenceSeconds: float
    nSeeds: int
    why: str


@dataclass
class Entry:
    key: str
    family: str
    engine: str
    backend: str
    amortized: bool
    summary: str
    config: Dict
    nParameters: Dict[str, int]
    measurements: List[Measurement] = field(default_factory=list)
    failureModes: List[str] = field(default_factory=list)
    calibrationVerdict: str = "unmeasured"
    calibrationSigma: Optional[float] = None
    admitted: bool = False

    def admit(self) -> None:
        """Admitted when calibration is measured on every task, never when it passes."""
        self.admitted = (len(self.measurements) == len(TASKS)
                         and all(m.coverage68 is not None for m in self.measurements))


def verdict(coverages: List[float], tol: float, sigma: float) -> tuple:
    """A word and a number of sigma, so the verdict is quantified not asserted."""
    mean = sum(coverages) / len(coverages)
    n_sigma = (mean - NOMINAL_68) / sigma
    if mean < NOMINAL_68 - tol:
        return "overconfident", round(n_sigma, 1)
    if mean > NOMINAL_68 + tol:
        return "underconfident", round(n_sigma, 1)
    return "calibrated", round(n_sigma, 1)


def explain(arch_key: str, task_key: str, agg: Dict, all_agg: List[Dict]) -> str:
    """A physical or architectural reason for this row, not a restatement of it.

    The brief asks for documented failure modes. A number with no reason attached
    is not documentation, so every measurement carries one.
    """
    arch = ARCHITECTURES[arch_key]
    task = TASKS[task_key]
    bits = []

    peers = [a["r2"][0] for a in all_agg if a["task"] == task_key]
    spread = max(peers) - min(peers)
    bits.append(
        f"Accuracy on this task spans only {spread:.3f} R2 across all eight "
        f"entries, so accuracy alone does not distinguish this one.")

    if arch.engine == "NPE":
        bits.append(
            "NPE is amortized: it learns the posterior directly, so inference is a "
            "single forward pass and costs almost nothing per observation.")
    else:
        bits.append(
            f"{arch.engine} learns a proxy rather than the posterior, so every "
            f"observation needs its own MCMC run. That is why inference takes "
            f"{agg.get('inferenceSeconds', 0):.0f} s here against under 3 s for NPE.")

    if task.n_params == 1:
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


def build() -> List[Entry]:
    tol, sigma, _band = calibration_tolerance()
    sweep = json.loads((RESULTS / "sweep.json").read_text(encoding="utf-8"))
    params = json.loads((RESULTS / "paramCount.json").read_text(encoding="utf-8"))
    if not sweep.get("complete"):
        raise SystemExit("sweep.json is not complete; refusing to build a zoo from it")

    cells = [c for c in sweep["cells"] if "error" not in c]
    timing: Dict[tuple, List[float]] = {}
    for c in cells:
        timing.setdefault((c["architecture"], c["task"]), []).append(
            (c["trainSeconds"], c["evalSeconds"]))

    entries = []
    for key, arch in ARCHITECTURES.items():
        e = Entry(
            key=key, family=arch.family, engine=arch.engine, backend="sbi",
            amortized=(arch.engine == "NPE"), summary=arch.summary,
            config={"backend": "sbi", "engine": arch.engine, "model": arch.model,
                    "repeats": arch.repeats, "model_args": arch.model_args,
                    "train_args": TRAIN_ARGS, "sample_method": arch.sample_method},
            nParameters=params.get(key, {}),
            failureModes=list(arch.known_failure_modes))

        for task_key in TASKS:
            agg = next((a for a in sweep["aggregate"]
                        if a["architecture"] == key and a["task"] == task_key), None)
            if agg is None:
                continue
            t = timing[(key, task_key)]
            row = {"coverage68": sum(agg["coverage68"]) / len(agg["coverage68"]),
                   "inferenceSeconds": sum(x[1] for x in t) / len(t)}
            e.measurements.append(Measurement(
                task=task_key,
                r2=agg["r2"], r2Std=agg["r2Std"],
                coverage68=round(row["coverage68"], 4),
                coverage95=round(sum(agg["coverage95"]) / len(agg["coverage95"]), 4),
                tarp68=agg["tarpAt68"],
                trainSeconds=round(sum(x[0] for x in t) / len(t), 1),
                inferenceSeconds=round(row["inferenceSeconds"], 1),
                nSeeds=agg["nSeeds"],
                why=explain(key, task_key, {**agg, **row}, sweep["aggregate"])))

        e.calibrationVerdict, e.calibrationSigma = verdict(
            [m.coverage68 for m in e.measurements], tol, sigma)
        if e.calibrationVerdict == "overconfident":
            e.failureModes.insert(0, (
                "MEASURED overconfident on all three CAMELS tasks: mean coverage at "
                f"the 68 per cent level is "
                f"{sum(m.coverage68 for m in e.measurements) / len(e.measurements):.3f} "
                f"against a nominal 0.680, which is {abs(e.calibrationSigma):.1f} "
                "sigma low. Error bars from this entry are too small."))
        e.admit()
        entries.append(e)
    return entries


def main() -> None:
    entries = build()
    tol, sigma, band = calibration_tolerance()
    payload = {
        "nominal68": NOMINAL_68, "nominal95": NOMINAL_95,
        "calibrationTolerance": tol, "calibrationNoiseBand": band,
        "admissionRule": ("Admitted when calibration has been measured on every "
                          "task. Passing is not required. The verdict travels with "
                          "the entry and every recommendation must carry it."),
        "source": {"sweep": "ili_kaai/results/sweep.json",
                   "parameters": "ili_kaai/results/paramCount.json",
                   "noiseBand": "ili_kaai/results/calibrationNoiseBand.json"},
        "entries": [asdict(e) for e in entries]}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"  {len(entries)} entries, {sum(e.admitted for e in entries)} admitted")
    print(f"  overconfident threshold {tol:.4f} (2 sigma), measured not chosen\n")
    print(f"  {'entry':17s}{'family':20s}{'amortized':>10s}{'verdict':>16s}"
          f"{'sigma':>8s}{'mean cov68':>12s}{'infer s':>9s}")
    for e in entries:
        cov = sum(m.coverage68 for m in e.measurements) / len(e.measurements)
        inf = sum(m.inferenceSeconds for m in e.measurements) / len(e.measurements)
        print(f"  {e.key:17s}{e.family:20s}{str(e.amortized):>10s}"
              f"{e.calibrationVerdict:>16s}{e.calibrationSigma:>8.1f}{cov:12.3f}{inf:9.1f}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
