"""Rank zoo entries for a described problem.

    python -m zoo.recommend "I have galaxy point clouds and need posteriors on
                             two cosmological parameters, a few GPU hours"

WHY GATING COMES BEFORE RANKING
-------------------------------
Two mismatches are category errors, not ranking mistakes, so they filter before
any score is compared:

  output kind   A point estimate and a posterior are not interchangeable. A
                cosmologist who needs an error bar is not helped by the model
                with the best R2.
  role          Encoders and inference heads compose rather than compete. Every
                posterior estimator needs something to turn raw data into a
                summary first, so ranking an encoder against a flow is
                meaningless.

WHY LEAKAGE IS A RANKING TERM AND NOT A FILTER
----------------------------------------------
Entries that leak the element count are kept, because on a dataset where the
count carries no information they are perfectly good. They are demoted and
labelled instead, so a user is never handed a number inflated by an artefact
without being told.

Measured example this exists to prevent: on CAMELS, sum pooling scores 0.8002 on
Omega_m and mean pooling scores 0.6654, identical architecture. On CAMELS-SAM,
where the count is fixed, the same swap is worth -0.0026. Ranking those two on
score alone would recommend the artefact.
"""

import argparse
import json
import re

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from zoo.registry import build
from zoo.schema import Entry, LeakStatus, Modality, OutputKind, Role

LEAK_PENALTY = 0.25        # demotion when an entry can see a count that varies
MISCALIBRATION_PENALTY = 0.20   # demotion when a posterior entry is overconfident


@dataclass
class Problem:
    """A parsed problem description."""
    modality: Optional[Modality] = None
    output: Optional[OutputKind] = None
    role: Optional[Role] = None
    count_varies: Optional[bool] = None
    minutes_budget: Optional[float] = None
    task: str = "camels_omega_m"
    prefer_small: Optional[bool] = None
    raw: str = ""


PATTERNS: List[Tuple[str, str, object]] = [
    (r"point cloud|galaxy position|halo position|particle position|3d position",
     "modality", Modality.POINT_CLOUD),
    (r"\bgraph\b|radius graph|neighbou?r graph", "modality", Modality.GRAPH),
    (r"summary statistic|correlation function|power spectrum|feature vector",
     "modality", Modality.SUMMARY_VECTOR),
    (r"merger tree|\btree\b", "modality", Modality.TREE),
    (r"density field|\bfield\b|\bgrid\b|\bimage\b", "modality", Modality.FIELD),
    (r"posterior|uncertaint|error bar|credible|calibrat|sbi|"
     r"simulation.based inference|likelihood.free", "output", OutputKind.POSTERIOR),
    (r"point estimate|regress|predict the value|best guess|r2|r squared",
     "output", OutputKind.POINT_ESTIMATE),
    (r"embedding|summary network|encoder|representation", "output", OutputKind.EMBEDDING),
    (r"encoder|summary network|feature extractor", "role", Role.ENCODER),
    (r"aggregat|pooling|readout", "role", Role.AGGREGATION),
    (r"inference head|flow|posterior estimator|density estimator", "role", Role.INFERENCE_HEAD),
    (r"end.to.end|whole pipeline|complete model", "role", Role.END_TO_END),
]


def parse(text: str) -> Problem:
    """Turn a plain-language description into a structured query."""
    low = text.lower()
    p = Problem(raw=text)
    for pattern, field, value in PATTERNS:
        if getattr(p, field) is None and re.search(pattern, low):
            setattr(p, field, value)

    if re.search(r"fixed number|same number|equal number|fixed count|top ?\d+", low):
        p.count_varies = False
    if re.search(r"varying number|different number|variable number|counts? var", low):
        p.count_varies = True
    if "camels-sam" in low or "camels sam" in low or "quijote" in low:
        p.count_varies = False
    elif "camels" in low:
        p.count_varies = True

    if re.search(r"smallest|fewest parameter|tiniest|most compact|parameter count|"
                 r"minimal model|as small as", low):
        p.prefer_small = True

    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:gpu[- ])?hour", low)
    mins = re.search(r"(\d+(?:\.\d+)?)\s*(?:gpu[- ])?min", low)
    if hours:
        p.minutes_budget = float(hours.group(1)) * 60
    elif mins:
        p.minutes_budget = float(mins.group(1))

    if "sigma" in low or "sigma_8" in low or "clustering amplitude" in low:
        p.task = "camels_sigma_8"
    return p


def _best_for(entry: Entry, task: str):
    """Best measurement on a task, matching across naming variants.

    Point-estimate tasks are named `camels_sigma_8` and posterior tasks
    `camels_sigma_8_posterior`. Without this, a query about sigma_8 fell through
    to the entry's FIRST measurement, which was Omega_m, and the recommender
    quoted a number for the wrong parameter. Caught 2026-08-25.
    """
    exact = entry.best(task)
    if exact:
        return exact
    hits = [m for m in entry.measurements if m.task.startswith(task)
            or task.startswith(m.task.replace("_posterior", ""))]
    if hits:
        return max(hits, key=lambda m: m.value)
    return None


def score(entry: Entry, problem: Problem) -> Tuple[float, List[str]]:
    """Rank score plus the reasons, so a recommendation can be explained."""
    reasons: List[str] = []
    m = _best_for(entry, problem.task)
    base = m.value if m else 0.0
    if m:
        reasons.append(f"measured {m.value:+.4f} on {m.task}"
                       + (f" +/- {m.spread:.4f} over {m.seeds} seeds" if m.spread else ""))
        if getattr(m, "why", ""):
            reasons.append(f"why: {m.why}")
    else:
        reasons.append("no measurement on this task yet")

    if entry.leak_screen.status is LeakStatus.LEAKS:
        if problem.count_varies is False:
            reasons.append("leaks the element count, but your count is fixed so it cannot bite")
        else:
            base -= LEAK_PENALTY
            reasons.append(f"DEMOTED: leaks the element count "
                           f"(probe recovers it at R2 {entry.leak_screen.r2_recovering_count:+.2f})")
    elif entry.leak_screen.status is LeakStatus.CLEAN:
        reasons.append("screened clean for count leakage")

    if entry.calibration is not None:
        c = entry.calibration
        if c.overconfident:
            base -= MISCALIBRATION_PENALTY
            reasons.append(
                f"DEMOTED: overconfident. Its 90% interval contains the truth only "
                f"{c.coverage_90:.0%} of the time, so the error bars are too tight. "
                f"Fitting a density needs more data than fitting a mean, and 600 "
                f"training examples is too few.")
        else:
            reasons.append(f"calibrated: 90% interval contains the truth "
                           f"{c.coverage_90:.0%} of the time")

    if problem.prefer_small:
        if entry.parameters:
            # A log scale, because the range runs from 49 to over a million and a
            # linear term would let one huge entry dominate everything else.
            bonus = max(0.0, (7 - np.log10(max(entry.parameters, 1))) * 0.25)
            base += bonus
            reasons.append(f"{entry.parameters:,} parameters, and you asked for small "
                           f"(+{bonus:.2f})")
        else:
            reasons.append("parameter count not recorded")

    if problem.minutes_budget and entry.minutes_per_fit:
        if entry.minutes_per_fit > problem.minutes_budget:
            base -= 0.5
            reasons.append(f"over budget: {entry.minutes_per_fit:.1f} min per fit")
        else:
            reasons.append(f"{entry.minutes_per_fit:.1f} min per fit, inside budget")
    return base, reasons


def recommend(text: str, top: int = 3) -> Dict:
    problem = parse(text)
    pool = [e for e in build() if e.admissible()]

    gated, rejected = [], []
    for e in pool:
        if problem.output and e.output is not problem.output and e.role is not Role.AGGREGATION:
            why = (f"produces a {e.output.value.replace('_',' ')}, you asked for a "
                   f"{problem.output.value.replace('_',' ')}")
            if problem.output is OutputKind.POSTERIOR:
                why += ". A point estimate cannot be turned into an error bar after the fact"
            rejected.append((e.key, why))
            continue
        if problem.role and e.role is not problem.role:
            rejected.append((e.key, f"is a {e.role.value}, you asked for a {problem.role.value}"))
            continue
        # An inference head consumes whatever summary sits in front of it, so its
        # own modality is not a reason to exclude it from a point-cloud problem.
        if (problem.modality and e.modality is not problem.modality
                and e.role is not Role.INFERENCE_HEAD):
            rejected.append((e.key, f"consumes {e.modality.value.replace('_',' ')}"))
            continue
        gated.append(e)

    ranked = sorted(((*score(e, problem), e) for e in gated), key=lambda t: -t[0])
    return {
        "understood": {k: (v.value if hasattr(v, "value") else v)
                       for k, v in vars(problem).items() if v is not None and k != "raw"},
        "recommendations": [
            {"rank": i + 1, "key": e.key, "name": e.name, "score": round(s, 4),
             "leak": e.leak_screen.status.value, "why": reasons,
             "failure_modes": e.failure_modes}
            for i, (s, reasons, e) in enumerate(ranked[:top])],
        "considered": len(gated), "excluded": rejected[:6],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("description", type=str, help="your problem, in plain language")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out = recommend(args.description, args.top)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f'\n  QUERY: "{args.description}"')
    print(f"  understood as: {out['understood']}")
    print(f"  {out['considered']} entries eligible after gating\n")
    for r in out["recommendations"]:
        flag = "  [LEAKS]" if r["leak"] == "leaks" else ""
        print(f"  {r['rank']}. {r['name']}{flag}   score {r['score']:+.4f}")
        for w in r["why"]:
            print(f"       - {w}")
        for f in r["failure_modes"][:1]:
            print(f"       ! {f}")
        print()


if __name__ == "__main__":
    main()
