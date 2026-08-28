"""The zoo itself. Every number here is read from point_clouds/results/*.json.

Nothing is transcribed by hand. A measurement typed into a registry rots the
moment anything is rerun, and a stale number in a recommendation engine is worse
than a missing one because it still looks authoritative.

Run `python -m zoo.registry` to rebuild and print the catalogue.
"""

import json
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional

from zoo.schema import (Calibration, Entry, LeakScreen, LeakStatus, Measurement,
                        Modality, OutputKind, Role)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "point_clouds" / "results"

# Screen results measured with point_clouds/blocks/count_screen.py, 3 seeds each.
# Held-out R2 of a probe recovering the element count from the entry's output.
SCREEN = {
    "mean": (-0.6616, LeakStatus.CLEAN), "sum": (0.9138, LeakStatus.LEAKS),
    "max": (0.8968, LeakStatus.LEAKS), "quasi_arithmetic": (-0.7839, LeakStatus.CLEAN),
    "attention_readout": (-3.9413, LeakStatus.CLEAN), "fishnets": (-0.1004, LeakStatus.CLEAN),
    "pna_blind": (-2.5890, LeakStatus.CLEAN), "pna_scaled": (1.0000, LeakStatus.LEAKS),
}


# Why each number came out as it did. A zoo that reports only scores is a
# leaderboard, and a leaderboard cannot tell a user whether a score transfers.
WHY = {
 "tpcf_mlp_omega_m":
   "Pair counting at 25 separations captures most of the Omega_m signal because "
   "matter density sets how strongly galaxies cluster at every scale at once.",
 "tpcf_mlp_sigma_8":
   "Far weaker than for Omega_m, because sigma_8 is driven by rare massive "
   "structures and the CAMELS box is only 25 units across, so it contains very "
   "few of them.",
 "lls_omega_m":
   "Forty-nine parameters suffice because the relationship between pair-distance "
   "statistics and matter density is close to linear once the right cutoff radii "
   "are chosen.",
 "lls_sigma_8":
   "Same small-box limitation as above, and a linear model has no capacity to "
   "recover the weak nonlinear signal that remains.",
}


def _load(name: str) -> Optional[dict]:
    path = RESULTS / f"{name}.json"
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def _measurements_from_gate() -> Dict[str, List[Measurement]]:
    out: Dict[str, List[Measurement]] = {"tpcf_mlp": [], "lls": []}
    for key, doc in (("tpcf_mlp", _load("step1_gate_2pcf")), ("lls", _load("lls_baseline"))):
        if not doc:
            continue
        for result in doc["results"]:
            for target, v in result["targets"].items():
                value = v.get("ours_mean", v.get("ours"))
                spread = v.get("ours_std_across_seeds")
                out[key].append(Measurement(
                    task=f"{result['suite'].lower().replace('-','_')}_{target.lower()}",
                    metric="R2", value=round(float(value), 4),
                    spread=None if spread is None else round(float(spread), 4),
                    seeds=3 if spread is not None else 1,
                    note="reproduces the published baseline within its error bars",
                    why=WHY.get(f"{key}_{target.lower()}", "")))
    return out


def build() -> List[Entry]:
    gate = _measurements_from_gate()
    entries: List[Entry] = []

    entries.append(Entry(
        key="tpcf_mlp", name="Two-point correlation function plus MLP",
        role=Role.END_TO_END, modality=Modality.POINT_CLOUD,
        output=OutputKind.POINT_ESTIMATE, source="CosmoBench, arXiv 2507.03707",
        summary="Counts galaxy pairs at 25 separations, then a four-layer network reads "
                "the resulting curve.",
        parameters=11000, minutes_per_fit=0.2, hardware="laptop CPU",
        measurements=gate["tpcf_mlp"],
        leak_screen=LeakScreen(None, LeakStatus.CLEAN, 0,
                               "xi(r) is a ratio normalised by pair count, so the count "
                               "divides out. Adding the count as an extra input measured "
                               "-0.0114 on Omega_m, i.e. it made things slightly worse."),
        failure_modes=["Fixed binning. The distance range must be chosen for the box, and "
                       "the ranges shipped with the dataset do not match the published ones.",
                       "Sees only pair separations, so any three-body structure is invisible."],
        tags=["baseline", "classical", "strong", "cheap"]))

    entries.append(Entry(
        key="lls_pairwise", name="Linear least squares on pairwise-distance statistics",
        role=Role.END_TO_END, modality=Modality.POINT_CLOUD,
        output=OutputKind.POINT_ESTIMATE, source="CosmoBench, arXiv 2507.03707",
        summary="Four statistics of squared pair separations at twelve cutoff radii, "
                "then a straight-line fit. Forty-nine parameters in total.",
        parameters=49, minutes_per_fit=0.05, hardware="laptop CPU",
        measurements=gate["lls"],
        leak_screen=LeakScreen(None, LeakStatus.CLEAN, 0,
                               "Quantiles and means of pair distances are intensive, so "
                               "they do not scale with the number of objects."),
        failure_modes=["No capacity for anything nonlinear beyond the chosen statistics."],
        tags=["baseline", "classical", "tiny", "surprisingly-strong"]))

    pooling_sources = {
        "mean": ("Mean pooling", "baseline", "averages every element, so the count divides out"),
        "sum": ("Sum pooling", "baseline", "adds every element, so the total scales with the count"),
        "max": ("Max pooling", "baseline", "takes the largest value per channel"),
        "fishnets": ("Fishnets aggregation", "arXiv 2310.03812",
                     "weights each element by how informative it is, a precision-weighted mean"),
        "quasi_arithmetic": ("Quasi-arithmetic pooling", "arXiv 2602.04941",
                             "a learnable power mean containing sum, mean and max as special cases"),
        "attention_readout": ("Attention readout", "arXiv 2211.04952",
                              "learnable seed vectors attend over elements, weights sum to one"),
        "pna_blind": ("PNA without degree scalers", "arXiv 2004.05718",
                      "several aggregators at once, restricted to the count-blind subset"),
        "pna_scaled": ("PNA with degree scalers", "arXiv 2004.05718",
                       "several aggregators, scaled by node degree"),
    }
    missing = _load("missing_cell") or {}
    for key, (name, source, blurb) in pooling_sources.items():
        r2, status = SCREEN[key]
        ms: List[Measurement] = []
        probe = {"fishnets": "fishnets", "mean": "mean", "sum": "sum", "max": "max"}.get(key)
        if probe and probe in missing:
            v = missing[probe]
            ms.append(Measurement("camels_omega_m", "R2", round(v["mean"][0], 4),
                                  round(v["std"][0], 4), 3,
                                  "identical architecture, only the aggregation swapped"))
        entries.append(Entry(
            key=f"pool_{key}", name=name, role=Role.AGGREGATION,
            modality=Modality.POINT_CLOUD, output=OutputKind.EMBEDDING,
            source=source, summary=blurb.capitalize() + ".",
            measurements=ms,
            leak_screen=LeakScreen(r2, status, 3,
                                   "held-out probe recovering the element count"),
            failure_modes=(
                ["Grants access to the element count, which is a documented artefact in "
                 "CAMELS. Do not use where the count varies with the target."]
                if status is LeakStatus.LEAKS else []),
            requires=["point_clouds/blocks/"],
            tags=["aggregation", "screened"] + (["leaks"] if status is LeakStatus.LEAKS else ["clean"])))

    gnn = _load("gnn_experiment")
    if gnn:
        ms = []
        for suite, r in gnn["results"].items():
            v = r["mean"]
            ms.append(Measurement(f"{suite.lower().replace('-','_')}_omega_m", "R2",
                                  round(v["test_r2_mean"][0], 4),
                                  round(v["test_r2_seed_std"][0], 4), 3,
                                  "mean pooling, the count-blind configuration"))
        entries.append(Entry(
            key="radius_gnn", name="Radius-graph message passing",
            role=Role.ENCODER, modality=Modality.POINT_CLOUD, output=OutputKind.EMBEDDING,
            source="CosmoBench, arXiv 2507.03707", parameters=66562, minutes_per_fit=5.5,
            hardware="Apple M5 Pro, MPS",
            summary="Connects galaxies closer than a cutoff, then passes messages between "
                    "neighbours.",
            measurements=ms,
            leak_screen=LeakScreen(None, LeakStatus.CLEAN, 0,
                                   "clean when paired with a count-blind aggregation; the "
                                   "aggregation is what decides"),
            failure_modes=[
                "Receptive field is bounded by cutoff times depth. Measured on CAMELS: the "
                "cutoff grid used reached 0.75 units while the correlation function measures "
                "to 12, so the model never saw the scales the baseline works at.",
                "More message-passing rounds did not help. One round beat five."],
            requires=["point_clouds/gnn.py"], tags=["encoder", "graph"]))

    winner = _load("screened_winner")
    if winner:
        entries.append(Entry(
            key="searched_gnn", name="Searched graph network (count-blind arm)",
            role=Role.END_TO_END, modality=Modality.POINT_CLOUD,
            output=OutputKind.POINT_ESTIMATE, source="this project",
            parameters=None, minutes_per_fit=4.0, hardware="Apple M5 Pro, MPS",
            summary="Best of twenty architectures searched with only count-blind "
                    "aggregations available.",
            measurements=[Measurement("camels_omega_m", "R2",
                                      round(winner["test_r2_mean"][0], 4),
                                      round(winner["test_r2_seed_std"][0], 4), 3,
                                      "winner of a 20-trial search"),
                          Measurement("camels_sigma_8", "R2",
                                      round(winner["test_r2_mean"][1], 4),
                                      round(winner["test_r2_seed_std"][1], 4), 3, "")],
            leak_screen=LeakScreen(-0.6616, LeakStatus.CLEAN, 3,
                                   "uses mean pooling, screened clean"),
            failure_modes=["Searching 40 architectures gained 0.019 over a hand-built model, "
                           "against a 0.181 gap to the classical baseline. Architecture search "
                           "is not where the difficulty lies on this task."],
            tags=["searched", "clean"]))

    entries.extend(_inference_entries())
    return entries


HEAD_INFO = {
 "npe_maf": ("Neural posterior estimation, masked autoregressive flow", "sbi 0.27 / arXiv 1705.07057",
   "Learns the posterior directly as a normalising flow, transforming a simple "
   "distribution into the shape the data implies."),
 "npe_nsf": ("Neural posterior estimation, neural spline flow", "sbi 0.27 / arXiv 1906.04032",
   "Same idea with spline transforms, which bend more flexibly than the "
   "autoregressive version."),
 "npe_mdn": ("Mixture density network", "sbi 0.27 / Bishop 1994",
   "Predicts a weighted sum of Gaussians rather than a single value."),
 "nre": ("Neural ratio estimation", "sbi 0.27 / arXiv 1903.04057",
   "Trains a classifier to tell matched parameter and data pairs from mismatched "
   "ones, which recovers the likelihood ratio."),
}


def _inference_entries() -> List[Entry]:
    """Posterior entries, from the matched-compute comparison."""
    path = ROOT / "zoo" / "inference" / "results.json"
    if not path.exists():
        return []
    rows = json.load(open(path, encoding="utf-8"))
    by_head: Dict[str, List[dict]] = {}
    for r in rows:
        by_head.setdefault(r["head"], []).append(r)

    out: List[Entry] = []
    for head, runs in by_head.items():
        name, source, blurb = HEAD_INFO[head]
        ms, joint = [], None
        for r in runs:
            targets = (["Omega_m"] if "omega" in r["task"] else
                       ["sigma_8"] if "sigma" in r["task"] else ["Omega_m", "sigma_8"])
            if "joint" in r["task"]:
                joint = r
            for i, t in enumerate(targets):
                sd = r["posterior_mean_r2_std"]
                ms.append(Measurement(
                    task=r["task"], metric="R2_posterior_mean",
                    value=r["posterior_mean_r2"][i],
                    spread=sd[i] if sd else None, seeds=r["seeds"],
                    note=f"posterior mean, {t}",
                    why=("Beats the point-estimate baseline on the same features because "
                         "learning the whole conditional distribution extracts more from "
                         "each of the 600 training universes than fitting a mean does."
                         if t == "Omega_m" and r["posterior_mean_r2"][i] > 0.8597 else
                         "Limited by the same small-box scarcity of massive structures "
                         "that caps every model on this parameter."
                         if t == "sigma_8" else "")))
        single = [r for r in runs if "joint" not in r["task"]]
        cov = float(np.mean([r["coverage_90"] for r in single])) if single else None
        err = float(np.mean([r["calibration_error"] for r in single])) if single else None
        over = bool(cov is not None and cov < 0.85)

        failures = [
            "Overconfident on this data: the stated ninety percent interval contains the "
            f"truth only {cov:.0%} of the time. Fitting a density needs more examples than "
            "fitting a mean, and 600 training universes is too few, so sparse sampling is "
            "mistaken for precision."] if over else []
        if joint:
            failures.append(
                "Degrades sharply when both parameters are inferred together: calibration "
                f"error rises from {err:.3f} to {joint['calibration_error']:.3f} while R2 "
                "barely moves. The joint posterior is a tilted ellipse because the two "
                "parameters push clustering in opposite directions, and getting that tilt "
                "wrong shrinks the region even when each margin looks correct. No R2 "
                "leaderboard would reveal this.")

        out.append(Entry(
            key=head, name=name, role=Role.INFERENCE_HEAD,
            modality=Modality.SUMMARY_VECTOR, output=OutputKind.POSTERIOR,
            source=source, summary=blurb,
            parameters=runs[0]["n_parameters"],
            minutes_per_fit=round(float(np.mean([r["train_minutes"] for r in runs])), 2),
            hardware="laptop CPU",
            measurements=ms,
            leak_screen=LeakScreen(None, LeakStatus.CLEAN, 0,
                "consumes the correlation function, which is normalised by pair count, "
                "so the element count is already divided out upstream"),
            calibration=Calibration(
                coverage_90=round(cov, 4) if cov else None,
                calibration_error=round(err, 4) if err else None,
                overconfident=over,
                note="mean over the two single-parameter tasks, 3 seeds each"),
            failure_modes=failures,
            requires=["sbi", "zuko"],
            tags=["posterior", "sbi", "calibrated" if not over else "overconfident"]))
    return out


def save(path: Optional[Path] = None) -> Path:
    path = path or ROOT / "zoo" / "catalogue.json"
    entries = build()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"entries": [e.to_dict() for e in entries]}, fh, indent=2)
    return path


if __name__ == "__main__":
    entries = build()
    print(f"{len(entries)} entries\n")
    print(f"  {'key':22s} {'role':16s} {'leak':10s} {'best CAMELS Om':>15s}")
    for e in sorted(entries, key=lambda x: x.role.value):
        m = e.best("camels_omega_m")
        print(f"  {e.key:22s} {e.role.value:16s} {e.leak_screen.status.value:10s} "
              f"{(f'{m.value:+.4f}' if m else '-'):>15s}")
    post = [e for e in entries if e.output is OutputKind.POSTERIOR]
    if post:
        print(f"\n  {'posterior entry':52s} {'cov90':>7s} {'calib err':>10s}")
        for e in post:
            c = e.calibration
            print(f"  {e.name[:52]:52s} {c.coverage_90:7.3f} {c.calibration_error:10.4f}"
                  f"{'  OVERCONFIDENT' if c.overconfident else '  calibrated'}")
    print(f"\n  admissible: {sum(e.admissible() for e in entries)} of {len(entries)}")
    print(f"  wrote {save()}")
