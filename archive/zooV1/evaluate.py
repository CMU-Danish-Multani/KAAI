"""Does the recommender work, and does the admission screen change what it says?

Two experiments.

HELD-OUT PROBLEM DESCRIPTIONS
    Five descriptions written the way a researcher would actually ask, each with
    the architecture we believe is correct for it. The brief's success criterion
    is four of five.

THE SCREEN ABLATION
    The same queries, run with the leakage screen switched off, to measure how
    the advice changes. This is the evidence for requiring a screen at admission
    rather than ranking on score alone.
"""

import json
from typing import Dict, List

from zoo import recommend as R

HELD_OUT: List[Dict] = [
    {"description": "I have point clouds of galaxy positions from CAMELS, the boxes have "
                    "different numbers of galaxies in them, and I want the best estimate of "
                    "Omega_m I can get in a few minutes.",
     "expect": "tpcf_mlp",
     "because": "strongest measured score, cheap, and unaffected by the count artefact"},
    {"description": "I need a pooling operation for a set model on CAMELS. It must not be "
                    "able to see how many objects are in the set.",
     "expect": "pool_fishnets",
     "because": "best measured count-blind aggregation, 0.6895 against 0.6654 for mean"},
    {"description": "Give me the smallest possible model for cosmological parameters from "
                    "point clouds. I care about parameter count, not accuracy.",
     "expect": "lls_pairwise",
     "because": "49 parameters, and still competitive"},
    {"description": "I am working on Quijote where every box has exactly 5000 halos. I want "
                    "the highest scoring aggregation available, and leakage is not a concern "
                    "because the count is fixed.",
     "expect": "pool_sum",
     "because": "highest measured aggregation score, and its leak cannot bite when the "
                "count is constant"},
    {"description": "I want an encoder that turns a galaxy point cloud into an embedding "
                    "for downstream use.",
     "expect": "radius_gnn",
     "because": "the only entry whose role is encoder producing an embedding"},
    {"description": "I need a posterior on Omega_m from CAMELS with error bars I can actually "
                    "trust in a paper.",
     "expect": "nre",
     "because": "highest posterior-mean R2 among the calibrated heads; the more accurate "
                "MAF flow is overconfident and is demoted"},
    {"description": "I want to infer both cosmological parameters jointly and report a "
                    "credible region.",
     "expect": "nre",
     "because": "best calibrated head, though every head degrades on the joint task and the "
                "recommendation must say so"},
    {"description": "Which density estimator gives the best calibrated uncertainty on "
                    "sigma_8?",
     "expect": "nre",
     "because": "lowest calibration error on the single-parameter tasks"},
]


def run_held_out(top: int = 3) -> Dict:
    rows, hits = [], 0
    for case in HELD_OUT:
        out = R.recommend(case["description"], top=top)
        got = [r["key"] for r in out["recommendations"]]
        ok = bool(got) and got[0] == case["expect"]
        hits += ok
        rows.append({"expected": case["expect"], "got": got, "top1_correct": ok,
                     "in_top3": case["expect"] in got, "because": case["because"]})
    return {"cases": rows, "top1": hits, "total": len(HELD_OUT),
            "in_top3": sum(r["in_top3"] for r in rows)}


def run_screen_ablation() -> Dict:
    """Rerun every query with the leak penalty removed, and diff the advice."""
    original = R.LEAK_PENALTY
    changed = []
    try:
        with_screen = {c["description"][:40]: R.recommend(c["description"])["recommendations"]
                       for c in HELD_OUT}
        R.LEAK_PENALTY = 0.0
        without = {c["description"][:40]: R.recommend(c["description"])["recommendations"]
                   for c in HELD_OUT}
    finally:
        R.LEAK_PENALTY = original

    for key in with_screen:
        a = [r["key"] for r in with_screen[key]]
        b = [r["key"] for r in without[key]]
        if a != b:
            leaky = [r["key"] for r in without[key] if r["leak"] == "leaks"]
            changed.append({"query": key, "with_screen": a, "without_screen": b,
                            "leaking_entries_promoted": leaky})
    return {"queries": len(with_screen), "advice_changed": len(changed), "detail": changed}


def run_calibration_ablation() -> Dict:
    """Rerun with the calibration penalty removed, and diff the advice.

    This is the evidence for the second admission check. Without it, entries are
    ranked on accuracy alone, and the most accurate posterior head on this data
    is also the one whose error bars are too tight.
    """
    original = R.MISCALIBRATION_PENALTY
    posterior_cases = [c for c in HELD_OUT
                       if "posterior" in c["description"].lower()
                       or "credible" in c["description"].lower()
                       or "uncertainty" in c["description"].lower()
                       or "error bar" in c["description"].lower()]
    changed = []
    try:
        with_check = {c["description"][:40]: R.recommend(c["description"])["recommendations"]
                      for c in posterior_cases}
        R.MISCALIBRATION_PENALTY = 0.0
        without = {c["description"][:40]: R.recommend(c["description"])["recommendations"]
                   for c in posterior_cases}
    finally:
        R.MISCALIBRATION_PENALTY = original

    for key in with_check:
        a = [r["key"] for r in with_check[key]]
        b = [r["key"] for r in without[key]]
        if a != b:
            changed.append({"query": key, "with_check": a, "without_check": b})
    return {"queries": len(with_check), "advice_changed": len(changed), "detail": changed}


if __name__ == "__main__":
    print("=" * 74)
    print("HELD-OUT PROBLEM DESCRIPTIONS")
    print("=" * 74)
    res = run_held_out()
    for i, r in enumerate(res["cases"], 1):
        mark = "PASS" if r["top1_correct"] else ("top-3" if r["in_top3"] else "MISS")
        print(f"  {i}. expected {r['expected']:22s} got {str(r['got'][:2]):46s} {mark}")
    print(f"\n  top-1 correct: {res['top1']} of {res['total']}   "
          f"(brief's criterion: 4 of 5)")
    print(f"  in top-3:      {res['in_top3']} of {res['total']}")

    print("\n" + "=" * 74)
    print("SCREEN ABLATION: what changes if entries are ranked on score alone?")
    print("=" * 74)
    ab = run_screen_ablation()
    print(f"  advice changed on {ab['advice_changed']} of {ab['queries']} queries\n")
    for d in ab["detail"]:
        print(f"  query: {d['query']}...")
        print(f"    with screen:    {d['with_screen']}")
        print(f"    without screen: {d['without_screen']}")
        print(f"    leaking entries promoted: {d['leaking_entries_promoted']}\n")
    print("\n" + "=" * 74)
    print("CALIBRATION ABLATION: what changes if posteriors are ranked on accuracy alone?")
    print("=" * 74)
    cb = run_calibration_ablation()
    print(f"  advice changed on {cb['advice_changed']} of {cb['queries']} "
          f"uncertainty-related queries\n")
    for d in cb["detail"]:
        print(f"  query: {d['query']}...")
        print(f"    with the check:    {d['with_check']}")
        print(f"    without the check: {d['without_check']}\n")

    json.dump({"held_out": res, "leak_ablation": ab, "calibration_ablation": cb},
              open("zoo/evaluation.json", "w"), indent=2)
    print("  wrote zoo/evaluation.json")
