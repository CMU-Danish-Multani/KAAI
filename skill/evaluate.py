"""Score the structured retrieval arm against the held out problems.

    conda run -n ltuili python -m skill.evaluate

The brief's success bar is 4 of 5 held out descriptions answered correctly, and the
held out problems come from published applications rather than from us, so the bar
means something.

WHAT THIS GRADES AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
Graded automatically: the recommended engine, the recommended entry key, and whether
the recommender correctly DECLINED a problem whose modality the zoo has never measured.
Those are checkable without judgement.

Not graded automatically: the required warnings. They are prose, and matching prose
against a list of expected strings measures vocabulary rather than understanding. A
grader that scored them by substring would reward a recommender that pasted the right
words. So they are printed next to what the structured arm actually warned, and scored
by a human or by the few shot arm reading the same catalogue.

The headline number here therefore covers the engine and key decision only. That is a
weaker claim than "answered correctly" and is reported as such.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from skill.query import Query, load_zoo, recommend

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = Path(__file__).resolve().parent / "heldOut.json"
OUT = ROOT / "ili_kaai" / "results" / "skillEvaluation.json"

# Which zoo entries belong to which engine, so a family level answer can be graded
# without hardcoding a mapping that could drift from architectures.py.
def engine_of(zoo: Dict, key: str) -> str:
    entry = next((e for e in zoo["entries"] if e["key"] == key), None)
    return entry["engine"] if entry else "unknown"


PROMPT_TEMPLATE = """You are recommending a simulation based inference architecture.

Read the catalogue at ili_kaai/results/zoo.json. Use each entry's summary, its
failureModes, and each measurement's `why` field. Do NOT run skill/query.py: this is
the few shot arm and the point is to compare your reading of the catalogue against
what the structured ranker computes.

Answer with exactly this JSON and nothing else:
  {{"engine": "NPE" | "NLE" | "NRE" | null,
    "key": "<zoo entry key>" | null,
    "reasoning": "<two or three sentences>"}}

Use null for engine or key if the catalogue cannot answer this problem. Declining is a
valid answer and is graded as correct when the zoo genuinely does not cover the case.

THE PROBLEM
-----------
{description}
"""


def emit_prompts(problems: List[Dict], out_dir: Path) -> None:
    """Write one blind prompt per problem: the description only, never the answer.

    The few shot arm has to be run by a Claude session that has not seen this file's
    acceptableEngines or acceptableKeys. Whoever wrote the answer key cannot also play
    the few shot arm, because they already know the answer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for prob in problems:
        text = PROMPT_TEMPLATE.format(description=prob["description"])
        (out_dir / f"{prob['id']}.txt").write_text(text, encoding="utf-8")
    print(f"  wrote {len(problems)} blind prompts to {out_dir}")
    print("  run each in a Claude session that has NOT read the held out file, save "
          "the JSON replies as {id: {engine, key, reasoning}}, then grade with "
          "--arm fewshot --answers <file>")


def grade_fewshot(problem: Dict, answer: Optional[Dict]) -> Dict:
    """Grade a recorded few shot answer with exactly the structured arm's rules."""
    out_of_scope = problem["query"]["modality"] not in ("summary_vector", "point_cloud")
    res = {"id": problem["id"], "modality": problem["query"]["modality"],
           "outOfScope": out_of_scope,
           "expectedEngines": problem["acceptableEngines"],
           "expectedKeys": problem["acceptableKeys"],
           "excludedByBudget": [],
           "requiredWarnings": problem["requiredWarnings"]}
    if answer is None:
        res.update({"recommended": None, "recommendedEngine": None,
                    "engineCorrect": False, "keyCorrect": False,
                    "actualWarnings": [], "verdict": "no answer recorded"})
        return res
    engine, key = answer.get("engine"), answer.get("key")
    res.update({
        "recommended": key, "recommendedEngine": engine,
        "engineCorrect": (engine in problem["acceptableEngines"]
                          if engine else out_of_scope),
        "keyCorrect": (key in problem["acceptableKeys"] if key
                       else (out_of_scope or not problem["acceptableKeys"])),
        "actualWarnings": [answer.get("reasoning", "")],
        "verdict": "answered" if engine else "declined"})
    return res


def grade_one(zoo: Dict, problem: Dict) -> Dict:
    q = problem["query"]
    query = Query(modality=q["modality"], nParams=q["nParams"],
                  nObservations=q["nObservations"],
                  computeSeconds=q.get("computeSeconds"),
                  downstream=q.get("downstream", False))
    query.iidTrials = q.get("iidTrials", False)
    result = recommend(zoo, query, top=3)
    recs, excluded = result.recommendations, result.excluded

    out_of_scope = result.outOfScope
    res = {"id": problem["id"], "modality": q["modality"],
              "outOfScope": out_of_scope,
              "expectedEngines": problem["acceptableEngines"],
              "expectedKeys": problem["acceptableKeys"],
              "excludedByBudget": [x["key"] for x in excluded],
              "requiredWarnings": problem["requiredWarnings"]}

    result_advice = result.advice
    if not recs:
        # Declining is correct exactly when the zoo has no measurement for this
        # modality. Declining a problem it could have answered is a miss.
        # An out of scope problem is answered correctly when the recommender
        # declines AND names the engine the literature points to, if it has a rule.
        engine_ok = out_of_scope and (
            result.advisedEngine in problem["acceptableEngines"]
            or result.advisedEngine is None)
        res.update({"recommended": None,
                    "recommendedEngine": result.advisedEngine,
                    "engineCorrect": engine_ok, "keyCorrect": out_of_scope,
                    "actualWarnings": result_advice,
                    "verdict": "declined, correct" if out_of_scope
                               else "declined, but the zoo covers this modality"})
        return res

    top = recs[0]
    engine = engine_of(zoo, top.key)
    res.update({
        "recommended": top.key,
        "recommendedEngine": engine,
        "runnerUp": [r.key for r in recs[1:]],
        "engineCorrect": (result.advisedEngine or engine)
                         in problem["acceptableEngines"],
        "keyCorrect": top.key in problem["acceptableKeys"],
        "actualWarnings": top.warnings + result_advice,
        "advisedEngine": result.advisedEngine,
        "verdict": "answered"})
    if out_of_scope:
        res["verdict"] = "answered a problem it should have declined"
        res["engineCorrect"] = False
        res["keyCorrect"] = False
    return res


def report(results: List[Dict]) -> str:
    lines = []
    for r in results:
        mark = "PASS" if r["engineCorrect"] else "FAIL"
        lines.append(f"\n[{mark}] {r['id']}   ({r['verdict']})")
        lines.append(f"   expected engine {r['expectedEngines']}, "
                     f"got {r['recommendedEngine']}")
        lines.append(f"   expected one of {r['expectedKeys']}")
        lines.append(f"   got {r['recommended']}"
                     + (f", then {r['runnerUp']}" if r.get("runnerUp") else "")
                     + f"   [key {'match' if r['keyCorrect'] else 'MISS'}]")
        if r["excludedByBudget"]:
            lines.append(f"   budget removed: {r['excludedByBudget']}")
        lines.append("   required warnings, scored by hand:")
        for w in r["requiredWarnings"]:
            lines.append(f"     REQUIRED  {w}")
        for w in r["actualWarnings"]:
            lines.append(f"     EMITTED   {w}")
        if not r["actualWarnings"]:
            lines.append("     EMITTED   none")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--held-out", type=str, default=str(HELD_OUT))
    p.add_argument("--out", type=str, default=str(OUT))
    p.add_argument("--arm", type=str, default="structured",
                   choices=["structured", "fewshot"])
    p.add_argument("--answers", type=str, default=None,
                   help="fewshot arm: JSON of {problemId: {engine, key, reasoning}}")
    p.add_argument("--emit-prompts", type=str, default=None,
                   help="write blind prompts for the few shot arm to this directory "
                        "and exit")
    args = p.parse_args()

    held = json.loads(Path(args.held_out).read_text(encoding="utf-8"))
    if args.emit_prompts:
        emit_prompts(held["problems"], Path(args.emit_prompts))
        return

    if args.arm == "fewshot":
        if not args.answers:
            raise SystemExit("--arm fewshot needs --answers with recorded replies")
        answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        results = [grade_fewshot(prob, answers.get(prob["id"]))
                   for prob in held["problems"]]
    else:
        zoo = load_zoo()
        results = [grade_one(zoo, prob) for prob in held["problems"]]

    engine_score = sum(r["engineCorrect"] for r in results)
    key_score = sum(r["keyCorrect"] for r in results)
    total = len(results)

    print(report(results))
    print(f"\n  engine correct {engine_score}/{total}, "
          f"entry key correct {key_score}/{total}")
    print(f"  the brief's bar is 4/5, and this covers the engine and key decision "
          f"only, not the warnings")
    print(f"  held out problems come from {held['provenance']['source']}")

    payload = {"bar": "4 of 5", "arm": args.arm, "engineCorrect": engine_score,
               "keyCorrect": key_score, "nProblems": total,
               "gradedAutomatically": ["engine", "entry key", "correct decline"],
               "gradedByHand": ["required warnings"],
               "heldOutSource": held["provenance"]["source"],
               "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
