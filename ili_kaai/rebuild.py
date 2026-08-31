"""Rebuild the catalogue and everything derived from it, in order, in one command.

    conda run -n ltuili python -m ili_kaai.rebuild
    conda run -n ltuili python -m ili_kaai.rebuild --with-param-count

WHY THIS EXISTS
---------------
Finishing a sweep leaves five things that must be regenerated in a fixed order: the
catalogue, the parameter counts, the generated facts the skill quotes, the check that
emitted configs still load, and the held out scores. Doing them by hand means doing
four of them and forgetting one.

That is not hypothetical. skill/SKILL.md carried "14 of the 15 admitted entries are
overconfident" while the rebuilt catalogue held 28 admitted entries and 72 overconfident
pairs out of 77, because the catalogue was rebuilt and the document was not.

Order matters. facts.py reads zoo.json, so the catalogue must be rebuilt first.
zoo.py reads paramCount.json, so parameter counts must exist before it. The emitted
config check and the held out evaluation both read the finished catalogue.

Parameter counting is off by default because it rebuilds every net on every task and
takes minutes. Pass --with-param-count after a sweep that added or changed an entry.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, module: str, args: List[str]) -> Tuple[str, bool, str]:
    """One stage. Returns its label, whether it succeeded, and its last output line."""
    proc = subprocess.run([sys.executable, "-m", module, *args],
                          cwd=ROOT, capture_output=True, text=True)
    stream = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln.strip() for ln in stream.splitlines() if ln.strip()]
    # Library warnings are often the last thing on the stream and say nothing about
    # whether the stage worked, so report the last line that is actually ours.
    noise = ("WARNING", "UserWarning", "FutureWarning", "warnings.warn")
    ours = [ln for ln in lines if not ln.startswith(noise) and "Warning" not in ln]
    tail = (ours[-1] if ours else (lines[-1] if lines else "(no output)"))
    return label, proc.returncode == 0, tail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--with-param-count", action="store_true",
                   help="recount trainable weights; slow, needed after an entry "
                        "is added or its config changes")
    p.add_argument("--skip-evaluation", action="store_true",
                   help="skip the held out scoring, which needs the skill package")
    args = p.parse_args()

    stages: List[Tuple[str, str, List[str]]] = []
    if args.with_param_count:
        # Before the catalogue: zoo.py reads paramCount.json.
        stages.append(("parameter counts", "ili_kaai.paramCount", []))
    stages.append(("catalogue", "ili_kaai.zoo", []))
    stages.append(("generated facts", "skill.facts", []))
    stages.append(("emitted configs load", "ili_kaai.checks.emittedConfig", []))
    if not args.skip_evaluation:
        stages.append(("held out, development set", "skill.evaluate", []))
        stages.append(("held out, clean set", "skill.evaluate",
                       ["--held-out", "skill/heldOutTwo.json",
                        "--out", "ili_kaai/results/skillEvaluationTwo.json"]))

    results = []
    for label, module, extra in stages:
        print(f"  running {label} ...", flush=True)
        results.append(run(label, module, extra))

    print()
    failed = 0
    for label, ok, tail in results:
        mark = "ok  " if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {label:28} {tail[:88]}")

    if not args.with_param_count:
        print("\n  NOTE parameter counts were NOT regenerated. Pass "
              "--with-param-count after adding or changing an entry.")
    print(f"\n  {len(results) - failed}/{len(results)} stages succeeded")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
