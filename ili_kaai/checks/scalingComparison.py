"""Did removing the batch dependent rescale change the point cloud results?

    conda run -n ltuili python -m ili_kaai.checks.scalingComparison

Compares sweepCloud.json, measured with `embeddings._to_unit_box` rescaling by the
minimum and maximum over the training batch, against sweepCloudFixedScaling.json,
measured after that rescale was removed.

WHY THIS IS A COMPARISON AND NOT A REPLACEMENT
----------------------------------------------
The old numbers are honest measurements of the old implementation. Whether the fix
helps is a question, and answering it by overwriting the file would destroy the
evidence needed to answer it. So the corrected sweep writes its own file, the
catalogue keeps reading the old one, and this decides which becomes canonical.

The defect was real and measured: training pooled the extremes over 32 clouds while
evaluation saw one cloud at a time, a scale difference of 1.003 to 1.006 per
coordinate, which flipped 6.7 per cent of k=16 neighbour slots to a different galaxy
and changed the neighbour set of all 32 clouds tested. Whether that costs accuracy is
what this measures.

Both arms of the pretraining comparison carried the defect, so the pretrained versus
from scratch conclusion does not depend on this. The absolute values may.
"""

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "ili_kaai" / "results"
OLD = RESULTS / "sweepCloud.json"
NEW = RESULTS / "sweepCloudFixedScaling.json"
OUT = RESULTS / "scalingComparison.json"


def cells_by_pair(path: Path) -> Dict[Tuple[str, str], List[Dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise SystemExit(f"{path.name} is not complete; refusing to compare a partial "
                         f"sweep against a finished one")
    out: Dict[Tuple[str, str], List[Dict]] = {}
    for c in data["cells"]:
        if "error" not in c:
            out.setdefault((c["architecture"], c["task"]), []).append(c)
    return out


def summarise(cells: List[Dict]) -> Dict[str, Optional[float]]:
    r2 = [c["r2"][0] for c in cells]
    cov = [sum(c["coverage68"]) / len(c["coverage68"]) for c in cells]
    return {"r2": st.mean(r2),
            # null, never zero: one seed has no spread
            "r2Std": st.pstdev(r2) if len(r2) > 1 else None,
            "coverage68": st.mean(cov),
            "nSeeds": len(cells)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--old", type=str, default=str(OLD))
    p.add_argument("--new", type=str, default=str(NEW))
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    new_path = Path(args.new)
    if not new_path.exists():
        raise SystemExit(f"{new_path.name} does not exist yet. The corrected sweep is "
                         f"stage 6 of the chain and has not run.")

    old, new = cells_by_pair(Path(args.old)), cells_by_pair(new_path)
    shared = sorted(set(old) & set(new))
    if not shared:
        raise SystemExit("no entry-task pair appears in both sweeps")

    rows = []
    print(f"  {'entry':30} {'task':15} {'old R2':>16} {'new R2':>16} {'delta':>8}")
    for key in shared:
        a, b = summarise(old[key]), summarise(new[key])
        delta = b["r2"] - a["r2"]
        # A change smaller than the combined seed spread is not a change.
        spread = max(a["r2Std"] or 0.0, b["r2Std"] or 0.0)
        significant = abs(delta) > spread
        rows.append({"architecture": key[0], "task": key[1], "old": a, "new": b,
                     "deltaR2": round(delta, 4),
                     "largerThanSeedSpread": significant})
        flag = "  *" if significant else ""
        print(f"  {key[0]:30} {key[1]:15} "
              f"{a['r2']:+8.3f} +/- {a['r2Std'] or 0:.3f} "
              f"{b['r2']:+8.3f} +/- {b['r2Std'] or 0:.3f} {delta:+8.3f}{flag}")

    moved = [r for r in rows if r["largerThanSeedSpread"]]
    deltas = [r["deltaR2"] for r in rows]
    print(f"\n  {len(moved)} of {len(rows)} pairs moved by more than the seed spread "
          f"(marked *)")
    print(f"  mean delta {st.mean(deltas):+.4f}, "
          f"spread {st.pstdev(deltas) if len(deltas) > 1 else 0:.4f}")
    print("  a mean near zero with individual pairs moving means the rescale added "
          "noise rather than bias")

    payload = {"old": args.old, "new": args.new,
               "nPairs": len(rows), "nMovedBeyondSeedSpread": len(moved),
               "meanDeltaR2": round(st.mean(deltas), 4), "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
