"""Every config the skill emits must actually load in ltu-ili.

    conda run -n ltuili python -m ili_kaai.checks.emittedConfig

The skill tells users its recommendation is runnable. That is a claim, and it was false
twice before this check existed.

Defect one: InferenceRunner.from_config dispatches on config['model']['backend'] and
the emitted config had no backend key, so every config raised KeyError. Nothing caught
it because the yaml looked plausible and was never loaded.

Defect two: point cloud entries emitted the EMBEDDINGS registry key as the class name,
so ili's getattr(module, class) raised "module has no attribute 'pairwiseGnn'" while
the class is PairwiseGnn.

Both were found by loading a config rather than by reading it. So this loads one for
every admitted entry, every time.
"""

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List

from ili.inference import InferenceRunner

from skill.query import emit_config, load_zoo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ili_kaai" / "results" / "emittedConfigCheck.json"


def check_one(zoo: Dict, key: str, n_params: int, out_dir: Path) -> Dict:
    """Emit a config for one entry and load it. Loading is the test, not parsing."""
    path = out_dir / f"{key}.yaml"
    row: Dict = {"key": key, "nParams": n_params}
    try:
        text = emit_config(zoo, key, [0.1] * n_params, [0.5] * n_params)
        path.write_text(text, encoding="utf-8")
    except Exception as exc:
        row.update({"emitted": False, "loaded": False,
                    "error": f"emit: {type(exc).__name__}: {exc}"})
        return row
    row["emitted"] = True
    try:
        runner = InferenceRunner.from_config(str(path))
        row.update({"loaded": True, "runner": type(runner).__name__,
                    "nNets": len(runner.nets), "error": None})
    except Exception as exc:
        row.update({"loaded": False, "runner": None, "nNets": None,
                    "error": f"load: {type(exc).__name__}: {exc}"})
    return row


def expected_nets(zoo: Dict, key: str) -> int:
    """An ensemble entry must build one net per member, or the config silently
    produces a single model while the catalogue promises an ensemble."""
    entry = next(e for e in zoo["entries"] if e["key"] == key)
    mixture = entry["config"].get("mixture") or ()
    return len(mixture) if mixture else max(1, entry["config"]["repeats"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    zoo = load_zoo()
    admitted = [e for e in zoo["entries"] if e["admitted"]]
    rows: List[Dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in admitted:
            n_params = len(entry["measurements"][0]["params"])
            row = check_one(zoo, entry["key"], n_params, Path(tmp))
            want = expected_nets(zoo, entry["key"])
            row["expectedNets"] = want
            row["netCountCorrect"] = (row.get("nNets") == want)
            rows.append(row)
            status = "LOADS " if row["loaded"] else "FAILS "
            nets = (f"{row['nNets']} of {want} nets"
                    if row["loaded"] else row["error"][:70])
            flag = "" if row["netCountCorrect"] or not row["loaded"] else "  NET COUNT WRONG"
            print(f"  {status} {row['key']:30} {nets}{flag}")

    loaded = sum(r["loaded"] for r in rows)
    nets_ok = sum(r["netCountCorrect"] for r in rows)
    print(f"\n  {loaded}/{len(rows)} emitted configs load in ltu-ili")
    print(f"  {nets_ok}/{len(rows)} build the number of nets the catalogue promises")
    payload = {"nChecked": len(rows), "nLoaded": loaded, "nNetCountCorrect": nets_ok,
               "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {args.out}")
    if loaded != len(rows) or nets_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
