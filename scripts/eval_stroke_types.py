"""Stroke-type accuracy against hand labels (``evaluation/labels/dji0009_stroke_types.json``).

    python scripts/eval_stroke_types.py events.json [strokes.json] [--labels evaluation/labels/....json]

``events.json`` gives the hits (time, hitter); types come from ``strokes.json`` when given, else from
``shot_type`` in the events. A label matches the nearest predicted hit within 0.3 s. Prints accuracy
over matched hits (all labels / ``sure`` only), how many labels had no hit, and the confusion table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = Path(__file__).resolve().parents[1] / "evaluation" / "labels" / "dji0009_stroke_types.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("strokes", nargs="?")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--window-ms", type=int, default=300)
    a = ap.parse_args()
    shots = json.loads(Path(a.events).read_text())["shots"]
    types = {s["shot_id"]: s.get("shot_type") for s in shots}
    if a.strokes:
        types = {s["shot_id"]: (s.get("stroke") or {}).get("shot_type") for s in json.loads(Path(a.strokes).read_text())["shots"]}
    labels = json.loads(Path(a.labels).read_text())["shots"]
    used: set[str] = set()
    hit = {"all": [0, 0], "sure": [0, 0]}
    missed = 0
    conf: dict[str, Counter] = defaultdict(Counter)
    for lab in labels:
        cands = [s for s in shots if s["shot_id"] not in used and abs(s["t_ms"] - lab["t_ms"]) <= a.window_ms]
        if not cands:
            missed += 1
            continue
        s = min(cands, key=lambda s: abs(s["t_ms"] - lab["t_ms"]))
        used.add(s["shot_id"])
        pred = types.get(s["shot_id"])
        conf[lab["shot_type"]][pred] += 1
        for k in ("all", "sure"):
            if k == "all" or lab["confidence"] == "sure":
                hit[k][0] += pred == lab["shot_type"]
                hit[k][1] += 1
    for k, (ok, n) in hit.items():
        print(f"{k:5s} {ok}/{n} = {100 * ok / n:.0f}%" if n else f"{k:5s} -")
    print(f"labels without a detected hit: {missed}/{len(labels)}")
    for t, c in sorted(conf.items()):
        print(f"  {t:12s} -> {dict(c.most_common())}")


if __name__ == "__main__":
    main()
