"""Full-pipeline threshold sweep on saved leave-one-video-out models (train_hits_v3.py lovo --tmp DIR).

    python scripts/lovo_sweep.py DIR --keep 0.2 0.3 --mc 0.1 0.15 --veto 0.1 0 --out sweep.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import amateur_hits as AH  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("models", type=Path)
ap.add_argument("--keep", type=float, nargs="+", default=[0.3])
ap.add_argument("--mc", type=float, nargs="+", default=[0.15])
ap.add_argument("--veto", type=float, nargs="+", default=[0.1])
ap.add_argument("--extra", default="{}")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--labels", default="~/Desktop/RallyIQ/train_v3/labels/amateur_v1.json")
ap.add_argument("--bundled", action="store_true", help="use the bundled scorer for every video instead of DIR models")
ap.add_argument("--scorer", help="with --bundled: this scorer JSON instead of rallyiq/events/models/hit_scorer.json")
a = ap.parse_args()
clips = AH.load_clips(Path("~/Desktop/RallyIQ/train_v3/inputs").expanduser(), Path(a.labels).expanduser())
tmp = a.models.expanduser() / "sweep"
tmp.mkdir(exist_ok=True)
for keep, mc, veto in itertools.product(a.keep, a.mc, a.veto):
    preds = {}
    for c in clips:
        src = (Path(a.scorer).expanduser() if a.scorer else ROOT / "rallyiq/events/models/hit_scorer.json") if a.bundled else a.models.expanduser() / f"lovo_{c.cid}.json"
        spec = json.loads(src.read_text())
        spec["thresholds"] = {**spec.get("thresholds", {}), "min_candidate_conf": mc, "keep_threshold": keep, "ok_threshold": max(0.7, keep)}
        mp = tmp / f"{c.cid}_{keep}_{mc}.json"
        mp.write_text(json.dumps(spec))
        cfg = replace(EventsConfig(), **{**AH.AMATEUR_BASE, "swing_veto": veto, "hit_scorer": str(mp), **json.loads(a.extra)})
        preds[c.cid] = c.predict_full(cfg)
    md, js = AH.report(clips, preds)
    t = js["total"]
    row = {"models": "bundled" if a.bundled else str(a.models), "keep": keep, "mc": mc, "veto": veto, "extra": a.extra,
           **{f"{k}_{m}": round(t[k][m], 4) for k in ("qa6", "am7", "all", "qa6_loose") for m in ("f1", "precision", "recall")},
           "dead_fp": t["all"]["dead_fp"], "side_acc": round(t["all"]["side_acc"], 4), "hitter_acc": round(t["am7"]["hitter_acc"], 4)}
    print(json.dumps(row), flush=True)
    with open(a.out.expanduser(), "a") as fh:
        fh.write(json.dumps(row) + "\n")
