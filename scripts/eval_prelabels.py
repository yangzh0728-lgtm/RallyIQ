"""Stroke-type accuracy of court_v1 (+ rally smoothing) vs court_v2 on hand-checked amateur hits.

    git fetch origin claude/project-thread-ah2yye && git checkout claude/project-thread-ah2yye
    python scripts/eval_prelabels.py ~/Desktop/RallyIQ/amateur_prelabels

<root>/artifacts/<video_key>/{probe,players,shuttle,calibration,events}.json[.gz] and <root>/labels.json
(or labels_compact.csv). Uses the trusted labels only: status sure, corrected/*, added_by_review.
Each label is matched to the nearest pipeline hit within 0.3 s; types come from re-running the strokes
classifier on that run's events with each court model. court_v1 is read from git (commit cbbac7c).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import EventsResult, PlayerTracks, ShuttleTrack  # noqa: E402
from rallyiq.strokes import court as CM  # noqa: E402
from rallyiq.strokes.classify import classify  # noqa: E402
from rallyiq.strokes.detail import get_model as get_attrs  # noqa: E402
from rallyiq.strokes.learned import BUNDLED, make_model_fn as video_model  # noqa: E402

TRUSTED = ("sure", "corrected", "added_by_review")


def read(d: Path, name: str):
    for p in (d / f"{name}.json", d / f"{name}.json.gz"):
        if p.exists():
            return json.loads(gzip.open(p).read() if p.suffix == ".gz" else p.read_text())
    return None


def labels(root: Path) -> list[dict]:
    rows: list[dict] = []
    if (root / "labels_compact.csv").exists():
        with open(root / "labels_compact.csv", encoding="utf-8") as fh:
            rows = [dict(r, t_ms=float(r["t"]) * 1000) for r in csv.DictReader(fh)]
    elif (root / "labels.json").exists():
        data = json.loads((root / "labels.json").read_text())

        def walk(o, video=None):
            if isinstance(o, dict):
                video = o.get("video") or o.get("key") or video
                if "stroke" in o and ("t" in o or "t_ms" in o):
                    t = o.get("t_ms", (o.get("t") or 0) * 1000)
                    rows.append({**o, "video": video, "t_ms": float(t)})
                    return
                for k, v in o.items():
                    walk(v, video if not isinstance(v, (dict, list)) or k in ("hits", "labels", "entries") else (k if isinstance(v, (list, dict)) and video is None else video))
            elif isinstance(o, list):
                for v in o:
                    walk(v, video)
        walk(data)
    return [r for r in rows if str(r.get("status", "")).split("/")[0] in TRUSTED and r.get("stroke")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--window-ms", type=float, default=300)
    a = ap.parse_args()
    root = a.root.expanduser()
    labs = labels(root)
    v1 = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", "cbbac7c:rallyiq/strokes/models/strokes_court_v1.json"]))
    models = {"court_v1 + smoothing (before)": CM.CourtModel(v1), "court_v2 (after)": CM.CourtModel(json.loads(CM.BUNDLED.read_text()))}
    attrs, vid = get_attrs(), video_model(BUNDLED)
    total = {k: [0, 0] for k in models}
    conf = {k: Counter() for k in models}
    print(f"{len(labs)} trusted labels in {len({r['video'] for r in labs})} videos")
    for video in sorted({r["video"] for r in labs}):
        d = root / "artifacts" / video
        probe, ev = read(d, "probe"), read(d, "events")
        if probe is None or ev is None:
            print(f"{video}: no artifacts in {d}, skipped")
            continue
        events = EventsResult.model_validate(ev)
        sh, pl = read(d, "shuttle"), read(d, "players")
        sh = ShuttleTrack.model_validate(sh) if sh else None
        pl = PlayerTracks.model_validate(pl) if pl else None
        fps = float(probe.get("fps") or 30.0)
        vl = [r for r in labs if r["video"] == video]
        line = [f"{video}: {len(vl)} labels"]
        for name, cm in models.items():
            fn = CM.make_model_fn(vid, cm, attrs, fps, seq=True)
            res = classify(events, sh, pl, fps, model=fn, model_name=name, detail=False, attrs=attrs)
            pred = {s.shot_id: (s.stroke.shot_type or (s.stroke.candidates[0].shot_type if s.stroke.candidates else None))
                    if s.stroke else None for s in res.shots}
            t = {s.shot_id: s.t_ms for s in events.shots}
            used: set[str] = set()
            ok = matched = 0
            for r in vl:
                c = [i for i in t if i not in used and abs(t[i] - r["t_ms"]) <= a.window_ms]
                if not c:
                    conf[name][(r["stroke"], "<no hit>")] += 1
                    continue
                i = min(c, key=lambda i: abs(t[i] - r["t_ms"]))
                used.add(i)
                matched += 1
                ok += pred.get(i) == r["stroke"]
                if pred.get(i) != r["stroke"]:
                    conf[name][(r["stroke"], pred.get(i))] += 1
            total[name][0] += ok
            total[name][1] += matched
            line.append(f"{name.split()[0]} {ok}/{matched}")
        print("  " + " | ".join(line))
    for name, (ok, n) in total.items():
        print(f"{name}: {ok}/{n} right among matched hits ({100 * ok / max(n, 1):.0f}%)")
        print("   errors:", conf[name].most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
