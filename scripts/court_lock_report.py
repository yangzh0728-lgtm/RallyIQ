"""Label-free report of the target-court lock on a finished pipeline run.

    python scripts/court_lock_report.py ~/rallyiq-data/runs/latest/data/videos/<video_id>
    python scripts/court_lock_report.py <dir> --json

``<dir>`` is a video folder of the store (it holds ``artifacts/`` and the shuttle
stage's ``shuttle_candidates.json``). Meant for the multi-court clips
(``ko26_court2`` / ``ko26_court3`` in ``evaluation/clips.yaml``) until they have labels.

It reports:

* calibration: how many other courts were found per camera segment;
* players: detections whose feet are outside the target play area (should be ~0);
* shuttle: the raw candidates are re-tracked with the court lock off and on, and the
  points only the unlocked tracker keeps are split into "over another court" and
  "elsewhere". Those are the trajectories the lock removed. This is a count, not
  accuracy; a human look at a few of them is still needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import CourtCalibration, PlayerTracks, ShuttleTrack, VideoMeta  # noqa: E402
from rallyiq.perception.players.geometry import PlayArea  # noqa: E402
from rallyiq.perception.shuttle.stage import CANDIDATES_FILE, build_track, court_lock  # noqa: E402
from rallyiq.perception.target_court import SegmentRegions  # noqa: E402


def _load(path: Path, model):
    return model.model_validate_json(path.read_text()) if path.exists() else None


def report(video_dir: Path) -> dict:
    art = video_dir / "artifacts"
    meta = _load(art / "probe.json", VideoMeta)
    cal = _load(art / "calibration.json", CourtCalibration)
    players = _load(art / "players.json", PlayerTracks)
    shuttle = _load(art / "shuttle.json", ShuttleTrack)
    out: dict = {"video_dir": str(video_dir)}

    if cal is not None:
        out["calibration"] = [
            {"frames": [s.start_frame, s.end_frame], "status": s.status, "method": s.method,
             "other_courts": len(s.other_courts)}
            for s in cal.segments
        ]

    if players is not None:
        area = PlayArea()
        n = outside = 0
        for f in players.frames:
            for p in f.players:
                if p.interpolated or p.foot_court is None:
                    continue
                n += 1
                outside += area.distance_outside(p.foot_court) > 0
        out["players"] = {"calibrated_detections": n, "outside_play_area": outside,
                          "outside_rate": round(outside / n, 4) if n else None}

    cand_path = video_dir / "work" / CANDIDATES_FILE
    if not cand_path.exists():
        cand_path = next(video_dir.rglob(CANDIDATES_FILE), cand_path)
    if meta is not None and cand_path.exists():
        doc = json.loads(cand_path.read_text())
        w, h = doc["frame_size"]
        cands = {int(f): [tuple(c) for c in v["candidates"]] for f, v in doc["frames"].items()}
        off, _ = build_track(meta, cands, w, h, doc.get("detector"))
        region, support, mode = court_lock(cal, players)
        on, _ = build_track(meta, cands, w, h, doc.get("detector"), None, region, support)
        kept = {p.frame: p for p in on.points if p.source == "detected"}
        on_frames = set(kept)

        def same(p) -> bool:
            q = kept.get(p.frame)
            return q is not None and abs(q.x - p.x) < 1.0 and abs(q.y - p.y) < 1.0

        removed = [p for p in off.points if p.source == "detected" and not same(p)]
        regions = SegmentRegions(cal)
        over_other = 0
        for p in removed:
            r = regions.for_frame(p.frame)
            if r is not None and r.others and r.air_weight(p.x, p.y) == 0.0:
                over_other += 1
        out["shuttle"] = {
            "lock_mode": mode,
            "detected_points_unlocked": sum(p.source == "detected" for p in off.points),
            "detected_points_locked": len(on_frames),
            "removed_points": len(removed),
            "removed_over_other_court": over_other,
            "removed_example_frames": [p.frame for p in removed[:: max(len(removed) // 10, 1)]][:10],
            "saved_track_points": len(shuttle.points) if shuttle else None,
        }
    else:
        out["shuttle"] = {"note": f"no {CANDIDATES_FILE} found; run the shuttle stage with weights first"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()
    reps = [report(d.expanduser()) for d in args.video_dir]
    if args.json:
        print(json.dumps(reps, indent=2, ensure_ascii=False))
        return
    for r in reps:
        print(f"== {r['video_dir']}")
        for s in r.get("calibration", []):
            print(f"  calibration frames {s['frames']}: {s['status']} ({s['method']}), other courts: {s['other_courts']}")
        if "players" in r:
            p = r["players"]
            print(f"  players: {p['outside_play_area']} of {p['calibrated_detections']} detections outside the play area")
        s = r["shuttle"]
        if "note" in s:
            print(f"  shuttle: {s['note']}")
        else:
            print(f"  shuttle ({s['lock_mode']}): {s['detected_points_unlocked']} -> {s['detected_points_locked']} "
                  f"detected points; removed {s['removed_points']} ({s['removed_over_other_court']} over another "
                  f"court), e.g. frames {s['removed_example_frames']}")


if __name__ == "__main__":
    main()
