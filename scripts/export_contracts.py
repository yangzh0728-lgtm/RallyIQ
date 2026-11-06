"""Regenerate contracts/schemas/*.schema.json and contracts/examples/*.json.

    python scripts/export_contracts.py          # write
    python scripts/export_contracts.py --check  # fail if files are stale (CI)
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rallyiq import contracts as c
from rallyiq.api.corrections import apply_corrections
from rallyiq.contracts.geometry import project_boundary_lines, project_court_lines
from rallyiq.pipeline import demo
from rallyiq.pipeline.overlay import COCO17_SKELETON
from rallyiq.report import compute_match_stats
from rallyiq.strokes import classify, merge_strokes
from rallyiq.report.coach import OfflineProvider, build_report

ROOT = Path(__file__).resolve().parents[1] / "contracts"

SCHEMAS = [
    c.VideoMeta, c.CourtCalibration, c.PlayerTracks, c.ShuttleTrack, c.EventsResult, c.ShotEvent, c.Rally,
    c.StrokesResult,
    c.CorrectionIn, c.Correction, c.OverlayData, c.OverlayFrame, c.Job, c.Video, c.HeatmapData,
    c.MatchFormatSetting,
    c.MatchStats, c.CoachReport, c.MatchReport,
]


def examples() -> dict[str, object]:
    meta = c.VideoMeta(video_id="vid_example", filename="match.mp4", fps=30.0, frame_count=75, width=1920,
                       height=1080, duration_ms=2500, codec="h264")
    long_meta = meta.model_copy(update={"frame_count": 450, "duration_ms": 15000})
    cal = demo.demo_calibration(meta)
    players = demo.demo_players(long_meta, cal)
    shuttle = demo.demo_shuttle(long_meta, players)
    events = demo.demo_events(long_meta, players)
    strokes = classify(events, shuttle, players, long_meta.fps)
    # keep examples small: 2 frames of players, ~1 s of shuttle
    players_small = players.model_copy(update={"frames": players.frames[30:32]})
    shuttle_small = shuttle.model_copy(update={"points": shuttle.points[:8],
                                               "gaps": [c.FrameGap(start_frame=60, end_frame=65, reason="occluded")]})
    overlay = c.OverlayData(
        video=meta, skeleton=COCO17_SKELETON,
        court=[c.CourtLines(start_frame=0, end_frame=74, status="ok",
                            lines=project_court_lines(cal.segments[0].homography),
                            boundary=project_boundary_lines(cal.segments[0].homography, "singles"))],
        frames=[c.OverlayFrame(frame=pf.frame, t_ms=pf.t_ms,
                               players=[c.OverlayPlayer(player_id=p.player_id, team=p.team, side=p.side, bbox=p.bbox,
                                                        keypoints=p.keypoints, foot_court=p.foot_court)
                                        for p in pf.players],
                               shuttle=next((c.OverlayShuttle(x=s.x, y=s.y, source=s.source, confidence=s.confidence)
                                             for s in shuttle.points if s.frame == pf.frame), None))
                for pf in players.frames[30:32]],
    )
    ts = datetime(2026, 11, 6, 8, 0, tzinfo=UTC)
    corrections = [
        c.Correction(correction_id="cor_1", video_id=meta.video_id, seq=1, created_at=ts, op="set_hitter",
                     payload={"shot_id": "s0002", "hitter": "P1"}),
        c.Correction(correction_id="cor_2", video_id=meta.video_id, seq=2, created_at=ts, op="delete_shot",
                     payload={"shot_id": "s0003"}),
        c.Correction(correction_id="cor_3", video_id=meta.video_id, seq=3, created_at=ts, op="add_shot",
                     payload={"frame": 100, "hitter": "P2", "shuttle_px": [900, 400]}),
        c.Correction(correction_id="cor_4", video_id=meta.video_id, seq=4, created_at=ts, op="set_shot_type",
                     payload={"shot_id": "s0001", "shot_type": "short_serve"}),
    ]
    effective, _ = apply_corrections(merge_strokes(events, strokes), corrections, long_meta)
    job = c.Job(job_id="job_example", video_id=meta.video_id, state="running", created_at=ts, started_at=ts,
                stages=[c.StageRun(name="probe", state="succeeded", progress=1.0, started_at=ts, finished_at=ts),
                        c.StageRun(name="calibration", state="succeeded", progress=1.0, started_at=ts, finished_at=ts),
                        c.StageRun(name="players", state="running", progress=0.42, started_at=ts),
                        c.StageRun(name="shuttle"), c.StageRun(name="events"), c.StageRun(name="strokes"),
                        c.StageRun(name="overlay")])
    doubles = demo.demo_players(long_meta, cal, "doubles")
    doubles_events = demo.demo_events(long_meta, doubles)
    # report: winners entered for every rally, coach from the offline rule provider
    decided = effective.model_copy(update={"rallies": [
        r.model_copy(update={"winner": "P1" if i % 3 else "P2"}) for i, r in enumerate(effective.rallies)]})
    # fingerprint hashes float-exact inputs (BLAS-dependent in the last digits): pin it for the example
    stats = compute_match_stats(decided, players, cal, long_meta).model_copy(update={"fingerprint": "0123456789abcdef"})
    coach = build_report(OfflineProvider().generate(stats), stats, OfflineProvider()).model_copy(
        update={"created_at": ts})
    report = c.MatchReport(stats=stats, coach=coach)
    return {
        "match_report": report,
        "players_doubles": doubles.model_copy(update={"frames": doubles.frames[30:31]}),
        "events_doubles": doubles_events.model_copy(update={
            "rallies": doubles_events.rallies[:1], "shots": [s for s in doubles_events.shots if s.rally_id == "r001"]}),
        "video_meta": meta, "calibration": cal, "players": players_small, "shuttle": shuttle_small,
        "events": events.model_copy(update={"rallies": events.rallies[:1],
                                            "shots": [s for s in events.shots if s.rally_id == "r001"]}),
        "strokes": strokes.model_copy(update={"shots": [s for s in strokes.shots
                                                         if any(e.shot_id == s.shot_id and e.rally_id == "r001"
                                                                for e in events.shots)]}),
        "corrections": corrections, "events_effective": effective.model_copy(
            update={"rallies": effective.rallies[:1], "shots": [s for s in effective.shots if s.rally_id == "r001"]}),
        "overlay": overlay, "job": job,
    }


def render() -> dict[Path, str]:
    out: dict[Path, str] = {}
    for m in SCHEMAS:
        out[ROOT / "schemas" / f"{m.__name__}.schema.json"] = json.dumps(m.model_json_schema(), indent=2) + "\n"
    for name, obj in examples().items():
        data = [o.model_dump(mode="json") for o in obj] if isinstance(obj, list) else obj.model_dump(mode="json")
        out[ROOT / "examples" / f"{name}.json"] = json.dumps(_round(data), indent=2) + "\n"
    return out


def _round(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: (_round_sig(v) if k == "homography" else _round(v)) for k, v in x.items()}
    return x


def _round_sig(x, sig: int = 9):
    if isinstance(x, list):
        return [_round_sig(v, sig) for v in x]
    return float(f"{x:.{sig}g}") if isinstance(x, float) else x


def _close(a, b, tol: float = 1e-6) -> bool:
    """Structural equality with float tolerance (numpy/BLAS builds differ in the last digits)."""
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol * max(1, abs(a), abs(b)) + 2e-3
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    return a == b


def _stale(p: Path, s: str) -> bool:
    if not p.exists():
        return True
    return not _close(json.loads(p.read_text()), json.loads(s))


def main() -> int:
    files = render()
    if "--check" in sys.argv:
        stale = [str(p) for p, s in files.items() if _stale(p, s)]
        if stale:
            print("stale contract files, run scripts/export_contracts.py:\n" + "\n".join(stale))
            return 1
        return 0
    for p, s in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
    print(f"wrote {len(files)} files under {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
