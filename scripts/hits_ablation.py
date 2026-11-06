"""Why did high-confidence hits drop? Re-run tracking + events offline on a finished run.

    python scripts/hits_ablation.py --clip ae26_ms_f --clip ko26_court3   # ~/rallyiq-data/runs/latest
    python scripts/hits_ablation.py ~/rallyiq-data/runs/latest/data/videos/<video_id>

Uses the saved raw shuttle candidates (``work/shuttle_candidates.json``), players and
calibration, so no model runs and it takes seconds. For each combination of

* shuttle peak threshold (which candidates the tracker may use): 0.2 (old) / 0.1 (new)
* events ``min_shuttle_conf`` (which track points the hit rules see): 0.2 (default) / 0.1 / 0.05

it prints the number of rallies, hits, ``ok`` hits and the reasons on the hits that are
not ok. Comparing rows tells which setting costs the ``ok`` hits. Label-free counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import CourtCalibration, PlayerTracks, VideoMeta
from rallyiq.events.adapters import (
    events_to_contract,
    poses_from_contract,
    shuttle_from_contract,
)
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from rallyiq.perception.shuttle.stage import (
    CANDIDATES_FILE,
    build_track,
    court_lock,
    tracker_config,
)

DEFAULT_RUN = Path.home() / "rallyiq-data" / "runs" / "latest"
PEAK = (0.2, 0.1)
MIN_CONF = (0.2, 0.1, 0.05)


def _load(path: Path, model):
    return model.model_validate_json(path.read_text()) if path.exists() else None


def ablate(video_dir: Path, peaks=PEAK, min_confs=MIN_CONF) -> list[dict]:
    art = video_dir / "artifacts"
    meta = _load(art / "probe.json", VideoMeta)
    calib = _load(art / "calibration.json", CourtCalibration)
    players = _load(art / "players.json", PlayerTracks)
    cand_path = video_dir / "work" / CANDIDATES_FILE
    if meta is None or not cand_path.exists():
        raise SystemExit(f"{video_dir}: needs artifacts/probe.json and work/{CANDIDATES_FILE}")
    doc = json.loads(cand_path.read_text())
    w, h = doc["frame_size"]
    cands = {int(f): [tuple(c) for c in v["candidates"]] for f, v in doc["frames"].items()}
    region, support, _ = court_lock(calib, players)
    poses = poses_from_contract(players)
    fmt = players.match_format if players is not None else "singles"
    rows = []
    for peak in peaks:
        kept = {f: [c for c in cs if c[2] >= peak] for f, cs in cands.items()}
        track, _ = build_track(meta, kept, w, h, doc.get("detector"), tracker_config(None, peak), region, support)
        det = [p for p in track.points if p.source == "detected"]
        for mc in min_confs:
            cfg = replace(EventsConfig(), min_shuttle_conf=mc)
            rallies = segment_events(shuttle_from_contract(track), poses, cfg, video_end_ms=float(meta.duration_ms))
            ev = events_to_contract(meta.video_id, rallies, fmt)
            not_ok = [s for s in ev.shots if s.status != "ok"]
            rows.append({
                "peak_threshold": peak, "events_min_shuttle_conf": mc,
                "detected_points": len(det),
                "points_below_min_conf": sum(p.confidence < mc for p in track.points),
                "rallies": len(ev.rallies), "hits": len(ev.shots), "hits_ok": len(ev.shots) - len(not_ok),
                "mean_hit_conf": round(sum(s.confidence for s in ev.shots) / len(ev.shots), 3) if ev.shots else None,
                "reasons_not_ok": dict(Counter(r for s in not_ok for r in s.reasons).most_common()),
                "not_ok_without_reason": sum(1 for s in not_ok if not s.reasons),
            })
    return rows


def render(name: str, rows: list[dict]) -> str:
    L = [f"## {name}", "", "| peak thr | events min conf | detected pts | pts < min conf | rallies | hits | ok | "
         "mean conf | not-ok reasons |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        reasons = ", ".join(f"{k} {v}" for k, v in r["reasons_not_ok"].items())
        if r["not_ok_without_reason"]:
            reasons += f", (low confidence only) {r['not_ok_without_reason']}"
        L.append(f"| {r['peak_threshold']} | {r['events_min_shuttle_conf']} | {r['detected_points']} | "
                 f"{r['points_below_min_conf']} | {r['rallies']} | {r['hits']} | {r['hits_ok']} | "
                 f"{r['mean_hit_conf']} | {reasons} |")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path, nargs="*")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--clip", action="append", default=[])
    ap.add_argument("--json", type=Path, help="also write the rows here")
    a = ap.parse_args(argv)
    targets = [(d.name, d.expanduser()) for d in a.video_dir]
    if a.clip:
        res = json.loads((a.run.expanduser() / "results.json").read_text())
        by_clip = {r["clip_id"]: Path(r["artifacts"]).parent for r in res}
        targets += [(c, by_clip[c]) for c in a.clip]
    if not targets:
        ap.error("give --clip or a video folder")
    out = {name: ablate(d) for name, d in targets}
    print("\n\n".join(render(n, rows) for n, rows in out.items()))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
