"""Label-free diagnosis of where shuttle detections are lost (numpy only).

Given the raw per-frame candidates of the heatmap detector, this answers:

* does the network see anything at all? (frames with a raw peak above a threshold)
* is what it sees over the target court? (peaks inside the court-lock air box)
* how much of that survives the tracker? (detected track points)
* how much of the rally time is covered? (when the events stage found rallies)

Comparing these numbers across detector settings (ROI crop on/off, peak threshold)
tells whether a weak camera is a detector problem (no peaks), a court-lock problem
(peaks rejected as another court's) or a tracker problem (peaks that never form
trajectories). Used by ``scripts/shuttle_sweep.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ...contracts import CourtCalibration, EventsResult, PlayerTracks, VideoMeta
from .tracker import Candidate, TrackerConfig


@dataclass
class Coverage:
    variant: str
    threshold: float
    frames: int
    raw_frac: float  # frames with any peak >= threshold
    raw_target_frac: float | None  # ... of which at least one is in the target air box
    detected_frac: float  # tracked points with source "detected"
    rally_frames: int | None
    rally_detected_frac: float | None  # detected points inside rallies / rally frames
    max_score_p50: float | None  # median of the per-frame best peak (frames with a peak)
    max_score_p90: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def filter_candidates(cands: dict[int, Sequence[Candidate]], threshold: float) -> dict[int, list[Candidate]]:
    return {f: [c for c in cs if c[2] >= threshold] for f, cs in cands.items()}


def rally_frames(events: EventsResult | None) -> set[int] | None:
    if events is None or not events.rallies:
        return None
    out: set[int] = set()
    for r in events.rallies:
        out.update(range(r.start_frame, r.end_frame + 1))
    return out


def coverage(
    variant: str,
    meta: VideoMeta,
    candidates: dict[int, Sequence[Candidate]],
    size: tuple[int, int],
    threshold: float,
    calibration: CourtCalibration | None = None,
    players: PlayerTracks | None = None,
    events: EventsResult | None = None,
    court_lock: bool = True,
) -> Coverage:
    from .stage import build_track
    from .stage import court_lock as make_lock

    n = max(meta.frame_count, 1)
    cands = filter_candidates(candidates, threshold)
    raw = {f for f, cs in cands.items() if cs}
    region = support = None
    if court_lock:
        region, support, _mode = make_lock(calibration, players)
    sx, sy = meta.width / size[0], meta.height / size[1]
    raw_target = None
    if region is not None:
        raw_target = sum(
            1 for f in raw if any(region(f, x * sx, y * sy) > 0.0 for x, y, _s in cands[f])
        ) / n
    # Let weak candidates down to the threshold form trajectories too.
    cfg = TrackerConfig(weak_score=min(TrackerConfig.weak_score, threshold))
    track, _sel = build_track(meta, cands, size[0], size[1], variant, cfg, region, support)
    det = {p.frame for p in track.points if p.source == "detected"}
    rf = rally_frames(events)
    best = np.array([max(c[2] for c in cs) for cs in cands.values() if cs], dtype=float)
    return Coverage(
        variant=variant,
        threshold=threshold,
        frames=n,
        raw_frac=round(len(raw) / n, 3),
        raw_target_frac=None if raw_target is None else round(raw_target, 3),
        detected_frac=round(len(det) / n, 3),
        rally_frames=len(rf) if rf is not None else None,
        rally_detected_frac=round(len(det & rf) / len(rf), 3) if rf else None,
        max_score_p50=round(float(np.percentile(best, 50)), 3) if best.size else None,
        max_score_p90=round(float(np.percentile(best, 90)), 3) if best.size else None,
    )


def roi_report(meta: VideoMeta, calibration: CourtCalibration | None) -> list[dict]:
    """Per calibrated segment: how big the court is in the frame and what the auto ROI would do."""
    from .detector import court_roi

    out = []
    for s in (calibration.segments if calibration else []):
        row: dict = {"frames": [s.start_frame, s.end_frame], "status": str(getattr(s.status, "value", s.status)),
                     "other_courts": len(s.other_courts)}
        if s.image_corners:
            pts = np.asarray(s.image_corners, dtype=float)
            row["court_w_frac"] = round(float(np.ptp(pts[:, 0])) / meta.width, 3)
            row["court_h_frac"] = round(float(np.ptp(pts[:, 1])) / meta.height, 3)
            roi = court_roi(s.image_corners, meta.width, meta.height)
            row["roi"] = list(roi) if roi else None
            row["roi_zoom"] = round(min(meta.width / (roi[2] - roi[0]), meta.height / (roi[3] - roi[1])), 2) if roi else 1.0
        out.append(row)
    return out


def format_table(rows: Sequence[Coverage]) -> str:
    def pct(v):
        return "–" if v is None else f"{v:.0%}"

    lines = [
        "| variant | thr | raw peak | on target court | tracked | in rallies | best score p50/p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sc = "–" if r.max_score_p50 is None else f"{r.max_score_p50:.2f} / {r.max_score_p90:.2f}"
        lines.append(f"| {r.variant} | {r.threshold:.2f} | {pct(r.raw_frac)} | {pct(r.raw_target_frac)} | "
                     f"{pct(r.detected_frac)} | {pct(r.rally_detected_frac)} | {sc} |")
    return "\n".join(lines)
