"""Entry point: shuttle track + poses -> rallies with hit candidates."""

from __future__ import annotations

import statistics
from typing import Optional, Sequence

from .config import EventsConfig
from .hits import R_SAME_HITTER, Candidate, detect_hits
from .players import PoseIndex
from .rallies import segment_rallies
from .trajectory import Track, usable_points
from .types import HitEvent, PoseFrame, RallySegment, ReviewStatus, ShuttleObs

R_NO_HITS = "no_hits_detected"
R_POSSIBLE_MISSED_HIT = "possible_missed_hit"
R_HITS_NEED_REVIEW = "hits_need_review"


def _fps_ms(shuttle: Sequence[ShuttleObs]) -> float:
    obs = sorted(shuttle, key=lambda o: o.frame)
    steps = [
        (b.t_ms - a.t_ms) / (b.frame - a.frame)
        for a, b in zip(obs, obs[1:])
        if b.frame > a.frame
    ]
    return statistics.median(steps) if steps else 1000.0 / 30.0


def segment_events(
    shuttle: Sequence[ShuttleObs],
    poses: Sequence[PoseFrame],
    cfg: Optional[EventsConfig] = None,
    video_end_ms: Optional[float] = None,
) -> list[RallySegment]:
    cfg = cfg or EventsConfig()
    pose_index = PoseIndex(poses, cfg)
    scale = pose_index.global_height
    track = Track(usable_points(shuttle, cfg, scale))
    if video_end_ms is None:
        ends = [o.t_ms for o in shuttle] + [f.t_ms for f in poses]
        video_end_ms = max(ends) if ends else 0.0
    fps_ms = _fps_ms(shuttle)

    rallies: list[RallySegment] = []
    for n, span in enumerate(segment_rallies(track, cfg, pose_index.scale_at, video_end_ms), start=1):
        rally_id = f"R{n:03d}"
        cands = detect_hits(track, span, pose_index, cfg, fps_ms)
        hits = [_to_hit(c, track, rally_id, k, cfg) for k, c in enumerate(cands, start=1)]
        s, e = track.points[span.start], track.points[span.end]
        rally = RallySegment(
            rally_id=rally_id,
            start_frame=s.frame,
            end_frame=e.frame,
            start_ms=s.t_ms,
            end_ms=e.t_ms,
            end_reason=span.end_reason,
            hits=hits,
        )
        _decide_rally(rally, cfg)
        rallies.append(rally)
    return rallies


def _to_hit(c: Candidate, track: Track, rally_id: str, k: int, cfg: EventsConfig) -> HitEvent:
    pt = track.points[c.idx]
    best = c.best
    hitter = best.player_id if best and best.score > 0 else None
    status = (
        ReviewStatus.OK
        if c.confidence >= cfg.ok_threshold and not c.reasons and hitter is not None
        else ReviewStatus.NEEDS_REVIEW
    )
    return HitEvent(
        hit_id=f"{rally_id}_S{k:02d}",
        rally_id=rally_id,
        index_in_rally=k,
        frame=pt.frame,
        t_ms=pt.t_ms,
        hitter_id=hitter,
        shuttle_xy_px=(round(pt.x, 1), round(pt.y, 1)),
        shuttle_court_xy=pt.court_xy,
        confidence=c.confidence,
        status=status,
        reasons=list(c.reasons),
        scores={
            "turn_deg": round(c.turn_deg, 1),
            "change": round(c.change, 3),
            "sharpness": round(c.sharpness, 3),
            "proximity": round(best.score, 3) if best else 0.0,
            "hitter_distance_rel": round(best.distance_rel, 3) if best else -1.0,
            "track_quality": round(c.track_quality, 3),
            "impulse_rel": round(c.impulse, 3),
        },
        hitter_court_xy=(best.pose.foot_court_xy if hitter is not None else None),
    )


def _decide_rally(r: RallySegment, cfg: EventsConfig) -> None:
    if not r.hits:
        r.reasons.append(R_NO_HITS)
        r.confidence = 0.0
        r.status = ReviewStatus.NEEDS_REVIEW
        return
    times = [r.start_ms] + [h.t_ms for h in r.hits] + [r.end_ms]
    if max(b - a for a, b in zip(times, times[1:])) > cfg.long_exchange_ms:
        r.reasons.append(R_POSSIBLE_MISSED_HIT)
    if any(R_SAME_HITTER in h.reasons for h in r.hits) and R_POSSIBLE_MISSED_HIT not in r.reasons:
        r.reasons.append(R_POSSIBLE_MISSED_HIT)
    if any(h.status != ReviewStatus.OK for h in r.hits):
        r.reasons.append(R_HITS_NEED_REVIEW)
    r.confidence = round(statistics.mean(h.confidence for h in r.hits), 3)
    r.status = ReviewStatus.OK if not r.reasons else ReviewStatus.NEEDS_REVIEW
