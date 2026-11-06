"""Offline clean-up after tracking: fill short gaps and flag them.

Filled frames are marked ``interpolated=True`` and get reduced confidence so
they are never mistaken for observations.
"""

from __future__ import annotations

import numpy as np

from .geometry import court_side
from .models import PlayerFrameResult, PlayerState


def _lerp(a: np.ndarray | None, b: np.ndarray | None, t: float) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (1 - t) * a + t * b


def fill_short_gaps(frames: list[PlayerFrameResult], max_gap: int = 6) -> list[PlayerFrameResult]:
    """Linearly interpolate a player across gaps of at most ``max_gap`` processed
    frames when the same ``track_id`` is seen on both sides of the gap."""
    order = sorted(frames, key=lambda f: f.frame)
    last_seen: dict[str, tuple[int, PlayerState]] = {}
    for fi, frame in enumerate(order):
        for p in list(frame.players):
            if p.player_id is None or p.interpolated:
                continue
            prev = last_seen.get(p.player_id)
            if prev is not None:
                pf, ps = prev
                gap = fi - pf - 1
                p_frame = frame.frame
                if 0 < gap <= max_gap and ps.track_id == p.track_id:
                    for g in range(1, gap + 1):
                        f = order[pf + g]
                        if any(q.player_id == p.player_id for q in f.players):
                            continue
                        span = p_frame - order[pf].frame
                        t = (f.frame - order[pf].frame) / span
                        fc = _lerp(ps.foot_court, p.foot_court, t)
                        kps = _lerp(ps.keypoints, p.keypoints, t)
                        kps[:, 2] = np.minimum(ps.keypoints[:, 2], p.keypoints[:, 2]) * 0.5
                        f.players.append(
                            PlayerState(
                                player_id=p.player_id,
                                track_id=p.track_id,
                                side=court_side(fc) if fc is not None else ps.side,
                                bbox=_lerp(ps.bbox, p.bbox, t),
                                keypoints=kps,
                                foot_px=_lerp(ps.foot_px, p.foot_px, t),
                                foot_court=fc,
                                on_court=ps.on_court if ps.on_court == p.on_court else None,
                                confidence=0.5 * min(ps.confidence, p.confidence),
                                interpolated=True,
                            )
                        )
            last_seen[p.player_id] = (fi, p)
    for f in frames:
        f.players.sort(key=lambda p: p.player_id or "")
    return frames
