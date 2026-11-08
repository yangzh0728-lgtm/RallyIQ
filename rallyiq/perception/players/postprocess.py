"""Offline clean-up after tracking: fill short gaps and flag them.

Filled frames are marked ``interpolated=True`` and get reduced confidence so
they are never mistaken for observations.
"""

from __future__ import annotations

import numpy as np

from .geometry import PlayArea, court_side
from .models import PlayerFrameResult, PlayerState
from .tracker import TEAM_OF


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
                                team=ps.team,
                            )
                        )
            last_seen[p.player_id] = (fi, p)
    for f in frames:
        f.players.sort(key=lambda p: p.player_id or "")
    return frames


def drop_offcourt_runs(
    frames: list[PlayerFrameResult],
    play_area: PlayArea | None = None,
    min_frames: int = 3,
    min_outside_frac: float = 0.6,
    min_median_outside_m: float = 0.5,
    still_min_frames: int = 90,
    still_outside_frac: float = 0.8,
    still_outside_m: float = 0.3,
    still_extent_m: float = 1.0,
) -> int:
    """Track-level court lock: remove whole runs (same ``player_id`` and
    ``track_id``) that mostly stand outside the target play area.

    Per-frame filtering is soft near the margin so a lunge past the run-off is
    not lost. A person on the next court can sit in that soft band for a long
    time though, and over a whole run the difference is clear: our players
    spend nearly all their time inside.

    A bystander right next to the court (an umpire or spectator by the net post)
    sits inside the run-off margins, so a second test drops runs that stay still
    (``still_extent_m`` across, 5th-95th percentile) outside the court lines
    themselves for at least ``still_min_frames`` observations. Returns the
    number of runs dropped.
    """
    area = play_area or PlayArea()
    lines = PlayArea(margin_side=0.0, margin_end=0.0)
    runs: dict[tuple[str, int], list[np.ndarray]] = {}
    for f in frames:
        for p in f.players:
            if p.player_id is None or p.interpolated or p.foot_court is None:
                continue
            runs.setdefault((p.player_id, p.track_id), []).append(np.asarray(p.foot_court, dtype=np.float64))
    bad = set()
    for key, pts in runs.items():
        if len(pts) < min_frames:
            continue
        arr = np.asarray([area.distance_outside(xy) for xy in pts])
        if (arr > 0).mean() >= min_outside_frac and float(np.median(arr)) >= min_median_outside_m:
            bad.add(key)
            continue
        if len(pts) < still_min_frames:
            continue
        xy = np.asarray(pts)
        out = np.asarray([lines.distance_outside(q) for q in xy])
        extent = np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0)
        if (out >= still_outside_m).mean() >= still_outside_frac and np.all(extent < still_extent_m):
            bad.add(key)
    if bad:
        for f in frames:
            f.players = [p for p in f.players if (p.player_id, p.track_id) not in bad]
    return len(bad)


def apply_relabels(frames: list[PlayerFrameResult], relabels: dict[int, str]) -> int:
    """Rename whole tracks the tracker moved to another slot (``PlayerTracker.relabels``).

    Frames emitted before the move carry the old id; afterwards the track's
    earlier frames follow it. Where that leaves two people with one id in a
    frame, the moved track wins (the slot's earlier holder was a bystander or
    was lost). Returns the number of observations renamed.
    """
    if not relabels:
        return 0
    n = 0
    for f in frames:
        for p in f.players:
            new = relabels.get(p.track_id)
            if new is not None and p.player_id != new:
                p.player_id = new
                p.team = TEAM_OF.get(new, p.team)
                n += 1
        ids = [p.player_id for p in f.players]
        if len(ids) != len(set(ids)):
            keep: dict[str, PlayerState] = {}
            for p in f.players:
                cur = keep.get(p.player_id)
                if cur is None or (p.track_id in relabels and cur.track_id not in relabels):
                    keep[p.player_id] = p
            f.players = list(keep.values())
    return n
