"""Rally segmentation: when is the shuttle in play?

A rally is a stretch where the shuttle is seen and moving. It ends when the
shuttle is unseen for ``end_gap_ms`` (lost / out of frame), when it stays
(nearly) still for ``static_ms`` (on the floor, picked up), or at the end of
the video. Short occlusions inside a rally are bridged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import EventsConfig
from .trajectory import Track, dist
from .types import RallyEndReason

# (frame, (x, y)) -> local pixel scale (player height in px)
ScaleFn = Callable[[int, tuple[float, float]], float]


@dataclass(frozen=True)
class RallySpan:
    start: int  # index into Track.points (inclusive)
    end: int  # inclusive
    end_reason: RallyEndReason


def _static_flags(track: Track, cfg: EventsConfig, scale_at: ScaleFn) -> list[bool]:
    """Point i is static if the shuttle barely moves over the next ~200 ms."""
    pts = track.points
    flags = []
    for i, p in enumerate(pts):
        j = track.index_at_or_after(p.t_ms + 200.0)
        j = min(j, len(pts) - 1)
        if j <= i or pts[j].t_ms - p.t_ms > cfg.end_gap_ms:
            j = i + 1 if i + 1 < len(pts) and pts[i + 1].t_ms - p.t_ms <= cfg.end_gap_ms else i
        if j == i:
            flags.append(False)
            continue
        dt = (pts[j].t_ms - p.t_ms) / 1000.0
        thresh = cfg.static_speed_rel * scale_at(p.frame, p.xy)  # px per second
        flags.append(dist(pts[j].xy, p.xy) / dt < thresh)
    return flags


def _moves(track: Track, k: int, cfg: EventsConfig, scale_at: ScaleFn) -> bool:
    """Does the shuttle leave point k within the next ~100 ms (beyond pixel jitter)?"""
    pts = track.points
    j = min(max(track.index_at_or_after(pts[k].t_ms + 100.0), k + 1), len(pts) - 1)
    a, b = pts[k], pts[j]
    dt = (b.t_ms - a.t_ms) / 1000.0
    if j == k or dt <= 0 or b.t_ms - a.t_ms >= cfg.end_gap_ms:
        return True  # cannot tell; stop refining
    return dist(a.xy, b.xy) / dt >= cfg.static_speed_rel * scale_at(a.frame, a.xy)


def segment_rallies(
    track: Track, cfg: EventsConfig, scale_at: ScaleFn, video_end_ms: float
) -> list[RallySpan]:
    pts = track.points
    n = len(pts)
    if n == 0:
        return []
    static = _static_flags(track, cfg, scale_at)

    spans: list[RallySpan] = []
    i = 0
    while i < n:
        # Skip still or isolated points until the shuttle starts moving.
        if static[i]:
            i += 1
            continue
        # The static flag looks ~200 ms ahead, so the first "moving" points can
        # still be the shuttle at rest in the server's hand. Advance to the last
        # point before it actually moves: that is where the serve is struck.
        start = i - 1 if i > 0 and static[i - 1] and pts[i].t_ms - pts[i - 1].t_ms < cfg.end_gap_ms else i
        while start + 1 < n and not _moves(track, start, cfg, scale_at):
            start += 1
        i = max(i, start)
        end_reason = None
        j = i
        while True:
            if j == n - 1:
                end_reason = (
                    RallyEndReason.VIDEO_END
                    if video_end_ms - pts[j].t_ms < cfg.end_gap_ms
                    else RallyEndReason.SHUTTLE_LOST
                )
                break
            if pts[j + 1].t_ms - pts[j].t_ms >= cfg.end_gap_ms:
                end_reason = RallyEndReason.SHUTTLE_LOST
                break
            if static[j + 1]:
                # Is this a sustained stop, or a slow moment (e.g. lob apex)?
                k = j + 1
                while k < n and static[k] and pts[k].t_ms - pts[k - 1].t_ms < cfg.end_gap_ms:
                    k += 1
                last_static = k - 1
                if pts[last_static].t_ms - pts[j + 1].t_ms >= cfg.static_ms or k == n:
                    end_reason = RallyEndReason.SHUTTLE_STATIC
                    j = j + 1  # the first still point is where it landed
                    break
                j = k - 1  # brief slow-down, keep going
            j += 1
        spans.append(RallySpan(start, j, end_reason))
        i = j + 1

    if cfg.bridge_gap_ms > 0:
        spans = _bridge(spans, pts, cfg.bridge_gap_ms)
    return [
        s
        for s in spans
        if pts[s.end].t_ms - pts[s.start].t_ms >= cfg.min_rally_ms
        and s.end - s.start + 1 >= cfg.min_rally_points
    ]


def _bridge(spans: list[RallySpan], pts: list, gap_ms: float) -> list[RallySpan]:
    """Join consecutive spans whose gap (last point of one to first of the next) is at most ``gap_ms``."""
    out: list[RallySpan] = []
    for sp in spans:
        if out and pts[sp.start].t_ms - pts[out[-1].end].t_ms <= gap_ms:
            out[-1] = RallySpan(out[-1].start, sp.end, sp.end_reason)
        else:
            out.append(sp)
    return out
