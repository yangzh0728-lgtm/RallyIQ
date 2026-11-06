"""Shuttle trajectory helpers: cleaning, velocities and turn angles."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import EventsConfig
from .types import ShuttleObs

Vec = tuple[float, float]


@dataclass(frozen=True)
class Point:
    """A usable shuttle observation (visible, with a pixel position)."""

    frame: int
    t_ms: float
    x: float
    y: float
    confidence: float
    interpolated: bool
    court_xy: Optional[tuple[float, float]]

    @property
    def xy(self) -> Vec:
        return (self.x, self.y)


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm(v: Vec) -> float:
    return math.hypot(v[0], v[1])


def angle_deg(a: Vec, b: Vec) -> Optional[float]:
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def usable_points(
    shuttle: Sequence[ShuttleObs], cfg: EventsConfig, scale_px: float
) -> list[Point]:
    """Visible observations sorted by frame, with lone outliers removed.

    A point is an outlier when it is far from both neighbours while the
    neighbours agree with each other: a typical false detection.
    """
    pts = [
        Point(o.frame, o.t_ms, float(o.x), float(o.y), o.confidence, o.interpolated, o.court_xy)
        for o in sorted(shuttle, key=lambda o: o.frame)
        if o.usable and o.confidence >= cfg.min_shuttle_conf
    ]
    if len(pts) < 3:
        return pts
    jump = cfg.outlier_jump_rel * scale_px
    keep = [True] * len(pts)
    for i in range(1, len(pts) - 1):
        a, p, b = pts[i - 1], pts[i], pts[i + 1]
        if b.frame - a.frame > 4:
            continue  # neighbours too far apart in time to judge
        if dist(p.xy, a.xy) > jump and dist(p.xy, b.xy) > jump and dist(a.xy, b.xy) < jump:
            keep[i] = False
    return [p for p, k in zip(pts, keep) if k]


class Track:
    """Usable points with time lookup and windowed velocities."""

    def __init__(self, points: list[Point]):
        self.points = points
        self.times = [p.t_ms for p in points]

    def __len__(self) -> int:
        return len(self.points)

    def index_at_or_after(self, t_ms: float) -> int:
        return bisect_left(self.times, t_ms)

    def _velocity(self, i: int, j: int) -> Optional[Vec]:
        """Least-squares velocity (px/s) over points i..j inclusive.

        A line fit rather than an endpoint difference, so a few pixels of
        detection jitter do not turn into large direction errors.
        """
        if j < i:
            i, j = j, i
        pts = self.points[i : j + 1]
        if len(pts) < 2:
            return None
        ts = [p.t_ms / 1000.0 for p in pts]
        tm = sum(ts) / len(ts)
        den = sum((t - tm) ** 2 for t in ts)
        if den <= 0:
            return None
        xm = sum(p.x for p in pts) / len(pts)
        ym = sum(p.y for p in pts) / len(pts)
        vx = sum((t - tm) * (p.x - xm) for t, p in zip(ts, pts)) / den
        vy = sum((t - tm) * (p.y - ym) for t, p in zip(ts, pts)) / den
        return (vx, vy)

    def incoming(self, i: int, window_ms: float, max_window_ms: float) -> Optional[Vec]:
        """Velocity (px/s) arriving at point i, from the earliest point inside the window.

        If no point lies inside the window (occlusion), fall back to the
        nearest earlier point within ``max_window_ms``.
        """
        if i <= 0:
            return None
        t = self.times[i]
        j = self.index_at_or_after(t - window_ms)
        if j >= i:
            j = i - 1
            if t - self.times[j] > max_window_ms:
                return None
        return self._velocity(j, i)

    def outgoing(self, i: int, window_ms: float, max_window_ms: float) -> Optional[Vec]:
        if i >= len(self.points) - 1:
            return None
        t = self.times[i]
        j = self.index_at_or_after(t + window_ms + 1e-6) - 1
        if j <= i:
            j = i + 1
            if self.times[j] - t > max_window_ms:
                return None
        return self._velocity(i, j)

    def gap_around(self, i: int) -> float:
        """Largest time gap (ms) to the immediate neighbours of point i."""
        g = 0.0
        if i > 0:
            g = max(g, self.times[i] - self.times[i - 1])
        if i < len(self.points) - 1:
            g = max(g, self.times[i + 1] - self.times[i])
        return g

    def quality(self, i: int, window_ms: float, fps_ms: float, before: bool = True) -> float:
        """Share of frames around point i backed by a real, confident detection.

        ``before=False`` looks only at point i and after it (e.g. a serve,
        where nothing is in play before the hit).
        """
        t = self.times[i]
        lo = self.index_at_or_after(t - window_ms) if before else i
        hi = self.index_at_or_after(t + window_ms + 1e-6)
        expected = max(1.0, (2 if before else 1) * window_ms / fps_ms + 1)
        score = sum(
            0.0 if p.interpolated else p.confidence for p in self.points[lo:hi]
        )
        return max(0.0, min(1.0, score / expected))
