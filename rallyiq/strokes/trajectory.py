"""Flight-shape evidence for the learned stroke-type model (feature set ``v3`` = v2 + these).

``rise`` (features.py) says how far a flight climbed above its higher end point. Pushes and
lifts, the pair the v2 model confuses most, both start low at the front court; what tells them
apart is the *shape* of the flight: a lift climbs steeply and peaks late, a push leaves flat and
fast. These features describe the detected shuttle points between a hit and the next hit (or
1.5 s after the last hit), in the hitter's body heights so camera distance cancels out:

* ``arc``: highest point of the flight above the straight line from contact to the end point;
* ``apex_frac``: when the flight peaks, as a share of the flight time;
* ``curv``: the quadratic coefficient of image height over time (spans / s^2, + = bends down);
* ``climb_v``: upward image speed over the first quarter of the flight;
* ``slow_ratio``: image speed in the last quarter / the first quarter (fast shots decelerate);
* ``coverage``: detected points / frames of the flight (how much of it was seen).

Missing evidence is NaN.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

import numpy as np

from ..contracts import EventsResult, PlayerTracks, ShuttleTrack
from .features import StrokeFeatures, _Players, body_span

TRAJ_FEATURES = ("arc", "apex_frac", "curv", "climb_v", "slow_ratio", "coverage")
LAST_MS = 1500.0
MIN_POINTS = 5
MIN_KP = 0.3


def _speed(pts) -> float | None:
    if len(pts) < 2:
        return None
    d = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:]))
    dt = (pts[-1].t_ms - pts[0].t_ms) / 1000.0
    return d / dt if dt > 0 else None


def traj_features(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    feats: Sequence[StrokeFeatures],
    fps: float | None = None,
) -> np.ndarray:
    """One row of ``TRAJ_FEATURES`` per entry of ``feats`` (same order). ``fps`` defaults to the
    shuttle track's own frame spacing."""
    pts = sorted((p for p in (shuttle.points if shuttle else []) if p.confidence >= 0.2 and p.source == "detected"),
                 key=lambda p: p.t_ms)
    ts = [p.t_ms for p in pts]
    shots = {s.shot_id: s for s in events.shots}
    nxt_of: dict[str, float | None] = {}
    by_rally: dict[str | None, list] = {}
    for s in sorted(events.shots, key=lambda s: s.frame):
        by_rally.setdefault(s.rally_id, []).append(s)
    for rid, ss in by_rally.items():
        for k, s in enumerate(ss):
            nxt_of[s.shot_id] = ss[k + 1].t_ms if rid is not None and k + 1 < len(ss) else None
    pl = _Players(players, 3)
    if fps and fps > 0:
        frame_ms = 1000.0 / fps
    else:
        allp = sorted(shuttle.points, key=lambda p: p.frame) if shuttle and shuttle.points else []
        steps = [(b.t_ms - a.t_ms) / (b.frame - a.frame) for a, b in zip(allp, allp[1:]) if b.frame > a.frame]
        frame_ms = float(np.median(steps)) if steps else 33.3
    rows = []
    for f in feats:
        r = dict.fromkeys(TRAJ_FEATURES, math.nan)
        s = shots[f.shot_id]
        t0 = float(s.t_ms)
        t1 = nxt_of.get(f.shot_id)
        t1 = float(t1) if t1 is not None else t0 + LAST_MS
        flight = pts[bisect_left(ts, t0 + 1):bisect_left(ts, t1)]
        hitter = pl.find(s.frame, s.hitter) if s.hitter else None
        sp = body_span(hitter, MIN_KP) if hitter is not None else None
        dur = t1 - t0
        if dur > 0:
            r["coverage"] = min(1.0, len(flight) / max(1.0, dur / frame_ms))
        if len(flight) >= MIN_POINTS and sp is not None and dur > 0:
            span = sp[1]
            t = np.array([p.t_ms for p in flight], float)
            y = np.array([p.y for p in flight], float)
            a = (t[0], y[0]) if s.shuttle_px is None else (t0, float(s.shuttle_px[1]))
            chord = a[1] + (y[-1] - a[1]) * (t - a[0]) / max(t[-1] - a[0], 1e-6)
            r["arc"] = float((chord - y).max() / span)
            r["apex_frac"] = float((t[int(np.argmin(y))] - t0) / dur)
            tt = (t - t0) / 1000.0
            if tt.max() - tt.min() > 0.05:
                r["curv"] = float(np.polyfit(tt, y, 2)[0] / span)
            q = max(2, len(flight) // 4)
            head, tail = flight[:q], flight[-q:]
            if head[-1].t_ms > head[0].t_ms:
                r["climb_v"] = float((head[0].y - head[-1].y) / span / ((head[-1].t_ms - head[0].t_ms) / 1000.0))
            v0, v1 = _speed(head), _speed(tail)
            if v0 and v1 is not None:
                r["slow_ratio"] = float(v1 / v0)
        rows.append([r[k] for k in TRAJ_FEATURES])
    return np.asarray(rows, dtype=float).reshape(-1, len(TRAJ_FEATURES))
