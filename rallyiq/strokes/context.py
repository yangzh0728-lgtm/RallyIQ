"""Extra per-shot evidence for the learned stroke-type model (feature set ``v2``).

``features.py`` describes a shot by its end points (contact height, depths, flight time,
rise). Real broadcast video showed three gaps the rules cannot use but a model can:

* **the shot's neighbours in the rally**: what came in (the previous shot's flight time,
  rise and contact height: a block answers a smash, a lift answers a net shot) and how the
  receiver met it (the next contact height: clears are met high, drops and net shots low);
* **direction**: hitter and receiver on the same or opposite halves of the court
  (straight vs cross-court), signed relative to the hitter;
* **the hitter's body and the first frames of the flight**: wrist above the head
  (overhead stroke), wrist distance from the body, and the shuttle's image velocity just
  after and just before contact, in the hitter's body heights per second.

Missing evidence is NaN, never a guess (``HistGradientBoostingClassifier`` handles NaN).
Only data the pipeline already produced is used (events, shuttle track, poses), so this
adds no compute to a run.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

import numpy as np

from ..contracts import (
    COURT_WIDTH_M,
    NET_Y_M,
    EventsResult,
    PlayerTracks,
    ShotEvent,
    ShuttleTrack,
)
from .features import StrokeFeatures, _kp, _Players, body_span

CONTEXT_FEATURES = (
    "side",               # +1 hitter on the near half (court y > net), -1 far
    "hitter_x_m",         # hitter's lateral position from the centre line, in the hitter's own left/right
    "receiver_x_m",       # receiver's lateral position from the centre line, same frame as hitter_x_m
    "cross",              # (hitter x - centre) * (receiver x - centre): < 0 = cross-court
    "prev_flight_ms",     # flight time of the incoming shot
    "prev_rise",          # rise of the incoming shot
    "prev_contact_h",     # where the opponent met it
    "prev_hitter_depth_m",
    "next_contact_h",     # where the receiver met this shot
    "next_flight_ms",     # how long the reply took
    "wrist_h",            # highest wrist above the ankles, in ankle-to-nose spans
    "wrist_out",          # that wrist's sideways distance from the hip centre, in spans
    "elbow_up",           # higher elbow above the shoulders, in spans
    "out_vx",             # shuttle image velocity just after contact, spans/s (x: away from hitter's body centre)
    "out_vy",             # shuttle image velocity just after contact, spans/s (+ = up in the image)
    "out_speed",          # |velocity| just after contact, spans/s
    "in_vy",              # shuttle image velocity just before contact, spans/s (+ = up)
    "in_speed",
)

KP_SHOULDERS, KP_ELBOWS, KP_WRISTS, KP_HIPS = (5, 6), (7, 8), (9, 10), (11, 12)
WINDOW_MS = 200.0  # shuttle motion window on each side of the contact
MIN_KP = 0.3


def _nan(v) -> float:
    return math.nan if v is None else float(v)


class _Track:
    def __init__(self, track: ShuttleTrack | None, min_conf: float = 0.2):
        pts = sorted((p for p in (track.points if track else []) if p.confidence >= min_conf and p.source == "detected"),
                     key=lambda p: p.t_ms)
        self.t = [p.t_ms for p in pts]
        self.pts = pts

    def window(self, t0: float, t1: float):
        return self.pts[bisect_left(self.t, t0):bisect_left(self.t, t1)]


def _velocity(pts, span: float) -> tuple[float, float] | None:
    """Least-squares image velocity (px/s -> spans/s) over a few points."""
    if len(pts) < 3 or span <= 0:
        return None
    t = np.array([p.t_ms for p in pts], dtype=float) / 1000.0
    if t.max() - t.min() < 1e-3:
        return None
    x = np.array([p.x for p in pts], dtype=float)
    y = np.array([p.y for p in pts], dtype=float)
    tc = t - t.mean()
    den = float((tc * tc).sum())
    return float((tc * (x - x.mean())).sum() / den / span), float((tc * (y - y.mean())).sum() / den / span)


def _pose(p) -> dict[str, float]:
    out = {"wrist_h": math.nan, "wrist_out": math.nan, "elbow_up": math.nan}
    sp = body_span(p, MIN_KP)
    if sp is None:
        return out
    ankle_y, span = sp
    wrists = [q for i in KP_WRISTS if (q := _kp(p, i, MIN_KP))]
    hips = [q for i in KP_HIPS if (q := _kp(p, i, MIN_KP))]
    cx = sum(q[0] for q in hips) / len(hips) if hips else (p.bbox[0] + p.bbox[2]) / 2.0
    if wrists:
        w = min(wrists, key=lambda q: q[1])
        out["wrist_h"] = (ankle_y - w[1]) / span
        out["wrist_out"] = abs(w[0] - cx) / span
    shoulders = [q for i in KP_SHOULDERS if (q := _kp(p, i, MIN_KP))]
    elbows = [q for i in KP_ELBOWS if (q := _kp(p, i, MIN_KP))]
    if shoulders and elbows:
        out["elbow_up"] = (min(q[1] for q in shoulders) - min(q[1] for q in elbows)) / span
    return out


def context_features(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    feats: Sequence[StrokeFeatures],
) -> np.ndarray:
    """One row of ``CONTEXT_FEATURES`` per entry of ``feats`` (same order)."""
    shots = {s.shot_id: s for s in events.shots}
    fmap = {f.shot_id: f for f in feats}
    by_rally: dict[str | None, list[ShotEvent]] = {}
    for s in sorted(events.shots, key=lambda s: s.frame):
        by_rally.setdefault(s.rally_id, []).append(s)
    pos: dict[str, tuple[ShotEvent | None, ShotEvent | None]] = {}
    for rid, ss in by_rally.items():
        for k, s in enumerate(ss):
            prev = ss[k - 1] if rid is not None and k > 0 else None
            nxt = ss[k + 1] if rid is not None and k + 1 < len(ss) else None
            pos[s.shot_id] = (prev, nxt)
    pl = _Players(players, 3)
    tr = _Track(shuttle)
    cx = COURT_WIDTH_M / 2.0

    rows = []
    for f in feats:
        s = shots[f.shot_id]
        prev, nxt = pos.get(f.shot_id, (None, None))
        r = dict.fromkeys(CONTEXT_FEATURES, math.nan)
        side = None
        if s.hitter_court is not None:
            side = 1.0 if s.hitter_court[1] > NET_Y_M else -1.0
            r["side"] = side
            # a near-half player facing the net has court +x on their right; far half mirrored
            r["hitter_x_m"] = (s.hitter_court[0] - cx) * side
            if nxt is not None and nxt.hitter_court is not None and (nxt.hitter_court[1] > NET_Y_M) != (side > 0):
                r["receiver_x_m"] = (nxt.hitter_court[0] - cx) * side
                r["cross"] = (s.hitter_court[0] - cx) * (nxt.hitter_court[0] - cx)
        if prev is not None and prev.shot_id in fmap:
            pf = fmap[prev.shot_id]
            r["prev_flight_ms"] = _nan(pf.flight_ms)
            r["prev_rise"] = _nan(pf.rise)
            r["prev_contact_h"] = _nan(pf.contact_h)
            r["prev_hitter_depth_m"] = _nan(pf.hitter_depth_m)
        if nxt is not None and nxt.shot_id in fmap:
            nf = fmap[nxt.shot_id]
            r["next_contact_h"] = _nan(nf.contact_h)
            r["next_flight_ms"] = _nan(nf.flight_ms)

        hitter = pl.find(s.frame, s.hitter) if s.hitter else None
        if hitter is not None:
            r.update(_pose(hitter))
            sp = body_span(hitter, MIN_KP)
            if sp is not None:
                span = sp[1]
                body_x = (hitter.bbox[0] + hitter.bbox[2]) / 2.0
                t_end = nxt.t_ms if nxt is not None else s.t_ms + WINDOW_MS
                t_beg = prev.t_ms if prev is not None else s.t_ms - WINDOW_MS
                out = tr.window(s.t_ms + 1, min(s.t_ms + WINDOW_MS, t_end - 1))
                v = _velocity(out, span)
                if v is not None:
                    away = 1.0 if (out[-1].x - body_x) >= 0 else -1.0
                    r["out_vx"], r["out_vy"] = v[0] * away, -v[1]
                    r["out_speed"] = math.hypot(*v)
                inc = tr.window(max(s.t_ms - WINDOW_MS, t_beg + 1), s.t_ms)
                v = _velocity(inc, span)
                if v is not None:
                    r["in_vy"], r["in_speed"] = -v[1], math.hypot(*v)
        rows.append([r[k] for k in CONTEXT_FEATURES])
    return np.asarray(rows, dtype=float).reshape(-1, len(CONTEXT_FEATURES))
