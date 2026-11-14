"""Court-level stroke model + rally sequence decoding (``court_v1``).

Why: the video model (``strokes_v3``) reads image cues learned on broadcast footage and gets about
half of the strokes wrong on amateur rear-camera video. This module adds two things that transfer
across cameras:

* **Court model.** A gradient-boosted classifier on what any calibrated camera measures the same
  way: where the hitter and the receiver stand (court metres), flight time (relative to the video's
  median, so slower amateur rallies look like pro ones), high/low contact of this, the previous and
  the next shot, and the position in the rally. It is trained on the full ShuttleSet annotation
  (~35k strokes of 44 matches, no video needed) by ``scripts/learn_court_strokes.py``, with position
  noise and contact flips so it tolerates our measurement errors.
* **Depth alignment.** Foot positions of the far player come out too deep on low cameras (a 13 px
  error is >1 m there). Before the court model, each side's hitter depths in this video are
  quantile-mapped onto the ShuttleSet hitter-depth distribution (unsupervised, no labels).
* **Sequence decoding.** Stroke types follow each other (a lift is answered from the back court,
  a smash by a block or lift...). The two models' probabilities are combined (geometric mean) and
  smoothed with a hybrid HMM (ShuttleSet transition counts, forward-backward), so each shot's
  candidates take its neighbours into account. Off by default (stage option ``sequence``): on hits
  found by the detector, missed and extra hits break the chain, and a rally's first detected hit is
  often not the serve. Held-out ShuttleSet clips: 69.6% without, 60.2% with.

The backend returns per-shot probabilities, so ``classify``'s review policy is unchanged.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import EventsResult, PlayerTracks, ShuttleTrack
from .features import NET_Y_M, StrokeFeatures

NAME = "court_v1"
BUNDLED = Path(__file__).with_name("models") / "strokes_court_v1.json"
MAX_DEPTH_M = 7.5
CENTER_X_M = 3.05

FEATURES = ("hd", "hx", "rd", "lat", "fl", "fl_rel", "spd", "hi", "nhi", "phi", "pfl_rel", "phd",
            "first", "last", "k", "nfl_rel", "nrd")


def _depth(p) -> float | None:
    return None if p is None else min(MAX_DEPTH_M, abs(p[1] - NET_Y_M))


def rally_features(locs: Sequence[tuple[float, float] | None], t_ms: Sequence[float],
                   high: Sequence[float], median_flight_ms: float) -> list[list[float]]:
    """One row of ``FEATURES`` per stroke of one rally (in order).

    ``locs``: hitter court position (metres) at each hit or None; ``high``: p(contact above the net)
    or NaN; ``median_flight_ms``: the video's median time between hits."""
    n = len(locs)
    med = median_flight_ms if median_flight_ms and median_flight_ms > 0 else 900.0
    fl = [float(t_ms[k + 1] - t_ms[k]) if k + 1 < n else math.nan for k in range(n)]
    out = []
    for k in range(n):
        loc, nxt = locs[k], locs[k + 1] if k + 1 < n else None
        lat = spd = None
        if loc is not None and nxt is not None:
            lat = abs(nxt[0] - loc[0])
            if fl[k] > 0:
                spd = math.hypot(nxt[0] - loc[0], nxt[1] - loc[1]) / (fl[k] / 1000.0)
        row = [
            _depth(loc), None if loc is None else abs(loc[0] - CENTER_X_M), _depth(nxt), lat, fl[k], fl[k] / med, spd,
            high[k], high[k + 1] if k + 1 < n else math.nan, high[k - 1] if k > 0 else math.nan,
            fl[k - 1] / med if k > 0 else math.nan, _depth(locs[k - 1]) if k > 0 else None,
            float(k == 0), float(k == n - 1), float(k),
            fl[k + 1] / med if k + 2 < n else math.nan, _depth(locs[k + 2]) if k + 2 < n else None,
        ]
        out.append([math.nan if v is None else float(v) for v in row])
    return out


def depth_map(ref_quantiles: Sequence[float], depths: Sequence[float]) -> Callable[[float], float]:
    """Quantile mapping of one side's depths onto the reference (identity with < 8 samples)."""
    ref = np.asarray(ref_quantiles, float)
    if len(depths) < 8:
        return lambda d: d
    qs = np.linspace(0.05, 0.95, len(ref))
    src = np.quantile(np.asarray(depths, float), qs)
    if np.any(np.diff(src) <= 0):
        src = src + np.arange(len(src)) * 1e-6
    return lambda d: float(np.interp(d, src, ref))


def aligned_positions(events: EventsResult, ref_quantiles: Sequence[float]) -> dict[str, tuple[float, float] | None]:
    """Hitter court position per shot id with each side's depths aligned to ShuttleSet."""
    sides: dict[int, list[float]] = {-1: [], 1: []}
    for s in events.shots:
        if s.hitter_court is not None:
            sides[1 if s.hitter_court[1] > NET_Y_M else -1].append(abs(s.hitter_court[1] - NET_Y_M))
    maps = {k: depth_map(ref_quantiles, v) for k, v in sides.items()}
    out: dict[str, tuple[float, float] | None] = {}
    for s in events.shots:
        if s.hitter_court is None:
            out[s.shot_id] = None
            continue
        x, y = s.hitter_court
        side = 1 if y > NET_Y_M else -1
        out[s.shot_id] = (float(x), NET_Y_M + side * maps[side](abs(y - NET_Y_M)))
    return out


def video_features(events: EventsResult, high: dict[str, float], ref_quantiles: Sequence[float]
                   ) -> tuple[list[str], np.ndarray, list[str | None]]:
    """(shot ids, feature rows, rally id per row) for every shot of the video."""
    pos = aligned_positions(events, ref_quantiles)
    by: dict[str | None, list] = {}
    for s in sorted(events.shots, key=lambda s: s.frame):
        by.setdefault(s.rally_id, []).append(s)
    gaps = [b.t_ms - a.t_ms for rid, rs in by.items() if rid is not None for a, b in itertools.pairwise(rs)]
    med = float(np.median(gaps)) if gaps else 900.0
    ids, rows, rids = [], [], []
    for rid, rs in by.items():
        groups = [rs] if rid is not None else [[s] for s in rs]  # shots outside rallies stand alone
        for g in groups:
            rows += rally_features([pos[s.shot_id] for s in g], [s.t_ms for s in g],
                                   [high.get(s.shot_id, math.nan) for s in g], med)
            ids += [s.shot_id for s in g]
            rids += [rid] * len(g)
    return ids, np.asarray(rows, float).reshape(-1, len(FEATURES)), rids


# ------------------------------------------------------------------ sequence
def smooth(P: np.ndarray, start: np.ndarray, trans: np.ndarray, prior: np.ndarray, weight: float = 1.0) -> np.ndarray:
    """Forward-backward marginals of a hybrid HMM over one rally.

    ``P``: [n, K] per-shot p(type | evidence); emissions are P / prior (scaled likelihoods);
    ``weight`` tempers the transition and start terms (0 = no smoothing)."""
    n, _ = P.shape
    if n == 0:
        return P
    E = np.clip(P, 1e-6, None) / np.clip(prior, 1e-6, None)
    A = trans ** weight
    alpha = np.zeros_like(E)
    a = (start ** weight) * E[0]
    alpha[0] = a / a.sum()
    for k in range(1, n):
        a = (alpha[k - 1] @ A) * E[k]
        alpha[k] = a / a.sum()
    beta = np.ones_like(E)
    for k in range(n - 2, -1, -1):
        b = A @ (E[k + 1] * beta[k + 1])
        beta[k] = b / b.sum()
    post = alpha * beta
    return post / post.sum(1, keepdims=True)


# ------------------------------------------------------------------ model
class CourtModel:
    def __init__(self, spec: dict[str, Any]):
        from ..gbt import GBT

        if spec.get("kind") != "strokes_court" or tuple(spec["features"]) != FEATURES:
            raise ValueError("not a court stroke model for these features; retrain it")
        self.gbt = GBT(spec)
        self.classes = list(spec["classes"])
        self.depth_q = list(spec["depth_quantiles"])
        self.start = np.asarray(spec["start"], float)
        self.trans = np.asarray(spec["trans"], float)
        self.prior = np.asarray(spec["prior"], float)
        self.seq_weight = float(spec.get("seq_weight", 1.0))
        self.meta = spec.get("meta", {})

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self.gbt.predict_proba(X) if len(X) else np.zeros((0, len(self.classes)))


@lru_cache(maxsize=4)
def _load(path: str) -> CourtModel | None:
    p = Path(path)
    return CourtModel(json.loads(p.read_text(encoding="utf-8"))) if p.is_file() else None


def get_model(path: str | Path | None = None) -> CourtModel | None:
    return _load(str(path or BUNDLED))


def contact_high(events, shuttle, players, feats: Sequence[StrokeFeatures], fps: float, attrs) -> dict[str, float]:
    """p(contact above the net) per shot from the attribute model's contact head (empty without it)."""
    if attrs is None or "contact" not in attrs.heads or not feats:
        return {}
    from .detail import attr_matrix

    pred = attrs.predict(attr_matrix(events, shuttle, players, feats, fps))["contact"]
    return {f.shot_id: (p if v == "high" else 1.0 - p) for f, (v, p) in zip(feats, pred)}


def make_model_fn(video_fn, court: CourtModel, attrs, fps: float, seq: bool = True):
    """A ``classify`` backend: video model x court model (geometric mean), then rally smoothing.

    ``video_fn`` may be None (court model alone)."""
    C = court.classes

    def fn(events: EventsResult, shuttle: ShuttleTrack | None, players: PlayerTracks | None,
           feats: list[StrokeFeatures]) -> dict[str, dict[str, float]]:
        high = contact_high(events, shuttle, players, feats, fps, attrs)
        ids, X, rids = video_features(events, high, court.depth_q)
        P = court.proba(X)
        if video_fn is not None:
            pv = video_fn(events, shuttle, players, feats)
            V = np.asarray([[pv.get(i, {}).get(c, 0.0) for c in C] if i in pv else [math.nan] * len(C) for i in ids])
            have = ~np.isnan(V[:, 0]) if len(V) else np.zeros(0, bool)
            if have.any():
                G = np.sqrt(np.clip(P[have], 1e-6, None) * np.clip(V[have], 1e-6, None))
                P[have] = G / G.sum(1, keepdims=True)
        if seq:
            rid_arr = np.asarray([str(r) for r in rids], dtype=object)
            for r in dict.fromkeys(rids):
                if r is None:
                    continue
                ix = np.where(rid_arr == str(r))[0]
                P[ix] = smooth(P[ix], court.start, court.trans, court.prior, court.seq_weight)
        return {i: {c: float(p) for c, p in zip(C, row)} for i, row in zip(ids, P)}

    return fn
