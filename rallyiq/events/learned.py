"""Learned hit scorer: replaces the hand-weighted hit confidence with a small model trained on labelled matches.

The rules in ``hits.py`` still propose the candidates (direction / speed changes after non-maximum
suppression) and still produce the reason codes the review UI shows. Only the number that decides
whether a candidate is a hit, ``Candidate.confidence``, comes from the model: the probability that a
labelled hit lies within a few frames of it.

The model is a one-hidden-layer MLP (or logistic regression when ``hidden`` is empty) stored as JSON,
evaluated with numpy only. ``scripts/learn_hits.py`` trains it on ShuttleSet clips; the bundled
weights live in ``models/hit_scorer.json`` next to this file. Selection via ``EventsConfig.hit_scorer``:
``"auto"`` (bundled model if present, else rules), ``"off"`` (rules), or a path to a JSON model.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .config import EventsConfig
    from .hits import Candidate
    from .players import PoseIndex
    from .rallies import RallySpan
    from .trajectory import Track

BUNDLED = Path(__file__).with_name("models") / "hit_scorer.json"

FEATURES: tuple[str, ...] = (
    "turn", "change", "sharpness", "track_quality", "log_impulse",
    "prox1", "dist1", "prox2", "prox_margin", "used_wrist", "has_pose",
    "gap", "interpolated", "shuttle_conf",
    "t_from_start", "t_to_end", "landing",
    "log_speed_in", "log_speed_out", "dir_in_y", "dir_out_y", "dir_in_x", "dir_out_x",
    "near_side", "rule_conf", "n_players", "y_rel",
)
# v3 additions: shuttle motion relative to the best player instead of image axes (camera-independent),
# speed ratio, wrist distance and how densely the shuttle was seen around the candidate.
EXTRA_FEATURES: tuple[str, ...] = (
    "toward_in", "away_out", "log_speed_ratio", "wrist_dist", "track_density",
    # pose-sequence swing model (pose_hits.py): best p_swing of the best player's side / the other side within 250 ms
    "p_swing_hitter", "p_swing_other",
)
SWING_WINDOW_MS = 250.0


def _swing(poses: "PoseIndex", t_ms: float, side: Optional[str]) -> tuple[float, float]:
    """(p_swing of ``side``, p_swing of the other side) from the bundled pose hit model; (0.5, 0.5) when unknown.
    The side series is built once per PoseIndex and cached on it."""
    from . import pose_hits as PH

    model = PH.get_model("auto")
    if model is None:
        return 0.5, 0.5
    cache = getattr(poses, "_swing_cache", None)
    if cache is None:
        frames = sorted(poses.by_frame.values(), key=lambda f: f.frame)
        ts = [f.t_ms for f in frames[:200]]
        fr = [f.frame for f in frames[:200]]
        frame_ms = (ts[-1] - ts[0]) / (fr[-1] - fr[0]) if len(ts) > 1 and fr[-1] > fr[0] else 1000.0 / 30.0
        cache = (PH.side_series(frames), frame_ms, {})
        poses._swing_cache = cache  # type: ignore[attr-defined]
    series, frame_ms, memo = cache
    out = []
    for sd in ("near", "far"):
        key = (round(t_ms), sd)
        if key not in memo:
            memo[key] = PH.swing_evidence(series, t_ms, sd, model, SWING_WINDOW_MS, frame_ms)[0] if series[sd].bodies else 0.5
        out.append(memo[key])
    near, far = out
    if side == "near":
        return near, far
    if side == "far":
        return far, near
    return max(near, far), min(near, far)
ALL_FEATURES: tuple[str, ...] = FEATURES + EXTRA_FEATURES


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def features(
    c: "Candidate", track: "Track", span: "RallySpan", poses: "PoseIndex", cfg: "EventsConfig",
    rule_conf: float, landing: bool, no_pose: bool,
) -> list[float]:
    """Feature vector of one candidate, in ``ALL_FEATURES`` order. Everything is scale-free (player heights,
    seconds), so near/far players, resolutions and frame rates share one model. A model uses the columns
    named in its own ``features`` list."""
    pts = track.points
    pt = pts[c.idx]
    scale = max(poses.scale_at(pt.frame, pt.xy), 1.0)
    vin = track.incoming(c.idx, cfg.window_ms, cfg.max_window_ms) or (0.0, 0.0)
    vout = track.outgoing(c.idx, cfg.window_ms, cfg.max_window_ms) or (0.0, 0.0)
    sin, sout = math.hypot(*vin), math.hypot(*vout)
    b = c.best
    p1 = b.score if b else 0.0
    p2 = c.prox[1].score if len(c.prox) > 1 else 0.0
    side = b.pose.side if b else None
    # Shuttle height relative to the best player's body: above the head (smash/clear contact) vs at the feet.
    y_rel = 0.0
    if b is not None:
        ys = [k.y for k in b.pose.keypoints if k.confidence >= cfg.min_kp_conf]
        if ys:
            y_rel = _clip((max(ys) - pt.y) / max(max(ys) - min(ys), 1.0), -1.0, 2.0)
    toward_in = away_out = 0.0
    wrist = 3.0
    if b is not None:
        kp = [k for k in b.pose.keypoints if k.confidence >= cfg.min_kp_conf]
        if kp:
            # body centre: mean of the confident keypoints; direction shuttle -> body
            cx, cy = sum(k.x for k in kp) / len(kp), sum(k.y for k in kp) / len(kp)
            dx, dy = cx - pt.x, cy - pt.y
            dn = math.hypot(dx, dy)
            if dn > 1e-6:
                if sin > 1e-6:
                    toward_in = (vin[0] * dx + vin[1] * dy) / (sin * dn)
                if sout > 1e-6:
                    away_out = -(vout[0] * dx + vout[1] * dy) / (sout * dn)
        wr = [b.pose.keypoints[i] for i in (9, 10) if len(b.pose.keypoints) > i and b.pose.keypoints[i].confidence >= cfg.min_kp_conf]
        if wr:
            wrist = _clip(min(math.hypot(k.x - pt.x, k.y - pt.y) for k in wr) / scale, 0.0, 3.0)
    lo, hi = pt.t_ms - 500.0, pt.t_ms + 500.0
    seen = sum(1 for q in pts[max(0, c.idx - 40): c.idx + 41] if lo <= q.t_ms <= hi and not q.interpolated)
    span_ms = max(pts[min(len(pts) - 1, c.idx + 40)].t_ms - pts[max(0, c.idx - 40)].t_ms, 1.0)
    fr_ms = span_ms / max(min(len(pts) - 1, c.idx + 40) - max(0, c.idx - 40), 1)
    density = _clip(seen * fr_ms / 1000.0)
    sw_h, sw_o = _swing(poses, pt.t_ms, side)
    return [
        c.turn_deg / 180.0,
        c.change,
        c.sharpness,
        c.track_quality,
        math.log1p(max(c.impulse, 0.0)),
        p1,
        _clip((b.distance_rel if b else 3.0) / 3.0),
        p2,
        p1 - p2,
        1.0 if (b is not None and b.used_wrist) else 0.0,
        0.0 if (no_pose or not c.prox) else 1.0,
        _clip(track.gap_around(c.idx) / 500.0, 0.0, 2.0),
        1.0 if pt.interpolated else 0.0,
        pt.confidence,
        _clip((pt.t_ms - pts[span.start].t_ms) / 1000.0, 0.0, 3.0),
        _clip((pts[span.end].t_ms - pt.t_ms) / 1000.0, 0.0, 3.0),
        1.0 if landing else 0.0,
        math.log1p(sin / scale),
        math.log1p(sout / scale),
        vin[1] / sin if sin > 1e-6 else 0.0,
        vout[1] / sout if sout > 1e-6 else 0.0,
        vin[0] / sin if sin > 1e-6 else 0.0,
        vout[0] / sout if sout > 1e-6 else 0.0,
        1.0 if side == "near" else (0.0 if side == "far" else 0.5),
        rule_conf,
        float(min(len(c.prox), 4)),
        y_rel,
        toward_in,
        away_out,
        _clip(math.log((sout + 1.0) / (sin + 1.0)), -3.0, 3.0),
        wrist,
        density,
        sw_h,
        sw_o,
    ]


class HitScorer:
    def __init__(self, spec: dict):
        names = tuple(spec["features"])
        if not set(names) <= set(ALL_FEATURES):
            raise ValueError("hit scorer was trained on a different feature set; retrain with scripts/learn_hits.py")
        self.cols = [ALL_FEATURES.index(n) for n in names]
        self.spec = spec
        # an ensemble lists its members (same features); each member has mean/std/layers or is a gbt
        self.members = [HitScorer({**m, "features": list(names)}) for m in spec.get("members", [])]
        # kind "gbt": gradient-boosted trees (rallyiq.gbt JSON, from scikit-learn's HistGradientBoostingClassifier)
        self.gbt = None
        if spec.get("kind") == "gbt":
            from ..gbt import GBT

            self.gbt = GBT({"classes": [0, 1], "baseline": [float(spec["baseline"])], "trees": [[t] for t in spec["trees"]]})
        if self.members or self.gbt is not None:
            spec = {**spec, "mean": [0.0], "std": [1.0], "layers": []}
        self.mean = np.asarray(spec["mean"], float)
        self.std = np.asarray(spec["std"], float)
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in spec["layers"]]
        self.thresholds: dict[str, float] = dict(spec.get("thresholds") or {})
        # optional second head: P(hit came from the near end), same features
        self.side: Optional[HitScorer] = HitScorer(spec["side"]) if spec.get("side") else None

    def predict(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """``x``: rows in ``ALL_FEATURES`` order (or already in this model's own column order)."""
        x = np.asarray(x, float)
        if x.shape[-1] == len(ALL_FEATURES) and len(self.cols) != len(ALL_FEATURES):
            x = x[..., self.cols]
        if "members" in self.spec:  # ensemble: mean probability of the members
            return np.mean([m.predict(x) for m in self.members], axis=0)
        if self.gbt is not None:
            return self.gbt.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(x.shape[:-1])
        h = (x - self.mean) / self.std
        for k, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if k < len(self.layers) - 1:
                h = np.tanh(h)
        return 1.0 / (1.0 + np.exp(-h[..., 0]))

    def to_json(self) -> str:
        return json.dumps(self.spec, separators=(",", ":"))


@lru_cache(maxsize=8)
def _load(path: str) -> Optional[HitScorer]:
    p = Path(path)
    if not p.exists():
        return None
    return HitScorer(json.loads(p.read_text(encoding="utf-8")))


def get_scorer(setting: str) -> Optional[HitScorer]:
    if not setting or setting == "off":
        return None
    return _load(str(BUNDLED if setting == "auto" else Path(setting).expanduser()))
