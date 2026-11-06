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


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def features(
    c: "Candidate", track: "Track", span: "RallySpan", poses: "PoseIndex", cfg: "EventsConfig",
    rule_conf: float, landing: bool, no_pose: bool,
) -> list[float]:
    """Feature vector of one candidate, in ``FEATURES`` order. Everything is scale-free (player heights,
    seconds), so near/far players, resolutions and frame rates share one model."""
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
    ]


class HitScorer:
    def __init__(self, spec: dict):
        if tuple(spec["features"]) != FEATURES:
            raise ValueError("hit scorer was trained on a different feature set; retrain with scripts/learn_hits.py")
        self.spec = spec
        self.mean = np.asarray(spec["mean"], float)
        self.std = np.asarray(spec["std"], float)
        self.layers = [(np.asarray(w, float), np.asarray(b, float)) for w, b in spec["layers"]]
        self.thresholds: dict[str, float] = dict(spec.get("thresholds") or {})
        # optional second head: P(hit came from the near end), same features
        self.side: Optional[HitScorer] = HitScorer(spec["side"]) if spec.get("side") else None

    def predict(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        h = (np.asarray(x, float) - self.mean) / self.std
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
