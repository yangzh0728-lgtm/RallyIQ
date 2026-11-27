"""Dead-ball filter: drop a shuttle-based hit that the rally around it says nobody played.

Between rallies players bounce the shuttle on the racket, scoop it off the floor or tap it over to the other side. Those
touches look like strokes to the hit detector (the shuttle turns at a racket), but the rally context does not: they are
isolated (long gaps to the previous and next hit), the opponent does not swing in reply, and the scorer is less sure of them.
A logistic model over 16 rally-context features (``FEATURES``) gives ``p_dead``; a hit with ``p_dead`` above the model's
threshold is removed. Weights: ``models/dead_ball.json``, fitted by docs/train_v7/scripts/g_fit.py on the hand-labelled
hits of 11 own videos only (qa6 and the fresh test videos never seen); see docs/train_v7/REPORT.md.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from .types import PoseFrame, RallySegment

BUNDLED = Path(__file__).with_name("models") / "dead_ball.json"
FEATURES = ("conf", "logn", "first", "last", "single", "gap_prev", "gap_next", "same_prev", "same_next", "ps", "wv",
            "ps_other", "wv_other", "dur", "t_start", "t_end")
R_DEAD_BALL = "dead_ball"


@lru_cache(maxsize=4)
def _load(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_model(setting: str) -> dict | None:
    if not setting or setting == "off":
        return None
    return _load(str(BUNDLED if setting == "auto" else Path(setting).expanduser()))


def _side_lookup(poses: Sequence[PoseFrame]):
    seq: dict[str, list[tuple[int, str]]] = {}
    for f in poses:
        for p in f.players:
            if p.player_id and p.side in ("near", "far"):
                seq.setdefault(p.player_id, []).append((f.frame, p.side))

    def side_at(pid: str | None, frame: int) -> str | None:
        s = seq.get(pid or "")
        if not s:
            return None
        i = bisect_left(s, (frame, ""))
        best = min((s[j] for j in (i - 1, i) if 0 <= j < len(s)), key=lambda x: abs(x[0] - frame))
        return best[1] if abs(best[0] - frame) <= 60 else None
    return side_at


def rally_features(rally: RallySegment, side_at, swing) -> list[dict[str, float]]:
    """One feature dict per hit of ``rally``. ``swing(t_ms, side, window_ms) -> (p_swing, wrist_v)``."""
    hits = rally.hits
    n = len(hits)
    sides = [side_at(h.hitter_id, h.frame) for h in hits]
    out = []
    for i, h in enumerate(hits):
        t = h.t_ms / 1000.0
        s = sides[i]
        o = {"near": "far", "far": "near"}.get(s or "")
        ps, wv = swing(h.t_ms, s, 250.0) if s else (0.0, 0.0)
        po, wo = swing(h.t_ms, o, 600.0) if o else (0.0, 0.0)
        gp = t - hits[i - 1].t_ms / 1000.0 if i else 5.0
        gn = hits[i + 1].t_ms / 1000.0 - t if i + 1 < n else 5.0
        out.append({
            "conf": h.confidence, "logn": math.log1p(n), "first": float(i == 0), "last": float(i == n - 1),
            "single": float(n == 1), "gap_prev": min(gp, 5.0), "gap_next": min(gn, 5.0),
            "same_prev": float(i > 0 and sides[i - 1] == s), "same_next": float(i + 1 < n and sides[i + 1] == s),
            "ps": ps, "wv": min(wv, 20.0), "ps_other": po, "wv_other": min(wo, 20.0),
            "dur": min((rally.end_ms - rally.start_ms) / 1000.0, 30.0),
            "t_start": min(t - rally.start_ms / 1000.0, 10.0), "t_end": min(rally.end_ms / 1000.0 - t, 10.0),
        })
    return out


def p_dead(model: dict, feats: list[dict[str, float]]) -> np.ndarray:
    if not feats:
        return np.zeros(0)
    X = np.array([[f[k] for k in model["features"]] for f in feats], float)
    z = ((X - np.asarray(model["mean"])) / np.asarray(model["std"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def apply(rallies: list[RallySegment], poses: Sequence[PoseFrame], model: dict | None, frame_ms: float) -> int:
    """Remove dead-ball hits in place. Returns how many were removed."""
    if model is None or not rallies or not poses:
        return 0
    from .pose_hits import get_model as pose_model, side_series, swing_evidence

    pm = pose_model(model.get("swing_model", "auto"))
    if pm is None:
        return 0
    series = side_series(poses)
    side_at = _side_lookup(poses)

    def swing(t_ms, side, window_ms):
        return swing_evidence(series, t_ms, side, pm, window_ms, frame_ms)

    thr = float(model["threshold"])
    dropped = 0
    for r in rallies:
        p = p_dead(model, rally_features(r, side_at, swing))
        keep = [h for h, q in zip(r.hits, p) if q <= thr or h.source != "rules_v1"]
        for h, q in zip(r.hits, p):
            h.scores["p_dead"] = round(float(q), 3)
        if len(keep) != len(r.hits):
            dropped += len(r.hits) - len(keep)
            r.hits = keep
            for k, h in enumerate(r.hits, start=1):
                h.index_in_rally = k
                h.hit_id = f"{r.rally_id}_S{k:02d}"
    return dropped
