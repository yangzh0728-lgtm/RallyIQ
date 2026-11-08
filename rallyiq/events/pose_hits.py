"""Pose-sequence hit model: recover hits the shuttle track cannot show.

The hit candidates in ``hits.py`` come from the shuttle's trajectory, so a hit is lost whenever the
shuttle is not seen around the contact: the serve (the shuttle is small and held against the body),
a shot behind the player or the net post, a hit the detector merged into its neighbour. The player's
body still shows it: the racket arm swings up and through, the wrists accelerate, the body loads and
unloads. This model reads the hitter's pose over ~0.5 s around a moment and gives the probability that
this player hits the shuttle right then (``p_swing``). It only needs the pose stage's keypoints.

``recover`` uses it after the shuttle-based detector, inside each rally, for three gaps:

* **serve**: no hit before the first detected one although the opposite player must have served;
* **same side twice**: two consecutive hits by the same side, so the other side's hit in between was missed.

(Correcting the side of a detected hit from the pose model was tried and lowered F1, so it is not done.)

A recovered hit carries reason ``pose_recovered`` (always ``needs_review``) and source ``pose_v1``.

The model is gradient-boosted trees exported to JSON and evaluated with numpy (no scikit-learn at run
time), trained by ``scripts/learn_pose_hits.py``. The bundled weights live in
``models/pose_hits.json``; ``EventsConfig.pose_hits`` selects it (``"auto"``, ``"off"`` or a path).
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..gbt import GBT
from ..gbt import export_hgb as gbt_export
from .types import HitEvent, PoseFrame, RallySegment, ReviewStatus

BUNDLED = Path(__file__).with_name("models") / "pose_hits.json"
SOURCE = "pose_v1"
R_POSE_RECOVERED = "pose_recovered"

# COCO-17 joints used (face reduced to the nose)
JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
WRISTS, ELBOWS, SHOULDERS, HIPS, ANKLES = (9, 10), (7, 8), (5, 6), (11, 12), (15, 16)
STEPS_MS = (-320.0, -240.0, -160.0, -80.0, 0.0, 80.0, 160.0)  # window around the moment scored
MIN_KP = 0.3


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for s in STEPS_MS:
        tag = f"{int(s):+d}"
        names += [f"wrist_h{tag}", f"wrist_out{tag}", f"elbow_up{tag}", f"wrist_v{tag}", f"lean{tag}", f"crouch{tag}",
                  f"seen{tag}"]
    return tuple(names) + ("wrist_v_max", "wrist_v_at", "wrist_h_max", "far", "n_seen")


FEATURES = feature_names()


# ------------------------------------------------------------------ pose series per side
@dataclass
class _Body:
    t_ms: float
    frame: int
    player_id: str
    kp: np.ndarray  # 17 x 3
    span: float  # ankle-to-nose px
    ankle_y: float
    hip_x: float


@dataclass
class SideSeries:
    side: str
    bodies: list[_Body] = field(default_factory=list)
    t: list[float] = field(default_factory=list)

    def near(self, t_ms: float, tol_ms: float) -> _Body | None:
        i = bisect_left(self.t, t_ms)
        best = None
        for j in (i - 1, i):
            if (0 <= j < len(self.t) and abs(self.t[j] - t_ms) <= tol_ms
                    and (best is None or abs(self.t[j] - t_ms) < abs(best.t_ms - t_ms))):
                best = self.bodies[j]
        return best


def _body(frame: int, t_ms: float, p) -> _Body | None:
    if len(p.keypoints) < 17:
        return None
    kp = np.array([[k.x, k.y, k.confidence] for k in p.keypoints[:17]], float)
    ok = kp[:, 2] >= MIN_KP
    head = kp[0, 1] if ok[0] else (kp[[1, 2, 3, 4], 1][ok[[1, 2, 3, 4]]].mean() if ok[[1, 2, 3, 4]].any() else None)
    if head is None:
        return None
    ank = kp[list(ANKLES), 1][ok[list(ANKLES)]]
    if not len(ank):
        return None
    ankle_y = float(ank.max())
    span = ankle_y - float(head)
    if span <= 8.0:
        return None
    hips = kp[list(HIPS)][ok[list(HIPS)]]
    hip_x = float(hips[:, 0].mean()) if len(hips) else float(kp[ok, 0].mean())
    return _Body(t_ms, frame, p.player_id, kp, span, ankle_y, hip_x)


def side_series(poses: Sequence[PoseFrame]) -> dict[str, SideSeries]:
    out = {"near": SideSeries("near"), "far": SideSeries("far")}
    for pf in poses:
        seen: set[str] = set()
        for p in pf.players:
            if p.side not in out or p.side in seen:
                continue
            b = _body(pf.frame, pf.t_ms, p)
            if b is not None:
                seen.add(p.side)
                out[p.side].bodies.append(b)
                out[p.side].t.append(pf.t_ms)
    return out


def _step(b: _Body | None, prev: _Body | None) -> list[float]:
    nan = math.nan
    if b is None:
        return [nan] * 6 + [0.0]
    kp, ok = b.kp, b.kp[:, 2] >= MIN_KP
    wr = [i for i in WRISTS if ok[i]]
    wrist_h = wrist_out = elbow_up = wrist_v = nan
    if wr:
        w = min(wr, key=lambda i: kp[i, 1])
        wrist_h = (b.ankle_y - kp[w, 1]) / b.span
        wrist_out = abs(kp[w, 0] - b.hip_x) / b.span
    sh = [i for i in SHOULDERS if ok[i]]
    el = [i for i in ELBOWS if ok[i]]
    if sh and el:
        elbow_up = (min(kp[i, 1] for i in sh) - min(kp[i, 1] for i in el)) / b.span
    if prev is not None and b.t_ms > prev.t_ms:
        pk, pok = prev.kp, prev.kp[:, 2] >= MIN_KP
        vs = [math.hypot(kp[i, 0] - pk[i, 0], kp[i, 1] - pk[i, 1]) for i in WRISTS if ok[i] and pok[i]]
        if vs:
            wrist_v = max(vs) / b.span / ((b.t_ms - prev.t_ms) / 1000.0)
    lean = crouch = nan
    if sh and any(ok[i] for i in HIPS):
        sx = float(np.mean([kp[i, 0] for i in sh]))
        lean = (sx - b.hip_x) / b.span
        hy = float(np.mean([kp[i, 1] for i in HIPS if ok[i]]))
        crouch = (b.ankle_y - hy) / b.span
    return [wrist_h, wrist_out, elbow_up, wrist_v, lean, crouch, 1.0]


def window_features(series: SideSeries, t_ms: float, frame_ms: float) -> list[float]:
    """Features of ``series``' player around ``t_ms`` (``FEATURES`` order; NaN where unseen)."""
    tol = max(1.6 * frame_ms, 45.0)
    rows = []
    for s in STEPS_MS:
        b = series.near(t_ms + s, tol)
        prev = series.near(t_ms + s - 80.0, tol) if b is not None else None
        rows.append(_step(b, prev if prev is not b else None))
    flat = [v for r in rows for v in r]
    wv = [r[3] for r in rows]
    wh = [r[0] for r in rows]
    fin = [v for v in wv if not math.isnan(v)]
    vmax = max(fin) if fin else math.nan
    at = STEPS_MS[int(np.nanargmax(wv))] / 1000.0 if fin else math.nan
    hfin = [v for v in wh if not math.isnan(v)]
    return flat + [vmax, at, max(hfin) if hfin else math.nan, 1.0 if series.side == "far" else 0.0,
                   float(sum(r[-1] for r in rows))]


# ------------------------------------------------------------------ model
class PoseHitModel:
    """Gradient-boosted trees exported to JSON (``kind: "gbt"``), evaluated with numpy by
    :class:`rallyiq.gbt.GBT`: ``baseline`` + the sum of ``trees``, through a sigmoid (as scikit-learn's
    ``HistGradientBoostingClassifier``, from which ``scripts/learn_pose_hits.py`` exports it)."""

    def __init__(self, spec: dict):
        if tuple(spec["features"]) != FEATURES:
            raise ValueError("pose hit model was trained on a different feature set; retrain with scripts/learn_pose_hits.py")
        if spec.get("kind") != "gbt":
            raise ValueError(f"unknown pose hit model kind {spec.get('kind')!r}")
        self.spec = spec
        self.gbt = GBT({"classes": [0, 1], "baseline": [float(spec["baseline"])], "trees": [[t] for t in spec["trees"]]})
        self.thresholds: dict[str, float] = dict(spec.get("thresholds") or {})

    def raw(self, X) -> np.ndarray:
        return self.gbt.raw(np.asarray(X, float).reshape(-1, len(FEATURES)))[:, 0]

    def predict(self, X) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw(X)))


def export_hgb(model, thresholds: dict | None = None, meta: dict | None = None) -> dict:
    """JSON spec of a fitted binary ``HistGradientBoostingClassifier`` (numerical features only)."""
    g = gbt_export(model)
    return {"kind": "gbt", "features": list(FEATURES), "baseline": g["baseline"][0],
            "trees": [it[0] for it in g["trees"]], "thresholds": dict(thresholds or {}), **(meta or {})}


@lru_cache(maxsize=8)
def _load(path: str) -> PoseHitModel | None:
    p = Path(path)
    if not p.exists():
        return None
    return PoseHitModel(json.loads(p.read_text(encoding="utf-8")))


def get_model(setting: str) -> PoseHitModel | None:
    if not setting or setting == "off":
        return None
    return _load(str(BUNDLED if setting == "auto" else Path(setting).expanduser()))


# ------------------------------------------------------------------ recovery
@dataclass(frozen=True)
class RecoverConfig:
    serve_threshold: float = 0.6  # tuned on out-of-fold training matches (scripts/learn_pose_hits.py)
    gap_threshold: float = 0.3
    serve_window_ms: float = 1500.0  # how far before the first hit a serve may lie
    min_gap_ms: float = 280.0  # a recovered hit keeps this far from its neighbours
    step_ms: float = 40.0  # scan resolution


def swing_curve(series: SideSeries, model: PoseHitModel, t0: float, t1: float, step_ms: float,
                frame_ms: float) -> tuple[np.ndarray, np.ndarray]:
    if t1 <= t0 or not series.bodies:
        return np.zeros(0), np.zeros(0)
    ts = np.arange(t0, t1 + 1e-6, step_ms)
    X = np.asarray([window_features(series, float(t), frame_ms) for t in ts], float)
    return ts, model.predict(X)


def _side_of(h: HitEvent, series: dict[str, SideSeries]) -> str | None:
    if h.hitter_id is None:
        return None
    for s in series.values():
        b = s.near(h.t_ms, 120.0)
        if b is not None and b.player_id == h.hitter_id:
            return s.side
    return None


def _new_hit(r: RallySegment, side: SideSeries, t: float, p: float, frame_ms: float) -> HitEvent:
    b = side.near(t, 200.0)
    frame0 = r.hits[0].frame - (r.hits[0].t_ms - t) / frame_ms if r.hits else r.start_frame
    return HitEvent(
        hit_id="", rally_id=r.rally_id, index_in_rally=0, frame=round(frame0), t_ms=float(t),
        hitter_id=b.player_id if b is not None else None, shuttle_xy_px=None, shuttle_court_xy=None,
        confidence=round(float(p), 3), status=ReviewStatus.NEEDS_REVIEW, reasons=[R_POSE_RECOVERED],
        scores={"p_swing": round(float(p), 3)}, source=SOURCE,
    )


def recover(rallies: list[RallySegment], poses: Sequence[PoseFrame], model: PoseHitModel | None,
            cfg: RecoverConfig | None = None, frame_ms: float | None = None) -> int:
    """Add / fix hits in place using the pose model. Returns how many hits were added or changed."""
    if model is None or not rallies or not poses:
        return 0
    cfg = cfg or RecoverConfig(**{k: v for k, v in model.thresholds.items() if k in RecoverConfig.__dataclass_fields__})
    if frame_ms is None:
        ts = [p.t_ms for p in poses[:50]]
        fr = [p.frame for p in poses[:50]]
        frame_ms = (ts[-1] - ts[0]) / (fr[-1] - fr[0]) if len(ts) > 1 and fr[-1] > fr[0] else 40.0
    series = side_series(poses)
    other = {"near": "far", "far": "near"}
    changed = 0
    for r in rallies:
        if not r.hits:
            continue
        sides = [_side_of(h, series) for h in r.hits]
        added: list[tuple[HitEvent, str]] = []
        # serve before the first hit, by the other side
        if sides[0] is not None:
            srv = other[sides[0]]
            t1 = r.hits[0].t_ms - cfg.min_gap_ms
            ts, p = swing_curve(series[srv], model, t1 - cfg.serve_window_ms, t1, cfg.step_ms, frame_ms)
            if len(p) and p.max() >= cfg.serve_threshold:
                i = int(p.argmax())
                added.append((_new_hit(r, series[srv], float(ts[i]), float(p[i]), frame_ms), srv))
        # same side twice in a row: the other side's hit between them was missed
        for k in range(len(r.hits) - 1):
            a, b = sides[k], sides[k + 1]
            if a is None or a != b:
                continue
            t0, t1 = r.hits[k].t_ms + cfg.min_gap_ms, r.hits[k + 1].t_ms - cfg.min_gap_ms
            ts, p = swing_curve(series[other[a]], model, t0, t1, cfg.step_ms, frame_ms)
            if len(p) and p.max() >= cfg.gap_threshold:
                i = int(p.argmax())
                added.append((_new_hit(r, series[other[a]], float(ts[i]), float(p[i]), frame_ms), other[a]))
        if added:
            ref = r.hits[0]
            for h, _ in added:
                h.frame = round(ref.frame + (h.t_ms - ref.t_ms) / frame_ms)
            r.hits = sorted([*r.hits, *(h for h, _ in added)], key=lambda h: h.t_ms)
            for k, h in enumerate(r.hits, start=1):
                h.index_in_rally = k
                h.hit_id = f"{r.rally_id}_S{k:02d}"
            if r.hits[0].frame < r.start_frame:
                r.start_frame, r.start_ms = r.hits[0].frame, r.hits[0].t_ms
            changed += len(added)
    return changed

