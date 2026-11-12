"""Layered description of a shot, on top of its stroke type.

A coach does not stop at "smash": they say *forehand, straight, met high, fast, attacking,
to push the opponent back*. This module adds those layers to every classified shot
(``StrokeInfo.detail``) and says for each one where it comes from:

=========  ====================================  =========================================
layer      values                                source
=========  ====================================  =========================================
family     clear smash drop drive block push      the stroke type (12 types -> 10 families)
           lift net_shot net_kill serve
subtype    cross_court_net_shot short_serve       the stroke type, only where ShuttleSet
           long_serve                             labels a subtype we can learn
hand       forehand backhand around_head          model (``models/stroke_attrs.json``)
contact    high low  (above / below the net)      model
direction  straight cross                         model
pace       fast medium slow                       measured: hitter -> receiver court speed,
                                                  against the usual speed of that family
role       attacking neutral defensive            rule on family + contact + incoming shot
intent     attack move control defend transition  rule on family + role
=========  ====================================  =========================================

The three model heads are scikit-learn ``HistGradientBoostingClassifier`` trained on ShuttleSet
(``scripts/learn_stroke_attrs.py``) and run as JSON trees (numpy only). ``role`` and ``intent``
have no labels anywhere: they are a coach's reading of the measured layers and are marked
``source="rule"``. Stick smash, slice smash/drop, spinning net shot and flick/drive serves are
not labelled in ShuttleSet (or not separable in its labels), so they are never output.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import (
    EventsResult,
    PlayerTracks,
    ShuttleTrack,
    StrokeAttr,
    StrokeDetail,
)
from ..gbt import GBT
from .features import StrokeFeatures, _kp, _Players, body_span

ATTRS_NAME = "attrs_v2"
BUNDLED = Path(__file__).with_name("models") / "stroke_attrs.json"

FAMILY = {"short_serve": "serve", "long_serve": "serve", "cross_net": "net_shot"}
SERVES = ("short_serve", "long_serve")
SUBTYPE = {"cross_net": "cross_court_net_shot", "short_serve": "short_serve", "long_serve": "long_serve"}

HAND_FEATURES = ("right_off", "near_wrist_right", "wrist_gap", "facing", "l_wrist_h", "r_wrist_h", "rw_off", "lw_off")
DOM_FEATURES = ("dom_off", "near_wrist_dom", "dom_gap", "right_handed", "dom_wrist_off")
# The hand head reads only where the wrists and the shuttle are relative to the body (plus contact
# height). The full feature set scores a little higher on broadcast ShuttleSet, but its pose-window
# and court features depend on the camera and pushed it to "backhand" on low phone/DJI footage.
HAND_HEAD_FEATURES = ("rw_off", "lw_off", "dom_wrist_off", "right_handed", "facing", "l_wrist_h", "r_wrist_h",
                      "contact_h", "right_off")
MIN_KP = 0.3
OVERHEAD_H = 1.2  # contact_h above which a shot counts as overhead (handedness votes)

ZH = {
    "clear": "高远球", "smash": "杀球", "drop": "吊球", "drive": "平抽", "block": "挡网", "push": "推球",
    "lift": "挑球", "net_shot": "放网", "net_kill": "扑球", "serve": "发球",
    "cross_court_net_shot": "勾对角", "short_serve": "发短球", "long_serve": "发高远球",
    "forehand": "正手", "backhand": "反手", "around_head": "头顶",
    "straight": "直线", "cross": "斜线",
    "high": "击球点较高", "low": "击球点较低",
    "fast": "出球快", "medium": "", "slow": "出球慢",
    "attacking": "主动进攻", "neutral": "相持", "defensive": "被动防守",
}
INTENT_ZH = {
    "attack": "直接施压，逼对手被动或失误",
    "move": "调动对手前后跑动，拉开空当",
    "control": "控制网前，逼对手起球，为下一拍进攻创造机会",
    "defend": "化解对手进攻，先保证回球质量",
    "transition": "争取时间回位，由被动转入相持",
}


# ------------------------------------------------------------------ features
def _hand_row(p, contact: tuple[float, float] | None, side: str | None = None) -> list[float]:
    """Where the shuttle and each wrist are relative to the hitter's body, in the hitter's own
    left/right. The wrist offsets need no shuttle: a forehand keeps the racket wrist on the racket
    side of the hips, a backhand brings it across the body."""
    out = dict.fromkeys(HAND_FEATURES, math.nan)
    if p is None:
        return list(out.values())
    ls, rs = _kp(p, 5, MIN_KP), _kp(p, 6, MIN_KP)
    lw, rw = _kp(p, 9, MIN_KP), _kp(p, 10, MIN_KP)
    sp = body_span(p, MIN_KP)
    ankle_y, span = sp if sp else (p.bbox[3], p.bbox[3] - p.bbox[1])
    span = max(span, 10.0)
    if ls and rs:
        axis = 1.0 if rs[0] >= ls[0] else -1.0  # +1: the player's right is image right (back to the camera)
        out["facing"] = (rs[0] - ls[0]) / span
        hips = [q for i in (11, 12) if (q := _kp(p, i, MIN_KP))]
        cx = sum(q[0] for q in hips) / len(hips) if hips else (ls[0] + rs[0]) / 2.0
        if contact:
            out["right_off"] = (contact[0] - cx) * axis / span
        # players face the net, so their court half fixes their left/right more reliably than
        # the shoulders do (side-on, the shoulders overlap and their order is noise)
        sa = {"near": 1.0, "far": -1.0}.get(side or "", axis)
        if rw:
            out["rw_off"] = (rw[0] - cx) * sa / span
        if lw:
            out["lw_off"] = (lw[0] - cx) * sa / span
    if lw:
        out["l_wrist_h"] = (ankle_y - lw[1]) / span
    if rw:
        out["r_wrist_h"] = (ankle_y - rw[1]) / span
    if contact and lw and rw:
        dl = math.hypot(contact[0] - lw[0], contact[1] - lw[1])
        dr = math.hypot(contact[0] - rw[0], contact[1] - rw[1])
        out["near_wrist_right"] = 1.0 if dr < dl else 0.0
        out["wrist_gap"] = (dl - dr) / span
    return list(out.values())


def _dom_rows(hand: np.ndarray, contact_h: np.ndarray, hitters: Sequence[str | None]) -> np.ndarray:
    """Handedness per player from their overhead shots (the racket wrist is the higher one),
    then the hand features re-expressed relative to the racket side."""
    votes: dict[str, list[int]] = defaultdict(list)
    li, ri = HAND_FEATURES.index("l_wrist_h"), HAND_FEATURES.index("r_wrist_h")
    for h, c, who in zip(hand, contact_h, hitters):
        if who and c > OVERHEAD_H and not (np.isnan(h[li]) or np.isnan(h[ri])):
            votes[who].append(1 if h[ri] > h[li] else -1)
    out = np.full((len(hand), len(DOM_FEATURES)), np.nan)
    for k, (h, who) in enumerate(zip(hand, hitters)):
        v = votes.get(who or "", [])
        d = float(np.sign(np.mean(v))) if v else 0.0
        if d == 0.0:
            out[k, 3] = 0.0
            continue
        nw = h[1]
        wo = h[HAND_FEATURES.index("rw_off")] if d > 0 else -h[HAND_FEATURES.index("lw_off")]
        out[k] = [h[0] * d, (nw if d > 0 else 1.0 - nw) if not np.isnan(nw) else np.nan, h[2] * d, d, wo]
    return out


def attr_names() -> tuple[str, ...]:
    from ..events.pose_hits import FEATURES as POSE
    from .learned import FEATURES_V2

    return (*FEATURES_V2, *(f"pw_{n}" for n in POSE), *HAND_FEATURES, *DOM_FEATURES)


def attr_matrix(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    feats: Sequence[StrokeFeatures],
    fps: float,
    X_v2: np.ndarray | None = None,
) -> np.ndarray:
    """One row of ``attr_names()`` per entry of ``feats``. ``feats`` should cover the whole video
    (handedness is voted over all of a player's shots)."""
    from ..events import pose_hits as PH
    from ..events.adapters import poses_from_contract
    from .learned import FEATURES_V2, full_matrix

    n = len(feats)
    if X_v2 is None:
        X_v2 = full_matrix(feats, events, shuttle, players, "v2") if n else np.zeros((0, len(FEATURES_V2)))
    shots = {s.shot_id: s for s in events.shots}
    ser = PH.side_series(poses_from_contract(players)) if players else {}
    pl = _Players(players, 3)
    frame_ms = 1000.0 / fps if fps > 0 else 33.3
    pose = np.full((n, len(PH.FEATURES)), np.nan)
    hand = np.full((n, len(HAND_FEATURES)), np.nan)
    hitters: list[str | None] = []
    for k, f in enumerate(feats):
        s = shots[f.shot_id]
        hitters.append(s.hitter)
        p = pl.find(s.frame, s.hitter) if s.hitter else None
        side = getattr(p, "side", None) if p is not None else None
        if side not in ser:
            side = next((sd for sd, se in ser.items()
                         if (b := se.near(s.t_ms, 120.0)) is not None and b.player_id == s.hitter), None)
        if side is not None:
            pose[k] = PH.window_features(ser[side], s.t_ms, frame_ms)
        hand[k] = _hand_row(p, tuple(s.shuttle_px) if s.shuttle_px else None, side)
    ci = FEATURES_V2.index("contact_h")
    dom = _dom_rows(hand, np.nan_to_num(X_v2[:, ci], nan=-1.0) if n else np.zeros(0), hitters)
    return np.hstack([X_v2, pose, hand, dom]) if n else np.zeros((0, len(attr_names())))


# ------------------------------------------------------------------ model
class AttrModel:
    """The bundled attribute heads (JSON trees) plus the pace reference speeds."""

    def __init__(self, spec: dict[str, Any]):
        if spec.get("kind") != "stroke_attrs":
            raise ValueError("not a stroke attribute model")
        if tuple(spec["features"]) != attr_names():
            raise ValueError("stroke attribute model was trained on different features; retrain it")
        self.heads = {k: GBT(v) for k, v in spec["heads"].items()}
        names = attr_names()
        cols = spec.get("head_features", {})
        self.cols = {k: [names.index(n) for n in cols[k]] if k in cols else None for k in self.heads}
        self.pace = {k: tuple(v) for k, v in spec.get("pace", {}).items()}  # family -> (p33, p67) m/s
        self.meta = spec.get("meta", {})

    def predict(self, X: np.ndarray) -> dict[str, list[tuple[str, float]]]:
        out = {}
        for k, m in self.heads.items():
            cols = self.cols[k]
            P = m.predict_proba(X if cols is None else X[:, cols]) if len(X) else np.zeros((0, len(m.classes)))
            out[k] = [(str(m.classes[i]), float(row[i])) for row in P for i in [int(np.argmax(row))]]
        return out


@lru_cache(maxsize=4)
def _load(path: str) -> AttrModel | None:
    p = Path(path)
    if not p.is_file():
        return None
    return AttrModel(json.loads(p.read_text(encoding="utf-8")))


def get_model(path: str | Path | None = None) -> AttrModel | None:
    return _load(str(path or BUNDLED))


# ------------------------------------------------------------------ rules
def pace_of(family: str | None, speed: float | None, ref: dict[str, tuple[float, float]]) -> str | None:
    if family is None or speed is None or family not in ref or (isinstance(speed, float) and math.isnan(speed)):
        return None
    lo, hi = ref[family]
    return "fast" if speed >= hi else "slow" if speed <= lo else "medium"


def role_of(shot_type: str | None, contact: str | None, prev_type: str | None) -> str | None:
    """Attacking / neutral / defensive: a coach's reading, not a learned label."""
    fam = FAMILY.get(shot_type or "", shot_type)
    if fam is None:
        return None
    if fam in ("smash", "net_kill"):
        return "attacking"
    if fam in ("push", "drive", "drop"):
        return "attacking" if contact == "high" else "neutral"
    if fam == "clear":
        return "defensive" if contact == "low" else "neutral"
    if fam in ("block", "lift"):
        return "defensive" if prev_type in ("smash", "net_kill", "push", "drive") or contact == "low" else "neutral"
    return "neutral"  # net shots, serves


def intent_of(shot_type: str | None, role: str | None) -> str | None:
    fam = FAMILY.get(shot_type or "", shot_type)
    if fam is None:
        return None
    if fam in ("smash", "net_kill") or (fam in ("push", "drive") and role == "attacking"):
        return "attack"
    if (fam in ("drop", "clear") and role != "defensive") or shot_type == "long_serve":
        return "move"
    if fam in ("net_shot", "serve", "push", "drive"):
        return "control"
    if fam == "block" or (fam == "lift" and role == "defensive"):
        return "defend"
    return "transition"  # defensive clears, neutral lifts


def short_label(shot_type: str, d: StrokeDetail | None) -> str:
    """The learned layers in front of the stroke name: 正手直线杀球 / 反手勾对角 / 头顶吊球."""
    name = ZH.get(SUBTYPE.get(shot_type, ""), "") or ZH.get(FAMILY.get(shot_type, shot_type), shot_type)
    if d is None:
        return name
    hand = d.hand.value if d.hand and FAMILY.get(shot_type) != "serve" else None
    direction = d.direction.value if d.direction and shot_type not in ("cross_net", *SERVES) else None
    return (ZH[hand] if hand else "") + (ZH[direction] if direction else "") + name


def describe(shot_type: str | None, detail: StrokeDetail | None) -> str:
    """One Chinese sentence: 正手直线杀球，击球点较高，出球快，属于主动进攻，意图是…"""
    if shot_type is None:
        return ""
    d = detail or StrokeDetail()
    v = lambda a: a.value if a is not None else None  # noqa: E731
    parts = [short_label(shot_type, d)]
    if v(d.contact):
        parts.append(ZH[v(d.contact)])
    if v(d.pace) and ZH.get(v(d.pace)):
        parts.append(ZH[v(d.pace)])
    if v(d.role):
        parts.append("属于" + ZH[v(d.role)])
    if v(d.intent):
        parts.append("意图是" + INTENT_ZH[v(d.intent)])
    return "，".join(parts) + "。"


def retype(d: StrokeDetail | None, shot_type: str | None) -> StrokeDetail | None:
    """The description after a person corrected the stroke type: hand / direction / contact are
    kept, family / subtype / role / intent / text follow the new type (pace only if the family is
    unchanged, since it is relative to the family's usual speed)."""
    if d is None or shot_type is None:
        return None
    fam = FAMILY.get(shot_type, shot_type)
    role = role_of(shot_type, d.contact.value if d.contact else None, None)
    intent = intent_of(shot_type, role)
    new = d.model_copy(update={
        "family": fam, "subtype": SUBTYPE.get(shot_type), "pace": d.pace if fam == d.family else None,
        "hand": None if fam == "serve" else d.hand,
        "direction": None if fam == "serve" else d.direction,
        "role": StrokeAttr(value=role, source="rule") if role else None,
        "intent": StrokeAttr(value=intent, source="rule") if intent else None,
    })
    return new.model_copy(update={"description_zh": describe(shot_type, new)})


# ------------------------------------------------------------------ stage helper
def details(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    feats: Sequence[StrokeFeatures],
    fps: float,
    types: dict[str, str | None],
    model: AttrModel | None,
    X_v2: np.ndarray | None = None,
) -> dict[str, StrokeDetail]:
    """``StrokeDetail`` per shot id (``types``: the chosen stroke type of each shot)."""
    if not feats:
        return {}
    preds: dict[str, list[tuple[str, float]]] = {}
    if model is not None:
        preds = model.predict(attr_matrix(events, shuttle, players, feats, fps, X_v2))
    shots = {s.shot_id: s for s in events.shots}
    prev_of: dict[str, str | None] = {}
    last: dict[str | None, str | None] = {}
    for f in sorted(feats, key=lambda f: shots[f.shot_id].frame):
        rid = shots[f.shot_id].rally_id
        prev_of[f.shot_id] = None if f.first_in_rally else last.get(rid)
        last[rid] = types.get(f.shot_id)
    out = {}
    for k, f in enumerate(feats):
        t = types.get(f.shot_id)
        fam = FAMILY.get(t or "", t)
        attr = {h: StrokeAttr(value=p[k][0], p=round(p[k][1], 3), source="model") for h, p in preds.items()}
        if fam == "serve":  # too few labelled serves to learn a hand for; singles serves go diagonal anyway
            attr.pop("hand", None)
            attr.pop("direction", None)
        pace = pace_of(fam, f.court_speed_mps, model.pace if model else {})
        contact = attr.get("contact")
        role = role_of(t, contact.value if contact else None, prev_of.get(f.shot_id))
        intent = intent_of(t, role)
        d = StrokeDetail(
            family=fam,
            subtype=SUBTYPE.get(t or ""),
            hand=attr.get("hand"),
            contact=contact,
            direction=attr.get("direction"),
            pace=StrokeAttr(value=pace, source="measured") if pace else None,
            role=StrokeAttr(value=role, source="rule") if role else None,
            intent=StrokeAttr(value=intent, source="rule") if intent else None,
            source=ATTRS_NAME if model is not None else None,
        )
        out[f.shot_id] = d.model_copy(update={"description_zh": describe(t, d)})
    return out
