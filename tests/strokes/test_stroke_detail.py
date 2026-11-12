"""Layered shot description (rallyiq/strokes/detail.py): rules, bundled attribute model, plumbing.

The synthetic rallies say nothing about accuracy (that is measured on ShuttleSet by
scripts/learn_stroke_attrs.py); these tests check shapes, sources and wording.
"""

from __future__ import annotations

import numpy as np

from rallyiq.contracts import StrokeAttr, StrokeDetail
from rallyiq.strokes import classify, shot_features
from rallyiq.strokes import detail as DT
from rallyiq.strokes.synthetic import make_rallies


def test_rules_role_and_intent():
    assert DT.role_of("smash", "high", None) == "attacking"
    assert DT.role_of("drop", "high", None) == "attacking"
    assert DT.role_of("drop", "low", None) == "neutral"
    assert DT.role_of("lift", "low", "smash") == "defensive"
    assert DT.role_of("clear", "low", None) == "defensive"
    assert DT.role_of("net_shot", "low", None) == "neutral"
    assert DT.role_of(None, None, None) is None
    assert DT.intent_of("smash", "attacking") == "attack"
    assert DT.intent_of("clear", "neutral") == "move"
    assert DT.intent_of("clear", "defensive") == "transition"
    assert DT.intent_of("cross_net", "neutral") == "control"
    assert DT.intent_of("block", "defensive") == "defend"


def test_pace_against_family_reference():
    ref = {"smash": (18.0, 22.0)}
    assert DT.pace_of("smash", 25.0, ref) == "fast"
    assert DT.pace_of("smash", 20.0, ref) == "medium"
    assert DT.pace_of("smash", 10.0, ref) == "slow"
    assert DT.pace_of("drop", 10.0, ref) is None
    assert DT.pace_of("smash", None, ref) is None


def test_describe_reads_like_a_coach():
    d = StrokeDetail(
        hand=StrokeAttr(value="forehand", p=0.8, source="model"),
        direction=StrokeAttr(value="straight", p=0.9, source="model"),
        contact=StrokeAttr(value="high", p=0.9, source="model"),
        pace=StrokeAttr(value="fast", source="measured"),
        role=StrokeAttr(value="attacking", source="rule"),
        intent=StrokeAttr(value="attack", source="rule"),
    )
    text = DT.describe("smash", d)
    assert text.startswith("正手直线杀球，击球点较高，出球快，属于主动进攻，意图是")
    assert DT.short_label("cross_net", d) == "正手勾对角"  # direction is already in the name
    assert DT.short_label("short_serve", d) == "发短球"  # no hand or direction on serves
    assert DT.describe(None, d) == ""


def test_details_without_model_has_only_rule_layers():
    v = make_rallies(n_rallies=2, seed=3)
    feats = shot_features(v.events, v.shuttle, v.players, v.fps)
    out = DT.details(v.events, v.shuttle, v.players, feats, v.fps, dict(v.truth), None)
    assert set(out) == {f.shot_id for f in feats}
    for sid, d in out.items():
        assert d.hand is None and d.direction is None and d.contact is None
        assert d.family == DT.FAMILY.get(v.truth[sid], v.truth[sid])
        assert d.role is not None and d.role.source == "rule"
        assert d.description_zh


def test_bundled_model_loads_and_predicts():
    m = DT.get_model()
    assert m is not None and set(m.heads) == {"hand", "contact", "direction"}
    assert set(m.heads["hand"].classes) == {"forehand", "backhand", "around_head"}
    v = make_rallies(n_rallies=2, seed=4)
    feats = shot_features(v.events, v.shuttle, v.players, v.fps)
    X = DT.attr_matrix(v.events, v.shuttle, v.players, feats, v.fps)
    assert X.shape == (len(feats), len(DT.attr_names()))
    pred = m.predict(X)
    assert all(len(p) == len(feats) and all(0.0 <= q <= 1.0 for _, q in p) for p in pred.values())
    assert DT.attr_matrix(v.events, v.shuttle, v.players, [], v.fps).shape == (0, len(DT.attr_names()))


def test_handedness_from_overhead_shots():
    li, ri = DT.HAND_FEATURES.index("l_wrist_h"), DT.HAND_FEATURES.index("r_wrist_h")
    hand = np.full((3, len(DT.HAND_FEATURES)), np.nan)
    hand[:, li], hand[:, ri] = [1.2, 1.1, 0.5], [1.8, 1.7, 0.6]  # right wrist higher on overhead shots
    hand[:, 0] = 0.4
    dom = DT._dom_rows(hand, np.array([1.5, 1.6, 0.5]), ["P1", "P1", "P1"])
    assert (dom[:, 3] == 1.0).all() and np.allclose(dom[:, 0], 0.4)
    dom = DT._dom_rows(hand, np.array([0.5, 0.5, 0.5]), ["P1", "P1", "P1"])  # no overhead shot: unknown
    assert (dom[:, 3] == 0.0).all() and np.isnan(dom[:, 0]).all()


def test_classify_attaches_detail_and_retype_keeps_measured_layers():
    v = make_rallies(n_rallies=2, seed=6)
    res = classify(v.events, v.shuttle, v.players, v.fps, detail=True, attrs=DT.get_model())
    typed = [s for s in res.shots if s.stroke.shot_type]
    assert typed and all(s.stroke.detail is not None and s.stroke.detail.source == DT.ATTRS_NAME for s in typed)
    assert all(s.stroke.detail.hand is None or s.stroke.detail.hand.source == "model" for s in typed)
    plain = classify(v.events, v.shuttle, v.players, v.fps)
    assert all(s.stroke.detail is None for s in plain.shots)

    d = StrokeDetail(family="drop", hand=StrokeAttr(value="backhand", p=0.7, source="model"),
                     contact=StrokeAttr(value="high", p=0.8, source="model"),
                     pace=StrokeAttr(value="slow", source="measured"))
    r = DT.retype(d, "smash")
    assert r.family == "smash" and r.hand == d.hand and r.contact == d.contact and r.pace is None
    assert r.role.value == "attacking" and r.description_zh.startswith("反手杀球")
    assert DT.retype(d, None) is None and DT.retype(None, "smash") is None


def _pose(rw_x: float, side: str | None, shoulders_flipped: bool = False):
    from types import SimpleNamespace

    kp = [[0.0, 0.0, 0.0] for _ in range(17)]
    kp[0] = [100.0, 0.0, 1.0]                                   # nose
    ls, rs = (110.0, 90.0) if shoulders_flipped else (90.0, 110.0)
    kp[5], kp[6] = [ls, 20.0, 1.0], [rs, 20.0, 1.0]             # shoulders
    kp[11], kp[12] = [95.0, 55.0, 1.0], [105.0, 55.0, 1.0]      # hips, centre x = 100
    kp[9], kp[10] = [80.0, 40.0, 1.0], [rw_x, 40.0, 1.0]        # wrists
    kp[15], kp[16] = [95.0, 100.0, 1.0], [105.0, 100.0, 1.0]    # ankles
    return SimpleNamespace(keypoints=kp, bbox=[80.0, 0.0, 120.0, 100.0], side=side)


def test_wrist_offsets_follow_the_court_side():
    ri = DT.HAND_FEATURES.index("rw_off")
    near_fh = DT._hand_row(_pose(130.0, "near"), None, "near")
    assert near_fh[ri] > 0.25                                    # racket wrist out on the right: forehand side
    assert DT._hand_row(_pose(85.0, "near"), None, "near")[ri] < 0  # wrist across the body: backhand side
    # side-on, the shoulder order is noise: the court side decides the player's left/right
    assert DT._hand_row(_pose(130.0, "near", shoulders_flipped=True), None, "near")[ri] > 0.25
    # facing the camera from the far half, image-left is the player's right
    assert DT._hand_row(_pose(70.0, "far"), None, "far")[ri] > 0.25
    # unknown side: fall back to the shoulders
    assert DT._hand_row(_pose(130.0, None), None, None)[ri] > 0.25


def test_hand_head_reads_its_own_columns():
    m = DT.get_model()
    names = DT.attr_names()
    assert [names[i] for i in m.cols["hand"]] == list(DT.HAND_HEAD_FEATURES)
    assert m.cols["contact"] is None and m.cols["direction"] is None
    dom = DT._dom_rows(np.array([[0.3, 1.0, 0.2, 0.1, 0.9, 1.6, 0.4, -0.2]]), np.array([1.5]), ["P1"])
    assert dom[0, DT.DOM_FEATURES.index("dom_wrist_off")] == 0.4  # right-handed: the right wrist
