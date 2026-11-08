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
