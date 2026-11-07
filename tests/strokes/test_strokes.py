"""Stroke-type classification: rules on synthetic 3D rallies, review policy, BST input prep.

The accuracy floors are regression floors on the synthetic generator, not claims
about real video.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rallyiq.api.corrections import CorrectionError, apply_corrections, validate
from rallyiq.contracts import (
    SHOT_TYPES,
    Correction,
    EventsResult,
    Rally,
    ShotEvent,
    StrokesResult,
    VideoMeta,
)
from rallyiq.pipeline.context import StageContext
from rallyiq.store import Store
from rallyiq.strokes import bst, classify, merge_strokes
from rallyiq.strokes.synthetic import make_rallies
from rallyiq.strokes.taxonomy import EN, SHUTTLESET, ZH, from_shuttleset


def _run(seeds=range(4), **kw):
    hits = Counter()
    total = Counter()
    ok = ok_right = 0
    statuses = Counter()
    for seed in seeds:
        v = make_rallies(n_rallies=8, seed=seed, **kw)
        res = classify(v.events, v.shuttle, v.players, v.fps)
        assert [s.shot_id for s in res.shots] == [s.shot_id for s in v.events.shots]
        for s in res.shots:
            truth = v.truth[s.shot_id]
            total[truth] += 1
            hits[truth] += s.stroke.shot_type == truth
            statuses[s.stroke.status] += 1
            if s.stroke.status == "ok":
                ok += 1
                ok_right += s.stroke.shot_type == truth
    acc = sum(hits.values()) / sum(total.values())
    return acc, (ok_right / ok if ok else None), {t: hits[t] / total[t] for t in total}, statuses


@pytest.mark.parametrize(
    "kw, min_acc",
    [
        ({}, 0.85),
        ({"dropout": 0.15}, 0.82),
        ({"frame_jitter": 2}, 0.8),
        ({"px_noise": 3.0, "dropout": 0.1, "frame_jitter": 1}, 0.82),
    ],
)
def test_rules_on_synthetic_rallies(kw, min_acc):
    acc, ok_precision, per_type, _ = _run(**kw)
    assert acc >= min_acc, per_type
    # "ok" must mean something: almost always right.
    assert ok_precision is not None and ok_precision >= 0.9


def test_clear_cut_types_are_recognised():
    _, _, per_type, _ = _run()
    for t in ("long_serve", "short_serve", "clear", "smash", "lift"):
        assert per_type[t] >= 0.9, (t, per_type)
    assert min(per_type.values()) >= 0.4, per_type


def test_without_court_positions_nothing_is_ok():
    acc, _, _, statuses = _run(seeds=range(2), court_known=False)
    assert statuses["ok"] == 0
    assert acc >= 0.5


def _bare_events(**shot_kw) -> EventsResult:
    shots = [ShotEvent(shot_id=f"s{i}", rally_id="r1", frame=10 + 20 * i, t_ms=333 + 667 * i, confidence=0.8,
                       status="ok", **shot_kw) for i in range(3)]
    rally = Rally(rally_id="r1", start_frame=0, end_frame=90, start_ms=0, end_ms=3000,
                  shot_ids=[s.shot_id for s in shots], confidence=0.8, status="ok")
    return EventsResult(video_id="v", rallies=[rally], shots=shots)


def test_no_evidence_is_undeterminable_not_a_guess():
    res = classify(_bare_events(), None, None, 30.0)
    for s in res.shots:
        assert s.stroke.status == "undeterminable"
        assert s.stroke.shot_type is None
        assert "insufficient_evidence" in s.stroke.reasons
        assert s.stroke.evidence["contact_h"] is None


def test_review_policy_follows_the_hit():
    v = make_rallies(n_rallies=3, seed=1)
    weak = v.events.model_copy(update={"shots": [s.model_copy(update={"status": "needs_review"}) for s in v.events.shots]})
    res = classify(weak, v.shuttle, v.players, v.fps)
    assert all(s.stroke.status != "ok" for s in res.shots)
    assert all("hit_needs_review" in s.stroke.reasons for s in res.shots if s.stroke.shot_type)


def test_candidates_are_ranked_probabilities():
    v = make_rallies(n_rallies=2, seed=3)
    for s in classify(v.events, v.shuttle, v.players, v.fps).shots:
        ps = [c.p for c in s.stroke.candidates]
        assert ps == sorted(ps, reverse=True) and 0 < ps[0] <= 1
        assert s.stroke.candidates[0].shot_type == s.stroke.shot_type
        assert s.stroke.classifier == "rules_v1"


def test_merge_and_human_label():
    v = make_rallies(n_rallies=1, seed=2)
    res = classify(v.events, v.shuttle, v.players, v.fps)
    merged = merge_strokes(v.events, res)
    assert v.events.shots[0].stroke is None  # raw events untouched
    s0 = merged.shots[0]
    assert s0.stroke is not None and s0.shot_type == s0.stroke.shot_type

    ts = datetime(2026, 11, 6, tzinfo=UTC)
    c = Correction(correction_id="c1", video_id="v", seq=1, created_at=ts, op="set_shot_type",
                   payload={"shot_id": s0.shot_id, "shot_type": "drop"})
    out, skipped = apply_corrections(merged, [c], VideoMeta(video_id="v", filename="x", fps=30, frame_count=1,
                                                            width=1920, height=1080, duration_ms=1))
    s = next(x for x in out.shots if x.shot_id == s0.shot_id)
    assert not skipped
    assert s.shot_type == "drop" and s.stroke.source == "human" and s.stroke.status == "ok"
    assert s.source == s0.source and s.status == s0.status  # the hit itself is not "confirmed" by a type label

    validate("set_shot_type", {"shot_id": "s1", "shot_type": None})
    with pytest.raises(CorrectionError):
        validate("set_shot_type", {"shot_id": "s1", "shot_type": "tweener"})


def test_taxonomy_matches_contract():
    assert set(ZH) == set(EN) == set(SHOT_TYPES)
    assert from_shuttleset("Top_殺球") == ("smash", "far")
    assert from_shuttleset("Bottom_發短球") == ("short_serve", "near")
    assert from_shuttleset("未知球種") == (None, None)
    assert from_shuttleset("Top_過度切球") == ("drop", "far")  # spelling used by the set CSVs
    assert {v for v in SHUTTLESET.values() if v} == set(SHOT_TYPES)


# ------------------------------------------------------------------------------------ BST


def test_bst_input_preparation():
    v = make_rallies(n_rallies=1, seed=4)
    cfg = bst.BSTConfig(pose_style="JnB_bone", seq_len=100)
    pose, pos, shut, n = bst.build_clip(v.events, v.events.shots[1].shot_id, v.shuttle, v.players, 1920, 1080, cfg)
    assert pose.shape == (100, 2, 17 + 19, 2) and pos.shape == (100, 2, 2) and shut.shape == (100, 2)
    assert 0 < n <= 100
    assert np.all(pose[n:] == 0)  # padding
    live = pos[:n]
    assert np.all((live >= -0.05) & (live <= 1.05))
    # Top (far) player first: smaller y in the authors' frame.
    assert np.all(live[:, 0, 1] < 0.5) and np.all(live[:, 1, 1] > 0.5)
    assert np.all((shut[:n] >= 0) & (shut[:n] <= 1))
    assert np.abs(pose[:n, :, :17]).max() < 1.0  # box-diagonal normalisation

    jn2b = bst.pose_features(np.ones((4, 2, 17, 2), np.float32), "Jn2B")
    assert jn2b.shape == (4, 2, 17 + 2 * 19, 2)


def test_bst_seq_len_subsampling_matches_authors():
    j = np.ones((260, 2, 17, 2)); p = np.ones((260, 2, 2)); s = np.ones((260, 2))  # noqa: E702
    j2, p2, s2, n = bst.make_seq_len_same(100, j, p, s)
    assert j2.shape[0] == p2.shape[0] == s2.shape[0] == 100 and n == 87  # 260 % 100 > 50: stride 3, then zero padding
    _, p3, _, n3 = bst.make_seq_len_same(100, j[:40], p[:40], s[:40])
    assert p3.shape[0] == 100 and n3 == 40


def test_bst_label_folding_uses_hitter_side():
    probs = np.zeros(25)
    probs[bst.CLASSES_25.index("Bottom_殺球")] = 0.6
    probs[bst.CLASSES_25.index("Top_長球")] = 0.3
    probs[0] = 0.1
    near = bst.fold_probs(probs, "near")
    assert near["smash"] == pytest.approx(0.6) and near["clear"] == 0.0
    both = bst.fold_probs(probs, None)
    assert both["clear"] == pytest.approx(0.3)


def test_stage_falls_back_to_rules_without_bst(tmp_path: Path):
    v = make_rallies(n_rallies=1, seed=5)
    store = Store(tmp_path)
    vid, dest = store.create_video("x.mp4", ".mp4")
    dest.write_bytes(b"")
    meta = VideoMeta(video_id=vid, filename="x.mp4", fps=v.fps, frame_count=len(v.players.frames), width=1920,
                     height=1080, duration_ms=int(len(v.players.frames) * 1000 / v.fps))
    store.save_artifact(vid, "events", v.events.model_copy(update={"video_id": vid}))
    store.save_artifact(vid, "shuttle", v.shuttle)
    store.save_artifact(vid, "players", v.players)
    from rallyiq.strokes.stage import run

    ctx = StageContext(video_id=vid, video_path=dest, work_dir=tmp_path, store=store, meta=meta,
                       config={"strokes": {"classifier": "bst", "bst": {"repo": None}}})
    res = run(ctx)
    assert isinstance(res, StrokesResult) and res.classifier == "rules_v1" and len(res.shots) == len(v.events.shots)
    ctx.config["strokes"]["require_model"] = True
    with pytest.raises(bst.BSTUnavailable):
        run(ctx)
