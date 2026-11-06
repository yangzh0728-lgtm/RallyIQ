"""Rally segmentation and hit-candidate rules on synthetic footage.

Acceptance (README): a hit matches when it is within ±3 frames of a true hit
by the same player. The thresholds below are regression floors measured on
the synthetic generator, not claims about real video.
"""

from __future__ import annotations

import json

import pytest

from rallyiq.contracts import EventsResult, VideoMeta
from rallyiq.events import (
    Keypoint,
    PlayerPose,
    PoseFrame,
    RallyEndReason,
    ReviewStatus,
    ShuttleObs,
    segment_events,
)
from rallyiq.events.adapters import events_to_contract, poses_from_contract, shuttle_from_contract
from rallyiq.events.config import EventsConfig
from rallyiq.events.hits import (
    R_INFERRED_SERVE,
    R_INTERPOLATED,
    R_NO_POSE,
    R_OCCLUDED,
    R_SAME_HITTER,
    Candidate,
    _alternating_subset,
    _check_alternation,
)
from rallyiq.events.metrics import match_hits
from rallyiq.events.players import Proximity
from rallyiq.events.synthetic import make_video


def _score(kw: dict, seeds=range(6), n_rallies: int = 4):
    tp = fp = fn = 0
    for seed in seeds:
        v = make_video(n_rallies=n_rallies, seed=seed, **kw)
        rallies = segment_events(v.shuttle, v.poses)
        pred = [(h.frame, h.hitter_id) for r in rallies for h in r.hits]
        s = match_hits(pred, v.gt_hits)
        tp, fp, fn = tp + s.tp, fp + s.fp, fn + s.fn
    from rallyiq.events.metrics import HitScore

    return HitScore(tp, fp, fn)


@pytest.mark.parametrize(
    "kw, min_f1",
    [
        ({}, 0.95),
        ({"dropout": 0.1, "outlier_rate": 0.02}, 0.92),
        ({"occlude_hits": 0.3, "dropout": 0.05}, 0.90),
        ({"occlude_hits": 0.3, "interpolate_gaps": True}, 0.92),
        ({"fps": 25.0, "dropout": 0.05}, 0.92),
        ({"px_noise": 4.0, "kp_noise": 6.0, "dropout": 0.15, "outlier_rate": 0.03}, 0.75),
    ],
)
def test_hit_f1_on_synthetic(kw, min_f1):
    s = _score(kw)
    assert s.f1 >= min_f1, (s, s.precision, s.recall)


def test_ok_hits_are_trustworthy():
    """Anything marked ``ok`` should almost always be right; doubt goes to review."""
    good = bad = 0
    for seed in range(6):
        v = make_video(n_rallies=4, seed=seed, dropout=0.1, outlier_rate=0.02)
        for r in segment_events(v.shuttle, v.poses):
            for h in r.hits:
                if h.status is ReviewStatus.OK:
                    hit = any(abs(h.frame - f) <= 3 and p == h.hitter_id for f, p in v.gt_hits)
                    good, bad = good + hit, bad + (not hit)
    assert good > 50 and bad / (good + bad) <= 0.03


def test_rally_boundaries_and_end_reasons():
    v = make_video(n_rallies=4, seed=3, end_modes=["static", "lost"])
    rallies = segment_events(v.shuttle, v.poses)
    assert len(rallies) == len(v.rallies)
    for got, want in zip(rallies, v.rallies):
        assert abs(got.start_ms - want.start_ms) <= 150
        assert abs(got.end_ms - want.end_ms) <= 250
        expected = RallyEndReason.SHUTTLE_STATIC if want.end == "static" else RallyEndReason.SHUTTLE_LOST
        assert got.end_reason == expected
        # the floor landing at a static end is not reported as a hit
        assert all(h.t_ms < want.end_ms - 100 for h in got.hits)


def test_video_cut_mid_rally_ends_with_video_end():
    v = make_video(n_rallies=1, seed=1, end_modes=["lost"], trailing_ms=0)
    cut_ms = v.rallies[0].end_ms - 400
    shuttle = [o for o in v.shuttle if o.t_ms <= cut_ms]
    poses = [p for p in v.poses if p.t_ms <= cut_ms]
    rallies = segment_events(shuttle, poses, video_end_ms=cut_ms)
    assert len(rallies) == 1 and rallies[0].end_reason is RallyEndReason.VIDEO_END


def test_noise_between_rallies_is_not_a_rally():
    v = make_video(n_rallies=2, seed=5)
    # a few stray detections in the dead time between rallies
    gap_start = v.rallies[0].end_ms + 1600
    extra = [
        ShuttleObs(int(t / (1000 / 30)), t, 300.0 + i * 40, 500.0, True, 0.7)
        for i, t in enumerate([gap_start, gap_start + 33.3, gap_start + 66.7])
    ]
    shuttle = sorted(
        [o for o in v.shuttle if o.frame not in {e.frame for e in extra}] + extra, key=lambda o: o.frame
    )
    assert len(segment_events(shuttle, v.poses)) == 2


def test_occluded_and_interpolated_hits_are_flagged():
    v = make_video(n_rallies=3, seed=2, occlude_hits=1.0)
    hits = [h for r in segment_events(v.shuttle, v.poses) for h in r.hits]
    assert hits and sum(R_OCCLUDED in h.reasons for h in hits) >= len(hits) // 2
    assert all(h.status is ReviewStatus.NEEDS_REVIEW for h in hits if R_OCCLUDED in h.reasons)

    v = make_video(n_rallies=3, seed=2, occlude_hits=1.0, interpolate_gaps=True)
    hits = [h for r in segment_events(v.shuttle, v.poses) for h in r.hits]
    flagged = [h for h in hits if R_INTERPOLATED in h.reasons]
    assert flagged and all(h.status is ReviewStatus.NEEDS_REVIEW for h in flagged)


def test_without_poses_hits_have_no_hitter_and_need_review():
    v = make_video(n_rallies=2, seed=4, with_poses=False)
    rallies = segment_events(v.shuttle, [])
    hits = [h for r in rallies for h in r.hits]
    assert len(rallies) == 2 and hits
    assert all(h.hitter_id is None and R_NO_POSE in h.reasons for h in hits)
    assert all(h.status is ReviewStatus.NEEDS_REVIEW for h in hits)
    assert match_hits([(h.frame, None) for h in hits], [(f, None) for f, _ in v.gt_hits]).recall >= 0.7


def test_serve_inferred_when_shuttle_appears_in_flight():
    """If the shuttle is first seen at the server's racket, propose the serve for review."""
    v = make_video(n_rallies=1, seed=7)
    serve_ms = v.rallies[0].start_ms
    shuttle = [o if o.t_ms >= serve_ms - 1 else ShuttleObs(o.frame, o.t_ms) for o in v.shuttle]
    hits = segment_events(shuttle, v.poses)[0].hits
    first = hits[0]
    assert abs(first.t_ms - serve_ms) <= 100 and first.hitter_id == v.rallies[0].hits[0][1]
    if R_INFERRED_SERVE in first.reasons:
        assert first.status is ReviewStatus.NEEDS_REVIEW


def _cand(idx: int, player: str, conf: float) -> Candidate:
    prox = [Proximity(player, 0.9, 0.1, True, PlayerPose(player, []))]
    return Candidate(idx, 90.0, 1.0, 1.0, 1.0, prox=prox, confidence=conf)


def test_alternation_drops_weak_double_hit_and_keeps_needed_weak_hit():
    cfg = EventsConfig()
    # P1 strong, P1 weak (spurious), P2 strong
    out = _alternating_subset([_cand(0, "P1", 0.8), _cand(10, "P1", 0.5), _cand(20, "P2", 0.8)], cfg)
    assert [c.idx for c in out] == [0, 20]
    # P1 strong, P2 weak but needed, P1 strong
    out = _alternating_subset([_cand(0, "P1", 0.8), _cand(10, "P2", 0.46), _cand(20, "P1", 0.8)], cfg)
    assert [c.idx for c in out] == [0, 10, 20]
    # strong same-hitter pair is kept but flagged for review
    cands = _alternating_subset([_cand(0, "P1", 0.8), _cand(10, "P1", 0.8)], cfg)
    _check_alternation(cands)
    assert len(cands) == 2 and all(R_SAME_HITTER in c.reasons for c in cands)


def test_match_hits():
    s = match_hits([(10, "P1"), (31, "P2"), (60, "P1")], [(12, "P1"), (30, "P1"), (50, "P2")])
    assert (s.tp, s.fp, s.fn) == (1, 2, 2)
    s = match_hits([(10, "P1"), (11, "P1")], [(10, "P1")])
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_far_player_scale_and_unknown_identity():
    """A shuttle near an unresolved track still gives a hit, reported without a hitter."""
    v = make_video(n_rallies=1, seed=8)
    poses = [
        PoseFrame(
            f.frame,
            f.t_ms,
            [PlayerPose(p.player_id if p.player_id == "P1" else "track7", p.keypoints, p.side) for p in f.players],
        )
        for f in v.poses
    ]
    rallies = segment_events(v.shuttle, poses)
    ev = events_to_contract("vid", rallies)
    far = [s for s in ev.shots if s.reasons and "hitter_identity_unresolved" in s.reasons]
    assert far and all(s.hitter is None and s.status == "needs_review" for s in far)


def test_contract_round_trip_from_demo_perception():
    from rallyiq.pipeline import demo

    meta = VideoMeta(
        video_id="v1", filename="a.mp4", fps=30, frame_count=600, width=1920, height=1080, duration_ms=20000
    )
    players = demo.demo_players(meta, demo.demo_calibration(meta))
    shuttle = demo.demo_shuttle(meta, players)
    rallies = segment_events(
        shuttle_from_contract(shuttle), poses_from_contract(players), video_end_ms=meta.duration_ms
    )
    ev = events_to_contract(meta.video_id, rallies)
    EventsResult.model_validate(json.loads(ev.model_dump_json()))
    assert [r.rally_id for r in ev.rallies] == ["r001", "r002"]
    shot_ids = [s.shot_id for s in ev.shots]
    assert shot_ids == [f"s{i:04d}" for i in range(1, len(shot_ids) + 1)]
    assert all(sid in shot_ids for r in ev.rallies for sid in r.shot_ids)
    truth = [(f, "P1" if k % 2 == 0 else "P2") for _, k, f in demo._shots(meta)]
    # every hit with a shuttle path after it is found (the final hit of each demo rally has none)
    s = match_hits([(sh.frame, sh.hitter) for sh in ev.shots], truth)
    assert s.fp == 0 and s.tp >= len(truth) - 2
