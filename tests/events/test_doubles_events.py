"""Doubles: hits alternate between teams, not players; hitter picked among four."""

from __future__ import annotations

import pytest

from rallyiq.contracts import VideoMeta, team_of
from rallyiq.events import PlayerPose, segment_events
from rallyiq.events.adapters import events_to_contract, poses_from_contract, shuttle_from_contract
from rallyiq.events.config import EventsConfig
from rallyiq.events.hits import (
    R_HITTER_FROM_ALTERNATION,
    R_SAME_HITTER,
    Candidate,
    _alternating_subset,
    _check_alternation,
    _reattribute,
)
from rallyiq.events.metrics import HitScore, match_hits
from rallyiq.events.players import Proximity
from rallyiq.events.synthetic import make_video


def _score(kw: dict, seeds=range(6)):
    tp = fp = fn = 0
    teams_ok = teams_n = 0
    for seed in seeds:
        v = make_video(n_rallies=4, seed=seed, doubles=True, **kw)
        rallies = segment_events(v.shuttle, v.poses)
        pred = [(h.frame, h.hitter_id) for r in rallies for h in r.hits]
        s = match_hits(pred, v.gt_hits)
        tp, fp, fn = tp + s.tp, fp + s.fp, fn + s.fn
        for f, pid in pred:
            m = [g for g in v.gt_hits if abs(g[0] - f) <= 3]
            if m:
                teams_n += 1
                teams_ok += team_of(m[0][1]) == team_of(pid)
    return HitScore(tp, fp, fn), teams_ok / max(teams_n, 1)


@pytest.mark.parametrize(
    "kw, min_f1",
    [
        ({}, 0.93),
        ({"dropout": 0.1, "outlier_rate": 0.02}, 0.90),
        ({"occlude_hits": 0.3, "dropout": 0.05}, 0.90),
    ],
)
def test_doubles_hit_f1_on_synthetic(kw, min_f1):
    """Matching needs the exact player of four (README: same player ±3 frames)."""
    s, team_acc = _score(kw)
    assert s.f1 >= min_f1, (s, s.precision, s.recall)
    assert team_acc >= 0.97


def _cand(idx: int, players: list[tuple[str, str, float]], conf: float) -> Candidate:
    prox = [Proximity(pid, sc, 0.1, True, PlayerPose(pid, [], team=team)) for pid, team, sc in players]
    return Candidate(idx, 90.0, 1.0, 1.0, 1.0, prox=prox, confidence=conf)


def test_alternation_is_by_team_in_doubles():
    cfg = EventsConfig()
    # P1 then P3 (partners) back to back: a spurious second hit by the same team.
    out = _alternating_subset(
        [_cand(0, [("P1", "A", 0.9)], 0.8), _cand(10, [("P3", "A", 0.9)], 0.5), _cand(20, [("P2", "B", 0.9)], 0.8)],
        cfg,
    )
    assert [c.idx for c in out] == [0, 20]
    # P1, P4, P3: different players of alternating teams, all kept, nothing flagged.
    out = _alternating_subset(
        [_cand(0, [("P1", "A", 0.9)], 0.8), _cand(10, [("P4", "B", 0.9)], 0.5), _cand(20, [("P3", "A", 0.9)], 0.8)],
        cfg,
    )
    _check_alternation(out)
    assert [c.idx for c in out] == [0, 10, 20] and not any(R_SAME_HITTER in c.reasons for c in out)


def test_reattribute_uses_other_team_when_nearly_as_close():
    cfg = EventsConfig()
    a = _cand(0, [("P2", "B", 0.9)], 0.8)
    # Near the net the partner P4 is marginally closer than opponent P1 on screen.
    b = _cand(10, [("P4", "B", 0.7), ("P1", "A", 0.55)], 0.8)
    _reattribute([a, b], cfg)
    assert b.best.player_id == "P1" and R_HITTER_FROM_ALTERNATION in b.reasons
    # A clearly closer same-team player is not overridden; alternation flags it instead.
    c = _cand(20, [("P1", "A", 0.9), ("P2", "B", 0.2)], 0.8)
    d = _cand(30, [("P3", "A", 0.9), ("P4", "B", 0.1)], 0.8)
    _reattribute([c, d], cfg)
    _check_alternation([c, d])
    assert d.best.player_id == "P3" and R_SAME_HITTER in d.reasons


def test_doubles_contract_round_trip_from_demo():
    from rallyiq.pipeline import demo

    meta = VideoMeta(video_id="v1", filename="a.mp4", fps=30, frame_count=600, width=1920, height=1080,
                     duration_ms=20000)
    players = demo.demo_players(meta, demo.demo_calibration(meta), "doubles")
    assert players.match_format == "doubles"
    assert {p.player_id for p in players.frames[0].players} == {"P1", "P2", "P3", "P4"}
    shuttle = demo.demo_shuttle(meta, players)
    rallies = segment_events(shuttle_from_contract(shuttle), poses_from_contract(players),
                             video_end_ms=meta.duration_ms)
    ev = events_to_contract(meta.video_id, rallies, players.match_format)
    assert ev.match_format == "doubles"
    truth = [(f, players.frames[f].players[demo._hitter_index(k, 4)].player_id) for _, k, f in demo._shots(meta)]
    s = match_hits([(sh.frame, sh.hitter) for sh in ev.shots], truth)
    assert {sh.hitter for sh in ev.shots} >= {"P1", "P2", "P3", "P4"}
    assert s.fp <= 1 and s.tp >= len(truth) - 3
