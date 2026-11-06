"""Doubles: format detection, four stable identities and team assignment."""

import numpy as np
import pytest

from rallyiq.perception.players.filtering import build_candidates
from rallyiq.perception.players.match_format import decide_format
from rallyiq.perception.players.runner import PlayerTrackingConfig, track_players
from rallyiq.perception.players.stage import build_tracks
from rallyiq.perception.players.tracker import DoublesTracker

from .synth import H_IMG_TO_COURT, IMAGE_SIZE, person_at_court


class FakeEstimator:
    def __init__(self, per_frame):
        self.per_frame = iter(per_frame)

    def __call__(self, img):
        return next(self.per_frame)


def _frames(n):
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    for i in range(n):
        yield i, i * 1000 / 30, img


def _doubles_positions(t):
    """Near pair side by side, far pair front/back, everyone moving."""
    return {
        "nl": (1.6 + 0.6 * np.sin(2 * t), 2.8 + 0.6 * np.cos(2.5 * t)),
        "nr": (4.5 + 0.6 * np.sin(1.7 * t + 1), 3.2 + 0.7 * np.sin(2 * t)),
        "ff": (3.0 + 1.2 * np.sin(1.3 * t), 8.6 + 0.4 * np.sin(3 * t)),
        "fb": (3.1 - 1.2 * np.sin(1.1 * t), 11.6 + 0.5 * np.cos(2 * t)),
    }


def _doubles_frames(n, drop=None):
    per_frame = []
    for i in range(n):
        pos = _doubles_positions(i / 30)
        dets = [person_at_court(*xy) for k, xy in pos.items() if not (drop and drop(i, k))]
        dets.append(person_at_court(-2.0, 6.0))  # coach by the side, outside the play area
        per_frame.append(dets)
    return per_frame


def _cands(per_frame):
    return [build_candidates(d, H_IMG_TO_COURT, IMAGE_SIZE) for d in per_frame]


def test_format_detection_singles_vs_doubles():
    d = decide_format(_cands(_doubles_frames(60)))
    assert d.match_format == "doubles" and d.confidence > 0.8
    singles = [[person_at_court(3.0, 2.5), person_at_court(3.0, 10.5), person_at_court(-2.0, 6.0)]] * 60
    d = decide_format(_cands(singles))
    assert d.match_format == "singles" and d.confidence > 0.8
    # Singles with a line judge standing just inside the run-off now and then.
    noisy = [s + ([person_at_court(6.6, 1.0)] if i % 5 == 0 else []) for i, s in enumerate(singles)]
    assert decide_format(_cands(noisy)).match_format == "singles"
    assert decide_format([]).confidence == 0.0


def test_doubles_tracks_four_players_with_stable_ids_and_teams():
    n = 120
    # Far back player missed for a few frames (occluded by the net post).
    per_frame = _doubles_frames(n, drop=lambda i, k: k == "fb" and 50 <= i < 54)
    res = track_players(_frames(n), FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    assert res.match_format == "doubles" and res.format_confidence > 0.8
    assert len(res) == n

    def who(r):
        out = {}
        for p in r.players:
            xy = p.foot_court
            key = ("n" if xy[1] < 6.7 else "f") + ("l" if xy[0] < 3.05 else "r") if xy[1] < 6.7 else (
                "ff" if xy[1] < 10.0 else "fb")
            out[key] = (p.player_id, p.team)
        return out

    first = who(res[0])
    # Team A (P1/P3) starts on the near half, team B (P2/P4) far.
    assert {first["nl"][1], first["nr"][1]} == {"A"} and {first["ff"][1], first["fb"][1]} == {"B"}
    assert {first["nl"][0], first["nr"][0]} == {"P1", "P3"}
    for r in res:
        assert len(r.players) == 4
        assert who(r) == first
    gap = [p for r in res[50:54] for p in r.players if p.player_id == first["fb"][0]]
    assert all(p.interpolated for p in gap)

    tracks = build_tracks("vid", res, "test")
    assert tracks.match_format == "doubles"
    assert {p.team for p in tracks.frames[0].players} == {"A", "B"}


def test_forced_format_skips_detection():
    per_frame = _doubles_frames(10)
    cfg = PlayerTrackingConfig(match_format="singles")
    res = track_players(_frames(10), FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT, cfg)
    assert res.match_format == "singles" and res.format_confidence is None
    assert all(len(r.players) == 2 for r in res)
    with pytest.raises(ValueError):
        track_players(_frames(1), FakeEstimator(per_frame), None, PlayerTrackingConfig(match_format="mixed"))


def test_doubles_change_of_ends_keeps_partners_together():
    """Both pairs walk to the other end in view; ids and teams follow the people."""
    tr = DoublesTracker()
    ys = np.linspace(0, 1, 61)
    seen = []
    for i, s in enumerate(ys):
        dets = [
            person_at_court(1.5, 2.5 + 8 * s), person_at_court(4.5, 3.5 + 8 * s),
            person_at_court(1.0, 10.5 - 8 * s), person_at_court(5.0, 9.5 - 8 * s),
        ]
        out = tr.update(i, build_candidates(dets, H_IMG_TO_COURT, IMAGE_SIZE), calibrated=True)
        seen.append({p.player_id: (round(float(p.foot_court[0]), 1), p.team, p.side) for p in out})
    assert seen[0]["P1"][1] == "A" and seen[0]["P1"][2] == "near"
    # Same x (lane) at the end as at the start for every id: nobody swapped.
    assert all(seen[0][k][0] == seen[-1][k][0] for k in seen[0])
    assert {seen[-1][k][2] for k in ("P1", "P3")} == {"far"}
    assert {seen[-1][k][2] for k in ("P2", "P4")} == {"near"}
