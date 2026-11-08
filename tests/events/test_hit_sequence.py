"""Hit-sequence rules: two hits by one side in quick succession, and the reach of a large near player."""

from dataclasses import replace

from rallyiq.events.config import EventsConfig
from rallyiq.events.hits import R_HITTER_FROM_ALTERNATION, R_HITTER_FROM_SEQUENCE, Candidate, resolve_sequence
from rallyiq.events.players import PoseIndex, Proximity
from rallyiq.events.types import Keypoint, PlayerPose, PoseFrame

CFG = replace(EventsConfig(), same_side_min_gap_ms=800.0)


def _pose(pid: str, side: str) -> PlayerPose:
    return PlayerPose(pid, [], side=side)


NEAR, FAR = _pose("P1", "near"), _pose("P2", "far")


def _cand(t: float, conf: float, *prox: tuple[PlayerPose, float]) -> Candidate:
    ps = [Proximity(p.player_id, s, 0.1, True, p) for p, s in prox]
    return Candidate(int(t // 10), 90.0, 1.0, 1.0, 1.0, prox=ps, confidence=conf, t_ms=t)


def _who(hits):
    return [(h.t_ms, h.best.player_id) for h in hits]


def test_quick_repeat_is_handed_to_the_far_player_when_he_is_near_the_shuttle():
    hits = [_cand(0, 0.9, (NEAR, 0.8)), _cand(400, 0.9, (NEAR, 0.6), (FAR, 0.2)), _cand(1500, 0.9, (NEAR, 0.8))]
    out = resolve_sequence(hits, CFG)
    assert _who(out) == [(0, "P1"), (400, "P2"), (1500, "P1")]
    assert R_HITTER_FROM_ALTERNATION in out[1].reasons


def test_quick_repeat_with_nobody_on_the_other_side_drops_the_weaker():
    hits = [_cand(0, 0.7, (NEAR, 0.8)), _cand(300, 0.95, (NEAR, 0.8)), _cand(1400, 0.9, (FAR, 0.8))]
    assert _who(resolve_sequence(hits, CFG)) == [(300, "P1"), (1400, "P2")]


def test_hand_over_prefers_the_hit_that_then_alternates_with_its_neighbours():
    # far, near, near(quick), far: handing the 2nd to the far player would clash with the 1st; the 3rd fits.
    hits = [_cand(0, 0.9, (FAR, 0.8)), _cand(1000, 0.9, (NEAR, 0.7), (FAR, 0.3)),
            _cand(1300, 0.9, (NEAR, 0.7), (FAR, 0.2)), _cand(2300, 0.9, (NEAR, 0.8))]
    assert [p for _, p in _who(resolve_sequence(hits, CFG))] == ["P2", "P1", "P2", "P1"]


def test_slow_repeats_are_left_alone_and_rule_is_off_by_default():
    slow = [_cand(0, 0.9, (NEAR, 0.8)), _cand(1200, 0.9, (NEAR, 0.8), (FAR, 0.3))]
    assert _who(resolve_sequence(slow, CFG)) == [(0, "P1"), (1200, "P1")]
    quick = [_cand(0, 0.9, (NEAR, 0.8)), _cand(300, 0.9, (NEAR, 0.8))]
    assert len(resolve_sequence(quick, EventsConfig())) == 2


# -- hits without a known hitter -------------------------------------------------------------

def test_hit_with_nobody_near_takes_the_player_the_alternation_calls_for():
    # middle hit: shuttle beyond everyone's reach (score 0), the far player is the one who alternates
    hits = [_cand(0, 0.9, (NEAR, 0.8)), _cand(1000, 0.9, (NEAR, 0.0), (FAR, 0.0)), _cand(2000, 0.9, (NEAR, 0.8))]
    out = resolve_sequence(hits, EventsConfig())
    assert _who(out) == [(0, "P1"), (1000, "P2"), (2000, "P1")]
    assert out[1].hitter_known and R_HITTER_FROM_SEQUENCE in out[1].reasons
    assert all(R_HITTER_FROM_SEQUENCE not in h.reasons for h in (out[0], out[2]))


def test_hit_without_a_pose_uses_the_wider_pose_search():
    # first hit of the rally: no pose at the contact frame, players found a few frames away
    first = _cand(0, 0.9)
    first.prox_wide = [Proximity("P1", 0.0, 1.2, True, NEAR), Proximity("P2", 0.0, 2.0, True, FAR)]
    out = resolve_sequence([first, _cand(1000, 0.9, (NEAR, 0.8))], EventsConfig())
    assert _who(out) == [(0, "P2"), (1000, "P1")]


def test_hitter_fill_can_be_switched_off_and_needs_a_known_neighbour():
    hits = [_cand(0, 0.9, (NEAR, 0.8)), _cand(1000, 0.9, (NEAR, 0.0), (FAR, 0.0))]
    out = resolve_sequence([replace(h, prox=list(h.prox), reasons=[]) for h in hits], replace(EventsConfig(), hitter_fill=False))
    assert not out[1].hitter_known
    lonely = resolve_sequence([_cand(0, 0.9, (NEAR, 0.0), (FAR, 0.0))], EventsConfig())
    assert not lonely[0].hitter_known


# -- reach of a large near player -------------------------------------------------------------

def _person(pid: str, side: str, x: float, feet_y: float, h: float, wrists: bool = True) -> PlayerPose:
    """COCO-17 standing person, right arm raised above the head (racket up)."""
    top = feet_y - h
    k = [Keypoint(x, top, 1.0)] * 5  # head
    k += [Keypoint(x - 0.12 * h, top + 0.2 * h, 1.0), Keypoint(x + 0.12 * h, top + 0.2 * h, 1.0)]  # shoulders
    c = 1.0 if wrists else 0.0
    k += [Keypoint(x - 0.15 * h, top + 0.4 * h, c), Keypoint(x + 0.15 * h, top + 0.05 * h, c)]  # elbows
    k += [Keypoint(x - 0.15 * h, top + 0.55 * h, c), Keypoint(x + 0.15 * h, top + 0.0 * h, c)]  # wrists
    k += [Keypoint(x - 0.1 * h, top + 0.55 * h, 1.0), Keypoint(x + 0.1 * h, top + 0.55 * h, 1.0)]  # hips
    k += [Keypoint(x - 0.1 * h, top + 0.78 * h, 1.0), Keypoint(x + 0.1 * h, top + 0.78 * h, 1.0)]  # knees
    k += [Keypoint(x - 0.1 * h, feet_y, 1.0), Keypoint(x + 0.1 * h, feet_y, 1.0)]  # ankles
    return PlayerPose(pid, k, side=side)


def _scores(cfg: EventsConfig, xy):
    frame = PoseFrame(0, 0.0, [_person("P1", "near", 600, 1000, 320), _person("P2", "far", 640, 420, 90, wrists=False)])
    return {p.player_id: round(p.score, 3) for p in PoseIndex([frame], cfg).proximities(0, xy)}


def test_mixed_scale_shrinks_the_near_players_reach_among_small_far_players():
    mixed = replace(EventsConfig(), hitter_scale="mixed")
    # Shuttle at the small far player (wrists not seen): the big near player's own-height reach still covers
    # it and he would be credited. With the mixed scale he no longer reaches that far.
    xy = (650, 440)
    by_player, by_mixed = _scores(EventsConfig(), xy), _scores(mixed, xy)
    assert by_player["P1"] > 0.2 and by_player["P1"] > by_player["P2"]
    assert by_mixed["P1"] == 0.0
    # At the near player's racket nothing changes.
    assert _scores(EventsConfig(), (650, 640))["P1"] == _scores(mixed, (650, 640))["P1"]


# -- close-ups and replays between rallies ----------------------------------------------------

def test_off_court_views_only_when_most_of_the_video_is_calibrated():
    from rallyiq.events.pipeline import off_court_views

    segs = [(0, 99, True), (100, 149, False), (150, 399, True)]
    assert off_court_views(segs, 0.5) == [(100, 149)]
    assert off_court_views([(0, 99, False), (100, 120, True)], 0.5) == []  # mostly uncalibrated footage
    assert off_court_views([], 0.5) == []


def test_rallies_in_off_court_views_are_masked():
    from rallyiq.events.pipeline import segment_events
    from rallyiq.events.synthetic import make_video

    v = make_video(n_rallies=3, seed=2)
    base = segment_events(v.shuttle, v.poses)
    assert len(base) == 3
    r = base[1]
    off = [(r.start_frame - 5, r.end_frame + 5)]
    on = replace(EventsConfig(), mask_off_court_views=True)
    masked = segment_events(v.shuttle, v.poses, on, off_court=off)
    assert [x.start_frame for x in masked] == [base[0].start_frame, base[2].start_frame]
    # switched off
    off_cfg = replace(EventsConfig(), mask_off_court_views=False)
    assert len(segment_events(v.shuttle, v.poses, off_cfg, off_court=off)) == 3
