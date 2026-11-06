import numpy as np
import pytest

from rallyiq.perception.players.filtering import build_candidates, is_seated
from rallyiq.perception.players.geometry import (
    PlayArea,
    apply_homography,
    court_side,
    foot_point,
    image_net_y,
)
from rallyiq.perception.players.models import KP, PersonDetection, PlayerFrameResult
from rallyiq.perception.players.postprocess import fill_short_gaps
from rallyiq.perception.players.runner import track_players
from rallyiq.perception.players.tracker import SinglesTracker

from .synth import H_IMG_TO_COURT, IMAGE_SIZE, person_at_court, person_at_pixel


# -- geometry -------------------------------------------------------------------

def test_homography_roundtrip_corners():
    img = np.array([[360, 1000], [1560, 1000], [1240, 250], [680, 250]], dtype=float)
    court = apply_homography(H_IMG_TO_COURT, img)
    np.testing.assert_allclose(court, [[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], atol=1e-6)


@pytest.mark.parametrize("x,y", [(3.05, 1.5), (1.0, 12.0), (5.5, 7.5)])
def test_foot_point_maps_back_to_court(x, y):
    det = person_at_court(x, y)
    fc = apply_homography(H_IMG_TO_COURT, foot_point(det)[None])[0]
    assert np.hypot(fc[0] - x, fc[1] - y) < 0.15


def test_foot_point_falls_back_to_bbox_without_ankles():
    det = person_at_pixel(500, 800, 200)
    det.keypoints[KP["left_ankle"], 2] = 0.0
    det.keypoints[KP["right_ankle"], 2] = 0.0
    fp = foot_point(det)
    assert fp[0] == pytest.approx(500, abs=1)
    assert fp[1] == pytest.approx(det.bbox[3])


def test_play_area_and_sides():
    area = PlayArea()
    assert area.contains(np.array([3.0, 6.0]))
    assert area.contains(np.array([-0.8, 14.8]))  # run-off behind far baseline
    assert not area.contains(np.array([-3.0, 6.0]))
    assert court_side(np.array([3.0, 2.0])) == "near"
    assert court_side(np.array([3.0, 10.0])) == "far"
    net_y = image_net_y(H_IMG_TO_COURT)
    assert 250 < net_y < 1000


# -- filtering ------------------------------------------------------------------

def test_spectators_outside_play_area_are_dropped():
    dets = [
        person_at_court(3.0, 2.0),   # near player
        person_at_court(3.0, 11.0),  # far player
        person_at_court(-4.0, 5.0),  # spectator beside the court
        person_at_court(3.0, 18.0),  # person behind the far run-off
    ]
    cands = build_candidates(dets, H_IMG_TO_COURT, IMAGE_SIZE)
    ys = sorted(round(float(c.foot_court[1])) for c in cands)
    assert ys == [2, 11]
    assert all(c.on_court for c in cands)


def test_seated_umpire_is_penalised():
    standing = person_at_court(-0.5, 6.7)
    seated = person_at_court(-0.5, 6.7, seated=True)
    assert is_seated(seated) and not is_seated(standing)
    cs = build_candidates([standing], H_IMG_TO_COURT, IMAGE_SIZE)[0]
    cz = build_candidates([seated], H_IMG_TO_COURT, IMAGE_SIZE)[0]
    assert cz.player_score < 0.5 * cs.player_score
    assert "seated" in cz.reasons


def test_implausible_size_is_penalised():
    ok = person_at_court(3.0, 9.0)
    giant = person_at_court(3.0, 9.0, height_m=6.0)
    s_ok = build_candidates([ok], H_IMG_TO_COURT, IMAGE_SIZE)[0].player_score
    s_big = build_candidates([giant], H_IMG_TO_COURT, IMAGE_SIZE)[0].player_score
    assert s_big < 0.5 * s_ok


def test_uncalibrated_keeps_pixel_output():
    dets = [person_at_pixel(900, 950, 400), person_at_pixel(950, 400, 150), person_at_pixel(100, 300, 40)]
    cands = build_candidates(dets, None, IMAGE_SIZE)
    assert all(c.foot_court is None and c.on_court is None for c in cands)
    assert cands[-1].det.height == pytest.approx(40, abs=1) and "small_relative_to_others" in cands[-1].reasons


# -- tracking -------------------------------------------------------------------

class FakeEstimator:
    """Returns pre-built detections per call, ignoring the image."""

    def __init__(self, per_frame):
        self.per_frame = iter(per_frame)

    def __call__(self, img):
        return next(self.per_frame)


def _frames(n, colours=None):
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    for i in range(n):
        yield i, i * 1000 / 30, (colours(i) if colours else img)


def _ids_by_side(res):
    return {p.side: p.player_id for p in res.players}


def test_two_players_keep_ids_with_spectators_and_misses():
    rng = np.random.default_rng(0)
    per_frame = []
    for i in range(90):
        t = i / 30
        near = person_at_court(3.05 + 1.5 * np.sin(2 * t), 2.5 + 0.8 * np.cos(3 * t))
        far = person_at_court(3.05 - 1.8 * np.sin(1.5 * t), 10.5 + 1.0 * np.sin(2.5 * t))
        crowd = [person_at_court(-3.5, 4 + rng.uniform(0, 5)), person_at_court(8.5, 9.0)]
        dets = [near, far] + crowd
        if 40 <= i < 43:  # far player missed by the detector (blur / occlusion)
            dets = [near] + crowd
        per_frame.append(dets)
    results = track_players(_frames(90), FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    assert len(results) == 90
    for r in results:
        assert len(r.players) == 2
        assert _ids_by_side(r) == {"near": "P1", "far": "P2"}
        for p in r.players:
            assert p.on_court
    gap = [p for r in results[40:43] for p in r.players if p.player_id == "P2"]
    assert all(p.interpolated for p in gap)
    assert len({p.track_id for r in results for p in r.players if p.player_id == "P2"}) == 1


def _solid(colour):
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    img[:] = colour
    return img


def test_identity_survives_change_of_ends_by_appearance():
    """Both players vanish (break), then return on swapped halves.
    Torso colours tell them apart: P1 wears red, P2 wears blue."""
    red, blue = (0, 0, 220), (220, 0, 0)

    def paint(img, det, colour):
        x1, y1, x2, y2 = det.bbox.astype(int)
        img[max(y1, 0):y2, max(x1, 0):x2] = colour

    frames, per_frame = [], []
    for i in range(160):
        img = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, np.uint8)
        if i < 60:
            a, b = person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)  # red near, blue far
        elif i < 120:
            a = b = None  # change of ends, camera pointed elsewhere
        else:
            a, b = person_at_court(3.0, 10.5), person_at_court(3.0, 2.5)  # red far, blue near
        dets = []
        if a is not None:
            paint(img, a, red)
            paint(img, b, blue)
            dets = [b, a]  # detector order is arbitrary
        frames.append((i, i * 33.3, img))
        per_frame.append(dets)
    results = track_players(iter(frames), FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    assert _ids_by_side(results[10]) == {"near": "P1", "far": "P2"}
    assert results[90].players == []
    assert _ids_by_side(results[130]) == {"far": "P1", "near": "P2"}
    # Re-acquisition after a long loss starts a new raw track.
    assert results[130].players[0].track_id != results[10].players[0].track_id


def test_tracker_rejects_teleport_to_spectator():
    tr = SinglesTracker()
    from rallyiq.perception.players.filtering import build_candidates as bc

    for i in range(10):
        c = bc([person_at_court(3.0, 2.0), person_at_court(3.0, 11.0)], H_IMG_TO_COURT, IMAGE_SIZE)
        tr.update(i, c, calibrated=True)
    # Near player missed; a line judge inside the run-off on the near half appears.
    c = bc([person_at_court(6.9, 0.2), person_at_court(3.0, 11.0)], H_IMG_TO_COURT, IMAGE_SIZE)
    out = tr.update(10, c, calibrated=True)
    p1 = [p for p in out if p.player_id == "P1"]
    assert p1 == []


def test_fill_short_gaps_respects_limit():
    def st(pid, x):
        from rallyiq.perception.players.models import PlayerState

        return PlayerState(pid, 0, "near", np.array([x, 0, x + 1, 1.0]), np.ones((17, 3)),
                           np.array([x, 1.0]), np.array([x, 1.0]), True, 0.9)

    frames = [PlayerFrameResult(i, i * 33.3, [st("P1", i)] if i in (0, 3, 20) else []) for i in range(21)]
    fill_short_gaps(frames, max_gap=4)
    assert [len(f.players) for f in frames[:4]] == [1, 1, 1, 1]
    assert frames[1].players[0].interpolated
    assert frames[1].players[0].foot_court[0] == pytest.approx(1.0)
    assert all(len(f.players) == 0 for f in frames[4:20])
