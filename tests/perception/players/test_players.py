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
    """Both players vanish (break between games), then return on swapped halves
    for the next game. Torso colours tell them apart: P1 wears red, P2 blue."""
    red, blue = (0, 0, 220), (220, 0, 0)

    def paint(img, det, colour):
        x1, y1, x2, y2 = det.bbox.astype(int)
        img[max(y1, 0):y2, max(x1, 0):x2] = colour

    frames, per_frame = [], []
    for i in range(0, 1800, 6):  # 60 s at 5 processed frames per second
        img = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 90, np.uint8)
        if i < 600:
            a, b = person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)  # red near, blue far
        elif i < 900:
            a = b = None  # change of ends, camera pointed elsewhere
        else:
            a, b = person_at_court(3.0, 10.5), person_at_court(3.0, 2.5)  # red far, blue near
        dets = []
        if a is not None:
            paint(img, a, red)
            paint(img, b, blue)
            dets = [b, a]  # detector order is arbitrary
        frames.append((i, i * 1000 / 30, img))
        per_frame.append(dets)
    results = track_players(iter(frames), FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    at = {r.frame: r for r in results}
    assert _ids_by_side(at[60]) == {"near": "P1", "far": "P2"}
    assert at[720].players == []
    assert _ids_by_side(at[1200]) == {"far": "P1", "near": "P2"}
    assert results.side_switches == [900]
    # Re-acquisition after a long loss starts a new raw track.
    assert at[1200].players[0].track_id != at[60].players[0].track_id


def test_lookalike_players_keep_halves_across_camera_cuts():
    """Broadcast: a cut to a close-up every rally, players dressed alike. No
    change of ends happens, so none may be reported and ids stay put."""
    per_frame, idx = [], []
    for i in range(0, 2700, 6):  # 90 s
        t = i / 30
        in_view = (t % 12) < 8  # 8 s rally, 4 s close-up / replay
        idx.append(i)
        if not in_view:
            per_frame.append([])
            continue
        near = person_at_court(3.05 + 1.5 * np.sin(t), 2.5 + 0.8 * np.cos(1.3 * t))
        far = person_at_court(3.05 - 1.6 * np.sin(0.9 * t), 10.5 + 0.9 * np.sin(1.1 * t))
        per_frame.append([far, near] if int(t) % 2 else [near, far])
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    frames = ((i, i * 1000 / 30, img) for i in idx)
    results = track_players(frames, FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    assert results.side_switches == []
    for r in results:
        if r.players:
            assert _ids_by_side(r) == {"near": "P1", "far": "P2"}


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


# -- contract adapter -------------------------------------------------------------

def test_stage_output_matches_contract_and_reports_side_switch():
    from rallyiq.contracts import PlayerTracks
    from rallyiq.perception.players.stage import build_tracks

    # Change of ends while still in view: P1 walks near -> far, P2 far -> near,
    # then the next game is played on the new halves.
    ys = np.concatenate([np.full(30, 2.5), np.linspace(2.5, 10.5, 61), np.full(150, 10.5)])
    per_frame = [[person_at_court(1.5, y), person_at_court(4.5, 13.0 - y)] for y in ys]
    idx = list(range(0, 3 * len(ys), 3))  # 10 processed frames per second
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    frames = ((i, i * 1000 / 30, img) for i in idx)
    results = track_players(frames, FakeEstimator(per_frame), lambda f: H_IMG_TO_COURT)
    tracks = build_tracks("vid", results, "test")
    again = PlayerTracks.model_validate_json(tracks.model_dump_json())
    assert again.frames[0].players[0].player_id == "P1"
    assert len(again.frames[0].players[0].keypoints) == 17
    assert again.frames[0].players[0].foot_court is not None
    assert len(tracks.side_switches) == 1
    # Reported where the walk takes the players clear of the net.
    k = idx.index(tracks.side_switches[0])
    assert 6.7 <= ys[k] <= 7.6
    assert {p.player_id for p in again.frames[-1].players if p.side == "far"} == {"P1"}


def test_stage_without_model_returns_empty_tracks(tmp_path, monkeypatch):
    from rallyiq.contracts import VideoMeta
    from rallyiq.perception.players import stage
    from rallyiq.pipeline.context import StageContext

    monkeypatch.delenv("RALLYIQ_POSE_MODEL", raising=False)
    monkeypatch.delenv("RALLYIQ_DEMO", raising=False)

    class NoStore:
        def load_artifact(self, *a):
            return None

    meta = VideoMeta(video_id="v", filename="v.mp4", fps=30, frame_count=10, width=640, height=360, duration_ms=333)
    ctx = StageContext(video_id="v", video_path=tmp_path / "v.mp4", work_dir=tmp_path, store=NoStore(), meta=meta)
    out = stage.run(ctx)
    assert out.frames == [] and out.model is None


def test_coreml_rejects_empty_frame_falls_back_to_cpu():
    from rallyiq.perception.players.pose import CpuFallbackSession

    class Rejects:
        def run(self, names, feed):
            raise RuntimeError("Input (onnx__Mul_1229) has a dynamic shape ({-1}) but the runtime shape ({0})")

        def get_inputs(self):
            return ["input"]

    class Cpu:
        def run(self, names, feed):
            return ["cpu"]

    made = []
    s = CpuFallbackSession(Rejects(), lambda: made.append(1) or Cpu())
    assert s.run(None, {}) == ["cpu"] and s.run(None, {}) == ["cpu"]
    assert s.fallbacks == 2 and made == [1] and s.get_inputs() == ["input"]
