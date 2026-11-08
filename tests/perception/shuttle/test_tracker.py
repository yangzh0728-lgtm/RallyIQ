import math

import numpy as np
import pytest

from rallyiq.contracts import ShuttleTrack, VideoMeta
from rallyiq.perception.shuttle.heatmap import extract_peaks
from rallyiq.perception.shuttle.stage import build_track, segments_for, tracker_config
from rallyiq.perception.shuttle.tracker import ShuttleTracker

W, H = 1920, 1080


def flight(f0, f1, a, b, apex, fps=30.0):
    """Parabolic-looking image trajectory from a to b (px), peaking ``apex`` px above."""
    out = {}
    for f in range(f0, f1 + 1):
        u = (f - f0) / (f1 - f0)
        x = a[0] + (b[0] - a[0]) * u
        y = a[1] + (b[1] - a[1]) * u - apex * 4 * u * (1 - u)
        out[f] = (x, y)
    return out


def rally(fps=30.0):
    """Three shots: near -> far, far -> near, near -> far (hits change direction)."""
    s = 1.0 / 30.0 * fps  # frames scale with fps
    k = lambda v: int(round(v * s))  # noqa: E731
    pts = {}
    pts.update(flight(k(10), k(40), (960, 900), (1000, 420), 500))
    pts.update(flight(k(40), k(70), (1000, 420), (900, 880), 350))
    pts.update(flight(k(70), k(100), (900, 880), (1050, 430), 600))
    return pts


def as_candidates(pts, score=0.9, noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return {f: [(x + rng.normal(0, noise), y + rng.normal(0, noise), score)] for f, (x, y) in pts.items()}


def by_frame(res):
    return {p.frame: p for p in res.points}


def test_clean_trajectory_is_kept_as_detected():
    pts = rally()
    res = ShuttleTracker(30, W, H).track(as_candidates(pts), 120)
    got = by_frame(res)
    assert set(got) == set(pts)
    assert all(p.source == "detected" for p in res.points)
    for f, (x, y) in pts.items():  # raw detections are never moved
        assert (got[f].x, got[f].y) == (x, y)


@pytest.mark.parametrize("every,max_stray", [(3, 0), (1, 10)])
@pytest.mark.parametrize("seed", range(5))
def test_false_positives_are_rejected_and_true_track_kept(seed, every, max_stray):
    pts = rally()
    cands = as_candidates(pts, noise=1.0)
    rng = np.random.default_rng(seed)
    for f in range(0, 120, every):
        # A random strong blob (above the usual 0.5 threshold) in many frames.
        cands.setdefault(f, []).append((float(rng.uniform(0, W)), float(rng.uniform(0, H)), 0.55))
    res = ShuttleTracker(30, W, H).track(cands, 120)
    got = by_frame(res)
    hits = sum(1 for f, (x, y) in pts.items() if f in got and math.hypot(got[f].x - x, got[f].y - y) < 12)
    assert hits / len(pts) > 0.97
    stray = [p for p in res.points if p.frame not in pts]
    assert len(stray) <= max_stray


def test_weak_far_court_points_kept_only_when_consistent():
    # Far court: tiny shuttle, heatmap peaks only ~0.3.
    far = flight(0, 20, (900, 300), (1100, 330), 120)
    cands = as_candidates(far, score=0.3)
    # Isolated weak blips elsewhere must not survive.
    cands[40] = [(200.0, 200.0, 0.3)]
    cands[60] = [(1700.0, 900.0, 0.3)]
    res = ShuttleTracker(30, W, H).track(cands, 80)
    got = by_frame(res)
    assert set(range(0, 21)) <= set(got)
    assert 40 not in got and 60 not in got


def test_short_gap_is_filled_as_interpolated_with_lower_confidence():
    pts = rally()
    cands = as_candidates(pts)
    missing = [20, 21, 22, 23]
    for f in missing:
        cands.pop(f)
    res = ShuttleTracker(30, W, H).track(cands, 120)
    got = by_frame(res)
    for f in missing:
        assert got[f].source == "interpolated"
        assert got[f].confidence < 0.6
        assert math.hypot(got[f].x - pts[f][0], got[f].y - pts[f][1]) < 25
    assert not any(g.start_frame <= 21 <= g.end_frame for g in res.gaps)


def test_long_gap_stays_a_gap():
    pts = rally()
    cands = as_candidates(pts)
    for f in range(45, 65):  # 0.66 s missing
        cands.pop(f)
    res = ShuttleTracker(30, W, H).track(cands, 120)
    got = by_frame(res)
    assert not any(45 <= f < 65 for f in got)
    assert any(g.start_frame == 45 and g.end_frame == 64 for g in res.gaps)


def test_out_of_frame_gap_reason():
    # A high clear leaves through the top of the image and comes back.
    up = {f: (900.0 + 3 * f, 600.0 - 60 * f) for f in range(0, 10)}  # y reaches 60 at f=9
    down = {f: (1000.0 + 3 * f, -1500.0 + 60 * f) for f in range(26, 40)}  # re-enters ~f=26
    pts = {**up, **{f: p for f, p in down.items() if p[1] >= 0}}
    res = ShuttleTracker(30, W, H).track(as_candidates(pts), 40)
    gap = next(g for g in res.gaps if g.start_frame == 10)
    assert gap.reason == "out_of_frame"
    # No fill outside the image.
    assert all(0 <= p.y < H for p in res.points)


def test_spike_removed_but_hit_kept():
    pts = rally()
    cands = as_candidates(pts)
    cands[25] = [(300.0, 150.0, 0.6)]  # wild jump mid-flight
    res = ShuttleTracker(30, W, H).track(cands, 120)
    got = by_frame(res)
    assert 25 not in got or got[25].source == "interpolated"
    assert got[40].source == "detected" and got[70].source == "detected"  # hit frames stay


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0, 120.0])
def test_frame_rate_independent(fps):
    pts = rally(fps)
    n = max(pts) + 10
    cands = as_candidates(pts, noise=0.5)
    # Drop the same ~0.1 s of time regardless of fps.
    k = int(round(0.1 * fps))
    start = int(round(25 * fps / 30))
    for f in range(start, start + k):
        cands.pop(f, None)
    res = ShuttleTracker(fps, W, H).track(cands, n)
    got = by_frame(res)
    recovered = sum(1 for f in pts if f in got)
    assert recovered / len(pts) > 0.97


def test_low_fps_fast_smash_is_linked():
    # 25 fps smash covering ~200 px/frame.
    pts = {f: (300.0 + 200 * f, 200.0 + 80 * f) for f in range(0, 8)}
    res = ShuttleTracker(25, W, H).track(as_candidates(pts, score=0.45), 10)
    assert {p.frame for p in res.points} == set(pts)


def test_extract_peaks_subpixel_and_multiple():
    hm = np.zeros((288, 512), dtype=np.float32)
    yy, xx = np.mgrid[0:288, 0:512]
    hm += 0.9 * np.exp(-((xx - 100.4) ** 2 + (yy - 50.6) ** 2) / (2 * 2.5 ** 2))
    hm += 0.3 * np.exp(-((xx - 400) ** 2 + (yy - 200) ** 2) / (2 * 2.5 ** 2))
    peaks = extract_peaks(hm, threshold=0.2, top_k=3)
    assert len(peaks) == 2
    (x, y, s), (x2, y2, s2) = peaks
    assert abs(x - 100.4) < 0.3 and abs(y - 50.6) < 0.3 and s > s2
    assert abs(x2 - 400) < 0.3 and abs(y2 - 200) < 0.3
    assert extract_peaks(hm, threshold=0.95) == []


def _meta(**kw):
    base = dict(video_id="v1", filename="a.mp4", fps=30.0, frame_count=120, width=W, height=H, duration_ms=4000)
    base.update(kw)
    return VideoMeta(**base)


def test_build_track_matches_contract_and_scales_coordinates():
    pts = rally()
    # Decoder produced half-size frames: coordinates are scaled back to probe resolution.
    cands = {f: [(x / 2, y / 2, 0.9)] for f, (x, y) in pts.items()}
    cands[20] = []
    track, selected = build_track(_meta(), cands, W // 2, H // 2, "test@0")
    ShuttleTrack.model_validate(track.model_dump())
    pf = {p.frame: p for p in track.points}
    assert abs(pf[30].x - pts[30][0]) < 1.5 and abs(pf[30].y - pts[30][1]) < 1.5
    assert pf[20].source == "interpolated"
    assert pf[30].t_ms == 1000
    assert selected[30] == 0 and 20 not in selected
    # Frames before the first and after the last detection are gaps.
    assert track.gaps[0].start_frame == 0 and track.gaps[-1].end_frame == 119


def test_build_track_rejects_unrotated_frames():
    with pytest.raises(ValueError, match="rotation"):
        build_track(_meta(), {}, H, W, "test@0")


def test_tracker_config_overrides():
    cfg = tracker_config({"max_fill_s": 0.5, "weak_score": "0.3"})
    assert cfg.max_fill_s == 0.5 and cfg.weak_score == 0.3
    with pytest.raises(ValueError):
        tracker_config({"nope": 1})


def test_segments_cover_video_without_calibration():
    segs = segments_for(_meta(), None, "auto")
    assert len(segs) == 1 and segs[0].start == 0 and segs[0].end == 119 and segs[0].roi is None


def test_court_roi_zooms_on_small_court():
    from rallyiq.perception.shuttle.detector import court_roi

    # Court occupies the centre third of a 4K frame.
    corners = [[1500, 1700], [2340, 1700], [2200, 1100], [1640, 1100]]
    roi = court_roi(corners, 3840, 2160)
    assert roi is not None
    x0, y0, x1, y1 = roi
    assert x0 <= 1500 and x1 >= 2340 and y1 >= 1700 and y0 < 1100
    assert abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.02
    # Court filling the frame: no crop.
    assert court_roi([[0, 1080], [1920, 1080], [1500, 300], [420, 300]], 1920, 1080) is None


def test_idle_weak_tracklets_rejected_but_weak_flight_kept():
    rng = np.random.default_rng(3)
    cands = {}
    # Handshake / mop: weak blob wobbling a few px around one spot for 2 s.
    for f in range(0, 60):
        cands[f] = [(800 + rng.normal(0, 3), 700 + rng.normal(0, 3), 0.25)]
    # Weak far-court flight later in the clip (slow near its apex).
    far = flight(100, 130, (900, 300), (1000, 320), 80)
    for f, (x, y) in far.items():
        cands[f] = [(x, y, 0.25)]
    res = ShuttleTracker(30, W, H).track(cands, 140)
    got = by_frame(res)
    assert not any(f < 60 for f in got)
    assert sum(1 for f in far if f in got) / len(far) > 0.95


def test_short_walking_pace_weak_tracklet_rejected_but_short_flight_kept():
    # Between rallies: a shuttle in a player's hand walking off court, 8 frames at
    # ~100 px/s (0.045 image diagonals per second). Slow enough to be idle when short.
    cands = {f: [(600 + 3.3 * f, 500 + 0.5 * f, 0.3)] for f in range(0, 8)}
    # A weak glimpse of a shuttle in play, also 8 frames, at ~600 px/s.
    cands.update({f: [(1200 + 20 * (f - 60), 300 + 4 * (f - 60), 0.3)] for f in range(60, 68)})
    got = by_frame(ShuttleTracker(30, W, H).track(cands, 80))
    assert not any(f < 8 for f in got)
    assert sum(1 for f in range(60, 68) if f in got) >= 7


def test_idle_strong_tracklet_is_kept():
    # A shuttle lying on the floor after a rally is real (rally-end evidence).
    cands = {f: [(800.0, 900.0, 0.8)] for f in range(0, 30)}
    res = ShuttleTracker(30, W, H).track(cands, 30)
    assert len(res.points) == 30


def test_tracker_weak_score_follows_detector_threshold():
    from rallyiq.perception.shuttle.detector import DetectorConfig

    assert tracker_config(None, 0.05).weak_score == 0.05
    assert tracker_config({"weak_score": 0.3}, 0.05).weak_score == 0.3
    assert tracker_config(None).weak_score == DetectorConfig.peak_threshold


def test_low_score_flight_is_tracked_when_threshold_is_lowered():
    meta = _meta()
    cands = as_candidates(rally(), score=0.08)
    default, _ = build_track(meta, cands, W, H, "x", tracker_config(None, 0.1))
    low, _ = build_track(meta, cands, W, H, "x", tracker_config(None, 0.05))
    assert not default.points
    assert sum(p.source == "detected" for p in low.points) > 80


def test_fixed_bright_spot_is_dropped_at_any_score_and_real_flight_survives():
    """A chink of light scores like a shuttle but never moves; it must not claim frames."""
    fps, n = 30.0, 120
    cands = {}
    for f in range(n):
        cands[f] = [(1650.0 + 0.2 * (f % 3), 322.0, 0.6)]  # stronger than the flight below
        if 40 <= f < 70:
            cands[f].append((300.0 + 20 * (f - 40), 700.0 - 8 * (f - 40), 0.4))
    res = ShuttleTracker(fps, 1920, 1080).track(cands, n)
    assert all(p.x < 1000 for p in res.points)
    assert {p.frame for p in res.points if p.source == "detected"} == set(range(40, 70))


def test_slow_apex_is_not_a_static_spot():
    """A flight that slows at the top of its arc keeps moving, so it stays."""
    fps, n = 30.0, 60
    cands = {f: [(400.0 + 6 * f, 300.0 + 0.04 * (f - 30) ** 2, 0.6)] for f in range(n)}
    res = ShuttleTracker(fps, 1920, 1080).track(cands, n)
    assert len([p for p in res.points if p.source == "detected"]) == n


def test_still_strong_object_over_shared_air_is_dropped():
    from rallyiq.perception.shuttle.tracker import ShuttleTracker

    # 1 s at 30 fps: a strong still blob (a shuttle lying on a neighbour's floor) where our
    # air space overlaps another court's (weight 0.6), always next to our player.
    cands = {f: [(500.0, 400.0, 0.9)] for f in range(30)}
    shared = ShuttleTracker(30, 1920, 1080, region=lambda f, x, y: 0.6, support=lambda f, x, y: True)
    assert not shared.track(cands, 30).points
    ours = ShuttleTracker(30, 1920, 1080, region=lambda f, x, y: 1.0, support=lambda f, x, y: True)
    assert ours.track(cands, 30).points  # a shuttle lying on our own court is kept


def test_saved_candidates_round_trip(tmp_path):
    from rallyiq.perception.shuttle.stage import load_candidates, save_candidates

    cands = {3: [(10.5, 20.25, 0.9), (30.0, 40.0, 0.3)], 7: [(1.0, 2.0, 0.5)]}
    save_candidates(tmp_path / "c.json", cands, {3: 0}, "det@1", (1920, 1080))
    got, size, det = load_candidates(tmp_path / "c.json")
    assert got == cands and size == (1920, 1080) and det == "det@1"


def test_still_object_kept_only_where_a_shuttle_can_rest():
    from rallyiq.perception.shuttle.tracker import ShuttleTracker

    cands = {f: [(500.0, 400.0, 0.9)] for f in range(30)}  # strong, never moves
    on_floor = ShuttleTracker(30, 1920, 1080, region=lambda f, x, y: 1.0, rest=lambda f, x, y: True)
    assert on_floor.track(cands, 30).points  # lying on our court / held by our player
    duct = ShuttleTracker(30, 1920, 1080, region=lambda f, x, y: 1.0, support=lambda f, x, y: True,
                          rest=lambda f, x, y: False)
    assert not duct.track(cands, 30).points  # a duct tip in our air space, near a big near player
    flying = {f: [(500.0 + 20 * f, 400.0 - 5 * f, 0.9)] for f in range(30)}
    assert duct.track(flying, 30).points  # moving shuttles are not judged by rest at all


def test_blinking_fixed_feature_is_dropped_and_crossing_flight_survives():
    """A duct tip fires on two frames in three, jittering a few px: too unsteady for the
    static-spot test, but it sits in the same cells far longer than a shuttle can."""
    rng = np.random.default_rng(5)
    fps, n = 30.0, 300
    cands = {f: [] for f in range(n)}
    for f in range(n):
        if f % 3:
            cands[f].append((620.0 + rng.normal(0, 4), 436.0 + rng.normal(0, 4), 0.7))
    shot = flight(100, 130, (300, 800), (1200, 300), 200)
    for f, (x, y) in shot.items():
        cands[f].append((x, y, 0.5))
    tr = ShuttleTracker(fps, W, H)
    got = by_frame(tr.track(cands, n))
    assert tr.hot_cell(620, 436) in tr.hot_cells
    assert set(got) <= set(shot)  # the feature claims no frame of its own
    assert sum(1 for f in shot if f in got) == len(shot)  # where the flight crosses it, it is filled


def test_shared_air_trajectory_needs_racket_reach_not_just_a_big_player_nearby():
    shot = as_candidates(flight(0, 30, (300, 300), (700, 250), 150))
    loose = lambda f, x, y: True  # noqa: E731  (everything is near the huge near player)
    shared = lambda f, x, y: 0.6  # noqa: E731
    far_from_racket = ShuttleTracker(30, W, H, region=shared, support=loose, strict_support=lambda f, x, y: False)
    assert not far_from_racket.track(shot, 31).points
    assert far_from_racket.log[-1][-1] == "ambiguous"
    hit = ShuttleTracker(30, W, H, region=shared, support=loose, strict_support=lambda f, x, y: f == 0)
    assert hit.track(shot, 31).points  # leaves our player's racket: ours
    ours = ShuttleTracker(30, W, H, region=lambda f, x, y: 1.0, support=loose, strict_support=lambda f, x, y: False)
    assert ours.track(shot, 31).points  # only shared air asks for racket reach


def test_rest_area_counts_a_held_shuttle_but_not_a_pile_at_the_feet():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from evaluation.multicourt import make_scene
    from rallyiq.contracts import PlayerFrame, PlayerTracks
    from rallyiq.perception.shuttle.stage import rest_area

    cal = make_scene("broadcast", "side_by_side", frames=10).calibration
    tracks = PlayerTracks.model_validate({"video_id": "v", "frames": [{
        "frame": 0, "t_ms": 0.0,
        "players": [{"player_id": "P1", "track_id": 1, "bbox": [40.0, 20.0, 100.0, 220.0], "confidence": 0.9}],
    }]})
    assert isinstance(tracks.frames[0], PlayerFrame)
    rest = rest_area(cal, tracks)
    assert rest(0, 70, 90)  # in hand, chest height (off any floor we own)
    assert not rest(0, 70, 210)  # at the feet: a pile on the floor next to the player
    assert rest(0, 960, 600)  # on our court's floor


def _players(frames):
    from rallyiq.contracts import PlayerTracks

    return PlayerTracks.model_validate({"video_id": "v", "frames": [
        {"frame": f, "t_ms": f * 33, "players": [
            {"player_id": "P1", "track_id": 1, "bbox": [800.0, 400.0, 1100.0, 1000.0], "confidence": 0.9,
             "keypoints": [[0.0, 0.0, 0.0]] * 9 + [list(w) + [0.9], [1100.0, 700.0, 0.9]] + [[0.0, 0.0, 0.0]] * 6}]}
        for f, w in frames.items()]})


def test_racket_contact_needs_a_swinging_wrist_nearby():
    from rallyiq.perception.shuttle.stage import RacketContact

    hanging = RacketContact(_players({f: (820.0, 700.0) for f in range(30)}), 30)
    assert not hanging.contact(15, 830, 600)  # right next to the box, but the arm hangs still
    assert hanging.held(15, 822, 702)
    swing = RacketContact(_players({f: (820.0 + 40 * f, 500.0) for f in range(30)}), 30)  # 2 h/s
    w = (820.0 + 40 * 15, 500.0)
    assert swing.contact(15, w[0] + 100, w[1] - 150)  # racket head above the wrist
    assert not swing.contact(15, w[0] + 600, w[1])  # far beyond racket length
    assert not swing.held(15, w[0] + 100, w[1] - 150)


def test_shared_air_support_is_only_checked_where_a_trajectory_ends_or_turns():
    shot = as_candidates(flight(0, 30, (300, 300), (700, 250), 150))
    shared = lambda f, x, y: 0.6  # noqa: E731
    mid_flight = ShuttleTracker(30, W, H, region=shared, support=lambda f, x, y: True,
                                strict_support=lambda f, x, y: f == 15)
    assert not mid_flight.track(shot, 31).points  # a player's racket brushed past it mid flight
    at_end = ShuttleTracker(30, W, H, region=shared, support=lambda f, x, y: True,
                            strict_support=lambda f, x, y: f == 30)
    assert at_end.track(shot, 31).points


def _air(sure_above_y):
    """Our air only above ``sure_above_y`` (high in the image), shared below."""
    return lambda f, x, y: 1.0 if y < sure_above_y else 0.6


def test_one_linked_tracklet_is_judged_flight_by_flight():
    # Our clear peaks high (air that is only ours), then the linker runs straight on into
    # a neighbour's shuttle crossing at mid height: a sharp turn, no racket of ours.
    ours = flight(0, 20, (900, 500), (700, 300), 300)
    theirs = {f: (700 - 25 * (f - 20), 300 + 12 * (f - 20)) for f in range(21, 45)}
    cands = as_candidates(ours | theirs)
    tr = ShuttleTracker(30, W, H, region=_air(200), support=lambda f, x, y: True,
                        strict_support=lambda f, x, y: False)
    got = by_frame(tr.track(cands, 45))
    assert sum(1 for f in ours if f in got) >= len(ours) - 1
    assert not any(f in got and got[f].source == "detected" for f in range(24, 45))


def test_flight_that_runs_onto_a_duct_tip_loses_only_the_duct_part():
    shot = {f: (300 + 20 * f, 700 - 9 * f) for f in range(0, 15)}  # ends at (580, 574)
    duct = {f: (600 + (f % 2), 560.0) for f in range(15, 45)}
    tr = ShuttleTracker(30, W, H, region=lambda f, x, y: 0.6, support=lambda f, x, y: True,
                        strict_support=lambda f, x, y: f == 0, rest=lambda f, x, y: False)
    got = by_frame(tr.track(as_candidates(shot | duct), 45))
    assert sum(1 for f in shot if f in got) >= len(shot) - 1
    assert not any(f in got for f in range(18, 45))


def test_racket_contact_found_behind_where_detection_resumes():
    # Hit at frame 10 at (500, 600); the shuttle is first detected 4 frames later.
    shot = {f: (500 + 30 * (f - 10), 600 - 40 * (f - 10)) for f in range(14, 30)}
    at_racket = lambda f, x, y: f == 10 and math.dist((x, y), (500, 600)) < 40  # noqa: E731
    tr = ShuttleTracker(30, W, H, region=lambda f, x, y: 0.6, support=lambda f, x, y: True,
                        strict_support=at_racket)
    assert tr.track(as_candidates(shot), 30).points


def test_shared_air_flight_carrying_on_from_ours_across_a_gap_is_kept():
    # Our shuttle in air that is only ours, lost for 5 frames, then on in shared air in
    # the same direction: the same flight, though no racket of ours touches that part.
    path = {f: (400 + 20 * f, 150 + 15 * f) for f in range(0, 40)}
    cands = as_candidates({f: xy for f, xy in path.items() if not 15 <= f < 20})
    tr = ShuttleTracker(30, W, H, region=_air(380), support=lambda f, x, y: True,
                        strict_support=lambda f, x, y: False)
    got = by_frame(tr.track(cands, 40))
    assert sum(1 for f in range(20, 40) if f in got) >= 19
    assert any(v == "chained" for *_, v in tr.log)
