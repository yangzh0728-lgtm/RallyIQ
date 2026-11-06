import math

import numpy as np
import pytest

from rallyiq.contracts import ShuttleTrack, VideoMeta
from rallyiq.perception.shuttle.heatmap import extract_peaks
from rallyiq.perception.shuttle.stage import build_track, segments_for, tracker_config
from rallyiq.perception.shuttle.tracker import ShuttleTracker, TrackerConfig

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
