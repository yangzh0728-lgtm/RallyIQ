from rallyiq.contracts import (
    CalibrationSegment,
    CourtCalibration,
    EventsResult,
    PlayerDetection,
    PlayerFrame,
    PlayerTracks,
    Rally,
    ShuttlePoint,
    ShuttleTrack,
    VideoMeta,
)
from rallyiq.contracts.geometry import homography_from_corners
from rallyiq.ingest.quality import assess_quality, camera_view

W, H = 1920, 1080
END_CENTER = [[560, 1000], [1360, 1000], [1160, 420], [760, 420]]


def meta(fps=30.0, w=W, h=H, frames=300):
    return VideoMeta(video_id="v", filename="x.mp4", fps=fps, frame_count=frames, width=w, height=h,
                     duration_ms=int(frames * 1000 / fps))


def calib(corners=END_CENTER, frames=300, status="ok"):
    seg = CalibrationSegment(start_frame=0, end_frame=frames - 1, homography=homography_from_corners(corners),
                             image_corners=corners, reprojection_error_px=1.2, confidence=0.9, status=status)
    return CourtCalibration(video_id="v", segments=[seg])


def test_camera_view_classes():
    assert camera_view(END_CENTER, W, H)[0] == "end_center"
    shifted = [[x + 600, y] for x, y in END_CENTER]
    assert camera_view(shifted, W, H)[0] == "end_oblique"
    # court long axis running left-right: near baseline is near-vertical in the image
    side = [[300, 900], [420, 300], [1500, 300], [1620, 900]]
    assert camera_view(side, W, H)[0] == "side"
    low = [[300, 1000], [1620, 1000], [1300, 760], [620, 760]]
    assert camera_view(low, W, H)[1] == "low"
    high = [[700, 1050], [1220, 1050], [1100, 80], [820, 80]]
    assert camera_view(high, W, H)[1] == "high"


def test_good_footage_is_usable():
    det = lambda side, pid: PlayerDetection(player_id=pid, track_id=1, side=side, bbox=[0, 0, 10, 10], confidence=0.9)
    players = PlayerTracks(video_id="v", frames=[
        PlayerFrame(frame=f, t_ms=f * 33, players=[det("near", "P1"), det("far", "P2")]) for f in range(300)])
    shuttle = ShuttleTrack(video_id="v", points=[
        ShuttlePoint(frame=f, t_ms=f * 33, x=1, y=1, confidence=0.9) for f in range(0, 300) if f % 4])
    events = EventsResult(video_id="v", rallies=[Rally(rally_id="r1", start_frame=0, end_frame=299, start_ms=0,
                                                       end_ms=9966, confidence=0.8)], shots=[])
    q = assess_quality(meta(), calib(), players, shuttle, events, {"brightness": 120, "sharpness": 300, "frames": 12})
    caps = {c.capability: c.status for c in q.capabilities}
    assert caps["court_metrics"] == "ok" and caps["players"] == "ok" and caps["shuttle"] == "ok"
    assert caps["hits"] == "needs_review"  # until the README accuracy bar is met
    assert q.camera_view == "end_center"
    assert "后场正中" in q.summary


def test_degrades_instead_of_refusing():
    q = assess_quality(meta(fps=15, w=640, h=360), None, None, None, None,
                       {"brightness": 30, "sharpness": 10, "frames": 12})
    st = {c.key: c.status for c in q.checks}
    assert st["fps"] == "needs_review" and st["resolution"] == "needs_review"
    assert st["brightness"] == "needs_review" and st["sharpness"] == "needs_review"
    assert st["calibration"] == "undeterminable"
    assert all(c.status == "undeterminable" for c in q.capabilities)


def test_far_player_loss_flags_players():
    det = PlayerDetection(player_id="P1", track_id=1, side="near", bbox=[0, 0, 10, 10], confidence=0.9)
    players = PlayerTracks(video_id="v", frames=[PlayerFrame(frame=f, t_ms=f * 33, players=[det]) for f in range(0, 300, 2)])
    q = assess_quality(meta(), calib(), players)
    checks = {c.key: c for c in q.checks}
    assert checks["near_player_rate"].score == 1.0  # stride 2 is detected, not counted as misses
    assert checks["far_player_rate"].status == "undeterminable"
    assert next(c for c in q.capabilities if c.capability == "players").status == "undeterminable"


def test_wrong_court_with_ok_calibration_is_caught():
    """sam's DJI clip: calibration 'ok' on the wrong court, players all filtered as off-court."""
    q = assess_quality(meta(), calib(status="ok"), PlayerTracks(video_id="v", frames=[]))
    checks = {c.key: c for c in q.checks}
    assert checks["court_choice"].status == "undeterminable"
    assert checks["calibration"].status == "undeterminable"
    caps = {c.capability: c for c in q.capabilities}
    assert caps["court_metrics"].status == "undeterminable"
    assert any("疑似选错场地" in r for r in caps["court_metrics"].reasons)
