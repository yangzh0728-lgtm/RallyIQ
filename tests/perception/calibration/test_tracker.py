import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rallyiq.perception.calibration import CourtTracker, TrackerConfig  # noqa: E402
from rallyiq.perception.calibration import court_model as cm  # noqa: E402

from .synthetic import Camera, Scene, keypoint_error, render  # noqa: E402

W, L = cm.COURT_WIDTH, cm.COURT_LENGTH
DIAG = float(np.hypot(1280, 720))


def _frames(cams):
    out = []
    for i, cam in enumerate(cams):
        img, H = render(Scene(cam, seed=i))
        out.append((img, H, cam))
    return out


def test_steady_camera_keeps_one_segment_and_follows_jitter():
    base = (W / 2, -6, 6)
    cams = [Camera((base[0] + 0.01 * k, base[1], base[2])) for k in range(4)]
    tr = CourtTracker(TrackerConfig())
    for i, (img, H, cam) in enumerate(_frames(cams)):
        cal = tr.update(i * 15, img)
        assert keypoint_error(cal.image_from_court, H, cam.size) < 0.012 * DIAG
    segs = tr.finish(59)
    assert len(segs) == 1 and segs[0].end_frame == 59


def test_camera_move_starts_a_new_accurate_segment_with_stable_labels():
    a = Camera((W / 2, -6, 6))
    b = Camera((-2, -5, 4.5), focal=900)  # phone re-positioned to a corner
    tr = CourtTracker()
    frames = _frames([a, a, b, b])
    for i, (img, _, _) in enumerate(frames):
        tr.update(i * 15, img)
    segs = tr.finish(59)
    assert [s.reason for s in segs] == ["initial", "camera_moved"]
    assert segs[0].end_frame == 29 and segs[1].start_frame == 30
    img, H, cam = frames[-1]
    assert keypoint_error(segs[1].calibration.image_from_court, H, cam.size) < 0.012 * DIAG
    # The near end stays y = 0 across the move (no silent flip of court sides).
    near_a = segs[0].calibration.to_image([[W / 2, 0.0]])[0]
    near_b = segs[1].calibration.to_image([[W / 2, 0.0]])[0]
    far_b = segs[1].calibration.to_image([[W / 2, L]])[0]
    assert near_b[1] > far_b[1] and near_a[1] > 360


def test_cut_to_close_up_and_back():
    a = Camera((W / 2, -6, 6))
    img_a, H = render(Scene(a))
    close = np.full_like(img_a, (40, 60, 90))
    cv2.circle(close, (640, 360), 250, (180, 200, 230), -1)  # a face filling the frame
    tr = CourtTracker()
    for i, img in enumerate([img_a, close, close, img_a]):
        tr.update(i * 15, img)
    segs = tr.finish(59)
    assert [s.calibration.status == "undeterminable" for s in segs] == [False, True, False]
    assert segs[1].reason == "scene_cut"
    assert keypoint_error(segs[2].calibration.image_from_court, H, a.size) < 0.012 * DIAG
