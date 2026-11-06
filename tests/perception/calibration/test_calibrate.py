import numpy as np
import pytest

pytest.importorskip("cv2")

from rallyiq.perception.calibration import calibrate_frame, calibrate_from_points  # noqa: E402
from rallyiq.perception.calibration import court_model as cm  # noqa: E402
from rallyiq.perception.calibration.homography import project  # noqa: E402

from .synthetic import Camera, Scene, keypoint_error, render  # noqa: E402

W, L = cm.COURT_WIDTH, cm.COURT_LENGTH

# Varied phone-style viewpoints; none of them is a "required" filming setup.
CAMERAS = {
    "behind_high": Camera((W / 2, -6, 6)),
    "behind_low_tripod": Camera((W / 2, -4, 1.5), focal=900),
    "corner_oblique": Camera((-3, -5, 4), focal=900),
    "side_on": Camera((-8, L / 2, 5), focal=750),
    "near_end_cropped": Camera((W / 2, -2, 3.5), focal=800),
    "rolled_1080p": Camera((W / 2 + 1, -6, 5), focal=1500, size=(1920, 1080), roll_deg=4),
    "portrait_phone": Camera((W / 2, -7, 4), focal=900, size=(720, 1280)),
}


def _diag(cam):
    return float(np.hypot(*cam.size))


@pytest.mark.parametrize("name", list(CAMERAS))
@pytest.mark.parametrize("hall", [False, True], ids=["clean", "hall"])
def test_auto_calibration_finds_the_court(name, hall):
    cam = CAMERAS[name]
    extra = dict(adjacent_courts=True, blur=1.2, noise=8) if hall else {}
    img, H_true = render(Scene(cam, **extra))
    cal = calibrate_frame(img)
    assert cal.ok, cal.reason
    # Every visible court intersection within ~1.2% of the image diagonal.
    assert keypoint_error(cal.image_from_court, H_true, cam.size) < 0.012 * _diag(cam)
    assert 0.0 <= cal.confidence <= 1.0


@pytest.mark.parametrize("name", ["behind_high", "corner_oblique", "portrait_phone"])
def test_clean_views_are_confident(name):
    img, _ = render(Scene(CAMERAS[name]))
    cal = calibrate_frame(img)
    assert cal.status == "ok"
    assert cal.metrics.verified_keypoints >= 10


def test_labelling_puts_y0_at_the_near_end_with_x_to_the_right():
    cam = CAMERAS["behind_high"]
    img, H_true = render(Scene(cam))
    cal = calibrate_frame(img)
    pts = np.array([k.xy for k in cm.keypoints()])
    est, _ = project(cal.image_from_court, pts)
    gt, _ = project(H_true, pts)
    assert np.max(np.linalg.norm(est - gt, axis=1)) < 0.012 * _diag(cam)  # no symmetry needed


def test_court_from_image_round_trip():
    img, _ = render(Scene(CAMERAS["corner_oblique"]))
    cal = calibrate_frame(img)
    court = np.array([[1.0, 2.0], [5.0, 11.0], [3.05, 6.7]])
    assert np.allclose(cal.to_court(cal.to_image(court)), court, atol=1e-6)


def test_no_court_is_undeterminable():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    cal = calibrate_frame(noise)
    assert cal.status == "undeterminable"
    assert cal.image_from_court is None or cal.confidence < 0.3


def test_close_up_without_court_lines_is_undeterminable():
    import cv2

    img = np.full((720, 1280, 3), (60, 120, 40), np.uint8)
    cv2.rectangle(img, (500, 100), (800, 700), (240, 240, 240), -1)  # a player's shirt filling the frame
    cv2.line(img, (0, 650), (1280, 640), (235, 235, 235), 6)  # one stray line
    assert calibrate_frame(img).status == "undeterminable"


def test_manual_corners_without_image():
    cam = CAMERAS["side_on"]
    _, H_true = render(Scene(cam, players=0))
    clicks, _ = project(H_true, np.array(list(cm.CORNERS.values())))
    cal = calibrate_from_points(clicks, image_size=cam.size)
    assert cal.method == "manual" and cal.status == "ok"
    assert keypoint_error(cal.image_from_court, H_true, cam.size) < 1.0


def test_manual_corners_are_snapped_to_lines():
    cam = CAMERAS["behind_low_tripod"]
    img, H_true = render(Scene(cam))
    clicks, _ = project(H_true, np.array(list(cm.CORNERS.values())))
    rng = np.random.default_rng(0)
    sloppy = clicks + rng.normal(0, 3.0, clicks.shape)  # a few px of clicking error
    cal = calibrate_from_points(sloppy, image=img)
    assert cal.status == "ok"
    assert keypoint_error(cal.image_from_court, H_true, cam.size) < 0.012 * _diag(cam)


def test_manual_points_by_keypoint_name():
    cam = CAMERAS["behind_high"]
    _, H_true = render(Scene(cam))
    names = [
        "near_baseline/left_singles_sideline",
        "near_baseline/right_singles_sideline",
        "near_short_service/right_doubles_sideline",
        "far_short_service/centre_line",
        "far_baseline/left_doubles_sideline",
    ]
    lookup = {k.name: k.xy for k in cm.keypoints()}
    clicks, _ = project(H_true, np.array([lookup[n] for n in names]))
    cal = calibrate_from_points(clicks, names, image_size=cam.size)
    assert keypoint_error(cal.image_from_court, H_true, cam.size) < 1.0
