"""Low cameras: a tripod just behind the baseline, and placements no real camera gives."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from rallyiq.perception.calibration import calibrate_frame  # noqa: E402
from rallyiq.perception.calibration import court_model as cm  # noqa: E402
from rallyiq.perception.calibration.calibrate import MIN_FRAME_SHARE, _frame_share  # noqa: E402
from rallyiq.perception.calibration.homography import MIN_CAMERA_HEIGHT, _plausible, camera_height  # noqa: E402

from .synthetic import Camera, Scene, keypoint_error, render  # noqa: E402

W, L = cm.COURT_WIDTH, cm.COURT_LENGTH


@pytest.mark.parametrize(
    "cam",
    [
        Camera((W / 2, -6, 6)),
        Camera((W / 2, -4, 1.5), focal=900),
        Camera((-3, -5, 4), focal=900),
        Camera((W / 2 + 1, -6, 5), focal=1500, size=(1920, 1080), roll_deg=4),
    ],
)
def test_camera_height_matches_the_camera(cam):
    h = camera_height(cam.homography(), *cam.size)[0]
    assert h == pytest.approx(cam.position[2], rel=0.15)


def test_camera_at_floor_level_is_implausible():
    # A court laid along the floor/wall junction of a gym implies a lens a few
    # centimetres above the floor; a real low tripod is fine.
    low = Camera((W / 2, -3, 0.2), look_at=(W / 2, L / 2, 0.0), focal=700)
    tripod = Camera((W / 2, -3, 1.2), look_at=(W / 2, L / 2, 0.0), focal=700)
    ok = _plausible(np.stack([low.homography(), tripod.homography()]), *low.size)
    assert camera_height(low.homography(), *low.size)[0] < MIN_CAMERA_HEIGHT
    assert ok.tolist() == [False, True]


def test_court_far_away_covers_too_little_of_the_frame():
    far = Camera((W / 2, -40, 3), look_at=(W / 2, L / 2, 0.0), focal=700)
    near = Camera((W / 2, -4, 1.5), focal=900)
    assert _frame_share(far.homography(), far.size) < MIN_FRAME_SHARE
    assert _frame_share(near.homography(), near.size) >= MIN_FRAME_SHARE


@pytest.mark.parametrize("pos", [(W / 2, -2.0, 1.5), (2.0, -1.5, 1.3)])
def test_low_camera_in_a_hall_is_never_confidently_wrong(pos):
    # Low courtside / behind-the-baseline views next to other courts are hard
    # (sam's high-school footage): it is fine to give up, not to report a
    # wrong court as ok. (Today these views come out needs_review or worse.)
    cam = Camera(pos, look_at=(W / 2, L / 2, 0.0), focal=770)
    img, H_true = render(Scene(cam, adjacent_courts=True, blur=1.2, noise=8))
    cal = calibrate_frame(img)
    if cal.status == "ok":
        assert keypoint_error(cal.image_from_court, H_true, cam.size) < 0.012 * float(np.hypot(*cam.size))
