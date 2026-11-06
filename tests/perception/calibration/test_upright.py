"""Court placements no upright camera could see are rejected (DJI_0009 case)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rallyiq.perception.calibration import court_model as cm  # noqa: E402
from rallyiq.perception.calibration.homography import _plausible, upright  # noqa: E402

COURT = np.array(list(cm.CORNERS.values()), dtype=np.float32)


def image_from_court(corners):
    return cv2.getPerspectiveTransform(COURT, np.array(corners, dtype=np.float32)).astype(np.float64)


# Corners near-left, near-right, far-right, far-left in a 1920x1080 frame.
DJI_BAND = [[1731, 526], [1980, 560], [470, 626], [-67, 627]]  # what auto calibration accepted
DJI_REAL = [[18, 1023], [1950, 1023], [1164, 638], [761, 638]]  # manual corners


def test_band_seen_from_below_is_rejected_in_every_labelling():
    for S in cm.SYMMETRIES.values():
        assert not upright(image_from_court(DJI_BAND) @ S)[0]
        assert upright(image_from_court(DJI_REAL) @ S)[0]


def test_plausible_filters_it_too():
    Hs = np.stack([image_from_court(DJI_BAND), image_from_court(DJI_REAL)])
    assert not _plausible(Hs, 1920, 1080)[0]


@pytest.mark.parametrize("corners", [
    [[360, 1000], [1560, 1000], [1240, 250], [680, 250]],  # raised, behind the baseline
    [[200, 700], [1700, 760], [1400, 380], [500, 360]],  # oblique
    [[300, 1000], [1600, 1000], [1600, 80], [300, 80]],  # straight down (horizon at infinity)
])
def test_ordinary_views_stay_upright(corners):
    assert upright(image_from_court(corners))[0]
