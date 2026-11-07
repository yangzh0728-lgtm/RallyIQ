"""Re-choosing the target court from where the players stand (DJI_0009 case)."""

from __future__ import annotations

import numpy as np

from rallyiq.contracts.geometry import apply_homography
from rallyiq.contracts.models import COURT_LENGTH_M as L
from rallyiq.contracts.models import COURT_WIDTH_M as W
from rallyiq.contracts.models import CalibrationSegment, CourtCalibration
from rallyiq.perception.court_choice import rechoose, rechoose_segment
from rallyiq.perception.players.filtering import build_candidates
from rallyiq.perception.target_court import CourtRegion
from tests.perception.players.synth import H_COURT_TO_IMG, H_IMG_TO_COURT, IMAGE_SIZE, person_at_court

DX = W + 2.0  # the neighbouring court, 2 m to the right


def _corners(dx: float):
    return apply_homography(H_COURT_TO_IMG, [[dx, 0], [W + dx, 0], [W + dx, L], [dx, L]])


def _cal(method="auto", status="ok"):
    seg = CalibrationSegment(start_frame=0, end_frame=299, homography=H_IMG_TO_COURT.tolist(),
                             image_corners=_corners(0.0), other_courts=[_corners(DX)],
                             method=method, status=status, confidence=0.9)
    return CourtCalibration(video_id="v", segments=[seg])


def _frames(court_dx: float, n: int = 12):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        out.append([
            person_at_court(court_dx + rng.uniform(1, 5), rng.uniform(1, 5)),
            person_at_court(court_dx + rng.uniform(1, 5), rng.uniform(8.5, 12.5)),
            person_at_court(-3.0, 6.0, seated=True),  # umpire / spectator
        ])
    return out


def test_switches_to_the_court_the_players_are_on():
    cal = _cal()
    seen = []

    def detect(idx):
        seen.append(idx)
        return _frames(DX, len(idx))

    new = rechoose(cal, detect)
    assert new is not None and len(seen[0]) == 12
    seg = new.segments[0]
    assert np.allclose(seg.image_corners, _corners(DX), atol=1e-6)
    assert np.allclose(seg.other_courts[0], _corners(0.0), atol=1e-6)  # old target now ignored
    assert seg.status == "needs_review" and "re-chosen" in seg.notes[-1]
    # With the new target our players pass the filter; before they were rejected.
    dets = _frames(DX, 1)[0]
    old_region = CourtRegion(H_IMG_TO_COURT, [_corners(DX)])
    assert len(build_candidates(dets[:2], H_IMG_TO_COURT, IMAGE_SIZE, region=old_region)) == 0
    H_new = np.asarray(seg.homography)
    kept = build_candidates(dets[:2], H_new, IMAGE_SIZE, region=CourtRegion(H_new, seg.other_courts))
    assert len(kept) == 2


def test_keeps_the_target_when_players_are_on_it():
    assert rechoose(_cal(), lambda idx: _frames(0.0, len(idx))) is None


def test_never_overrides_manual_or_switches_on_weak_evidence():
    assert rechoose(_cal(method="manual"), lambda idx: _frames(DX, len(idx))) is None
    seg = _cal().segments[0]
    # Only one person on the neighbouring court half the time: not a match there.
    sparse = [[person_at_court(DX + 3, 3)] if i % 2 else [] for i in range(12)]
    assert rechoose_segment(seg, sparse) is None
    assert rechoose_segment(seg, []) is None


def test_empty_calibrated_court_with_people_playing_is_undeterminable():
    # Calibration fitted some other lines (a court 20 m away); nobody listed elsewhere.
    seg = _cal().segments[0].model_copy(update={"other_courts": []})
    from rallyiq.contracts.geometry import fit_homography, COURT_CORNERS_M

    wrong = fit_homography(_corners(20.0), COURT_CORNERS_M)
    seg = seg.model_copy(update={"homography": wrong, "image_corners": _corners(20.0)})
    new = rechoose(CourtCalibration(video_id="v", segments=[seg]), lambda idx: _frames(0.0, len(idx)))
    assert new is not None
    s = new.segments[0]
    assert s.status == "undeterminable" and s.notes[-1].startswith("calibration contradicted")


def test_consistent_or_quiet_scenes_are_not_flagged():
    seg = _cal().segments[0].model_copy(update={"other_courts": []})
    cal = CourtCalibration(video_id="v", segments=[seg])
    assert rechoose(cal, lambda idx: _frames(0.0, len(idx))) is None  # players on the court
    assert rechoose(cal, lambda idx: [[] for _ in idx]) is None  # nobody there (warm-up, break)
    assert rechoose(_cal(method="manual"), lambda idx: [[] for _ in idx]) is None
