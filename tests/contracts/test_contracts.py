import json
from pathlib import Path

import pytest

from rallyiq import contracts as c
from rallyiq.contracts import geometry as g

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"
MODELS = {
    "video_meta": c.VideoMeta, "calibration": c.CourtCalibration, "players": c.PlayerTracks,
    "shuttle": c.ShuttleTrack, "events": c.EventsResult, "events_effective": c.EventsResult,
    "overlay": c.OverlayData, "job": c.Job, "feet": c.FootLandmarks,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_examples_validate(name):
    MODELS[name].model_validate(json.loads((EXAMPLES / f"{name}.json").read_text()))


def test_corrections_example_validates():
    for item in json.loads((EXAMPLES / "corrections.json").read_text()):
        c.Correction.model_validate(item)


def test_extra_fields_rejected():
    with pytest.raises(Exception):
        c.ShuttlePoint(frame=0, t_ms=0, x=1, y=2, confidence=0.5, bogus=1)


def test_homography_roundtrip():
    corners = [[400, 900], [1500, 900], [1200, 300], [700, 300]]
    h = g.homography_from_corners(corners)
    assert sum(g.apply_homography(h, corners), []) == pytest.approx(sum(g.COURT_CORNERS_M, []), abs=1e-6)
    assert g.court_to_image(h, [g.W, g.L]) == pytest.approx([1200, 300], abs=1e-6)
    assert g.reprojection_error_px(h, corners, g.COURT_CORNERS_M) < 1e-6
    assert len(g.project_court_lines(h)) == len(g.COURT_LINES_M)


def test_calibration_segment_lookup():
    cal = c.CourtCalibration(video_id="v", segments=[
        c.CalibrationSegment(start_frame=0, end_frame=9, homography=None),
        c.CalibrationSegment(start_frame=10, end_frame=20, homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ])
    assert cal.homography_for(5) is None
    assert cal.homography_for(15) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert cal.segment_for(99) is None


def test_contract_files_up_to_date():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, str(root / "scripts" / "export_contracts.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_doubles_boundaries_and_team_of():
    from rallyiq.contracts import team_of
    from rallyiq.contracts.geometry import boundary_corners_m, in_bounds

    assert boundary_corners_m("doubles")[1] == [6.1, 0.0]
    assert boundary_corners_m("singles")[0] == [0.46, 0.0]
    # Side alley: in for doubles, out for singles.
    assert in_bounds((0.2, 3.0), "doubles") and not in_bounds((0.2, 3.0), "singles")
    # Doubles serve: back alley (between long service line and baseline) is out.
    assert not in_bounds((3.0, 0.4), "doubles", serve=True) and in_bounds((3.0, 0.4), "singles", serve=True)
    assert in_bounds((3.0, 12.0), "doubles", serve=True) and not in_bounds((3.0, 6.0), "doubles", serve=True)
    assert [team_of(p) for p in ("P1", "P2", "P3", "P4", None)] == ["A", "B", "A", "B", None]
