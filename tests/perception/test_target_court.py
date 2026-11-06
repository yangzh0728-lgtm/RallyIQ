"""Target-court lock: reject players and shuttles of neighbouring matches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The benchmark and report live in evaluation/ and scripts/ at the repo root, which
# is not on sys.path when pytest runs as a console script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.multicourt import make_scene, run
from rallyiq.contracts.geometry import apply_homography, invert_homography
from rallyiq.contracts.models import COURT_LENGTH_M as L
from rallyiq.contracts.models import COURT_WIDTH_M as W
from rallyiq.perception.players.filtering import build_candidates
from rallyiq.perception.players.models import PlayerFrameResult, PlayerState
from rallyiq.perception.players.postprocess import drop_offcourt_runs
from rallyiq.perception.target_court import CourtRegion, PlayerSpanRegion, SegmentRegions
from tests.perception.players.synth import H_COURT_TO_IMG, H_IMG_TO_COURT, IMAGE_SIZE, person_at_court

GAP = 2.0  # BWF minimum between side-by-side courts


def _neighbour_corners(dx: float, dy: float = 0.0):
    return apply_homography(H_COURT_TO_IMG, [[dx, dy], [W + dx, dy], [W + dx, L + dy], [dx, L + dy]])


@pytest.fixture
def region():
    return CourtRegion(H_IMG_TO_COURT, [_neighbour_corners(W + GAP)])


def _img(x, y):
    return apply_homography(H_COURT_TO_IMG, [[x, y]])[0]


def test_floor_keeps_lunges_and_rejects_neighbour_court(region):
    assert region.floor_outside_m(_img(W + 0.9, 3.0)) == pytest.approx(0.0, abs=1e-6)
    assert not region.on_other_court(_img(W + 0.9, 3.0))  # lunge past our sideline
    assert region.on_other_court(_img(W + GAP + 0.2, 3.0))  # standing on the next court
    assert not region.on_other_court(_img(W / 2, 3.0))


def test_build_candidates_drops_people_on_other_court(region):
    ours = person_at_court(W / 2, 2.0)
    lunge = person_at_court(W + 0.8, 2.0)
    # Only 1.3 m past our run-off, so the per-frame margin alone would keep it.
    neighbour = person_at_court(W + GAP + 0.3, 4.0)
    without = build_candidates([ours, lunge, neighbour], H_IMG_TO_COURT, IMAGE_SIZE)
    assert len(without) == 3
    kept = build_candidates([ours, lunge, neighbour], H_IMG_TO_COURT, IMAGE_SIZE, region=region)
    feet = sorted(round(float(c.foot_court[0]), 1) for c in kept)
    assert len(kept) == 2 and max(feet) < W + 1.0


def _state(pid, tid, x, y):
    return PlayerState(pid, tid, None, np.zeros(4), np.zeros((17, 3)), np.zeros(2),
                       np.array([x, y], dtype=np.float32), None, 0.9)


def test_drop_offcourt_runs_is_track_level():
    frames = []
    for f in range(30):
        lunge = W + 1.6 if f % 10 == 0 else W - 0.5  # our P1 lunges out now and then
        frames.append(PlayerFrameResult(f, f * 33.3, [
            _state("P1", 0, lunge, 2.0),
            _state("P2", 5, W + 2.3, 9.0),  # someone parked just outside our run-off
        ]))
    assert drop_offcourt_runs(frames) == 1
    assert all([p.player_id for p in f.players] == ["P1"] for f in frames)


def test_air_weight_over_target_and_neighbour(region):
    Hinv = invert_homography(H_IMG_TO_COURT)
    centre = apply_homography(Hinv, [[W / 2, L / 2]])[0]
    scale = region.scale_px_per_m
    assert region.air_weight(centre[0], centre[1] - 5 * scale) == 1.0  # 5 m above mid court
    nb = apply_homography(Hinv, [[W + GAP + W / 2, 3.0]])[0]
    assert region.air_weight(nb[0], nb[1] - 1.0 * scale) == 0.0  # 1 m above the next court


def test_segment_regions_skip_failed_calibration():
    scene = make_scene("broadcast", "side_by_side", frames=10)
    seg = scene.calibration.segments[0]
    assert SegmentRegions(scene.calibration).for_frame(5) is not None
    seg.status = "undeterminable"
    assert not SegmentRegions(scene.calibration)


def test_player_span_region_from_tracks():
    scene = make_scene("broadcast", "side_by_side", frames=60)
    span = PlayerSpanRegion.from_players(scene.players)
    assert span is not None
    p = scene.players.frames[0].players[0]
    assert span.air_weight(p.foot_px[0], 100) == 1.0
    assert span.air_weight(span.x_hi + 10 * span.body, 100) == 0.0


@pytest.mark.parametrize("camera", ["behind_raised", "broadcast"])
def test_shuttle_lock_side_by_side(camera):
    off = run(camera, "side_by_side", lock=False)
    on = run(camera, "side_by_side", lock=True)
    assert off["foreign_rate"] > 0.3
    assert on["foreign_rate"] < 0.05
    assert on["recall"] >= off["recall"]
    # Without the neighbour listed by calibration, or without calibration at all.
    assert run(camera, "side_by_side", lock=True, report_other_court=False)["foreign_rate"] < 0.05
    assert run(camera, "side_by_side", lock=True, use_calibration=False)["foreign_rate"] < 0.05


def test_shuttle_lock_end_to_end_improves():
    off = run("behind_raised", "end_to_end", lock=False)
    on = run("behind_raised", "end_to_end", lock=True)
    assert on["precision"] > off["precision"] + 0.4


def test_court_lock_report_on_saved_run(tmp_path):
    import json

    from scripts.court_lock_report import report

    scene = make_scene("behind_raised", "side_by_side", frames=150)
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "probe.json").write_text(scene.meta.model_dump_json())
    (art / "calibration.json").write_text(scene.calibration.model_dump_json())
    (art / "players.json").write_text(scene.players.model_dump_json())
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "shuttle_candidates.json").write_text(json.dumps({
        "detector": "sim", "frame_size": [scene.meta.width, scene.meta.height],
        "frames": {str(f): {"candidates": [list(c) for c in cs], "selected": -1} for f, cs in scene.candidates.items()},
    }))
    r = report(tmp_path)
    assert r["calibration"][0]["other_courts"] == 1
    assert r["players"]["outside_rate"] == 0
    s = r["shuttle"]
    assert s["lock_mode"] == "calibration"
    assert s["removed_points"] > 20 and s["removed_over_other_court"] > 0.5 * s["removed_points"]
