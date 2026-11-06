"""Label-free shuttle diagnosis and the sweep script (no torch: detection is skipped)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.multicourt import make_scene
from rallyiq.contracts import EventsResult, Rally
from rallyiq.perception.shuttle.diagnose import (
    coverage,
    filter_candidates,
    format_table,
    roi_report,
)


def _scene():
    return make_scene("behind_raised", "side_by_side", frames=150)


def test_coverage_separates_raw_target_and_tracked():
    s = _scene()
    size = (s.meta.width, s.meta.height)
    with_lock = coverage("x", s.meta, s.candidates, size, 0.2, s.calibration, s.players)
    # The neighbour court's shuttle gives raw peaks that are not over the target court.
    assert with_lock.raw_frac >= with_lock.raw_target_frac > 0
    assert 0 < with_lock.detected_frac <= with_lock.raw_frac
    no_lock = coverage("x", s.meta, s.candidates, size, 0.2, s.calibration, s.players, court_lock=False)
    assert no_lock.raw_target_frac is None
    assert no_lock.detected_frac >= with_lock.detected_frac


def test_coverage_counts_rally_frames():
    s = _scene()
    events = EventsResult(video_id=s.meta.video_id, rallies=[
        Rally(rally_id="r1", start_frame=0, end_frame=99, start_ms=0, end_ms=3300, confidence=0.5)], shots=[])
    c = coverage("x", s.meta, s.candidates, (s.meta.width, s.meta.height), 0.2, s.calibration, s.players, events)
    assert c.rally_frames == 100 and c.rally_detected_frac is not None
    assert "| x | 0.20 |" in format_table([c])


def test_threshold_filter_and_empty_input():
    s = _scene()
    assert filter_candidates({1: [(0, 0, 0.1), (1, 1, 0.3)]}, 0.2) == {1: [(1, 1, 0.3)]}
    c = coverage("empty", s.meta, {}, (s.meta.width, s.meta.height), 0.2)
    assert c.raw_frac == 0 and c.detected_frac == 0 and c.max_score_p50 is None


def test_roi_report_lists_segments():
    s = _scene()
    rows = roi_report(s.meta, s.calibration)
    assert rows and rows[0]["other_courts"] == 1
    assert 0 < rows[0]["court_w_frac"] <= 1.5 and rows[0]["roi_zoom"] >= 1.0


def test_sweep_script_without_detection(tmp_path):
    from scripts.shuttle_sweep import main

    s = _scene()
    vdir = tmp_path / "data" / "videos" / "v1"
    art = vdir / "artifacts"
    art.mkdir(parents=True)
    (art / "probe.json").write_text(s.meta.model_dump_json())
    (art / "calibration.json").write_text(s.calibration.model_dump_json())
    (art / "players.json").write_text(s.players.model_dump_json())
    (vdir / "work").mkdir()
    (vdir / "work" / "shuttle_candidates.json").write_text(json.dumps({
        "detector": "sim", "frame_size": [s.meta.width, s.meta.height],
        "frames": {str(f): {"candidates": [list(c) for c in cs], "selected": -1} for f, cs in s.candidates.items()},
    }))
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps([{"clip_id": "c1", "artifacts": str(art)}]))
    assert main(["--run", str(run), "--clip", "c1", "--no-detect"]) == 0
    res = json.loads((run / "shuttle_sweep.json").read_text())
    assert [v["threshold"] for v in res[0]["variants"]] == [0.2, 0.1, 0.05]
    assert "c1" in (run / "shuttle_sweep.md").read_text()
