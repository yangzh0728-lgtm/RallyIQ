"""Smoke test: scripts/real_run.py runs every stage on a clip and writes a prefill + report."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_real_run_smoke(tmp_path, monkeypatch):
    monkeypatch.delenv("RALLYIQ_POSE_MODEL", raising=False)
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clips / "c1.mp4")], check=True)
    (clips / "c1.meta.json").write_text(json.dumps({"clip_id": "c1", "video_id": "v", "start_ms": 5000}))

    spec = importlib.util.spec_from_file_location("real_run", ROOT / "scripts" / "real_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    assert mod.main(["--clips", str(clips), "--out", str(out)]) == 0

    pred = json.loads((out / "preds" / "c1.json").read_text())
    assert pred["clip_id"] == "c1"
    res = json.loads((out / "results.json").read_text())[0]
    assert res["state"] == "succeeded" and res["frames"] == 25
    assert "c1" in (out / "report.md").read_text()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_real_run_uses_fixed_court_points(tmp_path, monkeypatch):
    monkeypatch.delenv("RALLYIQ_POSE_MODEL", raising=False)
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clips / "c1.mp4")], check=True)
    pts = {"near_baseline/left_doubles_sideline": [20, 175], "near_baseline/right_doubles_sideline": [300, 175],
           "far_baseline/right_doubles_sideline": [230, 60], "far_baseline/left_doubles_sideline": [90, 60]}
    (clips / "c1.meta.json").write_text(json.dumps({"clip_id": "c1", "video_id": "v", "start_ms": 0,
                                                    "duration_ms": 1000, "fps": 25, "court_points": pts}))
    spec = importlib.util.spec_from_file_location("real_run", ROOT / "scripts" / "real_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    assert mod.main(["--clips", str(clips), "--out", str(out)]) == 0
    res = json.loads((out / "results.json").read_text())[0]
    cal = json.loads((Path(res["artifacts"]) / "calibration.json").read_text())
    seg = cal["segments"][0]
    assert seg["method"] == "manual" and seg["status"] == "ok"
    assert [round(v) for v in seg["image_corners"][0]] == [20, 175]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_rerun_stages_picks_up_neighbour_courts(tmp_path, monkeypatch):
    monkeypatch.delenv("RALLYIQ_POSE_MODEL", raising=False)
    clips = tmp_path / "clips"
    clips.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clips / "c1.mp4")], check=True)
    pts = {"near_baseline/left_doubles_sideline": [20, 175], "near_baseline/right_doubles_sideline": [300, 175],
           "far_baseline/right_doubles_sideline": [230, 60], "far_baseline/left_doubles_sideline": [90, 60]}
    meta = {"clip_id": "c1", "video_id": "v", "start_ms": 0, "duration_ms": 1000, "fps": 25, "court_points": pts}
    (clips / "c1.meta.json").write_text(json.dumps(meta))
    from scripts import real_run, rerun_stages

    out = tmp_path / "out"
    assert real_run.main(["--clips", str(clips), "--out", str(out)]) == 0
    art = Path(json.loads((out / "results.json").read_text())[0]["artifacts"])
    players_before = (art / "players.json").stat().st_mtime_ns
    (clips / "c1.meta.json").write_text(json.dumps(meta | {"neighbour_courts": [[-7.3, 0], [7.3, 0]]}))
    assert rerun_stages.main(["--out", str(out), "--clips", str(clips), "--no-overlay"]) == 0
    res = json.loads((out / "results.json").read_text())[0]
    assert res["state"] == "succeeded" and res["rerun_from"] == "calibration"
    assert (Path(res["artifacts"]) / "players.json").stat().st_mtime_ns == players_before  # not re-run
    seg = json.loads((Path(res["artifacts"]) / "calibration.json").read_text())["segments"][0]
    assert len(seg["other_courts"]) == 2
    assert seg["other_courts"][0][3][0] < seg["image_corners"][3][0]  # left neighbour drawn left of ours
    # --from shuttle still rebuilds calibration.json from the (changed) manual court
    (clips / "c1.meta.json").write_text(json.dumps(meta | {"neighbour_courts": [[-7.3, 0]]}))
    assert rerun_stages.main(["--out", str(out), "--clips", str(clips), "--from", "shuttle", "--no-overlay"]) == 0
    seg = json.loads((Path(res["artifacts"]) / "calibration.json").read_text())["segments"][0]
    assert len(seg["other_courts"]) == 1
