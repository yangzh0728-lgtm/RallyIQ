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
