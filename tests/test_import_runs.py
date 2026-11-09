"""scripts/import_runs.py copies a finished run's video, meta, artifacts and job into another store."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from rallyiq.contracts.models import VideoMeta
from rallyiq.store import Store

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("import_runs", ROOT / "scripts" / "import_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finished_run(tmp_path: Path) -> tuple[Path, str]:
    run = tmp_path / "run"
    src = Store(run / "data")
    vid, dest = src.create_video("clip_a.mp4")
    dest.write_bytes(b"\x00fake-mp4")
    src.register_video(vid, "clip_a.mp4", dest)
    src.set_video_meta(VideoMeta(video_id=vid, filename="clip_a.mp4", width=320, height=180, fps=25.0,
                                 frame_count=25, duration_ms=1000))
    src.blobs.put_bytes(src.artifact_key(vid, "probe"), b'{"video_id": "%s"}' % vid.encode())
    src._set_artifact_flag(vid, "probe", True)
    job = src.create_job(vid)
    done = datetime.now(timezone.utc)
    src.update_job(job.model_copy(update={"state": "succeeded", "started_at": done, "finished_at": done,
                                          "stages": [s.model_copy(update={"state": "succeeded"}) for s in job.stages]}))
    (run / "results.json").write_text(json.dumps([
        {"clip_id": "clip_a", "video_id": vid, "state": "succeeded"},
        {"clip_id": "clip_b", "video_id": None, "state": "crashed"},
    ]))
    return run, vid


def test_imports_once_with_original_name(tmp_path):
    mod = _load()
    run, vid = _finished_run(tmp_path)
    meta_dir = tmp_path / "full"
    meta_dir.mkdir()
    (meta_dir / "clip_a.meta.json").write_text(json.dumps({"source_file": "DJI_0001_D.MP4"}))
    dest = tmp_path / "demo"
    assert mod.main(["--run", str(run), "--dest", str(dest), "--meta-dir", str(meta_dir)]) == 0

    dst = Store(dest)
    v = dst.get_video(vid)
    assert v is not None and v.filename == "DJI_0001_D (clip_a).mp4"
    assert v.meta is not None and v.meta.frame_count == 25
    assert "probe" in v.artifacts and dst.artifact_bytes(vid, "probe") is not None
    assert v.latest_job is not None and v.latest_job.state == "succeeded"
    assert dst.video_path(vid).read_bytes() == b"\x00fake-mp4"

    src = Store(run / "data")
    assert mod.import_video(src, dst, vid, "x.mp4") == "present"  # second run skips it
    assert len(dst.list_videos()) == 1
