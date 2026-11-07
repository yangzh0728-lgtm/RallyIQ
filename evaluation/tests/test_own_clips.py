"""Own footage (evaluation/own_clips.yaml): manifest merge, clip cutting, loop suggestions."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation import schema
from evaluation.tools import extract_clips

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_own_manifest_is_valid_and_merges_with_bwf():
    own = schema.load_manifest(schema.OWN_MANIFEST)
    assert own
    for c in own.values():
        assert schema.is_own(c) and c["split"] in ("eval", "train")
        assert c["conditions"]["source"] == "own"
        assert c["conditions"]["camera"].startswith("amateur_")
        assert not c["file"].startswith("._") and "/" not in c["file"]  # bare file name, no folders
    assert [c for c in own.values() if c.get("holdout")]  # DJI_0009 never trains
    merged = schema.load_manifests(extract_clips.HERE / "clips.yaml", schema.OWN_MANIFEST)
    assert set(own) <= set(merged) and len(merged) > len(own)
    assert not any(schema.is_own(c) for cid, c in merged.items() if cid not in own)


def test_duplicate_ids_across_manifests_are_rejected(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("clips:\n  - {id: x, video_id: abcdefghijk, split: eval}\n")
    b.write_text("clips:\n  - {id: x, file: X.MP4, split: eval}\n")
    with pytest.raises(schema.SchemaError):
        schema.load_manifests(a, b)
    assert schema.load_manifests(a, tmp_path / "missing.yaml") == schema.load_manifest(a)


def test_find_own_skips_resource_forks(tmp_path):
    (tmp_path / "day1").mkdir()
    (tmp_path / "day1" / "._DJI_1.MP4").write_bytes(b"x")
    (tmp_path / "day1" / "DJI_1.MP4").write_bytes(b"x")
    assert extract_clips.find_own(tmp_path, "DJI_1.MP4") == tmp_path / "day1" / "DJI_1.MP4"
    assert extract_clips.find_own(tmp_path, "nope.MP4") is None


def test_cut_cmd_scales_only_when_asked():
    cmd = extract_clips.cut_cmd(Path("a.mp4"), 1.5, 2.0, Path("b.mp4"), 1080)
    assert "scale=-2:'min(1080,ih)'" in cmd and cmd[cmd.index("-ss") + 1] == "1.500"
    assert "-vf" not in extract_clips.cut_cmd(Path("a.mp4"), 1.5, 2.0, Path("b.mp4"))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_extract_own_clip_downscales_and_writes_meta(tmp_path):
    footage = tmp_path / "T7" / "Badminton Film" / "some day"
    footage.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=2560x1440:rate=30",
                    "-t", "4", "-c:v", "libx264", "-preset", "ultrafast", str(footage / "DJI_X_0001_D.MP4")],
                   check=True)
    own = tmp_path / "own.yaml"
    own.write_text("clips:\n  - {id: o1, file: DJI_X_0001_D.MP4, start_s: 1, duration_s: 2, split: train,"
                   " conditions: {camera: amateur_baseline, source: own}}\n")
    bwf = tmp_path / "bwf.yaml"
    bwf.write_text("clips: []\n")
    data = tmp_path / "data"
    rc = extract_clips.main(["--manifest", str(bwf), "--own-manifest", str(own),
                             "--footage", str(tmp_path / "T7"), "--data", str(data)])
    assert rc == 0
    meta = json.loads((data / "clips" / "o1.meta.json").read_text())
    assert meta["start_ms"] == 1000 and meta["duration_ms"] == 2000 and meta["video_id"] == "DJI_X_0001_D"
    assert (meta["width"], meta["height"]) == (1920, 1080)


def test_extract_reports_missing_drive(tmp_path, capsys):
    own = tmp_path / "own.yaml"
    own.write_text("clips:\n  - {id: o1, file: A.MP4, start_s: 0, duration_s: 1, split: train}\n")
    bwf = tmp_path / "bwf.yaml"
    bwf.write_text("clips: []\n")
    extract_clips.main(["--manifest", str(bwf), "--own-manifest", str(own),
                        "--footage", str(tmp_path / "nodrive"), "--data", str(tmp_path / "data")])
    out = capsys.readouterr().out
    assert "硬盘没接上" in out and "o1" in out


def test_loop_prefers_own_footage_and_keeps_holdout_last(tmp_path):
    from scripts import shuttle_loop as loop

    cdir, labels = tmp_path / "clips", tmp_path / "labels"
    cdir.mkdir()
    labels.mkdir()
    for cid in ("bc", "own", "held"):
        (cdir / f"{cid}.mp4").write_bytes(b"")
    manifest = {
        "bc": {"id": "bc", "split": "train", "conditions": {"camera": "broadcast_main"}},
        "own": {"id": "own", "file": "A.MP4", "split": "train", "conditions": {"camera": "amateur_baseline"}},
        "held": {"id": "held", "file": "B.MP4", "split": "eval", "holdout": True,
                 "conditions": {"camera": "amateur_baseline"}},
        "uncut": {"id": "uncut", "file": "C.MP4", "split": "train", "conditions": {"camera": "amateur_side"}},
    }
    lines = loop.suggest_next(manifest, {}, cdir, tmp_path / "runs")
    assert [ln.split("（")[0] for ln in lines] == ["- own", "- bc", "- held"]
    assert loop.manifest_holdout(manifest) == ["held"]
    lines = loop.suggest_next({"uncut": manifest["uncut"]}, {}, cdir, tmp_path / "runs")
    assert "extract_clips --only uncut" in lines[0] and "--download" not in lines[0]
