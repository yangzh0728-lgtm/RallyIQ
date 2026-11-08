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
    assert "0:a?" in cmd and "-an" not in cmd
    no_audio = extract_clips.cut_cmd(Path("a.mov"), 0, 1, Path("b.mp4"), 1080, audio=False)
    assert "-an" in no_audio and "0:a?" not in no_audio  # iPhone apple_apac audio can't be decoded


def test_label_tool_hit_tag_handles_unknown_side():
    from evaluation.tools.label_tool import hit_tag

    assert hit_tag([]) == ""
    assert hit_tag([{"t_ms": 1, "side": "near"}]) == "  HIT near"
    assert hit_tag([{"t_ms": 1, "side": None}]) == "  HIT ?"  # pre-annotations leave side empty


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


def test_manual_calibration_from_named_points_recovers_corners():
    from rallyiq.contracts.geometry import COURT_CORNERS_M, apply_homography, fit_homography
    from scripts import real_run

    # a made-up camera: court metres -> image px
    true_corners = [[300.0, 1300.0], [1700.0, 1300.0], [1250.0, 620.0], [700.0, 620.0]]
    court_to_img = fit_homography(COURT_CORNERS_M, true_corners)
    look = real_run.court_lookup()
    names = ["near_short_service/left_doubles_sideline", "near_short_service/right_doubles_sideline",
             "far_baseline/left_doubles_sideline", "far_baseline/right_doubles_sideline",
             "far_short_service/centre_line"]
    pts = {n: apply_homography(court_to_img, [look[n]])[0] for n in names}
    cal = real_run.manual_calibration("v1", pts, 600)
    seg = cal.segments[0]
    assert seg.method == "manual" and seg.status == "ok" and seg.end_frame == 599
    for got, want in zip(seg.image_corners, true_corners):
        assert abs(got[0] - want[0]) < 0.5 and abs(got[1] - want[1]) < 0.5  # off-screen corners too
    assert seg.reprojection_error_px < 0.5
    with pytest.raises(ValueError):
        real_run.manual_calibration("v1", {"nope": [0, 0], **pts}, 600)


def test_manifest_court_points_fit_one_court():
    from scripts import real_run

    own = schema.load_manifest(schema.OWN_MANIFEST)
    with_points = {cid: c for cid, c in own.items() if c.get("court_points")}
    assert {"own0529_0006", "own0529_0008"} <= set(with_points)
    for cid, c in with_points.items():
        seg = real_run.manual_calibration(cid, c["court_points"], 7200).segments[0]
        assert seg.reprojection_error_px < 5, cid  # measured lines agree with one flat court
        (nlx, nly), (nrx, nry), (frx, fry), (flx, fly) = seg.image_corners
        assert nlx < flx < frx < nrx and fly < nly and fry < nry, cid  # left/right, near/far not swapped
