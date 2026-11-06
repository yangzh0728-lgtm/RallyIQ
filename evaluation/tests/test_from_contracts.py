import json
import shutil
from pathlib import Path

from evaluation import benchmark, from_contracts

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"


def test_convert_contract_examples(tmp_path):
    art = tmp_path / "artifacts"
    shutil.copytree(EXAMPLES, art)
    doc = from_contracts.convert(art, "c1", offset_ms=900_000)
    meta = json.loads((EXAMPLES / "video_meta.json").read_text())
    assert len(doc["court"]) == len(doc["shuttle"]) == meta["frame_count"]
    assert doc["range_ms"] == [900_000, 900_000 + meta["duration_ms"]]
    assert doc["court"][0]["t_ms"] == 900_000

    cal = json.loads((EXAMPLES / "calibration.json").read_text())
    seg = cal["segments"][0]
    if seg.get("image_corners"):
        nl, nr, fr, fl = seg["image_corners"]
        c = doc["court"][seg["start_frame"]]
        assert c["visible"] and c["corners"] == [fl, fr, nr, nl]

    shots = json.loads((EXAMPLES / "events.json").read_text())["shots"]
    assert len(doc["hits"]) == len(shots)
    assert doc["hits"][0]["t_ms"] == 900_000 + round(shots[0]["frame"] * 1000 / meta["fps"])
    assert doc["hits"][0]["side"] in ("near", "far")

    # 预测与自身比较应满分（以转换结果当作标注）
    gt = json.loads(json.dumps(doc)); gt["kind"] = "label"
    res = benchmark.evaluate_clip(gt, doc)
    assert res["hits"].summary()["f1"] in (1.0, None)
    assert res["court"].summary()["success_rate"] in (1.0, None)


def test_interpolated_points_excluded_by_default(tmp_path):
    art = tmp_path / "artifacts"
    shutil.copytree(EXAMPLES, art)
    sh = json.loads((art / "shuttle.json").read_text())
    sh["points"] = [dict(sh["points"][0], source="interpolated")]
    (art / "shuttle.json").write_text(json.dumps(sh))
    f = sh["points"][0]["frame"]
    assert from_contracts.convert(art, "c1", 0)["shuttle"][f]["visible"] is False
    assert from_contracts.convert(art, "c1", 0, include_interpolated=True)["shuttle"][f]["visible"] is True
