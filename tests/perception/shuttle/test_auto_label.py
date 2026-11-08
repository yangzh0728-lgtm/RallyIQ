"""scripts/auto_label.py: per-frame decisions, applying a plan, images, and one real proposal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import schema
from scripts import auto_label as al

START, FPS = 10_000, 30.0


def _doc(entries, status="draft"):
    doc = schema.empty_doc("c", "label", 1920, 1080, FPS, (START, START + 5000))
    doc["status"] = status
    doc["shuttle"] = [{"t_ms": al.t_of(f, START, FPS), "visible": v is not None, "xy": list(v) if v else None}
                      for f, v in entries.items()]
    return doc


def test_decide_rules():
    labels = {
        0: None, 1: None, 2: None,          # shuttle held still: model sees it in all three -> propose
        10: None,                           # isolated strong peak -> ignore
        20: None,                           # weak peak -> ignore
        30: (500.0, 500.0),                 # agree -> keep
        40: (500.0, 500.0),                 # model elsewhere -> ignore
        50: (500.0, 500.0),                 # model misses -> keep
        60: None,                           # edited by hand -> keep whatever the model says
    }
    peaks = {0: (100.0, 100.0, 0.9), 1: (101.0, 100.0, 0.8), 2: (102.0, 101.0, 0.3),
             10: (900.0, 300.0, 0.9), 20: (300.0, 300.0, 0.25), 21: (302.0, 300.0, 0.25),
             30: (505.0, 503.0, 0.9), 40: (900.0, 900.0, 0.9), 50: None, 60: (100.0, 100.0, 0.9),
             70: None, 71: (10.0, 10.0, 0.9)}
    out = {c["frame"]: c for c in al.decide(labels, {60}, peaks, 1080)}
    assert out[0]["action"] == "propose" and out[0]["xy"] == [100.0, 100.0]
    assert out[1]["action"] == "propose"
    assert out[2]["action"] == "ignore"  # consistent but below min_score
    assert out[10]["action"] == "ignore" and out[20]["action"] == "ignore" and out[40]["action"] == "ignore"
    assert 30 not in out and 50 not in out and 60 not in out
    assert out[70]["action"] == "invisible"  # unlabeled frame, model sees nothing
    assert 71 not in out  # unlabeled, isolated peak: stays unlabeled


def test_human_frames_and_find_prefill(tmp_path):
    pre = _doc({0: (1.0, 1.0), 1: None, 2: None})
    doc = _doc({0: (1.0, 1.0), 1: (5.0, 5.0), 2: None})
    assert al.human_frames(doc, pre, START) == {1}
    assert al.human_frames(doc, None, START) == set()
    runs = tmp_path / "runs"
    for name, d in (("old", pre), ("new", _doc({0: None, 1: None, 2: (9.0, 9.0)}))):
        (runs / name / "preds").mkdir(parents=True)
        schema.save(d, runs / name / "preds" / "c.json")
    assert al.find_prefill("c", doc, START, runs).parent.parent.name == "old"  # the one the labels came from
    assert al.find_prefill("c", None, START, runs) is not None
    assert al.find_prefill("x", None, START, runs) is None


def test_apply_plan():
    doc = _doc({0: None, 1: None, 2: None, 3: None, 4: (5.0, 5.0)})
    plan = {"start_ms": START, "fps": FPS, "model": "m.pt", "min_score": 0.4, "human_frames": 0, "changes": [
        {"frame": 0, "action": "propose", "xy": [1.0, 2.0], "score": 0.9, "tile": 1},
        {"frame": 1, "action": "propose", "xy": [1.0, 2.0], "score": 0.9, "tile": 2},
        {"frame": 2, "action": "propose", "xy": [1.0, 2.0], "score": 0.9, "tile": 3},
        {"frame": 4, "action": "ignore"},
        {"frame": 5, "action": "invisible"},
    ]}
    counts = al.apply_plan(plan, doc, not_shuttle={2}, unsure={3})
    assert counts == {"accepted": 1, "not_shuttle": 1, "unsure": 1, "ignored": 1, "invisible": 1}
    lab = al.shuttle_by_frame(doc, START)
    assert lab == {0: (1.0, 2.0), 1: None, 3: None, 5: None}  # 2 removed (unsure), 4 removed (ignore)
    assert doc["status"] == "reviewed" and doc["auto_label"]["accepted_frames"] == [0]
    schema.validate(doc)
    with pytest.raises(SystemExit):
        al.apply_plan(plan, doc, not_shuttle={9}, unsure=set())
    assert al.parse_ids("3, 17，5") == {3, 5, 17}


def _video(path, n=30, w=320, h=180):
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(img, (60, 90), 2, (255, 255, 255), -1)  # a shuttle held still
        if f >= 15:
            cv2.circle(img, (100 + 8 * f, 60), 2, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


def test_sheets_and_preview(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = _video(tmp_path / "c.mp4")
    props = [{"frame": f, "xy": [60.0, 90.0], "score": 0.8} for f in range(7)]
    paths = al.numbered_sheets(video, props, tmp_path / "out", "c", per=4, cols=2)
    assert [p.name for p in paths] == ["c-sheet-01.jpg", "c-sheet-02.jpg"]
    assert cv2.imread(str(paths[0])).shape == (2 * 240, 2 * 320, 3)
    doc = _doc({0: (60.0, 90.0), 1: None})
    doc["auto_label"] = {"accepted_frames": [0]}
    out = al.render_preview(video, doc, START, tmp_path / "out" / "c-preview.mp4")
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160


def test_propose_then_apply_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from rallyiq.perception.shuttle.model import build_heatmap_net

    clips, labels, runs, out = tmp_path / "clips", tmp_path / "labels", tmp_path / "runs", tmp_path / "out"
    clips.mkdir()
    labels.mkdir()
    _video(clips / "c.mp4")
    (clips / "c.meta.json").write_text(json.dumps({"start_ms": START, "duration_ms": 1000, "width": 320,
                                                   "height": 180, "fps": FPS}))
    pre = _doc({f: None for f in range(30)})
    pre["kind"] = "prediction"
    (runs / "r1" / "preds").mkdir(parents=True)
    schema.save(pre, runs / "r1" / "preds" / "c.json")
    torch.manual_seed(0)
    model = tmp_path / "m.pt"
    torch.save({"model": build_heatmap_net(3, "concat").state_dict(),
                "param_dict": {"seq_len": 3, "bg_mode": "concat"}}, model)
    common = ["--labels", str(labels), "--clips", str(clips), "--runs", str(runs), "--out", str(out),
              "--device", "cpu"]
    # min-score 0 so that the random network's peaks become proposals
    assert al.main(["c", "--model", str(model), "--min-score", "0", *common]) == 0
    plan = json.loads((out / "c" / "c-plan.json").read_text())
    props = [c for c in plan["changes"] if c["action"] == "propose"]
    assert plan["new_doc"] is not None and plan["frames"] == 30
    assert [c["tile"] for c in props] == list(range(1, len(props) + 1))
    assert not (labels / "c.json").exists()
    rejected = {1} if props else set()
    assert al.main(["c", "--apply", "--not-shuttle", ",".join(map(str, rejected)), *common]) == 0
    doc = schema.load(labels / "c.json")
    assert doc["status"] == "reviewed" and doc["auto_label"]["accepted"] == len(props) - len(rejected)
    assert al.main(["c", "--preview", *common]) == 0
    assert (out / "c" / "c-preview.mp4").exists()
    # a changed label file is not overwritten from a stale plan
    doc["shuttle"] = doc["shuttle"][:-1]
    schema.save(doc, labels / "c.json")
    with pytest.raises(SystemExit):
        al.main(["c", "--apply", *common])
