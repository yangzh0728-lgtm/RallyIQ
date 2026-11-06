import copy
import json

import numpy as np
import pytest

from evaluation import benchmark, metrics as M, schema
from evaluation.tools.label_tool import Session

FPS = 25.0
FM = 1000 / FPS
START = 900_000
CORNERS = [[700, 300], [1220, 300], [1500, 950], [420, 950]]  # 远左 远右 近右 近左


def gt_doc(cid="c1"):
    d = schema.empty_doc(cid, "label", 1920, 1080, FPS, (START, START + 60_000))
    d["status"] = "reviewed"
    d["court"] = [{"t_ms": START, "visible": True, "corners": CORNERS},
                  {"t_ms": START + 40 * FM, "visible": False, "corners": None}]
    # 远场球（画面上方）与近场球（画面下方）各 5 帧，再加 2 帧不可见
    d["shuttle"] = ([{"t_ms": START + i * FM, "visible": True, "xy": [960, 350]} for i in range(5)]
                    + [{"t_ms": START + i * FM, "visible": True, "xy": [960, 850]} for i in range(5, 10)]
                    + [{"t_ms": START + i * FM, "visible": False, "xy": None} for i in range(10, 12)])
    d["hits"] = [{"t_ms": START + 1000, "side": "near"}, {"t_ms": START + 2000, "side": "far"},
                 {"t_ms": START + 3000, "side": "near"}]
    d["rallies"] = [{"start_ms": START + 500, "end_ms": START + 10_000},
                    {"start_ms": START + 20_000, "end_ms": START + 30_000}]
    return schema.validate(d)


def test_perfect_prediction_scores_one():
    g = gt_doc()
    p = copy.deepcopy(g); p["kind"] = "prediction"
    res = benchmark.evaluate_clip(g, p)
    assert res["court"].summary()["success_rate"] == 1.0
    assert res["court"].summary()["false_calibration_rate"] == 0.0
    sh = res["shuttle"]["all"].summary()
    assert sh["precision"] == sh["recall"] == 1.0
    assert res["shuttle"]["far"].summary()["visible_frames"] == 5
    assert res["shuttle"]["near"].summary()["visible_frames"] == 5
    assert res["hits"].summary()["f1"] == 1.0
    assert res["rallies"].summary()["f1"] == 1.0


def test_missing_prediction_is_all_misses():
    res = benchmark.evaluate_clip(gt_doc(), None)
    assert res["shuttle"]["all"].summary()["recall"] == 0.0
    assert res["hits"].summary()["recall"] == 0.0
    assert res["court"].summary()["detection_rate"] == 0.0


def test_shuttle_pixel_threshold_scales_with_resolution():
    g = gt_doc()
    # 720p 片段：8px 偏差 = 12px@1080p，刚好命中；9px 不命中
    for doc in (g,):
        doc["video"]["height"], doc["video"]["width"] = 720, 1280
    p = copy.deepcopy(g)
    p["shuttle"] = [{"t_ms": START, "visible": True, "xy": [968, 350]}]
    g["shuttle"] = g["shuttle"][:1]
    assert M.eval_shuttle(g, p)["all"].tp == 1
    p["shuttle"][0]["xy"] = [969, 350]
    st = M.eval_shuttle(g, p)["all"]
    assert (st.tp, st.fp, st.fn) == (0, 1, 1)


def test_shuttle_false_positive_when_invisible():
    g = gt_doc()
    p = copy.deepcopy(g)
    p["shuttle"][10] = {"t_ms": p["shuttle"][10]["t_ms"], "visible": True, "xy": [10, 10]}
    st = M.eval_shuttle(g, p)["all"]
    assert st.fp == 1 and st.tn == 1


def test_hit_tolerance_three_frames_and_side():
    g = gt_doc()
    p = copy.deepcopy(g)
    p["hits"] = [{"t_ms": START + 1000 + 3 * FM, "side": "near"},      # 刚好 3 帧：命中
                 {"t_ms": START + 2000 + 4 * FM, "side": "far"},       # 4 帧：不命中
                 {"t_ms": START + 3000, "side": "far"}]                # 击球方错
    st = M.eval_hits(g, p)
    assert (st.tp, st.fp, st.fn) == (1, 2, 2)
    assert st.tp_any_side == 2
    assert st.summary()["f1_ignore_side"] == pytest.approx(2 / 3)


def test_hits_outside_range_ignored():
    g = gt_doc()
    p = copy.deepcopy(g)
    p["hits"].append({"t_ms": START + 120_000, "side": "near"})
    assert M.eval_hits(g, p).fp == 0


def test_rally_iou_matching():
    g = gt_doc()
    p = copy.deepcopy(g)
    p["rallies"] = [{"start_ms": START + 1000, "end_ms": START + 10_500},  # IoU 高
                    {"start_ms": START + 20_000, "end_ms": START + 23_000}]  # IoU 0.3
    st = M.eval_rallies(g, p)
    assert (st.tp, st.fp, st.fn) == (1, 1, 1)
    assert st.summary()["mean_abs_start_err_ms"] == 500


def test_court_error_and_false_calibration():
    g = gt_doc()
    p = copy.deepcopy(g)
    p["court"][0]["corners"] = [[x + 20, y] for x, y in CORNERS]
    p["court"][1] = {"t_ms": p["court"][1]["t_ms"], "visible": True, "corners": CORNERS}
    s = M.eval_court(g, p).summary()
    assert s["mean_corner_err_px"] == pytest.approx(20)
    assert s["success_rate"] == 0.0
    assert s["false_calibration_rate"] == 1.0


def test_homography_maps_corners():
    h = M.homography(np.array(CORNERS, float), M.COURT_WORLD)
    for c, w in zip(CORNERS, M.COURT_WORLD):
        assert np.allclose(M.apply_h(h, c), w, atol=1e-6)


def test_schema_rejects_bad_docs():
    d = gt_doc()
    d["hits"].append({"t_ms": 1, "side": "left"})
    with pytest.raises(schema.SchemaError):
        schema.validate(d)
    d = gt_doc(); d["court"][0]["corners"] = CORNERS[:3]
    with pytest.raises(schema.SchemaError):
        schema.validate(d)


def test_run_end_to_end(tmp_path):
    labels, preds = tmp_path / "labels", tmp_path / "preds"
    g1, g2 = gt_doc("a"), gt_doc("b")
    g2["status"] = "draft"
    schema.save(g1, labels / "a.json"); schema.save(g2, labels / "b.json")
    p = copy.deepcopy(g1); p["kind"] = "prediction"
    schema.save(p, preds / "a.json")
    manifest = {"a": {"id": "a", "split": "eval", "conditions": {"discipline": "MS", "camera": "broadcast_main"}},
                "b": {"id": "b", "split": "eval", "conditions": {"discipline": "MD"}}}
    rep = benchmark.run(labels, preds, manifest, "eval", include_draft=False)
    assert rep["overall"]["clips"] == 1 and "b" in rep["skipped"]
    rep = benchmark.run(labels, preds, manifest, "eval", include_draft=True)
    assert rep["overall"]["clips"] == 2
    assert rep["groups"]["discipline"]["MD"]["hits"]["recall"] == 0.0
    md = benchmark.to_markdown(rep)
    assert "按 discipline 分组" in md and "最弱条件" in md
    json.dumps(rep)  # 可序列化


def test_manifest_is_valid():
    clips = schema.load_manifest(benchmark.HERE / "clips.yaml")
    assert len(clips) >= 15
    disciplines = {c["conditions"]["discipline"] for c in clips.values()}
    assert {"MS", "WS", "MD", "WD", "XD"} <= disciplines
    for c in clips.values():
        assert c["split"] in ("eval", "train")
        assert len(c["video_id"]) == 11


def test_label_session_flow():
    doc = schema.empty_doc("x", "label", 1920, 1080, FPS, (START, START + 10_000))
    s = Session(doc, START, 250, shuttle_stride=2)
    for x, y in CORNERS:
        s.court_click(x, y)
    assert doc["court"][0]["corners"] == CORNERS
    s.seek(50); assert s.court_copy_prev(); assert len(doc["court"]) == 2
    s.seek(0); s.shuttle_click(100, 200); assert s.frame == 2
    s.shuttle_hidden(); assert doc["shuttle"][1]["visible"] is False and s.frame == 4
    s.seek(10); s.hit("near"); s.seek(12); assert s.hit_delete_nearest(); assert doc["hits"] == []
    s.seek(20); s.rally_start(); s.seek(100); assert s.rally_end()
    assert doc["rallies"] == [{"start_ms": START + 800, "end_ms": START + 4000}]
    assert s.toggle_status() == "reviewed"
    schema.validate(doc)


def test_uncertain_frames_and_jump():
    from evaluation.tools.label_tool import uncertain_frames

    fps = 30.0

    def item(f, score=None):
        t = int(round(START + f * 1000 / fps))
        return {"t_ms": t, "visible": score is not None, "xy": [1, 1] if score is not None else None,
                **({"score": score} if score is not None else {})}

    # detections 0-9 (frame 5 weak), short gap 10-14, detections 15-19, long gap 20-79, detection 80
    sh = [item(f, 0.3 if f == 5 else 0.9) for f in range(10)] + [item(f) for f in range(10, 15)]
    sh += [item(f, 0.9) for f in range(15, 20)] + [item(f) for f in range(20, 80)] + [item(80, 0.9)]
    doc = schema.empty_doc("x", "prediction", 1920, 1080, fps, (START, START + 10_000))
    doc["shuttle"] = sh
    u = uncertain_frames(doc, START)
    assert u == [5, 10, 11, 12, 13, 14, 20, 21, 22, 77, 78, 79]

    s = Session(schema.empty_doc("x", "label", 1920, 1080, fps, (START, START + 10_000)), START, 100)
    s.uncertain = u
    assert s.jump_uncertain() and s.frame == 5
    s.seek(15)
    assert s.jump_uncertain(forward=False) and s.frame == 14
    s.seek(79)
    assert not s.jump_uncertain()
