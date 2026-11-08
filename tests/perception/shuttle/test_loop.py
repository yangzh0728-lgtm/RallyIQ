"""Continuous training loop (scripts/shuttle_loop.py): decision rule, bookkeeping, one real round."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import shuttle_loop as loop


def _doc(clip_id, fps=30.0, start_ms=10_000, n=40, status="reviewed"):
    sh = []
    for f in range(n):
        t = round(start_ms + f * 1000 / fps)
        vis = f % 10 != 9
        sh.append({"t_ms": t, "visible": vis, "xy": [20.0 + 5 * f, 60.0] if vis else None})
    return {"schema_version": 1, "clip_id": clip_id, "kind": "label", "status": status,
            "video": {"width": 320, "height": 180, "fps": fps}, "range_ms": [start_ms, start_ms + 2000],
            "court": [], "shuttle": sh, "hits": [], "rallies": []}


def _res(f1, p, r=0.9):
    return {"0.2": {"f1": f1, "precision": p, "recall": r}}


def _setup(tmp_path, clips=(("a", "reviewed"), ("b", "draft")), video=b""):
    labels, cdir = tmp_path / "labels", tmp_path / "clips"
    labels.mkdir()
    cdir.mkdir()
    for cid, status in clips:
        (labels / f"{cid}.json").write_text(json.dumps(_doc(cid, status=status)))
    for cid in ("a", "b", "c", "d"):
        (cdir / f"{cid}.mp4").write_bytes(video)
        (cdir / f"{cid}.meta.json").write_text(json.dumps({"start_ms": 10_000}))
    manifest = tmp_path / "clips.yaml"
    manifest.write_text(
        "clips:\n"
        "  - {id: a, split: train, conditions: {camera: side_court_fixed}}\n"
        "  - {id: b, split: eval, conditions: {camera: broadcast}}\n"
        "  - {id: c, split: eval, conditions: {camera: broadcast}}\n"
        "  - {id: d, split: eval, conditions: {camera: side_court_fixed}}\n"
    )
    return labels, cdir, manifest


def test_decide_needs_gain_and_bounded_precision_drop():
    ok, why = loop.decide(_res(0.93, 0.90), _res(0.90, 0.90))
    assert ok and "0.900（阈值 0.2）-> 0.930（阈值 0.2）" in why
    ok, why = loop.decide(_res(0.905, 0.90), _res(0.90, 0.90))
    assert not ok and "提升不足" in why
    ok, why = loop.decide(_res(0.95, 0.80), _res(0.90, 0.90))
    assert not ok and "精确率下降" in why
    assert loop.decide(_res(0.95, 0.88), _res(0.90, 0.90))[0]  # small precision cost is allowed



def test_decide_compares_each_model_at_its_best_threshold():
    cand = {"0.2": {"f1": 0.85, "precision": 0.75, "recall": 0.98},
            "0.3": {"f1": 0.97, "precision": 0.96, "recall": 0.98}}
    champ = {"0.2": {"f1": 0.95, "precision": 0.93, "recall": 0.97},
             "0.3": {"f1": 0.92, "precision": 0.97, "recall": 0.88}}
    ok, why = loop.decide(cand, champ)
    assert ok and "0.950（阈值 0.2）-> 0.970（阈值 0.3）" in why
    assert not loop.decide(cand, champ, at=0.2)[0]  # the old fixed-threshold rule

def test_fingerprint_follows_shuttle_labels_only():
    d = {"a": _doc("a")}
    fp = loop.labels_fingerprint(d)
    d2 = {"a": dict(_doc("a"), hits=[{"t_ms": 1}])}
    assert loop.labels_fingerprint(d2) == fp
    d2["a"]["shuttle"][3]["xy"] = [1.0, 2.0]
    assert loop.labels_fingerprint(d2) != fp


def test_build_split_by_time_and_holdout(tmp_path):
    _, cdir, _ = _setup(tmp_path)
    docs = {"a": _doc("a"), "b": _doc("b"), "x": _doc("x")}
    tr, va, missing = loop.build_split(docs, cdir, 0.25, holdout=["b"])
    assert missing == ["x"]
    assert [c.clip_id for c in tr] == ["a[train]"] and len(tr[0].labels) == 30
    assert {c.clip_id: len(c.labels) for c in va} == {"a[val]": 10, "b": 40}
    assert max(tr[0].labels) < min(va[0].labels)


def test_suggest_next_prefers_started_then_side_court(tmp_path):
    labels, cdir, mpath = _setup(tmp_path)
    from evaluation import schema

    manifest = schema.load_manifest(mpath)
    runs = tmp_path / "runs"
    (runs / "r1" / "preds").mkdir(parents=True)
    (runs / "r1" / "preds" / "d.json").write_text("{}")
    lines = loop.suggest_next(manifest, schema.load_dir(labels), cdir, runs)
    assert [ln.split("（")[0] for ln in lines] == ["- b", "- d", "- c"]  # a is reviewed
    assert "已开始" in lines[0] and str(runs / "r1" / "preds" / "d.json") in lines[1]
    assert "real_run.py --only c" in lines[2]


def test_check_summary_and_contact_sheet(tmp_path):
    frames = [
        {"clip": "a", "frame": 5, "kind": "fn", "score": None, "xy": None, "label": [30.0, 60.0], "part": "val"},
        {"clip": "a", "frame": 3, "kind": "fp", "score": 0.3, "xy": [40.0, 60.0], "label": None, "part": "train"},
        {"clip": "a", "frame": 9, "kind": "fp", "score": 0.8, "xy": [60.0, 60.0], "label": None, "part": "val"},
    ]
    assert [m["frame"] for m in loop.check_order(frames)] == [9, 3, 5]
    doc = {"weights": "/w/round-001-last.pt", "frames": frames, "path": "/w/round-001-check.json", "sheet": None}
    text = loop.check_summary(doc)
    assert "共 3 帧（训练部分 1，验证部分 2）" in text and "误报（标注说看不见，模型说有球）：2" in text
    assert "label_tool a --check /w/round-001-check.json" in text

    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "a.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    for _ in range(12):
        vw.write(np.full((180, 320, 3), 40, np.uint8))
    vw.release()
    out = loop.contact_sheet(frames, tmp_path, tmp_path / "sheet.jpg", cols=2)
    img = cv2.imread(str(out))
    assert img.shape == (26 + 2 * 240, 2 * 320, 3)


def _args(tmp_path, labels, cdir, mpath, *extra):
    return ["--labels", str(labels), "--clips", str(cdir), "--manifest", str(mpath),
            "--weights-dir", str(tmp_path / "w"), "--runs", str(tmp_path / "runs"), *extra]


def test_status_skip_and_rollback_without_torch(tmp_path, capsys):
    labels, cdir, mpath = _setup(tmp_path)
    wdir = tmp_path / "w"
    d = loop.loop_dir(wdir)
    d.mkdir(parents=True)
    (d / "base.pt").write_bytes(b"base")
    (d / "round-001.pt").write_bytes(b"r1")
    from evaluation import schema

    fp = loop.labels_fingerprint(loop.reviewed_docs(labels))
    loop.save_state(wdir, {"base": str(d / "base.pt"), "base_source": "x", "champion": str(d / "round-001.pt"),
                           "previous": [str(d / "base.pt")], "holdout": ["b"],
                           "rounds": [{"round": 1, "time": "t", "labels": fp, "promoted": True, "reason": "r"}]})
    assert schema.load_dir(labels)

    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--status")) == 0
    out = capsys.readouterr().out
    assert "round-001.pt" in out and "a（40 帧）" in out and "只用于验证的片段：b" in out

    assert loop.main(_args(tmp_path, labels, cdir, mpath)) == 0  # labels unchanged: no training
    assert "没有变化" in capsys.readouterr().out
    assert len(loop.load_state(wdir)["rounds"]) == 1

    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--rollback")) == 0
    st = loop.load_state(wdir)
    assert st["champion"] == str(d / "base.pt") and st["previous"] == []
    assert (wdir / "heatmap.pt").read_bytes() == b"base"
    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--rollback")) == 1


def test_rounds_promote_compare_and_rollback(tmp_path):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from rallyiq.perception.shuttle.detector import HeatmapShuttleDetector
    from rallyiq.perception.shuttle.model import build_heatmap_net

    labels, cdir, mpath = _setup(tmp_path, clips=(("a", "reviewed"),))
    w, h = 320, 180
    vw = cv2.VideoWriter(str(cdir / "a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in range(40):
        img = np.full((h, w, 3), 40, np.uint8)
        if f % 10 != 9:
            cv2.circle(img, (20 + 5 * f, 60), 2, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    torch.manual_seed(0)
    base = tmp_path / "pretrained.pt"
    torch.save({"model": build_heatmap_net(3, "concat").state_dict(),
                "param_dict": {"seq_len": 3, "bg_mode": "concat"}}, base)
    fast = ["--epochs", "1", "--steps", "2", "--batch-size", "1", "--device", "cpu"]
    wdir = tmp_path / "w"

    # Round 1 with a rule that always accepts: the candidate becomes the installed model.
    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--base", str(base), "--min-gain", "-1",
                           "--max-precision-drop", "1", *fast)) == 0
    st = loop.load_state(wdir)
    r1 = loop.loop_dir(wdir) / "round-001.pt"
    assert st["champion"] == str(r1) and st["previous"] == [str(loop.loop_dir(wdir) / "base.pt")]
    assert st["rounds"][0]["promoted"] and st["rounds"][0]["val"] == {"a[val]": 12}
    assert (wdir / "heatmap.pt").read_bytes() == r1.read_bytes()
    assert HeatmapShuttleDetector(wdir / "heatmap.pt").seq_len == 3

    # Unchanged labels: skipped. --force: trains from base again and evaluates the champion too.
    assert loop.main(_args(tmp_path, labels, cdir, mpath, *fast)) == 0
    assert len(loop.load_state(wdir)["rounds"]) == 1
    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--force", "--min-gain", "1", *fast)) == 0
    st = loop.load_state(wdir)
    assert len(st["rounds"]) == 2 and not st["rounds"][1]["promoted"] and st["champion"] == str(r1)
    ck = torch.load(loop.loop_dir(wdir) / "round-002.pt", weights_only=False)
    assert ck["finetune"]["compare"]["champion"]["weights"] == str(r1)
    assert st["rounds"][1]["champion_val"] == ck["finetune"]["compare"]["champion"]["val"]
    names = {p.name for p in loop.loop_dir(wdir).iterdir()}
    assert {"history.md", "round-002-last.pt", "round-002-check.json", "round-002-check.jpg"} <= names
    chk = json.loads((loop.loop_dir(wdir) / "round-002-check.json").read_text())
    assert chk["weights"].endswith("round-002-last.pt") and st["rounds"][1]["check"]["frames"] == len(chk["frames"])
    assert all(m["clip"] == "a" and m["kind"] in loop.KINDS for m in chk["frames"])
    assert {m["part"] for m in chk["frames"]} <= {"train", "val"}
    c = chk["scores"][str(chk["threshold"])]  # the last epoch's own threshold
    assert c["tp"] + c["tn"] + len(chk["frames"]) == 40  # every labeled frame is right or listed once

    # Standalone check of any checkpoint, no training.
    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--check", str(base), "--device", "cpu")) == 0
    assert (loop.loop_dir(wdir) / "check-pretrained.json").exists()

    assert loop.main(_args(tmp_path, labels, cdir, mpath, "--rollback")) == 0
    assert (wdir / "heatmap.pt").read_bytes() == (loop.loop_dir(wdir) / "base.pt").read_bytes()
