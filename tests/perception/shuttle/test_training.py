"""Fine-tuning plumbing: label mapping (no torch) and a tiny training run (needs torch + OpenCV)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_shuttle import clip_labels, select, split_by_time


def _doc(clip_id, fps=30.0, start_ms=10_000, n=40, status="reviewed"):
    sh = []
    for f in range(n):
        t = int(round(start_ms + f * 1000 / fps))
        sh.append({"t_ms": t, "visible": f % 10 != 9, "xy": [20.0 + 5 * f, 60.0] if f % 10 != 9 else None})
    return {"schema_version": 1, "clip_id": clip_id, "kind": "label", "status": status,
            "video": {"width": 320, "height": 180, "fps": fps}, "range_ms": [start_ms, start_ms + 2000],
            "court": [], "shuttle": sh, "hits": [], "rallies": []}


def test_clip_labels_map_ms_to_frames():
    lab = clip_labels(_doc("c"), 10_000)
    assert len(lab) == 40 and lab[0] == (20.0, 60.0) and lab[9] is None and lab[38] == (210.0, 60.0)
    a, b = split_by_time(lab, 0.75)
    assert max(a) < min(b) and len(a) == 30 and len(b) == 10


def test_select_uses_split_and_review_status(tmp_path):
    labels, clips = tmp_path / "labels", tmp_path / "clips"
    labels.mkdir()
    clips.mkdir()
    for cid, status in (("tr", "reviewed"), ("ev", "reviewed"), ("dr", "draft")):
        (labels / f"{cid}.json").write_text(json.dumps(_doc(cid, status=status)))
        (clips / f"{cid}.mp4").write_bytes(b"")
        (clips / f"{cid}.meta.json").write_text(json.dumps({"start_ms": 10_000}))
    manifest = {"tr": {"split": "train"}, "ev": {"split": "eval"}, "dr": {"split": "train"}}
    tr, va, _ = select(labels, clips, manifest, False, [], [], None)
    assert [c.clip_id for c in tr] == ["tr"] and [c.clip_id for c in va] == ["ev"]
    tr, va, _ = select(labels, clips, manifest, True, [], [], "ev:0.5")
    assert {c.clip_id for c in tr} == {"dr", "tr", "ev[:0.5]"} and [c.clip_id for c in va] == ["ev[0.5:]"]


def test_tiny_finetune_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from rallyiq.perception.shuttle.detector import HeatmapShuttleDetector
    from rallyiq.perception.shuttle.model import build_heatmap_net
    from rallyiq.perception.shuttle.training import (
        ClipFrames,
        TrainClip,
        TrainConfig,
        make_input,
        train,
    )

    w, h, n = 320, 180, 24
    video = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    labels = {}
    for f in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        x, y = 20 + 10 * f, 60 + f
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
        vw.write(img)
        labels[f] = (float(x), float(y)) if f % 6 else None
    vw.release()

    torch.manual_seed(0)
    seq_len = 3
    ckpt = tmp_path / "base.pt"
    torch.save({"model": build_heatmap_net(seq_len, "concat").state_dict(),
                "param_dict": {"seq_len": seq_len, "bg_mode": "concat"}}, ckpt)

    clip = TrainClip("c", video, 30.0, labels)
    cf = ClipFrames(clip, seq_len, use_bg=True)
    idx, fr = cf.window(cf.starts[0])
    assert fr.shape == (seq_len, 288, 512, 3)
    assert make_input(fr, cf.median, True, 1.1, 0.0).shape == ((seq_len + 1) * 3, 288, 512)
    hm, mask = cf.target(idx, flip=False)
    k = next(i for i, f in enumerate(idx) if labels.get(f))
    yy, xx = np.unravel_index(hm[k].argmax(), hm[k].shape)
    x, y = labels[idx[k]]
    assert abs(xx - x * 512 / w) <= 1 and abs(yy - y * 288 / h) <= 1 and mask.sum() == seq_len

    out = tmp_path / "ft.pt"
    cfg = TrainConfig(epochs=1, steps_per_epoch=2, batch_size=1, device="cpu", select_at=None)
    val = TrainClip("v", video, 30.0, {f: v for f, v in labels.items() if f >= 12})
    val2 = TrainClip("v2", video, 30.0, {f: v for f, v in labels.items() if f >= 18})
    report = train(ckpt, [TrainClip("c", video, 30.0, {f: v for f, v in labels.items() if f < 12})], [val, val2],
                   out, cfg, progress=lambda s: None)
    assert [h["epoch"] for h in report["history"]] == [0, 1]
    # several clips in one pass, with per-frame peaks (a local name once shadowed the argument)
    from rallyiq.perception.shuttle.training import disagreements
    peaks: dict = {}
    _res, _mistakes = disagreements(out, [val, val2], TrainConfig(device="cpu"), progress=lambda s: None,
                                   peaks=peaks)
    assert set(peaks) == {"c"} and set(peaks["c"]) == {f for f in labels if f >= 12}
    assert set(report["history"][0]["val"]) == {"0.5", "0.4", "0.3", "0.2"}
    assert report["best_threshold"] in (0.5, 0.4, 0.3, 0.2)
    det = HeatmapShuttleDetector(out)  # loads as a normal detector checkpoint
    assert det.seq_len == seq_len and det.bg_mode == "concat"
    assert det.recommended_threshold == report["best_threshold"]  # travels with the checkpoint


def test_best_threshold_prefers_higher_on_ties():
    from rallyiq.perception.shuttle.training import ValResult, best_threshold

    res = {"0.5": {"f1": 0.9}, "0.3": {"f1": 0.95}, "0.2": {"f1": 0.95}}
    assert best_threshold(res) == (0.95, 0.3)
    assert ValResult(1, 0.0, res).best() == (0.95, 0.3)
    assert best_threshold({}) == (0.0, 0.2)
