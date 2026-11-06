"""Converters for public datasets (ShuttleSet, TrackNet) into evaluation labels."""
import csv
from pathlib import Path

import numpy as np

from evaluation import benchmark
from evaluation.public import shuttleset as SS
from evaluation.public import tracknet as TN

HEADER = ["rally", "ball_round", "time", "frame_num", "player", "player_location_x", "player_location_y"]


def _shuttleset(tmp: Path) -> Path:
    d = tmp / "set"
    (d / "m1").mkdir(parents=True)
    # identity homography: court coordinates == pixels, far baseline at y=300, near at y=700
    with open(d / "match.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "video", "tournament", "url"])
        w.writerow([1, "m1", "Test Open", "https://www.youtube.com/watch?v=abc123"])
    with open(d / "homography.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "video", "homography_matrix", "upleft_x", "upright_x", "downleft_x", "downright_x",
                    "upleft_y", "upright_y", "downleft_y", "downright_y"])
        w.writerow([1, "m1", "[[1,0,0],[0,1,0],[0,0,1]]", 400, 880, 300, 980, 300, 300, 700, 700])
    with open(d / "m1" / "set1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        frame = 3000
        for rally in range(1, 6):
            for k in range(4):
                near = k % 2 == 0
                w.writerow([rally, k + 1, f"00:{frame // 30 // 60:02d}:{frame // 30 % 60:02d}", frame, "A",
                            640, 650 if near else 350])
                frame += 30
            frame += 300
    return d


def test_shuttleset_label(tmp_path):
    d = _shuttleset(tmp_path)
    (m,) = SS.load_matches(d)
    st = SS.load_strokes(d / "m1" / "set1.csv")
    fps = SS.video_fps(st)
    assert fps == 30.0
    win = SS.pick_window(st, fps, 60)
    assert win[1] - win[0] <= 60
    doc = SS.build_label("ss01", m, st, fps, win, 1920, 1080)
    assert doc["status"] == "reviewed" and doc["hits"]
    assert [h["side"] for h in doc["hits"][:4]] == ["near", "far", "near", "far"]
    # corners scaled from 1280x720 to 1920x1080, ordered far-left, far-right, near-right, near-left
    assert np.allclose(doc["court"][0]["corners"], [[600, 450], [1320, 450], [1470, 1050], [450, 1050]])
    assert all(r["end_ms"] > r["start_ms"] for r in doc["rallies"])
    # a perfect prediction scores 1.0
    pred = {**doc, "kind": "prediction"}
    res = benchmark.evaluate_clip(doc, pred)
    assert res["hits"].summary()["f1"] == 1.0 and res["rallies"].summary()["f1"] == 1.0


def test_tracknet_label(tmp_path):
    c = tmp_path / "test" / "match1" / "csv"
    v = tmp_path / "test" / "match1" / "video"
    c.mkdir(parents=True)
    v.mkdir(parents=True)
    (v / "1_01_00.mp4").write_bytes(b"")
    with open(c / "1_01_00_ball.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        w.writerows([[0, 1, 10, 20], [1, 0, 0, 0], [2, 1, 12, 22]])
    ((cid, _, csv_path),) = TN.rallies(tmp_path)
    doc = TN.build_label(cid, csv_path, 1280, 720, 30.0, 3)
    assert cid == "tn_test_match1_1_01_00"
    assert [s["visible"] for s in doc["shuttle"]] == [True, False, True]
    assert doc["shuttle"][1]["xy"] is None and doc["shuttle"][2]["t_ms"] == 67


def test_public_diag_finds_time_shift_and_flipped_court(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import public_diag as D

    d = _shuttleset(tmp_path)
    (m,) = SS.load_matches(d)
    st = SS.load_strokes(d / "m1" / "set1.csv")
    doc = SS.build_label("ss01", m, st, 30.0, SS.pick_window(st, 30.0, 60), 1280, 720)
    shift = 500
    pred = {**doc, "kind": "prediction",
            "hits": [{**h, "t_ms": h["t_ms"] - shift} for h in doc["hits"]],
            "court": [{**c, "corners": [c["corners"][i] for i in (2, 3, 0, 1)]} for c in doc["court"]]}
    sync = D.time_sync(doc, pred)
    assert sync["f1_at_0"] < 0.5 and sync["f1_at_best"] == 1.0 and abs(sync["best_shift_ms"] - shift) <= 34
    c = D.court(doc, pred)
    assert c["as is"] > 100 and c["180°"] == 0.0 and c["share_ok_any_sym"] == 1.0
    h = D.hits(doc, pred)
    assert h["gt"] == len([x for x in doc["hits"]]) and h["missed"] == h["gt"] - h["tp_any_side"]
