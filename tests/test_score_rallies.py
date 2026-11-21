import json

from scripts import score_rallies

TRUTH = [{
    "clip": "c", "window_s": [0, 60],
    "rallies": [
        {"start_s": 10.0, "end_s": 20.0, "sure": True},
        {"start_s": 30.0, "end_s": 35.0, "sure": True},
        {"start_s": 40.0, "end_s": 42.0, "sure": False},
    ],
    "unsure": [[50.0, 55.0]],
}]


def _r(a, b, hits=0):
    return {"start_ms": a * 1000, "end_ms": b * 1000, "shot_ids": [f"s{i}" for i in range(hits)]}


def test_perfect_prediction():
    s = score_rallies.score_clip(TRUTH, [_r(10, 20), _r(30, 35), _r(40, 42)])
    assert s["found"] == 3 and s["missed"] == 0 and s["fragments"] == 0
    assert s["false"] == 0 and s["merged"] == 0
    assert s["start_err_s"] == 0.0 and s["end_err_s"] == 0.0


def test_fragments_false_merges_and_errors():
    pred = [
        _r(10.5, 14), _r(16, 19),   # truth 1 found, split in two, start 0.5 s late, end 1 s early
        _r(31, 41.5),               # truth 2 and 3 merged into one
        _r(24, 25),                 # dead time: false rally
        _r(51, 53),                 # inside an unsure span: ignored
        _r(70, 80),                 # outside the window: ignored
    ]
    s = score_rallies.score_clip(TRUTH, pred)
    assert s["found"] == 3 and s["missed"] == 0
    assert s["fragments"] == 1
    assert s["merged"] == 1
    assert s["false"] == 1
    # sure rallies only: truth 1 (0.5, 1.0), truth 2 (1.0, 6.5) -> medians 0.75 and 3.75
    assert s["start_err_s"] == 0.75 and s["end_err_s"] == 3.75


def test_missed_when_too_little_is_covered():
    s = score_rallies.score_clip(TRUTH, [_r(10, 12)])
    assert s["found"] == 0 and s["missed"] == 3


def test_main_reads_runs(tmp_path, capsys):
    art = tmp_path / "myrun" / "data" / "videos" / "v" / "artifacts"
    art.mkdir(parents=True)
    (tmp_path / "myrun" / "results.json").write_text(json.dumps([{"clip_id": "c", "video_id": "v"}]))
    (art / "events.json").write_text(json.dumps({"rallies": [_r(10, 20), _r(24, 25)]}))
    (tmp_path / "t.json").write_text(json.dumps(TRUTH))
    assert score_rallies.main(["--truth", str(tmp_path / "t.json"), "--run", str(tmp_path / "myrun")]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("myrun")).split()
    # truth found missed fragments merged false start_err end_err
    assert row[1:8] == ["3", "1", "2", "0", "0", "1", "0.00"]
