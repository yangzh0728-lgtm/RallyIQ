import json

from scripts import score_hits

TRUTH = {"windows": {"c": [[10, 20]]},
         "hits": {"c": [{"t": 11.0, "side": "far", "sure": True}, {"t": 12.0, "side": "near", "sure": True},
                        {"t": 13.0, "side": "far", "sure": False}, {"t": 14.0, "side": "near", "sure": True}]}}


def _run(tmp_path, shots):
    art = tmp_path / "myrun" / "data" / "videos" / "v" / "artifacts"
    art.mkdir(parents=True)
    (tmp_path / "myrun" / "results.json").write_text(json.dumps([{"clip_id": "c", "video_id": "v"}]))
    (art / "probe.json").write_text(json.dumps({"fps": 10.0}))
    (art / "events.json").write_text(json.dumps({"shots": [{"frame": f, "hitter_court": [3.0, y]} for f, y in shots]}))
    return tmp_path / "myrun"


def test_score_hits(tmp_path, capsys):
    (tmp_path / "t.json").write_text(json.dumps(TRUTH))
    # 11.1 far (found, right side), 12.0 far (found, wrong side), 13.0 unsure (ignored), 16.0 (false), 30 outside
    run = _run(tmp_path, [(111, 10.0), (120, 10.0), (130, 10.0), (160, 2.0), (300, 2.0)])
    s = score_hits.score_run(TRUTH, run)
    assert (s["sure"], s["found"], s["missed"], s["wrong_side"], s["false"]) == (3, 2, 1, 1, 1)
    assert (s["near_found"], s["near_sure"], s["far_found"], s["far_sure"]) == (1, 2, 1, 1)
    assert score_hits.main(["--truth", str(tmp_path / "t.json"), "--run", str(run)]) == 0
    assert any(line.startswith("myrun") for line in capsys.readouterr().out.splitlines())
