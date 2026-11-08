import json

from scripts import score_points


def _run(tmp_path, name, points, shots):
    art = tmp_path / name / "data" / "videos" / "v" / "artifacts"
    art.mkdir(parents=True)
    (tmp_path / name / "results.json").write_text(json.dumps([{"clip_id": "c", "video_id": "v"}]))
    (art / "shuttle.json").write_text(json.dumps({"points": [{"frame": f, "x": x, "y": y} for f, (x, y) in points.items()]}))
    (art / "events.json").write_text(json.dumps({"shots": [{"frame": f, "hitter_court": [3.0, y]} for f, y in shots]}))
    return tmp_path / name


def test_score_points_counts_each_verdict(tmp_path, capsys):
    truth = [
        {"clip": "c", "frame": 1, "real": [100, 100], "false": [[500, 500]], "hit": "near"},
        {"clip": "c", "frame": 2, "real": None, "false": [[500, 500]], "hit": None},
        {"clip": "c", "frame": 3, "real": [100, 100], "false": []},
        {"clip": "c", "frame": 9, "real": "?", "false": [[500, 500]]},  # only false points scored
    ]
    (tmp_path / "t.json").write_text(json.dumps(truth))
    good = _run(tmp_path, "good", {1: (105, 98), 3: (101, 100)}, [(2, 2.0)])
    bad = _run(tmp_path, "bad", {1: (500, 505), 2: (500, 500)}, [(1, 10.0)])
    assert score_points.main(["--truth", str(tmp_path / "t.json"), "--run", str(good), "--run", str(bad), "--frames"]) == 0
    out = capsys.readouterr().out.splitlines()
    good_row = next(line for line in out if line.startswith("good")).split()[1:]
    bad_row = next(line for line in out if line.startswith("bad")).split()[1:]
    # real_found real_missed false_shown shown_when_none hit_found hit_missed hit_wrong_side false_hit
    assert good_row == ["2", "0", "0", "0", "1", "0", "0", "1"]
    assert bad_row == ["0", "2", "2", "1", "0", "0", "1", "1"]
    assert any("c f1:" in line for line in out)


def test_tracker_settings_from_the_command_line():
    from rallyiq.perception.shuttle.stage import tracker_config
    from scripts.diag_tracklets import parse_settings

    got = parse_settings(["flight_split=true", "contact_gate=ends", "hot_s=1.5"])
    assert got == {"flight_split": True, "contact_gate": "ends", "hot_s": 1.5}
    cfg = tracker_config(got | {"chain": "false"})
    assert cfg.flight_split is True and cfg.chain is False and cfg.contact_gate == "ends"
