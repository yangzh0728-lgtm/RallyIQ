import time

import pytest
from fastapi.testclient import TestClient

from rallyiq.api.app import create_app
from rallyiq.store import Store


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RALLYIQ_DEMO", "1")
    app = create_app(Store(tmp_path / "data"), start_worker=True)
    with TestClient(app) as c:
        yield c


def wait_job(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("succeeded", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job timed out")


def upload(client, path):
    with path.open("rb") as f:
        r = client.post("/api/videos", files={"file": ("my clip.mp4", f, "video/mp4")})
    assert r.status_code == 201, r.text
    return r.json()


def test_upload_analyze_and_read_results(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    job = wait_job(client, v["latest_job"]["job_id"])
    assert job["state"] == "succeeded", job
    assert all(s["state"] == "succeeded" for s in job["stages"])

    v = client.get(f"/api/videos/{vid}").json()
    assert v["meta"]["fps"] == pytest.approx(30)
    assert v["meta"]["frame_count"] == 360
    assert set(v["artifacts"]) >= {"calibration", "players", "shuttle", "events", "strokes", "overlay"}

    ov = client.get(f"/api/videos/{vid}/overlay").json()
    assert ov["court"] and ov["frames"][0]["players"]
    part = client.get(f"/api/videos/{vid}/overlay", params={"start_ms": 1000, "end_ms": 2000}).json()
    assert all(1000 <= f["t_ms"] <= 2000 for f in part["frames"]) and part["frames"]

    ev = client.get(f"/api/videos/{vid}/events").json()
    assert ev["rallies"] and ev["shots"]

    r = client.get(f"/api/videos/{vid}/file", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    hm = client.get(f"/api/videos/{vid}/heatmap", params={"player_id": "P1"}).json()
    assert hm["samples"] == 360 and sum(map(sum, hm["counts"])) == 360

    q = client.get(f"/api/videos/{vid}/quality").json()
    assert q["camera_view"] in ("end_center", "end_oblique", "side", "unknown")
    keys = {c["key"] for c in q["checks"]}
    assert {"fps", "resolution", "calibration"} <= keys
    assert {c["capability"] for c in q["capabilities"]} == {"court_metrics", "players", "shuttle", "rallies", "hits"}
    assert next(c for c in q["capabilities"] if c["capability"] == "hits")["status"] != "ok"
    assert client.get(f"/api/videos/{vid}/quality").json()["fingerprint"] == q["fingerprint"]  # cached


def test_corrections_layer_on_raw_events(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    raw = client.get(f"/api/videos/{vid}/events").json()
    first, second = raw["shots"][0], raw["shots"][1]

    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "delete_shot", "payload": {"shot_id": first["shot_id"]}}).status_code == 201
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "set_hitter", "payload": {"shot_id": second["shot_id"], "hitter": "P1"}}).status_code == 201
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "add_shot", "payload": {"frame": 50, "hitter": "P2"}}).status_code == 201
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "set_hitter", "payload": {"shot_id": "x"}}).status_code == 422

    ev = client.get(f"/api/videos/{vid}/events").json()
    ids = {s["shot_id"]: s for s in ev["shots"]}
    assert first["shot_id"] not in ids
    assert ids[second["shot_id"]]["hitter"] == "P1" and ids[second["shot_id"]]["source"] == "human"
    added = [s for s in ev["shots"] if s["source"] == "human" and s["frame"] == 50]
    assert added and added[0]["rally_id"] == raw["rallies"][0]["rally_id"]

    # raw prediction is untouched
    assert client.get(f"/api/videos/{vid}/events", params={"raw": True}).json()["shots"][0] == first

    # stroke types: model prediction merged into events, human label via set_shot_type
    assert "stroke" in first and first["stroke"] is not None and first["stroke"]["source"] == "model"
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "set_shot_type", "payload": {"shot_id": second["shot_id"], "shot_type": "smash"}}).status_code == 201
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "set_shot_type", "payload": {"shot_id": second["shot_id"], "shot_type": "tweener"}}).status_code == 422
    s2 = {s["shot_id"]: s for s in client.get(f"/api/videos/{vid}/events").json()["shots"]}[second["shot_id"]]
    assert s2["shot_type"] == "smash" and s2["stroke"]["source"] == "human"

    csv_text = client.get(f"/api/videos/{vid}/export", params={"format": "csv"}).text
    assert csv_text.splitlines()[0].startswith("shot_id,rally_id")
    assert csv_text.splitlines()[0].endswith("shot_type,shot_type_confidence,shot_type_status,shot_type_source")
    exp = client.get(f"/api/videos/{vid}/export").json()
    assert len(exp["corrections"]) == 4 and exp["raw_events"]["shots"][0] == first


def test_manual_calibration_reruns_pipeline(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    corners = [[100, 340], [540, 340], [430, 120], [210, 120]]
    job = client.put(f"/api/videos/{vid}/calibration/manual", json={"corners": corners}).json()
    assert [s["name"] for s in job["stages"] if s["state"] != "skipped"][0] == "calibration"
    job = wait_job(client, job["job_id"])
    assert job["state"] == "succeeded"
    cal = client.get(f"/api/videos/{vid}/artifacts/calibration").json()
    assert cal["segments"][0]["method"] == "manual"


def test_rejects_unknown_type_and_missing(client, tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with p.open("rb") as f:
        assert client.post("/api/videos", files={"file": ("x.txt", f)}).status_code == 415
    assert client.get("/api/videos/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_confirm_ops_and_hard_example_flag(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    raw = client.get(f"/api/videos/{vid}/events").json()
    shot, rally = raw["shots"][0], raw["rallies"][0]
    r = client.post(f"/api/videos/{vid}/corrections",
                    json={"op": "confirm_shot", "payload": {"shot_id": shot["shot_id"]}, "hard_example": False})
    assert r.status_code == 201 and r.json()["hard_example"] is False
    client.post(f"/api/videos/{vid}/corrections", json={"op": "confirm_rally", "payload": {"rally_id": rally["rally_id"]}})
    ev = client.get(f"/api/videos/{vid}/events").json()
    s = next(x for x in ev["shots"] if x["shot_id"] == shot["shot_id"])
    assert s["status"] == "ok" and s["source"] == "human" and s["hitter"] == shot["hitter"] and s["frame"] == shot["frame"]
    rr = next(x for x in ev["rallies"] if x["rally_id"] == rally["rally_id"])
    assert rr["status"] == "ok" and rr["start_frame"] == rally["start_frame"]
    assert [c["hard_example"] for c in client.get(f"/api/videos/{vid}/corrections").json()] == [False, True]


def test_doubles_match_format_rerun_heatmap_and_corrections(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    assert client.get(f"/api/videos/{vid}/match_format").json() == {"match_format": "auto"}
    assert client.get(f"/api/videos/{vid}/artifacts/players").json()["match_format"] == "singles"

    job = client.put(f"/api/videos/{vid}/match_format", json={"match_format": "doubles"}).json()
    assert [s["name"] for s in job["stages"] if s["state"] != "skipped"][0] == "players"
    assert wait_job(client, job["job_id"])["state"] == "succeeded"
    assert client.put(f"/api/videos/{vid}/match_format", json={"match_format": "mixed"}).status_code == 422

    players = client.get(f"/api/videos/{vid}/artifacts/players").json()
    assert players["match_format"] == "doubles"
    assert {p["player_id"] for p in players["frames"][0]["players"]} == {"P1", "P2", "P3", "P4"}
    ov = client.get(f"/api/videos/{vid}/overlay").json()
    assert ov["match_format"] == "doubles" and len(ov["court"][0]["boundary"]) == 4
    ev = client.get(f"/api/videos/{vid}/events").json()
    assert ev["match_format"] == "doubles"

    team = client.get(f"/api/videos/{vid}/heatmap", params={"team": "A"}).json()
    p3 = client.get(f"/api/videos/{vid}/heatmap", params={"player_id": "P3"}).json()
    assert team["team"] == "A" and team["samples"] == 720 and p3["samples"] == 360

    shot = next(s for s in ev["shots"] if s["hitter"] == "P1")
    r = client.post(f"/api/videos/{vid}/corrections", json={"op": "set_hitter",
                                                             "payload": {"shot_id": shot["shot_id"], "hitter": "P3"}})
    assert r.status_code == 201
    rid = ev["rallies"][0]["rally_id"]
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "set_rally_winner",
                                                               "payload": {"rally_id": rid, "winner": "A"}}).status_code == 201
    # Partners mixed up from frame 0: swap P1/P3. Team winner stays.
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "swap_player_ids",
                                                               "payload": {"from_frame": 0, "a": "P1", "b": "P3"}}).status_code == 201
    assert client.post(f"/api/videos/{vid}/corrections", json={"op": "swap_player_ids",
                                                               "payload": {"from_frame": 0, "a": "P1", "b": "P1"}}).status_code == 422
    ev2 = client.get(f"/api/videos/{vid}/events").json()
    s2 = next(s for s in ev2["shots"] if s["shot_id"] == shot["shot_id"])
    assert s2["hitter"] == "P1"
    assert ev2["rallies"][0]["winner"] == "A"
    csv = client.get(f"/api/videos/{vid}/export", params={"format": "csv"}).text
    assert csv.splitlines()[0].split(",")[6] == "team"


def test_match_report_and_coach(client, sample_video, monkeypatch):
    monkeypatch.setenv("RALLYIQ_COACH_PROVIDER", "offline")
    v = upload(client, sample_video)
    vid = v["video_id"]
    assert wait_job(client, v["latest_job"]["job_id"])["state"] == "succeeded"

    rep = client.get(f"/api/videos/{vid}/report").json()
    keys = {m["key"] for m in rep["stats"]["metrics"]}
    assert {"match.rallies", "match.rally_len.mean"} <= keys
    assert rep["coach"] is None and rep["coach_stale"] is False
    assert [p["player_id"] for p in rep["stats"]["players"]] == ["P1", "P2"]

    coach = client.post(f"/api/videos/{vid}/report/coach", json={}).json()
    assert coach["provider"] == "offline" and coach["summary"]
    rep = client.get(f"/api/videos/{vid}/report").json()
    assert rep["coach"]["stats_fingerprint"] == rep["stats"]["fingerprint"] and not rep["coach_stale"]

    # a correction changes the stats -> the cached coach report is marked stale
    rally_id = client.get(f"/api/videos/{vid}/events").json()["rallies"][0]["rally_id"]
    client.post(f"/api/videos/{vid}/corrections", json={"op": "set_rally_winner",
                                                         "payload": {"rally_id": rally_id, "winner": "P1"}})
    rep = client.get(f"/api/videos/{vid}/report").json()
    assert rep["coach_stale"] is True
    won = next(m for m in rep["stats"]["players"][0]["metrics"] if m["key"] == "P1.win.won")
    assert won["value"] == 1

    md = client.get(f"/api/videos/{vid}/report/export", params={"format": "md"})
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert "赛后报告" in md.text and "已过时" in md.text
    js = client.get(f"/api/videos/{vid}/report/export", params={"format": "json"}).json()
    assert js["coach"]["provider"] == "offline"

    assert client.post(f"/api/videos/{vid}/report/coach", json={"provider": "nope"}).status_code == 503
    providers = client.get("/api/coach/providers").json()
    assert providers["default"] == "offline" and {"anthropic", "offline"} <= set(providers["providers"])


def test_hard_examples_collected_from_corrections(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    raw = client.get(f"/api/videos/{vid}/events").json()
    s0, s1 = raw["shots"][0], raw["shots"][1]
    other = "P2" if s1["hitter"] == "P1" else "P1"
    post = lambda op, payload, **kw: client.post(f"/api/videos/{vid}/corrections", json={"op": op, "payload": payload, **kw})
    post("delete_shot", {"shot_id": s0["shot_id"]})
    post("set_hitter", {"shot_id": s1["shot_id"], "hitter": other})
    post("add_shot", {"frame": 200, "hitter": "P1"})
    post("confirm_rally", {"rally_id": raw["rallies"][0]["rally_id"]})
    post("move_shot", {"shot_id": s1["shot_id"], "frame": s1["frame"] + 2}, hard_example=False)

    ex = client.get("/api/hard_examples").json()
    kinds = [e["kind"] for e in ex]
    assert kinds == ["false_hit", "wrong_hitter", "missed_hit", "confirmed"]  # move_shot opted out
    assert ex[0]["model"]["shot_id"] == s0["shot_id"] and ex[0]["frame"] == s0["frame"]
    assert ex[2]["frame"] == 200 and ex[2]["model"] is None
    assert ex[0]["conditions"]["fps"] == "25-35" and "camera_view" in ex[0]["conditions"]

    summ = client.get("/api/hard_examples/summary").json()
    assert summ["total"] == 4 and summ["errors"] == 3 and summ["videos"] == 1
    fps = next(g for g in summ["groups"] if g["condition"] == "fps")
    assert fps["errors"] == {"hits": 3} and fps["confirmed"] == 1
    assert fps["hit_error_rate"] == pytest.approx(3 / fps["model_shots"], abs=1e-3)

    jl = client.get("/api/hard_examples", params={"format": "jsonl"}).text.strip().splitlines()
    assert len(jl) == 4


def test_clip_export(client, sample_video):
    v = upload(client, sample_video)
    vid = v["video_id"]
    wait_job(client, v["latest_job"]["job_id"])
    rally = client.get(f"/api/videos/{vid}/events").json()["rallies"][0]
    r = client.get(f"/api/videos/{vid}/clip", params={"rally_id": rally["rally_id"], "pad_ms": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert "attachment" in r.headers["content-disposition"] and r.content[4:8] == b"ftyp"
    r2 = client.get(f"/api/videos/{vid}/clip", params={"start_ms": 1000, "end_ms": 2000})
    assert r2.status_code == 200 and len(r2.content) > 1000
    assert client.get(f"/api/videos/{vid}/clip").status_code == 422
    assert client.get(f"/api/videos/{vid}/clip", params={"start_ms": 2000, "end_ms": 1000}).status_code == 422
    assert client.get(f"/api/videos/{vid}/clip", params={"rally_id": "nope"}).status_code == 404


def test_memory_commit_profile_and_opponent(client, sample_video):
    vids = []
    for _ in range(2):
        v = upload(client, sample_video)
        wait_job(client, v["latest_job"]["job_id"])
        vids.append(v["video_id"])
    assert client.post(f"/api/videos/{vids[0]}/memory/commit").status_code == 422  # nobody named yet
    assert client.put(f"/api/videos/{vids[0]}/people", json={"people": {"P1": "我"}, "self_player": "P2"}).status_code == 422
    for vid in vids:
        r = client.put(f"/api/videos/{vid}/people", json={"people": {"P1": " 我 ", "P2": "小李"}, "self_player": "P1"})
        assert r.status_code == 200 and r.json()["people"]["P1"] == "我"
        client.post(f"/api/videos/{vid}/report/coach", json={"provider": "offline"})
        ev = client.post(f"/api/videos/{vid}/memory/commit")
        assert ev.status_code == 201 and {e["person"] for e in ev.json()} == {"我", "小李"}
    me = client.get("/api/profile").json()
    assert me["person"] == "我" and me["is_self"] and len(me["videos"]) == 2
    shots = next(m for m in me["metrics"] if m["key"] == "shots")
    assert len(shots["points"]) == 2 and shots["trend"] in ("flat", "up", "down")
    assert all(r["to_video_id"] == vids[1] or r["status"] == "pending" for r in me["retests"])

    ppl = client.get("/api/people").json()
    assert [p["person"] for p in ppl] == ["我", "小李"] and ppl[0]["is_self"]
    opp = client.get("/api/opponents/小李").json()
    assert opp["person"] == "小李" and len(opp["videos"]) == 2 and opp["shots"] > 0

    # re-commit appends; withdraw hides the video but keeps the log
    client.post(f"/api/videos/{vids[0]}/memory/commit")
    client.post(f"/api/videos/{vids[1]}/memory/withdraw")
    assert len(client.get("/api/profile").json()["videos"]) == 1
    log = client.get("/api/memory/log").json()
    assert len(log) == 7 and [e["seq"] for e in log] == list(range(1, 8))
