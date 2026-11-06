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
    assert set(v["artifacts"]) >= {"calibration", "players", "shuttle", "events", "overlay"}

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

    csv_text = client.get(f"/api/videos/{vid}/export", params={"format": "csv"}).text
    assert csv_text.splitlines()[0].startswith("shot_id,rally_id")
    exp = client.get(f"/api/videos/{vid}/export").json()
    assert len(exp["corrections"]) == 3 and exp["raw_events"]["shots"][0] == first


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
