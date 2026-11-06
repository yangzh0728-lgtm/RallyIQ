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
