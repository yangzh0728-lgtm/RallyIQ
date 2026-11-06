"""Online-deployment pieces: lease-based job queue, object storage, accounts, direct upload, wake hook."""

import http.server
import json
import threading
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from rallyiq.api.app import create_app
from rallyiq.api.auth import AuthConfig, hash_password, verify_password
from rallyiq.contracts import EventsResult
from rallyiq.pipeline.worker import Worker
from rallyiq.store import Store, now


def make_video(store: Store, tmp_path, owner=None, data=b"\0" * 1000):
    vid, dest = store.create_video("clip.mp4", ".mp4")
    dest.write_bytes(data)
    store.register_video(vid, "clip.mp4", dest, owner_id=owner)
    return vid


# ------------------------------------------------------------------ job queue

def test_claim_is_exclusive_and_leased(tmp_path, db_url):
    store = Store(tmp_path / "data", db_url=db_url)
    vid = make_video(store, tmp_path)
    a, b = store.create_job(vid), store.create_job(vid)
    j1 = store.claim_next_job("w1")
    j2 = store.claim_next_job("w2")
    assert {j1.job_id, j2.job_id} == {a.job_id, b.job_id}
    assert j1.job_id == a.job_id  # FIFO
    assert store.claim_next_job("w3") is None
    assert store.heartbeat(j1.job_id, "w1")
    assert not store.heartbeat(j1.job_id, "w2")  # not w2's job
    assert store.queue_depth() == {"queued": 0, "running": 2}


def test_parallel_claims_never_double_assign(tmp_path, db_url):
    store = Store(tmp_path / "data", db_url=db_url)
    vid = make_video(store, tmp_path)
    for _ in range(12):
        store.create_job(vid)
    stores = [Store(tmp_path / "data", db_url=db_url) for _ in range(4)]  # separate connections
    got: list[str] = []
    lock = threading.Lock()

    def claim(s, wid):
        while (j := s.claim_next_job(wid)) is not None:
            with lock:
                got.append(j.job_id)

    ts = [threading.Thread(target=claim, args=(s, f"w{i}")) for i, s in enumerate(stores)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(got) == 12 and len(set(got)) == 12


def test_stale_jobs_are_requeued_then_failed(tmp_path, db_url):
    store = Store(tmp_path / "data", db_url=db_url)
    vid = make_video(store, tmp_path)
    job = store.create_job(vid)
    old = (now() - timedelta(minutes=30)).isoformat()

    claimed = store.claim_next_job("dead-worker")
    claimed.stages[0].state = "running"
    store.update_job(claimed, "dead-worker")
    store.db.q("UPDATE jobs SET heartbeat_at=? WHERE job_id=?", (old, job.job_id))
    assert store.recover_stale_jobs(lease_s=60, max_attempts=2) == {"requeued": 1, "failed": 0}
    j = store.get_job(job.job_id)
    assert j.state == "queued" and j.stages[0].state == "pending"

    # the dead worker comes back: its writes are ignored, the job now belongs to w2
    store.claim_next_job("w2")
    claimed.state = "failed"
    store.update_job(claimed, "dead-worker")
    assert store.get_job(job.job_id).state == "running"

    store.db.q("UPDATE jobs SET heartbeat_at=? WHERE job_id=?", (old, job.job_id))
    assert store.recover_stale_jobs(lease_s=60, max_attempts=2) == {"requeued": 0, "failed": 1}
    j = store.get_job(job.job_id)
    assert j.state == "failed" and "worker stopped" in j.error


def test_worker_runs_jobs_and_exits_when_idle(tmp_path, db_url, sample_video, monkeypatch):
    monkeypatch.setenv("RALLYIQ_DEMO", "1")
    store = Store(tmp_path / "data", db_url=db_url)
    vid = make_video(store, tmp_path, data=sample_video.read_bytes())
    job = store.create_job(vid)
    w = Worker(store, poll_s=0.05, idle_exit_s=0.3, single_process=False, worker_id="gpu-1")
    t0 = time.monotonic()
    w.loop()  # returns by itself once the queue stays empty
    assert time.monotonic() - t0 < 60
    assert w.jobs_done == 1
    assert store.get_job(job.job_id).state == "succeeded"
    assert store.get_video(vid).artifacts == ["probe", "calibration", "players", "shuttle", "events", "overlay"]


# ------------------------------------------------------------- object storage

@pytest.fixture()
def s3_env(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("RALLYIQ_STORAGE", "s3://rallyiq-test/prod")
    monkeypatch.setenv("RALLYIQ_S3_REGION", "us-east-1")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="rallyiq-test")
        yield


def test_s3_store_roundtrip(tmp_path, s3_env, sample_video, monkeypatch):
    monkeypatch.setenv("RALLYIQ_DEMO", "1")
    store = Store(tmp_path / "api-data")
    assert store.blobs.kind == "s3"
    vid = make_video(store, tmp_path, data=sample_video.read_bytes())
    assert store.blobs.exists(f"videos/{vid}/source.mp4")

    # a worker on another machine: empty cache, same database and bucket
    worker_store = Store(tmp_path / "gpu-cache", db_url=store.db.url)
    job = store.create_job(vid)
    Worker(worker_store, single_process=False).run_once()
    assert store.get_job(job.job_id).state == "succeeded"
    assert (tmp_path / "gpu-cache" / "videos" / vid / "source.mp4").exists()  # downloaded for the stages
    assert store.load_artifact(vid, "events", EventsResult) is not None
    assert "overlay" in store.get_video(vid).artifacts
    assert "X-Amz-Signature" in store.video_download_url(vid)

    store.delete_video(vid)
    assert store.get_video(vid) is None
    assert not store.blobs.exists(f"videos/{vid}/source.mp4")
    assert not store.blobs.exists(f"videos/{vid}/artifacts/events.json")


def test_direct_upload_flow(tmp_path, s3_env, sample_video):
    store = Store(tmp_path / "data")
    app = create_app(store, start_worker=False, auth=AuthConfig(enabled=False))
    with TestClient(app) as c:
        assert c.get("/api/config").json()["direct_upload"] is True
        data = sample_video.read_bytes()
        t = c.post("/api/uploads", json={"filename": "match.mp4", "size_bytes": len(data)})
        assert t.status_code == 201, t.text
        ticket = t.json()
        assert ticket["headers"]["Content-Type"] == "video/mp4" and "X-Amz-Signature" in ticket["url"]
        vid = ticket["video_id"]
        # not visible until the upload completes
        assert c.get("/api/videos").json() == []
        assert c.post(f"/api/uploads/{vid}/complete").status_code == 409
        store.blobs.put_bytes(f"videos/{vid}/source.mp4", data)  # what the browser's PUT does
        v = c.post(f"/api/uploads/{vid}/complete").json()
        assert v["size_bytes"] == len(data) and v["latest_job"]["state"] == "queued"
        assert [x["video_id"] for x in c.get("/api/videos").json()] == [vid]
        r = c.get(f"/api/videos/{vid}/file", follow_redirects=False)
        assert r.status_code == 307 and "X-Amz-Signature" in r.headers["location"]
        assert c.post("/api/uploads", json={"filename": "x.exe", "size_bytes": 5}).status_code == 415


def test_direct_upload_unavailable_with_local_storage(tmp_path):
    app = create_app(Store(tmp_path / "data"), start_worker=False, auth=AuthConfig(enabled=False))
    with TestClient(app) as c:
        assert c.get("/api/config").json()["direct_upload"] is False
        assert c.post("/api/uploads", json={"filename": "a.mp4", "size_bytes": 5}).status_code == 409


# ------------------------------------------------------------------- accounts

def test_password_hash():
    h = hash_password("correct horse")
    assert verify_password("correct horse", h) and not verify_password("wrong", h)
    assert not verify_password("x", "garbage")


@pytest.fixture()
def auth_app(tmp_path, db_url, monkeypatch):
    monkeypatch.setenv("RALLYIQ_MAX_ACTIVE_JOBS", "2")
    store = Store(tmp_path / "data", db_url=db_url)
    cfg = AuthConfig(enabled=True, signup="invite", invite_code="shuttle")
    return create_app(store, start_worker=False, auth=cfg)


def signup(c, email, code="shuttle", pw="password123"):
    return c.post("/api/auth/signup", json={"email": email, "password": pw, "invite_code": code})


def test_accounts_isolate_videos(auth_app, sample_video):
    with TestClient(auth_app) as a, TestClient(auth_app) as b:
        assert a.get("/api/videos").status_code == 401
        assert a.get("/api/health").status_code == 200
        assert a.get("/api/config").json()["auth"] is True
        assert signup(a, "sam@example.com", code="nope").status_code == 403
        assert signup(a, "not-an-email").status_code == 422
        assert signup(a, "sam@example.com", pw="short").status_code == 422
        r = signup(a, "Sam@Example.com")
        assert r.status_code == 201 and r.json()["is_admin"] is True
        assert signup(b, "sam@example.com").status_code == 409
        assert signup(b, "friend@example.com").json()["is_admin"] is False

        with sample_video.open("rb") as f:
            v = a.post("/api/videos", files={"file": ("m.mp4", f, "video/mp4")}).json()
        vid, job_id = v["video_id"], v["latest_job"]["job_id"]
        assert [x["video_id"] for x in a.get("/api/videos").json()] == [vid]
        assert b.get("/api/videos").json() == []
        for path in (f"/api/videos/{vid}", f"/api/videos/{vid}/file", f"/api/jobs/{job_id}",
                     f"/api/videos/{vid}/corrections"):
            assert b.get(path).status_code == 404, path
        assert b.delete(f"/api/videos/{vid}").status_code == 404
        assert a.get(f"/api/videos/{vid}/file", headers={"Range": "bytes=0-9"}).status_code == 206

        # per-user cap on queued + running jobs
        assert a.post(f"/api/videos/{vid}/jobs").status_code == 201
        assert a.post(f"/api/videos/{vid}/jobs").status_code == 429

        # bearer token works too; logout ends the session
        assert a.get("/api/auth/me").json()["email"] == "sam@example.com"
        assert a.post("/api/auth/logout").status_code == 204
        assert a.get("/api/videos").status_code == 401
        r = a.post("/api/auth/login", json={"email": "sam@example.com", "password": "wrong-password"})
        assert r.status_code == 401
        r = a.post("/api/auth/login", json={"email": "sam@example.com", "password": "password123"})
        assert r.status_code == 200
        assert a.delete(f"/api/videos/{vid}").status_code == 204
        assert a.get("/api/videos").json() == []


def test_closed_signup_and_auth_off(tmp_path):
    store = Store(tmp_path / "data")
    closed = create_app(store, start_worker=False, auth=AuthConfig(enabled=True, signup="closed"))
    with TestClient(closed) as c:
        assert signup(c, "a@b.co").status_code == 403
    off = create_app(store, start_worker=False, auth=AuthConfig(enabled=False))
    with TestClient(off) as c:
        assert c.get("/api/videos").status_code == 200
        assert c.get("/api/auth/me").status_code == 404


def test_upload_size_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("RALLYIQ_MAX_UPLOAD_MB", "1")
    app = create_app(Store(tmp_path / "data"), start_worker=False, auth=AuthConfig(enabled=False))
    with TestClient(app) as c:
        r = c.post("/api/videos", files={"file": ("big.mp4", b"\0" * (3 << 20), "video/mp4")})
        assert r.status_code == 413
        assert c.get("/api/videos").json() == []


# ------------------------------------------------------------------ wake hook

def test_enqueue_wakes_on_demand_worker(tmp_path, monkeypatch, sample_video):
    hits: list[tuple[str, str | None]] = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length") or 0))
            hits.append((self.path, self.headers.get("authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("RALLYIQ_WORKER_WAKE_URL", f"http://127.0.0.1:{srv.server_port}/wake")
    monkeypatch.setenv("RALLYIQ_WORKER_WAKE_TOKEN", "s3cret")
    app = create_app(Store(tmp_path / "data"), start_worker=False, auth=AuthConfig(enabled=False))
    try:
        with TestClient(app) as c:
            with sample_video.open("rb") as f:
                c.post("/api/videos", files={"file": ("m.mp4", f, "video/mp4")})
            for _ in range(50):
                if hits:
                    break
                time.sleep(0.05)
    finally:
        srv.shutdown()
    assert hits == [("/wake", "Bearer s3cret")]
