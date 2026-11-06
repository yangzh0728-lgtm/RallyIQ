"""End-to-end smoke test of a running online stack (docker compose, or the real deployment).

    python scripts/smoke_online.py --base http://localhost:8000 --video clip.mp4 --invite rallyiq

Signs up (or logs in), uploads the video the way the browser does (direct-to-bucket when the
server offers it), waits for the analysis job and checks the results endpoints. Stdlib only.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--email", default=f"smoke-{uuid.uuid4().hex[:6]}@example.com")
    ap.add_argument("--password", default="smoke-password")
    ap.add_argument("--invite", default=None)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def call(method: str, path: str, body: dict | None = None, ok=(200, 201, 204)):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(args.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with opener.open(req, timeout=60) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            if e.code in ok:
                return e.code, None
            raise SystemExit(f"{method} {path} -> {e.code} {e.read()[:300]!r}") from e

    _, health = call("GET", "/api/health")
    _, cfg = call("GET", "/api/config")
    print("health", health, "config", cfg)
    if cfg["auth"]:
        try:
            call("POST", "/api/auth/signup", {"email": args.email, "password": args.password,
                                              "invite_code": args.invite})
        except SystemExit:
            call("POST", "/api/auth/login", {"email": args.email, "password": args.password})
        print("signed in as", call("GET", "/api/auth/me")[1]["email"])

    data = args.video.read_bytes()
    if cfg["direct_upload"]:
        _, t = call("POST", "/api/uploads", {"filename": args.video.name, "size_bytes": len(data)})
        put = urllib.request.Request(t["url"], data=data, method="PUT", headers=t["headers"])
        with urllib.request.urlopen(put, timeout=120) as r:  # straight to the bucket, no cookies
            assert r.status == 200, r.status
        _, video = call("POST", f"/api/uploads/{t['video_id']}/complete")
        print("direct upload ok", video["video_id"])
    else:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{args.video.name}\"\r\n"
                f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(args.base + "/api/videos", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with opener.open(req, timeout=120) as r:
            video = json.loads(r.read())
        print("api upload ok", video["video_id"])

    vid, job_id = video["video_id"], video["latest_job"]["job_id"]
    t0 = time.time()
    while True:
        _, job = call("GET", f"/api/jobs/{job_id}")
        if job["state"] in ("succeeded", "failed"):
            break
        if time.time() - t0 > args.timeout:
            print("timed out waiting for job", job, file=sys.stderr)
            return 1
        time.sleep(2)
    print("job", job["state"], [(s["name"], s["state"]) for s in job["stages"]], job.get("error") or "")
    if job["state"] != "succeeded":
        return 1
    _, ev = call("GET", f"/api/videos/{vid}/events")
    _, ov = call("GET", f"/api/videos/{vid}/overlay")
    print(f"events: {len(ev['rallies'])} rallies, {len(ev['shots'])} shots; overlay frames: {len(ov['frames'])}")
    req = urllib.request.Request(args.base + f"/api/videos/{vid}/file", headers={"Range": "bytes=0-99"})
    with opener.open(req, timeout=60) as r:  # follows the redirect to the bucket when using S3
        assert r.status == 206 and len(r.read()) == 100, r.status
    print("video streaming (Range) ok")
    call("DELETE", f"/api/videos/{vid}")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
