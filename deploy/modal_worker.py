"""On-demand GPU worker on Modal (billed per second, scales to zero).

How it fits together:
  * The API queues a job in PostgreSQL and POSTs ``RALLYIQ_WORKER_WAKE_URL`` (the ``wake_app`` endpoint below).
  * ``wake`` spawns ``drain``: a GPU container running ``python -m rallyiq.pipeline worker --idle-exit``,
    which works through the queue and exits once it has been empty for IDLE_EXIT seconds.
  * ``sweep`` runs every 10 minutes as a safety net: if jobs are queued (a wake was lost) or a running
    job's lease expired (a container died), it starts ``drain`` too.

Setup (once; needs a Modal account):
    pip install modal && modal setup
    modal secret create rallyiq RALLYIQ_DATABASE_URL=... RALLYIQ_STORAGE=s3://rallyiq \\
        RALLYIQ_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com RALLYIQ_S3_REGION=auto \\
        AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... RALLYIQ_WORKER_WAKE_TOKEN=<random>
    modal deploy deploy/modal_worker.py        # run from the repo root
    # then set RALLYIQ_WORKER_WAKE_URL on the API to the printed URL of `wake_app`
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("RALLYIQ_MODAL_GPU", "L4")   # T4 is cheaper and slower; A10G faster
IDLE_EXIT = 120                                     # seconds of empty queue before the GPU shuts down
MAX_GPUS = 2                                        # hard cap on parallel GPU containers (cost guard)

app = modal.App("rallyiq-worker")
secret = modal.Secret.from_name("rallyiq")
worker_image = modal.Image.from_dockerfile(ROOT / "docker" / "worker.Dockerfile", context_dir=ROOT).entrypoint([])
light_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "psycopg[binary]")


@app.function(image=worker_image, gpu=GPU, secrets=[secret], timeout=6 * 3600, max_containers=MAX_GPUS)
def drain() -> None:
    subprocess.run(["python", "-m", "rallyiq.pipeline", "worker", "--idle-exit", str(IDLE_EXIT),
                    "--lease", "300"], check=True)


@app.function(image=light_image, secrets=[secret])
@modal.asgi_app()
def wake_app():
    import hmac

    from fastapi import FastAPI, Header, HTTPException

    api = FastAPI()

    @api.post("/")
    def wake(authorization: str = Header(default="")):
        want = os.environ.get("RALLYIQ_WORKER_WAKE_TOKEN", "")
        if not want or not hmac.compare_digest(authorization, f"Bearer {want}"):
            raise HTTPException(401, "bad token")
        drain.spawn()
        return {"ok": True}

    return api


@app.function(image=light_image, secrets=[secret], schedule=modal.Period(minutes=10))
def sweep() -> None:
    from datetime import UTC, datetime, timedelta

    import psycopg

    cutoff = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    with psycopg.connect(os.environ["RALLYIQ_DATABASE_URL"]) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM jobs WHERE state='queued' OR (state='running' AND heartbeat_at < %s)", (cutoff,)
        ).fetchone()[0]
    if n:
        print(f"{n} job(s) waiting, starting a worker")
        drain.spawn()
