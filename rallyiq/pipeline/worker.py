"""Job queue worker.

Jobs live in the database (state=queued). Any number of workers may run against the same
database: each claims one job at a time under a lease that it renews with a heartbeat. If a
worker dies (crash, preempted spot GPU, scale-down), another process re-queues its job once
the lease expires (``recover_stale_jobs``); a job is retried at most ``max_attempts`` times.

Run standalone with ``python -m rallyiq.pipeline worker`` (online: on a GPU machine), or let
the API start one in-process (default locally; set ``RALLYIQ_WORKER=external`` to disable).

``--idle-exit SECONDS`` makes the worker exit after that long with an empty queue, so an
on-demand GPU container is only billed while there is work (see docs/deploy.md).
"""

from __future__ import annotations

import logging
import threading
import time

from ..store import Store, default_worker_id
from .runner import run_job

log = logging.getLogger("rallyiq.worker")


class Worker:
    def __init__(self, store: Store, poll_s: float = 0.5, config: dict | None = None, *,
                 worker_id: str | None = None, lease_s: float = 300.0, max_attempts: int = 2,
                 idle_exit_s: float | None = None, single_process: bool = True):
        self.store = store
        self.poll_s = poll_s
        self.config = config or {}
        self.worker_id = worker_id or default_worker_id()
        self.lease_s = lease_s
        self.max_attempts = max_attempts
        self.idle_exit_s = idle_exit_s
        # single_process: this is the only worker on this database (local / in-process mode),
        # so anything 'running' at startup was orphaned by a crash and is failed right away.
        self.single_process = single_process
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_recover = 0.0
        self.jobs_done = 0

    def _heartbeat_loop(self, job_id: str, done: threading.Event) -> None:
        interval = max(1.0, self.lease_s / 5)
        while not done.wait(interval):
            try:
                if not self.store.heartbeat(job_id, self.worker_id):
                    log.warning("lost lease on %s", job_id)
                    return
            except Exception:  # noqa: BLE001
                log.exception("heartbeat failed for %s", job_id)

    def recover(self) -> None:
        if time.monotonic() - self._last_recover < min(60.0, self.lease_s / 2):
            return
        self._last_recover = time.monotonic()
        res = self.store.recover_stale_jobs(self.lease_s, self.max_attempts)
        if any(res.values()):
            log.warning("recovered stale jobs: %s", res)

    def run_once(self) -> bool:
        self.recover()
        job = self.store.claim_next_job(self.worker_id)
        if job is None:
            return False
        log.info("running %s for %s (worker %s)", job.job_id, job.video_id, self.worker_id)
        done = threading.Event()
        hb = threading.Thread(target=self._heartbeat_loop, args=(job.job_id, done), daemon=True)
        hb.start()
        try:
            run_job(self.store, job, self.config, worker_id=self.worker_id)
        finally:
            done.set()
            hb.join(1)
        self.jobs_done += 1
        return True

    def notify(self) -> None:
        self._wake.set()

    def loop(self) -> None:
        idle_since = time.monotonic()
        while not self._stop.is_set():
            try:
                if self.run_once():
                    idle_since = time.monotonic()
                    continue
            except Exception:  # noqa: BLE001
                log.exception("worker iteration failed")
            if self.idle_exit_s is not None and time.monotonic() - idle_since >= self.idle_exit_s:
                log.info("queue empty for %.0fs, exiting after %d job(s)", self.idle_exit_s, self.jobs_done)
                return
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def start(self) -> None:
        if self.single_process:
            self.store.fail_orphaned_jobs()
        self._thread = threading.Thread(target=self.loop, name="rallyiq-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
