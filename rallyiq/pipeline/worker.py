"""Job queue worker. Jobs live in SQLite (state=queued); one worker runs them one at a time.

Run standalone with ``python -m rallyiq.pipeline worker``, or let the API start one
in-process (default; set ``RALLYIQ_WORKER=external`` to disable).
"""

from __future__ import annotations

import logging
import threading

from ..store import Store
from .runner import run_job

log = logging.getLogger("rallyiq.worker")


class Worker:
    def __init__(self, store: Store, poll_s: float = 0.5, config: dict | None = None):
        self.store = store
        self.poll_s = poll_s
        self.config = config or {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> bool:
        job = self.store.claim_next_job()
        if job is None:
            return False
        log.info("running %s for %s", job.job_id, job.video_id)
        run_job(self.store, job, self.config)
        return True

    def notify(self) -> None:
        self._wake.set()

    def loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.run_once():
                    continue
            except Exception:  # noqa: BLE001
                log.exception("worker iteration failed")
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def start(self) -> None:
        self.store.fail_orphaned_jobs()
        self._thread = threading.Thread(target=self.loop, name="rallyiq-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
