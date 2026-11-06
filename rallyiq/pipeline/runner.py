"""Runs the stages of one job in order and records progress in the store."""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from ..contracts import STAGES, CourtCalibration, Job
from ..store import Store, now
from .context import StageContext
from .registry import STAGE_PREFETCH, load_prefetch, load_stage

log = logging.getLogger("rallyiq.pipeline")

MANUAL_CALIBRATION = "calibration_manual"
MATCH_FORMAT = "match_format"  # MatchFormatSetting chosen in the UI; read by the players stage


def run_job(store: Store, job: Job, config: dict[str, Any] | None = None, worker_id: str | None = None) -> Job:
    """Run ``job`` to completion. With ``worker_id`` every progress write also renews the job lease
    (see ``Store.claim_next_job``) and is dropped if the lease was lost to another worker."""

    def save(j: Job) -> None:
        store.update_job(j, worker_id)

    try:
        video_path = store.video_path(job.video_id)  # downloads from object storage when needed
    except Exception as e:  # noqa: BLE001
        log.exception("could not fetch video %s", job.video_id)
        job.state, job.error, job.finished_at = "failed", f"could not fetch video: {e}", now()
        save(job)
        return job
    if video_path is None:
        job.state, job.error, job.finished_at = "failed", "video not found", now()
        save(job)
        return job

    ctx = StageContext(
        video_id=job.video_id, video_path=video_path, work_dir=store.work_dir(job.video_id),
        store=store, meta=store.get_video_meta(job.video_id), config=dict(config or {}),
    )
    job.state = "running"
    job.started_at = job.started_at or now()
    save(job)

    cancel = threading.Event()
    try:
        return _run_stages(store, job, ctx, cancel, save)
    finally:
        cancel.set()  # stop early-started work the job no longer needs
        for wait in ctx.prefetched.values():
            try:
                wait()
            except BaseException:  # noqa: BLE001 - its own stage reports real errors
                pass
        ctx.prefetched.clear()


def parallel_enabled(config: dict[str, Any]) -> bool:
    v = str(config.get("parallel_stages", os.environ.get("RALLYIQ_PARALLEL_STAGES", "auto"))).lower()
    return v not in ("off", "0", "false", "no")


def _start_prefetch(store: Store, job: Job, ctx: StageContext, trigger: str, cancel: threading.Event,
                    save: Callable[[Job], None]) -> None:
    """Start the stages that may overlap ``trigger`` (see ``registry.STAGE_PREFETCH``)."""
    if not parallel_enabled(ctx.config):
        return
    for target, (start_with, _entry) in STAGE_PREFETCH.items():
        run = next((r for r in job.stages if r.name == target), None)
        if start_with != trigger or run is None or run.state != "pending":
            continue

        def report(p: float, _run=run) -> None:  # main thread persists it with its own updates
            _run.progress = max(0.0, min(1.0, float(p)))

        try:
            wait = load_prefetch(target)(ctx, report, cancel)
        except Exception:  # noqa: BLE001 - the stage itself runs (and reports) normally later
            log.warning("could not start %s early; it will run after %s", target, trigger, exc_info=True)
            continue
        if wait is not None:
            ctx.prefetched[target] = wait
            run.state, run.started_at, run.progress, run.error = "running", now(), 0.0, None
            save(job)
            log.info("%s started alongside %s", target, trigger)


def _run_stages(store: Store, job: Job, ctx: StageContext, cancel: threading.Event,
                save: Callable[[Job], None]) -> Job:
    for run in job.stages:
        if run.state == "skipped":
            continue
        name = run.name
        if name in ctx.prefetched and run.state == "running":
            run.error = None  # already started alongside an earlier stage; keep its start time
        else:
            run.state, run.started_at, run.progress, run.error = "running", now(), 0.0, None
        save(job)
        _start_prefetch(store, job, ctx, name, cancel, save)

        last = [0.0]

        def report(p: float, _run=run) -> None:
            _run.progress = max(0.0, min(1.0, float(p)))
            if time.monotonic() - last[0] > 0.5:  # throttle DB writes
                last[0] = time.monotonic()
                save(job)

        ctx.report = report
        try:
            manual = store.load_artifact(job.video_id, MANUAL_CALIBRATION, CourtCalibration)
            if name == "calibration" and manual is not None:
                result = manual
            else:
                result = load_stage(name)(ctx)
            store.save_artifact(job.video_id, name, result)
            run.state, run.progress = "succeeded", 1.0
        except Exception as e:  # noqa: BLE001 - a stage failure fails the job, not the worker
            log.exception("stage %s failed for %s", name, job.video_id)
            run.state, run.error = "failed", f"{type(e).__name__}: {e}"
            job.state, job.error = "failed", f"stage {name} failed: {e}"
            job.finished_at = run.finished_at = now()
            for later in job.stages[STAGES.index(name) + 1:]:
                if later.state in ("pending", "running"):
                    later.state = "skipped"
            save(job)
            log.debug(traceback.format_exc())
            return job
        run.finished_at = now()
        save(job)

    job.state, job.finished_at = "succeeded", now()
    save(job)
    return job


def stages_from(first: str) -> list[str]:
    """``first`` and every stage after it."""
    return STAGES[STAGES.index(first):]
