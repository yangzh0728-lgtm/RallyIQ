"""FastAPI backend for the web UI.

    uvicorn rallyiq.api.app:app --reload --port 8000

Env: ``RALLYIQ_DATA`` (data dir), ``RALLYIQ_DEMO=1`` (stub stages emit synthetic data),
``RALLYIQ_WORKER=external`` (don't run the queue worker in-process),
``RALLYIQ_CORS_ORIGINS`` (comma separated, default http://localhost:3000,http://localhost:5173).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from ..contracts import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    STAGES,
    CalibrationSegment,
    Correction,
    CorrectionIn,
    CourtCalibration,
    CourtKeypoint,
    EventsResult,
    HeatmapData,
    Job,
    OverlayData,
    PlayerTracks,
    Point2,
    CoachReport,
    MatchReport,
    MatchStats,
    Video,
)
from ..contracts.geometry import COURT_CORNERS_M, homography_from_corners
from ..pipeline.runner import MANUAL_CALIBRATION, stages_from
from ..pipeline.worker import Worker
from ..report import CoachUnavailable, compute_match_stats, generate_coach_report, list_providers, render_markdown
from ..report.coach import default_provider_name
from ..store import Store
from .corrections import CorrectionError, apply_corrections, validate

COACH_ARTIFACT = "coach"
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


class JobRequest(BaseModel):
    from_stage: Literal["probe", "calibration", "players", "shuttle", "events", "overlay"] = "probe"


class EffectiveEvents(EventsResult):
    corrections_applied: int = 0
    corrections_skipped: list[str] = Field(default_factory=list)


class CoachRequest(BaseModel):
    provider: str | None = Field(default=None, description="null = server default (Claude, offline fallback)")


class ManualCalibration(BaseModel):
    corners: list[Point2] = Field(min_length=4, max_length=4,
                                  description="near-left, near-right, far-right, far-left (px)")
    start_frame: int = 0
    end_frame: int | None = None


def create_app(store: Store | None = None, start_worker: bool | None = None) -> FastAPI:
    store = store or Store()
    if start_worker is None:
        start_worker = os.environ.get("RALLYIQ_WORKER", "inprocess") != "external"
    worker = Worker(store) if start_worker else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if worker:
            worker.start()
        yield
        if worker:
            worker.stop()

    app = FastAPI(title="RallyIQ API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.worker = worker
    origins = os.environ.get("RALLYIQ_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])

    def video_or_404(video_id: str) -> Video:
        v = store.get_video(video_id)
        if v is None:
            raise HTTPException(404, "video not found")
        return v

    def enqueue(video_id: str, stages: list[str] | None = None) -> Job:
        job = store.create_job(video_id, stages)
        if worker:
            worker.notify()
        return job

    def effective_events(video_id: str) -> tuple[EventsResult, list[str]]:
        raw = store.load_artifact(video_id, "events", EventsResult)
        if raw is None:
            raise HTTPException(404, "events not available yet")
        return apply_corrections(raw, store.list_corrections(video_id), store.get_video_meta(video_id))

    # ---------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "stages": STAGES, "worker": bool(worker)}

    # ---------------------------------------------------------------- videos

    @app.post("/api/videos", response_model=Video, status_code=201)
    def upload_video(file: UploadFile = File(...), analyze: bool = Query(True)) -> Video:
        name = Path(file.filename or "video.mp4").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"unsupported file type {ext!r}")
        video_id, dest = store.create_video(name, ext)
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=8 << 20)
        store.register_video(video_id, name, dest)
        if analyze:
            enqueue(video_id)
        return store.get_video(video_id)

    @app.get("/api/videos", response_model=list[Video])
    def list_videos() -> list[Video]:
        return store.list_videos()

    @app.get("/api/videos/{video_id}", response_model=Video)
    def get_video(video_id: str) -> Video:
        return video_or_404(video_id)

    @app.get("/api/videos/{video_id}/file")
    def video_file(video_id: str):
        video_or_404(video_id)
        path = store.video_path(video_id)
        if path is None or not path.exists():
            raise HTTPException(404, "video file missing")
        media = "video/mp4" if path.suffix in {".mp4", ".m4v"} else "video/quicktime" if path.suffix == ".mov" else None
        return FileResponse(path, media_type=media)

    # ------------------------------------------------------------------ jobs

    @app.post("/api/videos/{video_id}/jobs", response_model=Job, status_code=201)
    def create_job(video_id: str, req: JobRequest | None = None) -> Job:
        video_or_404(video_id)
        first = req.from_stage if req else "probe"
        if first != "probe" and store.get_video_meta(video_id) is None:
            first = "probe"
        return enqueue(video_id, stages_from(first))

    @app.get("/api/videos/{video_id}/jobs", response_model=list[Job])
    def list_jobs(video_id: str) -> list[Job]:
        video_or_404(video_id)
        return store.list_jobs(video_id)

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    # ------------------------------------------------------------- artifacts

    def artifact_response(video_id: str, name: str) -> Response:
        video_or_404(video_id)
        p = store.artifact_path(video_id, name)
        if not p.exists():
            raise HTTPException(404, f"{name} not available yet")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/videos/{video_id}/artifacts/{stage}")
    def get_artifact(video_id: str, stage: str):
        if stage not in STAGES or stage == "probe":
            raise HTTPException(404, "unknown artifact")
        return artifact_response(video_id, stage)

    @app.get("/api/videos/{video_id}/overlay", response_model=OverlayData)
    def get_overlay(video_id: str, start_ms: int | None = None, end_ms: int | None = None):
        if start_ms is None and end_ms is None:
            return artifact_response(video_id, "overlay")
        video_or_404(video_id)
        ov = store.load_artifact(video_id, "overlay", OverlayData)
        if ov is None:
            raise HTTPException(404, "overlay not available yet")
        lo, hi = start_ms or 0, end_ms if end_ms is not None else 1 << 62
        ov.frames = [f for f in ov.frames if lo <= f.t_ms <= hi]
        return ov

    # ---------------------------------------------------------------- events

    @app.get("/api/videos/{video_id}/events", response_model=EffectiveEvents)
    def get_events(video_id: str, raw: bool = False):
        video_or_404(video_id)
        if raw:
            return artifact_response(video_id, "events")
        ev, skipped = effective_events(video_id)
        return EffectiveEvents(**ev.model_dump(), corrections_applied=len(store.list_corrections(video_id)),
                               corrections_skipped=skipped)

    @app.get("/api/videos/{video_id}/corrections", response_model=list[Correction])
    def list_corrections(video_id: str) -> list[Correction]:
        video_or_404(video_id)
        return store.list_corrections(video_id)

    @app.post("/api/videos/{video_id}/corrections", response_model=Correction, status_code=201)
    def add_correction(video_id: str, c: CorrectionIn) -> Correction:
        video_or_404(video_id)
        try:
            validate(c.op, c.payload)
        except CorrectionError as e:
            raise HTTPException(422, str(e)) from e
        return store.add_correction(video_id, c)

    # ----------------------------------------------------------- calibration

    @app.put("/api/videos/{video_id}/calibration/manual", response_model=Job)
    def manual_calibration(video_id: str, req: ManualCalibration) -> Job:
        video_or_404(video_id)
        meta = store.get_video_meta(video_id)
        try:
            h = homography_from_corners(req.corners)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise HTTPException(422, f"bad corners: {e}") from e
        end = req.end_frame if req.end_frame is not None else max((meta.frame_count - 1) if meta else 0, 0)
        names = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"]
        seg = CalibrationSegment(
            start_frame=req.start_frame, end_frame=end, homography=h, image_corners=req.corners,
            keypoints=[CourtKeypoint(name=n, image=i, court=c) for n, i, c in zip(names, req.corners, COURT_CORNERS_M)],
            reprojection_error_px=0.0, method="manual", confidence=1.0, status="ok",
            notes=["4 corners picked by user"],
        )
        cal = CourtCalibration(video_id=video_id, segments=[seg])
        store.save_artifact(video_id, MANUAL_CALIBRATION, cal)
        return enqueue(video_id, stages_from("calibration" if meta else "probe"))

    @app.delete("/api/videos/{video_id}/calibration/manual", response_model=Job)
    def clear_manual_calibration(video_id: str) -> Job:
        video_or_404(video_id)
        store.delete_artifact(video_id, MANUAL_CALIBRATION)
        return enqueue(video_id, stages_from("calibration"))

    # --------------------------------------------------------------- heatmap

    @app.get("/api/videos/{video_id}/heatmap", response_model=HeatmapData)
    def heatmap(video_id: str, player_id: Literal["P1", "P2"] | None = None, cell_m: float = Query(0.5, gt=0.05),
                start_ms: int | None = None, end_ms: int | None = None) -> HeatmapData:
        video_or_404(video_id)
        players = store.load_artifact(video_id, "players", PlayerTracks)
        if players is None:
            raise HTTPException(404, "players not available yet")
        nx, ny = int(np.ceil(COURT_WIDTH_M / cell_m)), int(np.ceil(COURT_LENGTH_M / cell_m))
        grid = np.zeros((ny, nx), dtype=int)
        n = 0
        for pf in players.frames:
            if (start_ms is not None and pf.t_ms < start_ms) or (end_ms is not None and pf.t_ms > end_ms):
                continue
            for p in pf.players:
                if p.foot_court is None or (player_id and p.player_id != player_id):
                    continue
                ix, iy = int(p.foot_court[0] // cell_m), int(p.foot_court[1] // cell_m)
                if 0 <= ix < nx and 0 <= iy < ny:
                    grid[iy, ix] += 1
                    n += 1
        return HeatmapData(video_id=video_id, player_id=player_id, cell_m=cell_m, nx=nx, ny=ny,
                           counts=grid.tolist(), samples=n)

    # ---------------------------------------------------------------- report

    def match_stats(video_id: str) -> MatchStats:
        ev, _ = effective_events(video_id)
        return compute_match_stats(
            ev, store.load_artifact(video_id, "players", PlayerTracks),
            store.load_artifact(video_id, "calibration", CourtCalibration), store.get_video_meta(video_id),
        )

    def match_report(video_id: str) -> MatchReport:
        stats = match_stats(video_id)
        coach = store.load_artifact(video_id, COACH_ARTIFACT, CoachReport)
        return MatchReport(stats=stats, coach=coach,
                           coach_stale=coach is not None and coach.stats_fingerprint != stats.fingerprint)

    @app.get("/api/coach/providers")
    def coach_providers() -> dict:
        return {"default": default_provider_name(), "providers": list_providers()}

    @app.get("/api/videos/{video_id}/report", response_model=MatchReport)
    def get_report(video_id: str) -> MatchReport:
        video_or_404(video_id)
        return match_report(video_id)

    @app.post("/api/videos/{video_id}/report/coach", response_model=CoachReport)
    def create_coach_report(video_id: str, req: CoachRequest | None = None) -> CoachReport:
        """Generate (or regenerate) the AI coach comments for the current stats. Synchronous."""
        video_or_404(video_id)
        stats = match_stats(video_id)
        try:
            coach = generate_coach_report(stats, req.provider if req else None)
        except CoachUnavailable as e:
            raise HTTPException(503, str(e)) from e
        store.save_artifact(video_id, COACH_ARTIFACT, coach)
        return coach

    @app.get("/api/videos/{video_id}/report/export")
    def export_report(video_id: str, format: Literal["md", "json"] = "md"):
        v = video_or_404(video_id)
        rep = match_report(video_id)
        if format == "json":
            return Response(
                content=rep.model_dump_json(indent=1), media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{video_id}_report.json"'},
            )
        md = render_markdown(rep.stats, rep.coach, title=f"赛后报告 · {v.filename}", coach_stale=rep.coach_stale)
        return PlainTextResponse(md, media_type="text/markdown; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{video_id}_report.md"'})

    # ---------------------------------------------------------------- export

    @app.get("/api/videos/{video_id}/export")
    def export(video_id: str, format: Literal["json", "csv"] = "json"):
        v = video_or_404(video_id)
        ev, skipped = effective_events(video_id)
        if format == "json":
            raw = store.load_artifact(video_id, "events", EventsResult)
            body = {
                "video": v.model_dump(mode="json", exclude={"latest_job"}),
                "events": ev.model_dump(mode="json"),
                "raw_events": raw.model_dump(mode="json") if raw else None,
                "corrections": [c.model_dump(mode="json") for c in store.list_corrections(video_id)],
                "corrections_skipped": skipped,
            }
            return Response(
                content=json.dumps(body, ensure_ascii=False, indent=1), media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{video_id}_events.json"'},
            )
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["shot_id", "rally_id", "frame", "t_ms", "time", "hitter", "confidence", "status", "source",
                    "shuttle_x", "shuttle_y", "hitter_court_x", "hitter_court_y"])
        for s in ev.shots:
            secs = s.t_ms / 1000
            w.writerow([s.shot_id, s.rally_id or "", s.frame, s.t_ms, f"{int(secs // 60):02d}:{secs % 60:06.3f}",
                        s.hitter or "", f"{s.confidence:.3f}", s.status, s.source,
                        *(s.shuttle_px or ["", ""]), *(s.hitter_court or ["", ""])])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{video_id}_shots.csv"'})

    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    # ``uvicorn rallyiq.api.app:app`` builds the app lazily so importing this module
    # (e.g. in tests) does not create a data dir or start a worker.
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
