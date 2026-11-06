"""FastAPI backend for the web UI.

    uvicorn rallyiq.api.app:app --reload --port 8000

Env: ``RALLYIQ_DATA`` (data dir), ``RALLYIQ_DEMO=1`` (stub stages emit synthetic data),
``RALLYIQ_WORKER=external`` (don't run the queue worker in-process),
``RALLYIQ_CORS_ORIGINS`` (comma separated, default http://localhost:3000,http://localhost:5173).

Online deployment (docs/deploy.md) adds: ``RALLYIQ_DATABASE_URL`` (PostgreSQL), ``RALLYIQ_STORAGE``
(S3 / R2 bucket), ``RALLYIQ_AUTH=on`` (accounts, see auth.py), ``RALLYIQ_WEB_DIR`` (serve the built
web UI from the same origin), ``RALLYIQ_MAX_UPLOAD_MB`` (default 4096), ``RALLYIQ_MAX_ACTIVE_JOBS``
(queued + running jobs per user, default 3), ``RALLYIQ_WORKER_WAKE_URL`` / ``RALLYIQ_WORKER_WAKE_TOKEN``
(POSTed when a job is queued, to start an on-demand GPU worker).
"""

from __future__ import annotations

import csv
import io
import json
import os
import logging
import shutil
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..contracts import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    MatchFormatSetting,
    team_of,
    STAGES,
    CalibrationSegment,
    Correction,
    CorrectionIn,
    CourtCalibration,
    CourtKeypoint,
    EventsResult,
    StrokesResult,
    HeatmapData,
    Job,
    OverlayData,
    PlayerTracks,
    Point2,
    CoachReport,
    MatchReport,
    MatchStats,
    Video,
    VideoQuality,
    HardExample,
    HardExampleSummary,
    AnalysisEvent,
    OpponentReport,
    PeopleAssignment,
    PlayerProfile,
)
from ..contracts.geometry import COURT_CORNERS_M, homography_from_corners
from ..ingest import quality_for
from ..memory import StoreMemoryLog, build_commit, build_opponent_report, build_profile, people as list_people
from ..review import collect as collect_hard_examples
from ..review import summarize as summarize_hard_examples
from ..review import video_conditions
from ..pipeline.clips import ClipError, cut_clip
from ..pipeline.runner import MANUAL_CALIBRATION, MATCH_FORMAT, stages_from
from ..pipeline.worker import Worker
from ..report import CoachUnavailable, compute_match_stats, generate_coach_report, list_providers, render_markdown
from ..report.coach import default_provider_name
from ..store import CONTENT_TYPES, Store, now
from ..strokes.classify import merge_strokes
from .auth import AuthConfig, User, build_auth
from .corrections import CorrectionError, apply_corrections, validate

COACH_ARTIFACT = "coach"
PEOPLE_ARTIFACT = "people"
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
log = logging.getLogger("rallyiq.api")


class UploadRequest(BaseModel):
    filename: str = Field(max_length=255)
    size_bytes: int = Field(ge=1)
    content_type: str | None = None


class UploadTicket(BaseModel):
    video_id: str
    method: Literal["PUT"] = "PUT"
    url: str
    headers: dict[str, str]
    expires_s: int


class PublicConfig(BaseModel):
    auth: bool
    signup: str
    direct_upload: bool
    max_upload_mb: int


class WorkerWaker:
    """Pings an on-demand GPU worker endpoint when a job is queued (at most every ``min_interval_s``)."""

    def __init__(self, url: str | None, token: str | None, min_interval_s: float = 20.0):
        self.url, self.token, self.min_interval_s = url, token, min_interval_s
        self._last = -1e9
        self._lock = threading.Lock()

    def __call__(self) -> None:
        if not self.url:
            return
        with self._lock:
            if time.monotonic() - self._last < self.min_interval_s:
                return
            self._last = time.monotonic()
        threading.Thread(target=self._post, daemon=True).start()

    def _post(self) -> None:
        req = urllib.request.Request(self.url, data=b"{}", method="POST",  # noqa: S310 - operator-set URL
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
                log.info("worker wake -> %s", r.status)
        except Exception as e:  # noqa: BLE001
            log.warning("worker wake failed: %s", e)


class PeopleRequest(BaseModel):
    people: dict[Literal["P1", "P2", "P3", "P4"], str] = Field(default_factory=dict)
    self_player: Literal["P1", "P2", "P3", "P4"] | None = None


class JobRequest(BaseModel):
    from_stage: Literal["probe", "calibration", "players", "shuttle", "events", "strokes", "overlay"] = "probe"


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
    other_courts: list[list[Point2]] = Field(
        default_factory=list,
        description="other courts in view, 4 corners each (e.g. the court analysed before switching), "
        "so their players and shuttles keep being rejected")


def create_app(store: Store | None = None, start_worker: bool | None = None,
               auth: AuthConfig | None = None) -> FastAPI:
    store = store or Store()
    if start_worker is None:
        start_worker = os.environ.get("RALLYIQ_WORKER", "inprocess") != "external"
    worker = Worker(store) if start_worker else None
    auth = auth or AuthConfig.from_env()
    auth_router, current_user = build_auth(store, auth)
    max_upload = int(os.environ.get("RALLYIQ_MAX_UPLOAD_MB", "4096")) << 20
    max_active = int(os.environ.get("RALLYIQ_MAX_ACTIVE_JOBS", "3"))
    wake = WorkerWaker(os.environ.get("RALLYIQ_WORKER_WAKE_URL"), os.environ.get("RALLYIQ_WORKER_WAKE_TOKEN"))
    direct_upload = store.blobs.presign_put("probe", "video/mp4", 60) is not None

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
    origins = [o.strip() for o in origins if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials="*" not in origins,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(auth_router)

    def owner(user: User | None) -> str | None:
        return user.user_id if user else None

    def check_owner(video_id: str, user: User | None) -> None:
        if user is not None and store.video_owner(video_id) != user.user_id:
            raise HTTPException(404, "video not found")

    def video_or_404(video_id: str, user: User | None) -> Video:
        if store.video_status(video_id) != "ready":
            raise HTTPException(404, "video not found")
        check_owner(video_id, user)
        v = store.get_video(video_id)
        if v is None:
            raise HTTPException(404, "video not found")
        return v

    def enqueue(video_id: str, stages: list[str] | None = None, user: User | None = None) -> Job:
        if user is not None and store.count_active_jobs(user.user_id) >= max_active:
            raise HTTPException(429, f"同时排队或分析中的任务最多 {max_active} 个，请等前面的完成")
        job = store.create_job(video_id, stages)
        if worker:
            worker.notify()
        wake()
        return job

    def model_events(video_id: str) -> EventsResult | None:
        """Raw model prediction: the ``events`` artifact with the ``strokes`` stage merged in."""
        raw = store.load_artifact(video_id, "events", EventsResult)
        if raw is None:
            return None
        return merge_strokes(raw, store.load_artifact(video_id, "strokes", StrokesResult))

    def effective_events(video_id: str) -> tuple[EventsResult, list[str]]:
        raw = model_events(video_id)
        if raw is None:
            raise HTTPException(404, "events not available yet")
        return apply_corrections(raw, store.list_corrections(video_id), store.get_video_meta(video_id))

    # ---------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "stages": STAGES, "worker": bool(worker), "queue": store.queue_depth(),
                "db": store.db.dialect, "storage": store.blobs.kind}

    @app.get("/api/config", response_model=PublicConfig)
    def public_config() -> PublicConfig:
        return PublicConfig(auth=auth.enabled, signup=auth.signup, direct_upload=direct_upload,
                            max_upload_mb=max_upload >> 20)

    # ---------------------------------------------------------------- videos

    @app.post("/api/videos", response_model=Video, status_code=201)
    def upload_video(request: Request, file: UploadFile = File(...), analyze: bool = Query(True),
                     user: User | None = Depends(current_user)) -> Video:
        """Upload through the API (local mode). Online, prefer ``POST /api/uploads`` (direct to bucket)."""
        name = Path(file.filename or "video.mp4").name
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"unsupported file type {ext!r}")
        if int(request.headers.get("content-length") or 0) > max_upload + (1 << 20):
            raise HTTPException(413, f"视频超过 {max_upload >> 20} MB 上限")
        video_id, dest = store.create_video(name, ext)
        written = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(8 << 20):
                written += len(chunk)
                if written > max_upload:
                    out.close()
                    shutil.rmtree(dest.parent, ignore_errors=True)
                    raise HTTPException(413, f"视频超过 {max_upload >> 20} MB 上限")
                out.write(chunk)
        store.register_video(video_id, name, dest, owner_id=owner(user))
        if store.blobs.local_path(f"videos/{video_id}/source{ext}") is None:
            shutil.rmtree(store.video_dir(video_id), ignore_errors=True)  # bytes now live in the bucket
        if analyze:
            enqueue(video_id, user=user)
        return store.get_video(video_id)

    @app.post("/api/uploads", response_model=UploadTicket, status_code=201)
    def start_upload(req: UploadRequest, user: User | None = Depends(current_user)) -> UploadTicket:
        """Direct upload, step 1: get a presigned URL, then ``PUT`` the file to it with ``headers``."""
        if not direct_upload:
            raise HTTPException(409, "direct upload needs object storage; use POST /api/videos")
        name = Path(req.filename).name or "video.mp4"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"unsupported file type {ext!r}")
        if req.size_bytes > max_upload:
            raise HTTPException(413, f"视频超过 {max_upload >> 20} MB 上限")
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        video_id, key = store.reserve_upload(name, ext, owner_id=owner(user))
        expires = 6 * 3600
        url = store.blobs.presign_put(key, ctype, expires)
        assert url is not None
        return UploadTicket(video_id=video_id, url=url, headers={"Content-Type": ctype}, expires_s=expires)

    @app.post("/api/uploads/{video_id}/complete", response_model=Video)
    def complete_upload(video_id: str, analyze: bool = Query(True),
                        user: User | None = Depends(current_user)) -> Video:
        """Direct upload, step 2: the PUT finished; register the video and queue analysis."""
        status = store.video_status(video_id)
        if status is None:
            raise HTTPException(404, "upload not found")
        check_owner(video_id, user)
        if status == "ready":
            return video_or_404(video_id, user)
        key = store.source_key(video_id)
        size = store.blobs.size(key) if key else None
        if size is None:
            raise HTTPException(409, "file not uploaded yet")
        if size > max_upload:
            store.delete_video(video_id)
            raise HTTPException(413, f"视频超过 {max_upload >> 20} MB 上限")
        v = store.complete_upload(video_id)
        if analyze:
            enqueue(video_id, user=user)
        return store.get_video(video_id) or v

    @app.get("/api/videos", response_model=list[Video])
    def list_videos(user: User | None = Depends(current_user)) -> list[Video]:
        return store.list_videos(owner_id=owner(user))

    @app.get("/api/videos/{video_id}", response_model=Video)
    def get_video(video_id: str, user: User | None = Depends(current_user)) -> Video:
        return video_or_404(video_id, user)

    @app.delete("/api/videos/{video_id}", status_code=204)
    def delete_video(video_id: str, user: User | None = Depends(current_user)) -> Response:
        """Delete the video, its analysis, corrections and stored files."""
        video_or_404(video_id, user)
        store.delete_video(video_id)
        return Response(status_code=204)

    @app.get("/api/videos/{video_id}/file")
    def video_file(video_id: str, user: User | None = Depends(current_user)):
        v = video_or_404(video_id, user)
        key = store.source_key(video_id)
        local = store.blobs.local_path(key) if key else store.video_path(video_id)
        if local is None:  # object storage: let the browser stream (with Range) straight from the bucket
            url = store.video_download_url(video_id, v.filename)
            if url is None:
                raise HTTPException(404, "video file missing")
            return RedirectResponse(url, status_code=307)
        if not local.exists():
            raise HTTPException(404, "video file missing")
        return FileResponse(local, media_type=CONTENT_TYPES.get(local.suffix.lower()))

    @app.get("/api/videos/{video_id}/clip")
    def video_clip(video_id: str, rally_id: str | None = None, start_ms: int | None = Query(None, ge=0),
                   end_ms: int | None = Query(None, ge=0), pad_ms: int = Query(1000, ge=0, le=10_000),
                   user: User | None = Depends(current_user)):
        """MP4 clip of one rally (``rally_id``, from the corrected events) or of ``start_ms``..``end_ms``,
        padded by ``pad_ms`` on both sides. Downloads as an attachment."""
        video_or_404(video_id, user)
        key = store.source_key(video_id)
        if key is not None and store.blobs.local_path(key) is None:
            # Object storage: ffmpeg seeks inside the presigned URL (HTTP Range), no full download.
            path = store.blobs.presign_get(key, expires_s=900)
        else:
            path = store.video_path(video_id)
            if path is None or not path.exists():
                raise HTTPException(404, "video file missing")
        name = video_id
        if rally_id is not None:
            ev, _ = effective_events(video_id)
            r = next((x for x in ev.rallies if x.rally_id == rally_id), None)
            if r is None:
                raise HTTPException(404, f"rally {rally_id} not found")
            start_ms, end_ms, name = r.start_ms, r.end_ms, f"{video_id}_{rally_id}"
        elif start_ms is None or end_ms is None:
            raise HTTPException(422, "give rally_id, or start_ms and end_ms")
        if end_ms <= start_ms:
            raise HTTPException(422, "end_ms must be after start_ms")
        meta = store.get_video_meta(video_id)
        lo = max(0, start_ms - pad_ms)
        hi = end_ms + pad_ms if meta is None else min(meta.duration_ms, end_ms + pad_ms)
        try:
            clip = cut_clip(path, store.work_dir(video_id) / "clips", lo, hi)
        except ClipError as e:
            raise HTTPException(422, str(e)) from e
        return FileResponse(clip, media_type="video/mp4", filename=f"{name}_{lo}-{hi}ms.mp4")

    # ------------------------------------------------------------------ jobs

    @app.post("/api/videos/{video_id}/jobs", response_model=Job, status_code=201)
    def create_job(video_id: str, req: JobRequest | None = None, user: User | None = Depends(current_user)) -> Job:
        video_or_404(video_id, user)
        first = req.from_stage if req else "probe"
        if first != "probe" and store.get_video_meta(video_id) is None:
            first = "probe"
        return enqueue(video_id, stages_from(first), user)

    @app.get("/api/videos/{video_id}/jobs", response_model=list[Job])
    def list_jobs(video_id: str, user: User | None = Depends(current_user)) -> list[Job]:
        video_or_404(video_id, user)
        return store.list_jobs(video_id)

    @app.get("/api/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str, user: User | None = Depends(current_user)) -> Job:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if user is not None and store.video_owner(job.video_id) != user.user_id:
            raise HTTPException(404, "job not found")
        return job

    # ------------------------------------------------------------- artifacts

    def artifact_response(video_id: str, name: str, user: User | None) -> Response:
        video_or_404(video_id, user)
        local = store.blobs.local_path(store.artifact_key(video_id, name))
        if local is not None:
            if not local.exists():
                raise HTTPException(404, f"{name} not available yet")
            return FileResponse(local, media_type="application/json")
        data = store.artifact_bytes(video_id, name)
        if data is None:
            raise HTTPException(404, f"{name} not available yet")
        return Response(content=data, media_type="application/json")

    @app.get("/api/videos/{video_id}/artifacts/{stage}")
    def get_artifact(video_id: str, stage: str, user: User | None = Depends(current_user)):
        if stage not in STAGES or stage == "probe":
            raise HTTPException(404, "unknown artifact")
        return artifact_response(video_id, stage, user)

    @app.get("/api/videos/{video_id}/overlay", response_model=OverlayData)
    def get_overlay(video_id: str, start_ms: int | None = None, end_ms: int | None = None,
                    user: User | None = Depends(current_user)):
        if start_ms is None and end_ms is None:
            return artifact_response(video_id, "overlay", user)
        video_or_404(video_id, user)
        ov = store.load_artifact(video_id, "overlay", OverlayData)
        if ov is None:
            raise HTTPException(404, "overlay not available yet")
        lo, hi = start_ms or 0, end_ms if end_ms is not None else 1 << 62
        ov.frames = [f for f in ov.frames if lo <= f.t_ms <= hi]
        return ov

    # ---------------------------------------------------------------- events

    @app.get("/api/videos/{video_id}/events", response_model=EffectiveEvents)
    def get_events(video_id: str, raw: bool = False, user: User | None = Depends(current_user)):
        video_or_404(video_id, user)
        if raw:
            ev = model_events(video_id)
            if ev is None:
                raise HTTPException(404, "events not available yet")
            return Response(ev.model_dump_json(), media_type="application/json")
        ev, skipped = effective_events(video_id)
        return EffectiveEvents(**ev.model_dump(), corrections_applied=len(store.list_corrections(video_id)),
                               corrections_skipped=skipped)

    @app.get("/api/videos/{video_id}/corrections", response_model=list[Correction])
    def list_corrections(video_id: str, user: User | None = Depends(current_user)) -> list[Correction]:
        video_or_404(video_id, user)
        return store.list_corrections(video_id)

    @app.post("/api/videos/{video_id}/corrections", response_model=Correction, status_code=201)
    def add_correction(video_id: str, c: CorrectionIn, user: User | None = Depends(current_user)) -> Correction:
        video_or_404(video_id, user)
        try:
            validate(c.op, c.payload)
        except CorrectionError as e:
            raise HTTPException(422, str(e)) from e
        return store.add_correction(video_id, c)

    # ----------------------------------------------------------- calibration

    @app.put("/api/videos/{video_id}/calibration/manual", response_model=Job)
    def manual_calibration(video_id: str, req: ManualCalibration, user: User | None = Depends(current_user)) -> Job:
        video_or_404(video_id, user)
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
            other_courts=[c for c in req.other_courts if len(c) == 4],
            notes=["4 corners picked by user"],
        )
        cal = CourtCalibration(video_id=video_id, segments=[seg])
        store.save_artifact(video_id, MANUAL_CALIBRATION, cal)
        return enqueue(video_id, stages_from("calibration" if meta else "probe"), user)

    @app.delete("/api/videos/{video_id}/calibration/manual", response_model=Job)
    def clear_manual_calibration(video_id: str, user: User | None = Depends(current_user)) -> Job:
        video_or_404(video_id, user)
        store.delete_artifact(video_id, MANUAL_CALIBRATION)
        return enqueue(video_id, stages_from("calibration"), user)

    # ---------------------------------------------------------- match format

    @app.get("/api/videos/{video_id}/match_format", response_model=MatchFormatSetting)
    def get_match_format(video_id: str, user: User | None = Depends(current_user)) -> MatchFormatSetting:
        video_or_404(video_id, user)
        return store.load_artifact(video_id, MATCH_FORMAT, MatchFormatSetting) or MatchFormatSetting()

    @app.put("/api/videos/{video_id}/match_format", response_model=Job)
    def set_match_format(video_id: str, req: MatchFormatSetting, user: User | None = Depends(current_user)) -> Job:
        """Force singles / doubles (or back to auto) and re-run from the players stage."""
        video_or_404(video_id, user)
        store.save_artifact(video_id, MATCH_FORMAT, req)
        meta = store.get_video_meta(video_id)
        return enqueue(video_id, stages_from("players" if meta else "probe"), user)

    # --------------------------------------------------------------- heatmap

    @app.get("/api/videos/{video_id}/heatmap", response_model=HeatmapData)
    def heatmap(video_id: str, player_id: Literal["P1", "P2", "P3", "P4"] | None = None,
                team: Literal["A", "B"] | None = None, cell_m: float = Query(0.5, gt=0.05),
                start_ms: int | None = None, end_ms: int | None = None,
                user: User | None = Depends(current_user)) -> HeatmapData:
        video_or_404(video_id, user)
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
                if team and (p.team or team_of(p.player_id)) != team:
                    continue
                ix, iy = int(p.foot_court[0] // cell_m), int(p.foot_court[1] // cell_m)
                if 0 <= ix < nx and 0 <= iy < ny:
                    grid[iy, ix] += 1
                    n += 1
        return HeatmapData(video_id=video_id, player_id=player_id, team=team, cell_m=cell_m, nx=nx, ny=ny,
                           counts=grid.tolist(), samples=n)

    # --------------------------------------------------------------- quality

    @app.get("/api/videos/{video_id}/quality", response_model=VideoQuality)
    def video_quality(video_id: str, user: User | None = Depends(current_user)) -> VideoQuality:
        """Input assessment: camera view, frame rate, picture, visibility, and per-output reliability."""
        video_or_404(video_id, user)
        q = quality_for(store, video_id)
        if q is None:
            raise HTTPException(404, "video not probed yet")
        return q

    # --------------------------------------------------------- hard examples

    def hard_example_owner(user: User | None) -> str | None:
        return None if user is None or user.is_admin else user.user_id  # admins see every account's

    @app.get("/api/hard_examples", response_model=list[HardExample])
    def hard_examples(format: Literal["json", "jsonl"] = "json", user: User | None = Depends(current_user)):
        """Corrections marked as hard examples across all videos, with model item and video conditions."""
        examples, _ = collect_hard_examples(store, hard_example_owner(user))
        if format == "jsonl":
            return PlainTextResponse("".join(e.model_dump_json() + "\n" for e in examples),
                                     media_type="application/x-ndjson",
                                     headers={"Content-Disposition": 'attachment; filename="hard_examples.jsonl"'})
        return examples

    @app.get("/api/hard_examples/summary", response_model=HardExampleSummary)
    def hard_example_summary(user: User | None = Depends(current_user)) -> HardExampleSummary:
        """Model errors per video condition (camera view, frame rate, ...) and the weakest scenarios."""
        return summarize_hard_examples(*collect_hard_examples(store, hard_example_owner(user)))

    # ---------------------------------------------------------------- memory

    def memory(user: User | None) -> StoreMemoryLog:
        return StoreMemoryLog(store, owner(user))  # one log per account (the whole store when accounts are off)

    @app.get("/api/videos/{video_id}/people", response_model=PeopleAssignment)
    def get_people(video_id: str, user: User | None = Depends(current_user)) -> PeopleAssignment:
        video_or_404(video_id, user)
        return store.load_artifact(video_id, PEOPLE_ARTIFACT, PeopleAssignment) or PeopleAssignment(video_id=video_id)

    @app.put("/api/videos/{video_id}/people", response_model=PeopleAssignment)
    def set_people(video_id: str, req: PeopleRequest, user: User | None = Depends(current_user)) -> PeopleAssignment:
        """Name the players of this video (e.g. P1 = 我, P2 = 对手名字) and say which one is the user."""
        video_or_404(video_id, user)
        a = PeopleAssignment(video_id=video_id, updated_at=now(), self_player=req.self_player,
                             people={k: v.strip() for k, v in req.people.items() if v and v.strip()})
        if a.self_player and a.self_player not in a.people:
            raise HTTPException(422, "self_player must be one of the named players")
        store.save_artifact(video_id, PEOPLE_ARTIFACT, a)
        return a

    @app.post("/api/videos/{video_id}/memory/commit", response_model=list[AnalysisEvent], status_code=201)
    def commit_to_memory(video_id: str, user: User | None = Depends(current_user)) -> list[AnalysisEvent]:
        """Write this match (after review) into the long-term log: one event per named player.
        Re-committing appends a newer version; the old one stays in the log."""
        v = video_or_404(video_id, user)
        a = store.load_artifact(video_id, PEOPLE_ARTIFACT, PeopleAssignment)
        if a is None or not a.people:
            raise HTTPException(422, "name the players first (PUT /people)")
        ev, _ = effective_events(video_id)
        stats = match_stats(video_id)
        coach = store.load_artifact(video_id, COACH_ARTIFACT, CoachReport)
        if coach is not None and coach.stats_fingerprint != stats.fingerprint:
            coach = None  # stale: its training items refer to other numbers
        quality = quality_for(store, video_id, sample_pictures=False)
        conditions = video_conditions(store.get_video_meta(video_id), quality, ev.match_format)
        try:
            rows = build_commit(v, a, stats, ev, coach, conditions, len(store.list_corrections(video_id)))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return memory(user).append(rows)

    @app.post("/api/videos/{video_id}/memory/withdraw", response_model=list[AnalysisEvent], status_code=201)
    def withdraw_from_memory(video_id: str, user: User | None = Depends(current_user)) -> list[AnalysisEvent]:
        """Exclude this video from profiles from now on (appends a withdrawal; nothing is deleted)."""
        v = video_or_404(video_id, user)
        return memory(user).append([dict(kind="match_withdrawn", video_id=video_id, filename=v.filename)])

    @app.get("/api/memory/log", response_model=list[AnalysisEvent])
    def memory_log(user: User | None = Depends(current_user)) -> list[AnalysisEvent]:
        return memory(user).read()

    @app.get("/api/people")
    def people_list(user: User | None = Depends(current_user)) -> list[dict]:
        """People with committed matches (the user first)."""
        return list_people(memory(user).read())

    @app.get("/api/profile", response_model=PlayerProfile)
    def profile(person: str | None = None, user: User | None = Depends(current_user)) -> PlayerProfile:
        """Profile of the user (default) or of ``person``: per-metric series, trends, changes, retests."""
        return build_profile(memory(user).read(), person)

    @app.get("/api/opponents/{person}", response_model=OpponentReport)
    def opponent_report(person: str, user: User | None = Depends(current_user)) -> OpponentReport:
        return build_opponent_report(memory(user).read(), person)

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
    def coach_providers(_user: User | None = Depends(current_user)) -> dict:
        return {"default": default_provider_name(), "providers": list_providers()}

    @app.get("/api/videos/{video_id}/report", response_model=MatchReport)
    def get_report(video_id: str, user: User | None = Depends(current_user)) -> MatchReport:
        video_or_404(video_id, user)
        return match_report(video_id)

    @app.post("/api/videos/{video_id}/report/coach", response_model=CoachReport)
    def create_coach_report(video_id: str, req: CoachRequest | None = None,
                            user: User | None = Depends(current_user)) -> CoachReport:
        """Generate (or regenerate) the AI coach comments for the current stats. Synchronous."""
        video_or_404(video_id, user)
        stats = match_stats(video_id)
        try:
            coach = generate_coach_report(stats, req.provider if req else None)
        except CoachUnavailable as e:
            raise HTTPException(503, str(e)) from e
        store.save_artifact(video_id, COACH_ARTIFACT, coach)
        return coach

    @app.get("/api/videos/{video_id}/report/export")
    def export_report(video_id: str, format: Literal["md", "json"] = "md", user: User | None = Depends(current_user)):
        v = video_or_404(video_id, user)
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
    def export(video_id: str, format: Literal["json", "csv"] = "json", user: User | None = Depends(current_user)):
        v = video_or_404(video_id, user)
        ev, skipped = effective_events(video_id)
        if format == "json":
            raw = model_events(video_id)
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
        w.writerow(["shot_id", "rally_id", "frame", "t_ms", "time", "hitter", "team", "confidence", "status", "source",
                    "shuttle_x", "shuttle_y", "hitter_court_x", "hitter_court_y",
                    "shot_type", "shot_type_confidence", "shot_type_status", "shot_type_source"])
        for s in ev.shots:
            secs = s.t_ms / 1000
            st = s.stroke
            w.writerow([s.shot_id, s.rally_id or "", s.frame, s.t_ms, f"{int(secs // 60):02d}:{secs % 60:06.3f}",
                        s.hitter or "", team_of(s.hitter) or "", f"{s.confidence:.3f}", s.status, s.source,
                        *(s.shuttle_px or ["", ""]), *(s.hitter_court or ["", ""]),
                        s.shot_type or "", f"{st.confidence:.3f}" if st else "", st.status if st else "",
                        st.source if st else ""])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{video_id}_shots.csv"'})

    # ------------------------------------------------------------- web UI

    web_dir = os.environ.get("RALLYIQ_WEB_DIR")
    if web_dir and Path(web_dir, "index.html").exists():
        # Same-origin hosting of the built web UI (hash routing, so no SPA fallback needed).
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

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
