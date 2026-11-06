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
)
from ..contracts.geometry import COURT_CORNERS_M, homography_from_corners
from ..ingest import quality_for
from ..review import collect as collect_hard_examples
from ..review import summarize as summarize_hard_examples
from ..pipeline.clips import ClipError, cut_clip
from ..pipeline.runner import MANUAL_CALIBRATION, MATCH_FORMAT, stages_from
from ..pipeline.worker import Worker
from ..report import CoachUnavailable, compute_match_stats, generate_coach_report, list_providers, render_markdown
from ..report.coach import default_provider_name
from ..store import Store
from ..strokes.classify import merge_strokes
from .corrections import CorrectionError, apply_corrections, validate

COACH_ARTIFACT = "coach"
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


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

    @app.get("/api/videos/{video_id}/clip")
    def video_clip(video_id: str, rally_id: str | None = None, start_ms: int | None = Query(None, ge=0),
                   end_ms: int | None = Query(None, ge=0), pad_ms: int = Query(1000, ge=0, le=10_000)):
        """MP4 clip of one rally (``rally_id``, from the corrected events) or of ``start_ms``..``end_ms``,
        padded by ``pad_ms`` on both sides. Downloads as an attachment."""
        video_or_404(video_id)
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
            ev = model_events(video_id)
            if ev is None:
                raise HTTPException(404, "events not available yet")
            return Response(ev.model_dump_json(), media_type="application/json")
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
            other_courts=[c for c in req.other_courts if len(c) == 4],
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

    # ---------------------------------------------------------- match format

    @app.get("/api/videos/{video_id}/match_format", response_model=MatchFormatSetting)
    def get_match_format(video_id: str) -> MatchFormatSetting:
        video_or_404(video_id)
        return store.load_artifact(video_id, MATCH_FORMAT, MatchFormatSetting) or MatchFormatSetting()

    @app.put("/api/videos/{video_id}/match_format", response_model=Job)
    def set_match_format(video_id: str, req: MatchFormatSetting) -> Job:
        """Force singles / doubles (or back to auto) and re-run from the players stage."""
        video_or_404(video_id)
        store.save_artifact(video_id, MATCH_FORMAT, req)
        meta = store.get_video_meta(video_id)
        return enqueue(video_id, stages_from("players" if meta else "probe"))

    # --------------------------------------------------------------- heatmap

    @app.get("/api/videos/{video_id}/heatmap", response_model=HeatmapData)
    def heatmap(video_id: str, player_id: Literal["P1", "P2", "P3", "P4"] | None = None,
                team: Literal["A", "B"] | None = None, cell_m: float = Query(0.5, gt=0.05),
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
    def video_quality(video_id: str) -> VideoQuality:
        """Input assessment: camera view, frame rate, picture, visibility, and per-output reliability."""
        video_or_404(video_id)
        q = quality_for(store, video_id)
        if q is None:
            raise HTTPException(404, "video not probed yet")
        return q

    # --------------------------------------------------------- hard examples

    @app.get("/api/hard_examples", response_model=list[HardExample])
    def hard_examples(format: Literal["json", "jsonl"] = "json"):
        """Corrections marked as hard examples across all videos, with model item and video conditions."""
        examples, _ = collect_hard_examples(store)
        if format == "jsonl":
            return PlainTextResponse("".join(e.model_dump_json() + "\n" for e in examples),
                                     media_type="application/x-ndjson",
                                     headers={"Content-Disposition": 'attachment; filename="hard_examples.jsonl"'})
        return examples

    @app.get("/api/hard_examples/summary", response_model=HardExampleSummary)
    def hard_example_summary() -> HardExampleSummary:
        """Model errors per video condition (camera view, frame rate, ...) and the weakest scenarios."""
        return summarize_hard_examples(*collect_hard_examples(store))

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
