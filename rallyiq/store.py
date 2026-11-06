"""Local storage: SQLite for videos/jobs/corrections, files for videos and stage artifacts.

Layout under the data dir (``RALLYIQ_DATA``, default ``./data``)::

    rallyiq.db
    videos/<video_id>/source<.ext>
    videos/<video_id>/artifacts/<stage>.json
    videos/<video_id>/artifacts/calibration_manual.json   (human override, optional)
    videos/<video_id>/work/                                (scratch for stages)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .contracts import (
    STAGES,
    Correction,
    CorrectionIn,
    Job,
    StageRun,
    Video,
    VideoMeta,
)

T = TypeVar("T", bound=BaseModel)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    state TEXT NOT NULL,
    stages TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS jobs_video ON jobs(video_id, created_at);
CREATE TABLE IF NOT EXISTS corrections (
    correction_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    seq INTEGER NOT NULL,
    op TEXT NOT NULL,
    payload TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(video_id, seq)
);
"""


def now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_data_dir() -> Path:
    return Path(os.environ.get("RALLYIQ_DATA", "data")).resolve()


class Store:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).resolve() if root else default_data_dir()
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "rallyiq.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ paths

    def video_dir(self, video_id: str) -> Path:
        return self.root / "videos" / video_id

    def artifact_path(self, video_id: str, name: str) -> Path:
        return self.video_dir(video_id) / "artifacts" / f"{name}.json"

    def work_dir(self, video_id: str) -> Path:
        d = self.video_dir(video_id) / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----------------------------------------------------------------- videos

    def create_video(self, filename: str, ext: str = ".mp4") -> tuple[str, Path]:
        """Reserve a video id and return the path the upload must be written to."""
        video_id = new_id("vid")
        d = self.video_dir(video_id)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        return video_id, d / f"source{ext.lower() or '.mp4'}"

    def register_video(self, video_id: str, filename: str, path: Path) -> Video:
        self._q(
            "INSERT INTO videos(video_id, filename, path, size_bytes, created_at) VALUES (?,?,?,?,?)",
            (video_id, filename, str(path), path.stat().st_size, now().isoformat()),
        )
        return self.get_video(video_id)  # type: ignore[return-value]

    def video_path(self, video_id: str) -> Path | None:
        rows = self._q("SELECT path FROM videos WHERE video_id=?", (video_id,))
        return Path(rows[0]["path"]) if rows else None

    def set_video_meta(self, meta: VideoMeta) -> None:
        self._q("UPDATE videos SET meta=? WHERE video_id=?", (meta.model_dump_json(), meta.video_id))

    def get_video_meta(self, video_id: str) -> VideoMeta | None:
        rows = self._q("SELECT meta FROM videos WHERE video_id=?", (video_id,))
        return VideoMeta.model_validate_json(rows[0]["meta"]) if rows and rows[0]["meta"] else None

    def _video_from_row(self, r: sqlite3.Row) -> Video:
        vid = r["video_id"]
        return Video(
            video_id=vid,
            filename=r["filename"],
            size_bytes=r["size_bytes"],
            created_at=datetime.fromisoformat(r["created_at"]),
            meta=VideoMeta.model_validate_json(r["meta"]) if r["meta"] else None,
            latest_job=self.latest_job(vid),
            artifacts=[s for s in STAGES if self.artifact_path(vid, s).exists()],
        )

    def get_video(self, video_id: str) -> Video | None:
        rows = self._q("SELECT * FROM videos WHERE video_id=?", (video_id,))
        return self._video_from_row(rows[0]) if rows else None

    def list_videos(self) -> list[Video]:
        return [self._video_from_row(r) for r in self._q("SELECT * FROM videos ORDER BY created_at DESC")]

    # -------------------------------------------------------------- artifacts

    def save_artifact(self, video_id: str, name: str, obj: BaseModel) -> Path:
        p = self.artifact_path(video_id, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(obj.model_dump_json())
        tmp.replace(p)
        return p

    def has_artifact(self, video_id: str, name: str) -> bool:
        return self.artifact_path(video_id, name).exists()

    def load_artifact(self, video_id: str, name: str, model: type[T]) -> T | None:
        p = self.artifact_path(video_id, name)
        return model.model_validate_json(p.read_text()) if p.exists() else None

    def delete_artifact(self, video_id: str, name: str) -> None:
        self.artifact_path(video_id, name).unlink(missing_ok=True)

    # ------------------------------------------------------------------- jobs

    def create_job(self, video_id: str, stages: list[str] | None = None) -> Job:
        selected = set(stages or STAGES)
        runs = [StageRun(name=s, state="pending" if s in selected else "skipped") for s in STAGES]
        job = Job(job_id=new_id("job"), video_id=video_id, state="queued", stages=runs, created_at=now())
        self._q(
            "INSERT INTO jobs(job_id, video_id, state, stages, created_at) VALUES (?,?,?,?,?)",
            (job.job_id, video_id, job.state, json.dumps([r.model_dump(mode="json") for r in runs]),
             job.created_at.isoformat()),
        )
        return job

    def update_job(self, job: Job) -> None:
        self._q(
            "UPDATE jobs SET state=?, stages=?, started_at=?, finished_at=?, error=? WHERE job_id=?",
            (
                job.state,
                json.dumps([r.model_dump(mode="json") for r in job.stages]),
                job.started_at.isoformat() if job.started_at else None,
                job.finished_at.isoformat() if job.finished_at else None,
                job.error,
                job.job_id,
            ),
        )

    @staticmethod
    def _job_from_row(r: sqlite3.Row) -> Job:
        return Job(
            job_id=r["job_id"],
            video_id=r["video_id"],
            state=r["state"],
            stages=[StageRun.model_validate(s) for s in json.loads(r["stages"])],
            created_at=datetime.fromisoformat(r["created_at"]),
            started_at=datetime.fromisoformat(r["started_at"]) if r["started_at"] else None,
            finished_at=datetime.fromisoformat(r["finished_at"]) if r["finished_at"] else None,
            error=r["error"],
        )

    def get_job(self, job_id: str) -> Job | None:
        rows = self._q("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        return self._job_from_row(rows[0]) if rows else None

    def latest_job(self, video_id: str) -> Job | None:
        rows = self._q(
            "SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (video_id,)
        )
        return self._job_from_row(rows[0]) if rows else None

    def list_jobs(self, video_id: str | None = None) -> list[Job]:
        if video_id:
            rows = self._q("SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC", (video_id,))
        else:
            rows = self._q("SELECT * FROM jobs ORDER BY created_at DESC")
        return [self._job_from_row(r) for r in rows]

    def claim_next_job(self) -> Job | None:
        """Atomically move the oldest queued job to running and return it."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT job_id FROM jobs WHERE state='queued' ORDER BY created_at, rowid LIMIT 1"
                ).fetchone()
                if not row:
                    self._conn.execute("COMMIT")
                    return None
                self._conn.execute(
                    "UPDATE jobs SET state='running', started_at=? WHERE job_id=? AND state='queued'",
                    (now().isoformat(), row["job_id"]),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return self.get_job(row["job_id"])

    def fail_orphaned_jobs(self) -> int:
        """Jobs left 'running' by a crashed worker are marked failed on startup."""
        rows = self._q("SELECT job_id FROM jobs WHERE state='running'")
        for r in rows:
            job = self.get_job(r["job_id"])
            if job:
                job.state, job.error, job.finished_at = "failed", "worker stopped while running", now()
                for s in job.stages:
                    if s.state == "running":
                        s.state, s.error = "failed", job.error
                self.update_job(job)
        return len(rows)

    # ------------------------------------------------------------ corrections

    def add_correction(self, video_id: str, c: CorrectionIn) -> Correction:
        with self._lock:
            seq = self._q("SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM corrections WHERE video_id=?", (video_id,))[0]["n"]
            corr = Correction(
                correction_id=new_id("cor"), video_id=video_id, created_at=now(), seq=seq,
                op=c.op, payload=c.payload, note=c.note,
            )
            self._q(
                "INSERT INTO corrections(correction_id, video_id, seq, op, payload, note, created_at) VALUES (?,?,?,?,?,?,?)",
                (corr.correction_id, video_id, seq, corr.op, json.dumps(corr.payload), corr.note,
                 corr.created_at.isoformat()),
            )
        return corr

    def list_corrections(self, video_id: str) -> list[Correction]:
        rows = self._q("SELECT * FROM corrections WHERE video_id=? ORDER BY seq", (video_id,))
        return [
            Correction(
                correction_id=r["correction_id"], video_id=r["video_id"], seq=r["seq"], op=r["op"],
                payload=json.loads(r["payload"]), note=r["note"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]
