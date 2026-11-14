"""Storage: a SQL database for videos / jobs / corrections / users, and a blob store for files.

Locally (default) both live in the data dir (``RALLYIQ_DATA``, default ``./data``)::

    rallyiq.db
    videos/<video_id>/source<.ext>
    videos/<video_id>/artifacts/<stage>.json
    videos/<video_id>/artifacts/calibration_manual.json   (human override, optional)
    videos/<video_id>/work/                                (scratch for stages, never uploaded)

Online, ``RALLYIQ_DATABASE_URL`` points at PostgreSQL and ``RALLYIQ_STORAGE`` at an S3 bucket
(see ``db.py`` / ``blobs.py``); the data dir is then only a local cache for the worker.
"""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .blobs import BlobStore, blob_store_from_env
from .contracts import (
    STAGES,
    Correction,
    CorrectionIn,
    Job,
    StageRun,
    Video,
    VideoMeta,
)
from .db import Database, Row, database_url

T = TypeVar("T", bound=BaseModel)

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
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
    hard_example INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE(video_id, seq)
);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    owner_id TEXT,
    seq INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_owner ON memory_events(owner_id, seq);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""

# Columns added after the first release; created on startup if missing (additive only).
VIDEO_COLUMNS = {
    "source_key": "TEXT",          # blob key of the source video
    "owner_id": "TEXT",            # users.user_id; NULL when accounts are off
    "artifacts": "TEXT",           # JSON list of saved artifact names (avoids listing the bucket)
    "status": "TEXT",              # 'uploading' (direct upload not finished) | 'ready'
}
JOB_COLUMNS = {
    "worker_id": "TEXT",
    "heartbeat_at": "TEXT",
    "attempts": "INTEGER NOT NULL DEFAULT 0",
}

CONTENT_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
                 ".mkv": "video/x-matroska", ".avi": "video/x-msvideo"}


def now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def default_data_dir() -> Path:
    import os

    return Path(os.environ.get("RALLYIQ_DATA", "data")).resolve()


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"


class Store:
    def __init__(self, root: Path | str | None = None, db_url: str | None = None, blobs: BlobStore | None = None):
        self.root = Path(root).resolve() if root else default_data_dir()
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        self.db = Database(db_url or database_url(self.root))
        self.blobs: BlobStore = blobs or blob_store_from_env(self.root)
        with self.db.schema_lock():
            self.db.executescript(SCHEMA)
            self.db.add_columns("videos", VIDEO_COLUMNS)
            self.db.add_columns("jobs", JOB_COLUMNS)
            self.db.q("CREATE INDEX IF NOT EXISTS videos_owner ON videos(owner_id, created_at)")

    def _q(self, sql: str, args: tuple = ()) -> list[Row]:
        return self.db.q(sql, args)

    @property
    def _order_tiebreak(self) -> str:
        return "rowid" if self.db.dialect == "sqlite" else "job_id"

    # ------------------------------------------------------------------ keys / local paths

    @staticmethod
    def artifact_key(video_id: str, name: str) -> str:
        return f"videos/{video_id}/artifacts/{name}.json"

    def video_dir(self, video_id: str) -> Path:
        return self.root / "videos" / video_id

    def artifact_path(self, video_id: str, name: str) -> Path:
        """Local path of an artifact. Only meaningful with local storage; see :meth:`artifact_bytes`."""
        return self.video_dir(video_id) / "artifacts" / f"{name}.json"

    def work_dir(self, video_id: str) -> Path:
        d = self.video_dir(video_id) / "work"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----------------------------------------------------------------- videos

    def create_video(self, filename: str, ext: str = ".mp4") -> tuple[str, Path]:
        """Reserve a video id and return the local path the upload must be written to."""
        video_id = new_id("vid")
        d = self.video_dir(video_id)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        return video_id, d / f"source{ext.lower() or '.mp4'}"

    def register_video(self, video_id: str, filename: str, path: Path, owner_id: str | None = None) -> Video:
        """Record a video whose bytes are at local ``path`` (uploaded to object storage if configured)."""
        ext = Path(path).suffix.lower()
        key = f"videos/{video_id}/source{ext}"
        self.blobs.put_file(key, Path(path), CONTENT_TYPES.get(ext))
        self._q(
            "INSERT INTO videos(video_id, filename, path, size_bytes, created_at, source_key, owner_id, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (video_id, filename, key, Path(path).stat().st_size, now().isoformat(), key, owner_id, "ready"),
        )
        return self.get_video(video_id)  # type: ignore[return-value]

    def reserve_upload(self, filename: str, ext: str, owner_id: str | None = None) -> tuple[str, str]:
        """Direct-to-bucket upload, step 1: create a placeholder row and return ``(video_id, key)``."""
        video_id = new_id("vid")
        key = f"videos/{video_id}/source{ext.lower() or '.mp4'}"
        self._q(
            "INSERT INTO videos(video_id, filename, path, size_bytes, created_at, source_key, owner_id, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (video_id, filename, key, 0, now().isoformat(), key, owner_id, "uploading"),
        )
        return video_id, key

    def complete_upload(self, video_id: str) -> Video | None:
        """Direct-to-bucket upload, step 2: the object exists now; mark the video ready."""
        key = self.source_key(video_id)
        size = self.blobs.size(key) if key else None
        if size is None:
            return None
        self._q("UPDATE videos SET size_bytes=?, status='ready' WHERE video_id=?", (size, video_id))
        return self.get_video(video_id)

    def video_status(self, video_id: str) -> str | None:
        rows = self._q("SELECT status FROM videos WHERE video_id=?", (video_id,))
        return (rows[0]["status"] or "ready") if rows else None

    def source_key(self, video_id: str) -> str | None:
        rows = self._q("SELECT path, source_key FROM videos WHERE video_id=?", (video_id,))
        if not rows:
            return None
        return rows[0]["source_key"]

    def video_owner(self, video_id: str) -> str | None:
        rows = self._q("SELECT owner_id FROM videos WHERE video_id=?", (video_id,))
        return rows[0]["owner_id"] if rows else None

    def video_path(self, video_id: str) -> Path | None:
        """Local path of the source video, downloading it into the cache from object storage if needed."""
        rows = self._q("SELECT path, source_key FROM videos WHERE video_id=?", (video_id,))
        if not rows:
            return None
        key = rows[0]["source_key"]
        if not key:  # rows written before object storage existed hold an absolute path
            return Path(rows[0]["path"])
        local = self.blobs.local_path(key)
        if local is not None:
            return local
        cached = self.root / key
        if not cached.exists():
            self.blobs.get_file(key, cached)
        return cached

    @contextmanager
    def local_video(self, video_id: str) -> Iterator[Path | None]:
        """A local copy of the source for one-off reads (e.g. by the API). A file downloaded from
        object storage just for this is removed afterwards; a worker's cached copy is kept."""
        key = self.source_key(video_id)
        if key is None or self.blobs.local_path(key) is not None:
            yield self.video_path(video_id)
            return
        cached = self.root / key
        existed = cached.exists()
        try:
            yield self.video_path(video_id)
        finally:
            if not existed:
                cached.unlink(missing_ok=True)

    def video_stamp(self, video_id: str) -> list[int] | None:
        """Changes when the source video changes; works without downloading it."""
        key = self.source_key(video_id)
        if key is None:
            p = self.video_path(video_id)
            try:
                st = p.stat() if p else None
            except OSError:
                st = None
            return [st.st_mtime_ns, st.st_size] if st else None
        return self.blobs.stamp(key)

    def artifact_stamp(self, video_id: str, name: str) -> list[int] | None:
        return self.blobs.stamp(self.artifact_key(video_id, name))

    def video_download_url(self, video_id: str, filename: str | None = None) -> str | None:
        key = self.source_key(video_id)
        return self.blobs.presign_get(key, filename=filename) if key else None

    def set_video_meta(self, meta: VideoMeta) -> None:
        self._q("UPDATE videos SET meta=? WHERE video_id=?", (meta.model_dump_json(), meta.video_id))

    def get_video_meta(self, video_id: str) -> VideoMeta | None:
        rows = self._q("SELECT meta FROM videos WHERE video_id=?", (video_id,))
        return VideoMeta.model_validate_json(rows[0]["meta"]) if rows and rows[0]["meta"] else None

    def _artifact_names(self, r: Row) -> list[str]:
        if r["artifacts"] is not None:
            names = set(json.loads(r["artifacts"]))
            return [s for s in STAGES if s in names]
        return [s for s in STAGES if self.blobs.exists(self.artifact_key(r["video_id"], s))]

    def _video_from_row(self, r: Row) -> Video:
        vid = r["video_id"]
        return Video(
            video_id=vid,
            filename=r["filename"],
            size_bytes=r["size_bytes"],
            created_at=datetime.fromisoformat(r["created_at"]),
            meta=VideoMeta.model_validate_json(r["meta"]) if r["meta"] else None,
            latest_job=self.latest_job(vid),
            artifacts=self._artifact_names(r),
        )

    def get_video(self, video_id: str) -> Video | None:
        rows = self._q("SELECT * FROM videos WHERE video_id=?", (video_id,))
        return self._video_from_row(rows[0]) if rows else None

    def list_videos(self, owner_id: str | None = None) -> list[Video]:
        """Ready videos, newest first. ``owner_id`` restricts to one user's videos."""
        where, args = "WHERE COALESCE(status, 'ready')='ready'", ()
        if owner_id is not None:
            where, args = where + " AND owner_id=?", (owner_id,)
        rows = self._q(f"SELECT * FROM videos {where} ORDER BY created_at DESC", args)
        return [self._video_from_row(r) for r in rows]

    def delete_video(self, video_id: str) -> None:
        with self.db.transaction():
            self.db.q("DELETE FROM corrections WHERE video_id=?", (video_id,))
            self.db.q("DELETE FROM jobs WHERE video_id=?", (video_id,))
            self.db.q("DELETE FROM videos WHERE video_id=?", (video_id,))
        self.blobs.delete_prefix(f"videos/{video_id}/")
        cache = self.video_dir(video_id)
        if cache.exists():
            import shutil

            shutil.rmtree(cache, ignore_errors=True)

    # -------------------------------------------------------------- artifacts

    def _set_artifact_flag(self, video_id: str, name: str, present: bool) -> None:
        with self.db.transaction():
            rows = self.db.q("SELECT artifacts FROM videos WHERE video_id=?", (video_id,))
            if not rows:
                return
            cur = rows[0]["artifacts"]
            names = set(json.loads(cur)) if cur is not None else {
                s for s in STAGES if self.blobs.exists(self.artifact_key(video_id, s))
            }
            names = names | {name} if present else names - {name}
            self.db.q("UPDATE videos SET artifacts=? WHERE video_id=?", (json.dumps(sorted(names)), video_id))

    def save_artifact(self, video_id: str, name: str, obj: BaseModel) -> str:
        key = self.artifact_key(video_id, name)
        self.blobs.put_bytes(key, obj.model_dump_json().encode(), "application/json")
        self._set_artifact_flag(video_id, name, True)
        return key

    def has_artifact(self, video_id: str, name: str) -> bool:
        return self.blobs.exists(self.artifact_key(video_id, name))

    def artifact_bytes(self, video_id: str, name: str) -> bytes | None:
        return self.blobs.get_bytes(self.artifact_key(video_id, name))

    def load_artifact(self, video_id: str, name: str, model: type[T]) -> T | None:
        data = self.artifact_bytes(video_id, name)
        return model.model_validate_json(data) if data is not None else None

    def delete_artifact(self, video_id: str, name: str) -> None:
        self.blobs.delete(self.artifact_key(video_id, name))
        self._set_artifact_flag(video_id, name, False)

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

    def update_job(self, job: Job, worker_id: str | None = None) -> None:
        """Persist job progress. With ``worker_id`` the write also refreshes the lease, and is ignored
        if another worker has taken the job over (e.g. after this one was presumed dead)."""
        args = [
            job.state,
            json.dumps([r.model_dump(mode="json") for r in job.stages]),
            job.started_at.isoformat() if job.started_at else None,
            job.finished_at.isoformat() if job.finished_at else None,
            job.error,
        ]
        if worker_id is None:
            self._q("UPDATE jobs SET state=?, stages=?, started_at=?, finished_at=?, error=? WHERE job_id=?",
                    (*args, job.job_id))
        else:
            self._q(
                "UPDATE jobs SET state=?, stages=?, started_at=?, finished_at=?, error=?, heartbeat_at=?"
                " WHERE job_id=? AND worker_id=?",
                (*args, now().isoformat(), job.job_id, worker_id),
            )

    @staticmethod
    def _job_from_row(r: Row) -> Job:
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
            f"SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC, {self._order_tiebreak} DESC LIMIT 1",
            (video_id,),
        )
        return self._job_from_row(rows[0]) if rows else None

    def list_jobs(self, video_id: str | None = None) -> list[Job]:
        if video_id:
            rows = self._q("SELECT * FROM jobs WHERE video_id=? ORDER BY created_at DESC", (video_id,))
        else:
            rows = self._q("SELECT * FROM jobs ORDER BY created_at DESC")
        return [self._job_from_row(r) for r in rows]

    def count_active_jobs(self, owner_id: str | None) -> int:
        rows = self._q(
            "SELECT COUNT(*) AS n FROM jobs j JOIN videos v ON v.video_id=j.video_id"
            " WHERE j.state IN ('queued','running') AND v.owner_id IS NOT DISTINCT FROM ?"
            if self.db.dialect == "postgres" else
            "SELECT COUNT(*) AS n FROM jobs j JOIN videos v ON v.video_id=j.video_id"
            " WHERE j.state IN ('queued','running') AND v.owner_id IS ?",
            (owner_id,),
        )
        return int(rows[0]["n"])

    def queue_depth(self) -> dict[str, int]:
        rows = self._q("SELECT state, COUNT(*) AS n FROM jobs WHERE state IN ('queued','running') GROUP BY state")
        out = {"queued": 0, "running": 0}
        out.update({r["state"]: int(r["n"]) for r in rows})
        return out

    def claim_next_job(self, worker_id: str | None = None) -> Job | None:
        """Atomically move the oldest queued job to running, lease it to ``worker_id`` and return it.

        Safe with many workers: PostgreSQL uses ``FOR UPDATE SKIP LOCKED``; SQLite takes the write lock.
        """
        worker_id = worker_id or default_worker_id()
        ts = now().isoformat()
        lock = " FOR UPDATE SKIP LOCKED" if self.db.dialect == "postgres" else ""
        with self.db.transaction():
            rows = self.db.q(
                f"SELECT job_id FROM jobs WHERE state='queued' ORDER BY created_at, {self._order_tiebreak}"
                f" LIMIT 1{lock}"
            )
            if not rows:
                return None
            job_id = rows[0]["job_id"]
            self.db.q(
                "UPDATE jobs SET state='running', started_at=COALESCE(started_at, ?), worker_id=?, heartbeat_at=?,"
                " attempts=attempts+1 WHERE job_id=? AND state='queued'",
                (ts, worker_id, ts, job_id),
            )
        return self.get_job(job_id)

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Extend the lease. Returns False if the job is no longer ours."""
        rows = self._q(
            "UPDATE jobs SET heartbeat_at=? WHERE job_id=? AND worker_id=? AND state='running' RETURNING job_id",
            (now().isoformat(), job_id, worker_id),
        )
        return bool(rows)

    def recover_stale_jobs(self, lease_s: float = 300, max_attempts: int = 2) -> dict[str, int]:
        """Jobs whose worker stopped heart-beating (crash, preempted GPU, scale-down) are re-queued,
        or failed once they have used ``max_attempts``. Safe to call from any process."""
        cutoff = (now() - timedelta(seconds=lease_s)).isoformat()
        out = {"requeued": 0, "failed": 0}
        with self.db.transaction():
            rows = self.db.q(
                "SELECT * FROM jobs WHERE state='running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)", (cutoff,)
            )
            for r in rows:
                job = self._job_from_row(r)
                retry = int(r["attempts"] or 0) < max_attempts
                for s in job.stages:
                    if s.state == "running":
                        s.state, s.progress = ("pending", 0.0) if retry else ("failed", s.progress)
                        s.error = None if retry else "worker stopped while running"
                if retry:
                    job.state, job.error = "queued", None
                    out["requeued"] += 1
                else:
                    job.state, job.error, job.finished_at = "failed", "worker stopped while running", now()
                    out["failed"] += 1
                self.db.q(
                    "UPDATE jobs SET state=?, stages=?, finished_at=?, error=?, worker_id=NULL WHERE job_id=?",
                    (job.state, json.dumps([s.model_dump(mode="json") for s in job.stages]),
                     job.finished_at.isoformat() if job.finished_at else None, job.error, job.job_id),
                )
        return out

    def fail_orphaned_jobs(self) -> int:
        """Single-process mode: anything still 'running' at startup was left by a crash; fail it."""
        return self.recover_stale_jobs(lease_s=-1, max_attempts=0)["failed"]

    # ------------------------------------------------------------ corrections

    def add_correction(self, video_id: str, c: CorrectionIn) -> Correction:
        with self.db.transaction():
            seq = self.db.q(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM corrections WHERE video_id=?", (video_id,)
            )[0]["n"]
            corr = Correction(
                correction_id=new_id("cor"), video_id=video_id, created_at=now(), seq=seq,
                op=c.op, payload=c.payload, note=c.note, hard_example=c.hard_example,
            )
            self.db.q(
                "INSERT INTO corrections(correction_id, video_id, seq, op, payload, note, hard_example, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (corr.correction_id, video_id, seq, corr.op, json.dumps(corr.payload), corr.note,
                 int(corr.hard_example), corr.created_at.isoformat()),
            )
        return corr

    def list_corrections(self, video_id: str) -> list[Correction]:
        rows = self._q("SELECT * FROM corrections WHERE video_id=? ORDER BY seq", (video_id,))
        return [
            Correction(
                correction_id=r["correction_id"], video_id=r["video_id"], seq=r["seq"], op=r["op"],
                payload=json.loads(r["payload"]), note=r["note"], hard_example=bool(r["hard_example"]),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ---------------------------------------------------------- memory (append-only)

    def _owner_clause(self) -> str:
        return "owner_id IS NOT DISTINCT FROM ?" if self.db.dialect == "postgres" else "owner_id IS ?"

    def memory_read(self, owner_id: str | None) -> list[str]:
        """Raw JSON of one account's analysis events, in log order."""
        rows = self._q(f"SELECT body FROM memory_events WHERE {self._owner_clause()} ORDER BY seq", (owner_id,))
        return [r["body"] for r in rows]

    def memory_append(self, owner_id: str | None, make_rows) -> list:
        """Append events atomically. ``make_rows(last_seq)`` returns ``[(event_id, seq, recorded_at, body,
        obj)]``; the ``obj`` list is returned. Never updates or deletes existing rows."""
        with self.db.transaction():
            last = self.db.q(
                f"SELECT COALESCE(MAX(seq), 0) AS n FROM memory_events WHERE {self._owner_clause()}", (owner_id,)
            )[0]["n"]
            rows = make_rows(int(last))
            for event_id, seq, recorded_at, body, _ in rows:
                self.db.q("INSERT INTO memory_events(event_id, owner_id, seq, recorded_at, body) VALUES (?,?,?,?,?)",
                          (event_id, owner_id, seq, recorded_at, body))
        return [r[4] for r in rows]

    def memory_count(self) -> int:
        return int(self._q("SELECT COUNT(*) AS n FROM memory_events")[0]["n"])

    # ------------------------------------------------------------------ users

    def create_user(self, email: str, password_hash: str, is_admin: bool = False) -> str:
        user_id = new_id("usr")
        self._q("INSERT INTO users(user_id, email, password_hash, created_at, is_admin) VALUES (?,?,?,?,?)",
                (user_id, email, password_hash, now().isoformat(), int(is_admin)))
        return user_id

    def get_user_by_email(self, email: str) -> Row | None:
        rows = self._q("SELECT * FROM users WHERE email=?", (email,))
        return rows[0] if rows else None

    def get_user(self, user_id: str) -> Row | None:
        rows = self._q("SELECT * FROM users WHERE user_id=?", (user_id,))
        return rows[0] if rows else None

    def count_users(self) -> int:
        return int(self._q("SELECT COUNT(*) AS n FROM users")[0]["n"])

    def create_session(self, token_hash: str, user_id: str, ttl: timedelta) -> None:
        t = now()
        self._q("INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                (token_hash, user_id, t.isoformat(), (t + ttl).isoformat()))

    def session_user(self, token_hash: str) -> Row | None:
        rows = self._q(
            "SELECT u.* FROM sessions s JOIN users u ON u.user_id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
            (token_hash, now().isoformat()),
        )
        return rows[0] if rows else None

    def delete_session(self, token_hash: str) -> None:
        self._q("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
