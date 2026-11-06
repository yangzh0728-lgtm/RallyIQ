"""Append-only analysis event log (README 个人记忆: "只追加的分析事件日志").

``StoreMemoryLog`` keeps the events in the database (table ``memory_events``, one log per
account), so the log survives container restarts online and is shared by every API instance.
``MemoryLog`` is the original JSON-lines file ``<data>/memory/events.jsonl``; a local log
from before the database version is imported once, on first use. Events
are only ever appended; profiles and opponent reports are recomputed from the
whole log, using the latest committed event per (video, person) unless a later
``match_withdrawn`` event removed that video.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import AnalysisEvent

_LOCK = threading.Lock()


class MemoryLog:
    def __init__(self, root: Path | str):
        self.path = Path(root) / "memory" / "events.jsonl"

    def read(self) -> list[AnalysisEvent]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(AnalysisEvent.model_validate_json(line))
        return out

    def append(self, events: list[dict]) -> list[AnalysisEvent]:
        """Append events (dicts without ``event_id``/``seq``/``recorded_at``); returns them as stored."""
        with _LOCK:
            existing = self.read()
            seq = existing[-1].seq if existing else 0
            now = datetime.now(timezone.utc)
            stored = []
            for e in events:
                seq += 1
                stored.append(AnalysisEvent(event_id=f"ae_{uuid.uuid4().hex[:12]}", seq=seq, recorded_at=now, **e))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                for e in stored:
                    f.write(e.model_dump_json() + "\n")
            return stored


class StoreMemoryLog:
    """Same interface as ``MemoryLog``, backed by ``Store`` (SQLite locally, PostgreSQL online)."""

    def __init__(self, store, owner_id: str | None = None):
        self.store, self.owner_id = store, owner_id
        self._import_legacy_file()

    def _import_legacy_file(self) -> None:
        legacy = MemoryLog(self.store.root)
        if self.owner_id is not None or not legacy.path.exists():
            return
        with _LOCK:
            if self.store.memory_count():
                return
            old = legacy.read()
            self.store.memory_append(None, lambda _last: [
                (e.event_id, e.seq, e.recorded_at.isoformat(), e.model_dump_json(), e) for e in old
            ])

    def read(self) -> list[AnalysisEvent]:
        return [AnalysisEvent.model_validate_json(b) for b in self.store.memory_read(self.owner_id)]

    def append(self, events: list[dict]) -> list[AnalysisEvent]:
        now = datetime.now(timezone.utc)

        def rows(last: int):
            out = []
            for i, e in enumerate(events, start=1):
                ev = AnalysisEvent(event_id=f"ae_{uuid.uuid4().hex[:12]}", seq=last + i, recorded_at=now, **e)
                out.append((ev.event_id, ev.seq, now.isoformat(), ev.model_dump_json(), ev))
            return out

        return self.store.memory_append(self.owner_id, rows)


def current_events(events: list[AnalysisEvent]) -> list[AnalysisEvent]:
    """Latest committed event per (video, player), minus videos withdrawn afterwards. Log order."""
    latest: dict[tuple[str, str | None], AnalysisEvent] = {}
    withdrawn_at: dict[str, int] = {}
    for e in events:
        if e.kind == "match_withdrawn":
            withdrawn_at[e.video_id] = e.seq
        else:
            latest[(e.video_id, e.player_id)] = e
    keep = [e for e in latest.values() if e.seq > withdrawn_at.get(e.video_id, -1)]
    # A re-commit names a fresh set of people: drop players of an older commit of the same video.
    newest_commit = {}
    for e in keep:
        newest_commit[e.video_id] = max(newest_commit.get(e.video_id, 0), e.seq)
    batch = {}
    for e in keep:
        if e.seq == newest_commit[e.video_id]:
            batch[e.video_id] = e.recorded_at
    keep = [e for e in keep if e.recorded_at == batch[e.video_id]]
    return sorted(keep, key=lambda e: (e.video_created_at or e.recorded_at, e.seq))
