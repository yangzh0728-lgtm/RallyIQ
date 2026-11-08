"""Thin SQL layer: SQLite (local, default) or PostgreSQL (online deployment).

``RALLYIQ_DATABASE_URL`` selects the backend::

    (unset)                                  -> SQLite at <data dir>/rallyiq.db
    sqlite:///abs/path/rallyiq.db            -> SQLite at that path
    postgresql://user:pass@host:5432/rallyiq -> PostgreSQL (needs ``pip install '.[server]'``)

Queries are written once with ``?`` placeholders; they are translated for psycopg.
Only portable SQL is used, plus ``Database.dialect`` checks where the two differ
(the job-claim query uses ``FOR UPDATE SKIP LOCKED`` on PostgreSQL).
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


_SCHEMA_LOCK = 0x5241_4C4C  # pg advisory lock key for schema creation ("RALL")

class Row(Protocol):
    def __getitem__(self, key: str) -> Any: ...


def database_url(data_dir: Path) -> str:
    return os.environ.get("RALLYIQ_DATABASE_URL") or f"sqlite:///{data_dir / 'rallyiq.db'}"


class Database:
    """Serialises access to one connection per process; enough for the API and a worker.

    Every statement autocommits unless it runs inside :meth:`transaction`.
    """

    dialect: str

    def __init__(self, url: str):
        self.url = url
        self._lock = threading.RLock()
        self._local_tx = threading.local()
        if url.startswith("sqlite:///"):
            self.dialect = "sqlite"
            path = url[len("sqlite:///"):]
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn: Any = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        elif url.startswith(("postgres://", "postgresql://")):
            self.dialect = "postgres"
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as e:  # pragma: no cover - depends on install
                raise RuntimeError("PostgreSQL needs psycopg: pip install '.[server]'") from e
            self._psycopg = psycopg
            self._dict_row = dict_row
            self._conn = None
            self._connect_pg()
        else:
            raise ValueError(f"unsupported RALLYIQ_DATABASE_URL {url!r}")

    def _connect_pg(self) -> None:
        self._conn = self._psycopg.connect(self.url, autocommit=True, row_factory=self._dict_row)

    # ------------------------------------------------------------------ sql

    _QMARK = re.compile(r"\?")

    def _sql(self, sql: str) -> str:
        return self._QMARK.sub("%s", sql) if self.dialect == "postgres" else sql

    def _execute(self, sql: str, args: Sequence[Any]) -> Any:
        if self.dialect == "postgres":
            in_tx = getattr(self._local_tx, "active", False)
            try:
                return self._conn.execute(self._sql(sql), tuple(args))
            except self._psycopg.OperationalError:
                # Managed Postgres drops idle connections; reconnect once outside transactions.
                if in_tx or not self._conn.closed:
                    raise
                self._connect_pg()
                return self._conn.execute(self._sql(sql), tuple(args))
        return self._conn.execute(sql, tuple(args))

    def q(self, sql: str, args: Sequence[Any] = ()) -> list[Row]:
        with self._lock:
            cur = self._execute(sql, args)
            return cur.fetchall() if cur.description else []

    def executescript(self, script: str) -> None:
        with self._lock:
            if self.dialect == "sqlite":
                self._conn.executescript(script)
            else:
                # The API and the worker start together and both create the schema; two
                # concurrent CREATE TABLE IF NOT EXISTS can still collide in Postgres's
                # catalog (duplicate pg_type), so serialise them with an advisory lock.
                self._execute("SELECT pg_advisory_lock(?)", (_SCHEMA_LOCK,))
                try:
                    for stmt in (s.strip() for s in script.split(";")):
                        if stmt:
                            self._execute(stmt, ())
                finally:
                    self._execute("SELECT pg_advisory_unlock(?)", (_SCHEMA_LOCK,))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Exclusive write transaction (``BEGIN IMMEDIATE`` on SQLite)."""
        with self._lock:
            if getattr(self._local_tx, "active", False):  # nested: join the outer transaction
                yield
                return
            if self.dialect == "postgres":
                if self._conn.closed:
                    self._connect_pg()
                self._local_tx.active = True
                try:
                    with self._conn.transaction():
                        yield
                finally:
                    self._local_tx.active = False
                return
            self._execute("BEGIN IMMEDIATE", ())
            self._local_tx.active = True
            try:
                yield
            except BaseException:
                self._execute("ROLLBACK", ())
                raise
            else:
                self._execute("COMMIT", ())
            finally:
                self._local_tx.active = False

    def columns(self, table: str) -> set[str]:
        if self.dialect == "sqlite":
            return {r["name"] for r in self.q(f"PRAGMA table_info({table})")}
        return {
            r["column_name"]
            for r in self.q("SELECT column_name FROM information_schema.columns WHERE table_name=?", (table,))
        }

    def add_columns(self, table: str, cols: dict[str, str]) -> None:
        """Additive migration: add any missing ``name -> type`` columns."""
        have = self.columns(table)
        for name, typ in cols.items():
            if name not in have:
                self.q(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
