"""The API and a worker start together on a fresh database: schema setup must not race."""

import threading

from rallyiq.store import Store


def test_stores_opened_at_once_on_a_fresh_database(db_url, tmp_path):
    # Before the schema lock, 4 concurrent opens on a fresh PostgreSQL database failed in 20 of 20
    # rounds (UniqueViolation on pg_type / DuplicateTable from CREATE TABLE IF NOT EXISTS).
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def open_store(i: int) -> None:
        try:
            start.wait()
            Store(root=tmp_path / f"s{i}", db_url=db_url).db.close()
        except BaseException as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    threads = [threading.Thread(target=open_store, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    s = Store(root=tmp_path / "again", db_url=db_url)  # reopening an existing schema still works
    assert {"users", "videos", "jobs"} <= {t for t in ("users", "videos", "jobs") if s.db.columns(t)}
