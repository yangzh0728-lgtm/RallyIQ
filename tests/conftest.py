import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Tests import shared helpers as ``tests.…`` and the repo-level ``evaluation`` /
# ``scripts`` packages; the repo root is not on sys.path when pytest runs as a
# console script (as in CI).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    p = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
                    "-t", "12", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request, tmp_path):
    """SQLite always; PostgreSQL when RALLYIQ_TEST_DATABASE_URL points at a server (CI does).
    Each test gets a fresh database."""
    if request.param == "sqlite":
        yield f"sqlite:///{tmp_path / 'test.db'}"
        return
    import os
    import uuid

    base = os.environ.get("RALLYIQ_TEST_DATABASE_URL")
    if not base:
        pytest.skip("RALLYIQ_TEST_DATABASE_URL not set")
    psycopg = pytest.importorskip("psycopg")
    name = f"rallyiq_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    root, _, _ = base.rpartition("/")
    yield f"{root}/{name}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
