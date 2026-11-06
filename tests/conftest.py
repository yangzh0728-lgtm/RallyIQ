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
