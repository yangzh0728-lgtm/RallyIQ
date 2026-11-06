"""Download the pretrained shuttle heatmap checkpoint into weights/shuttle/.

    bash scripts/fetch_shuttle_weights.sh   (or: pip install gdown && python scripts/fetch_shuttle_weights.py)

The checkpoint is MIT licensed (see rallyiq/perception/shuttle/LICENSE-THIRD-PARTY).
Without gdown, download the zip from DRIVE_URL by hand and pass it with --zip.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

DRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
DRIVE_URL = f"https://drive.google.com/file/d/{DRIVE_ID}/view"
MEMBER = "TrackNet_best.pt"  # file name inside the released zip
DEST = Path(__file__).resolve().parents[1] / "weights" / "shuttle"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="already downloaded checkpoint zip")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        z = args.zip
        if z is None:
            if not shutil.which("gdown"):
                print(f"gdown not installed. Run `pip install gdown`, or download {DRIVE_URL} "
                      "and rerun with --zip <file>", file=sys.stderr)
                return 2
            z = Path(tmp) / "ckpts.zip"
            subprocess.run(["gdown", DRIVE_ID, "-O", str(z)], check=True)
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith(MEMBER)]
            if not names:
                print(f"{MEMBER} not found in {z}", file=sys.stderr)
                return 1
            args.dest.mkdir(parents=True, exist_ok=True)
            out = args.dest / "heatmap.pt"
            with zf.open(names[0]) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"saved {out} (sha256 {sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
