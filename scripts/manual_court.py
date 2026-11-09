"""Turn a hand-labelled court file into a video's calibration_manual.json override.

    python scripts/manual_court.py COURT.json --video-id vid_abc [--data ~/rallyiq-data]
    python scripts/manual_court.py COURT.json --video-id vid_abc --size 1920x1080 --frames 3596 --out cal.json

COURT.json is the hand-labelling format used under notes/hs_courts/ (and
notes/dji0009_calibration.json):

    {"image_size": [3840, 2160],            # optional, default 3840x2160
     "frames": {"2697": {"image_from_court": [[...], [...], [...]]}, ...}}
or  {"image_size": [w, h], "image_from_court": [[...], ...]}   # one pose for the whole clip

image_from_court maps court metres (x across 0..6.1 from the left doubles
sideline, y along 0..13.4 from the near baseline) to pixels of a frame of
image_size. Labels are rescaled to the video's own resolution, and each
labelled frame covers the frames nearest to it (segments split halfway
between labelled frames), so a camera that drifts a little keeps the right court.

The pipeline uses calibration_manual.json in place of its own calibration on
the next run (rallyiq/pipeline/runner.py), and the player and shuttle stages
then keep only what is on that court.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts.geometry import COURT_CORNERS_M
from rallyiq.contracts.models import (
    CalibrationSegment,
    CourtCalibration,
    CourtKeypoint,
)

CORNER_NAMES = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"]


def load_labels(path: Path) -> tuple[tuple[int, int], dict[int, np.ndarray]]:
    d = json.loads(path.read_text())
    size = tuple(d.get("image_size") or (3840, 2160))
    if "frames" in d:
        poses = {int(f): np.array(v["image_from_court"], float) for f, v in d["frames"].items()}
    else:
        poses = {0: np.array(d["image_from_court"], float)}
    if not poses:
        raise ValueError(f"{path}: no labelled frames")
    return size, poses


def build(video_id: str, size: tuple[int, int], poses: dict[int, np.ndarray],
          video_size: tuple[int, int], frame_count: int, source: str = "") -> CourtCalibration:
    sx, sy = video_size[0] / size[0], video_size[1] / size[1]
    S = np.diag([sx, sy, 1.0])
    keys = sorted(poses)
    last = max(frame_count - 1, keys[-1])
    segments = []
    for i, f in enumerate(keys):
        start = 0 if i == 0 else (keys[i - 1] + f) // 2 + 1
        end = last if i == len(keys) - 1 else (f + keys[i + 1]) // 2
        H = S @ poses[f]  # court -> video px
        H = H / H[2, 2]
        corners = []
        for x, y in COURT_CORNERS_M:
            q = H @ [x, y, 1.0]
            corners.append([float(q[0] / q[2]), float(q[1] / q[2])])
        Hi = np.linalg.inv(H)
        Hi = Hi / Hi[2, 2]
        segments.append(CalibrationSegment(
            start_frame=start, end_frame=end, homography=Hi.tolist(), image_corners=corners,
            keypoints=[CourtKeypoint(name=n, image=c, court=list(m))
                       for n, c, m in zip(CORNER_NAMES, corners, COURT_CORNERS_M)],
            reprojection_error_px=0.0, method="manual", confidence=1.0, status="ok",
            notes=[f"hand-labelled court from frame {f}" + (f" ({source})" if source else "")],
        ))
    return CourtCalibration(video_id=video_id, segments=segments)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--data", default=None, help="store root (default: the pipeline's)")
    ap.add_argument("--size", help="video WxH; default from the video's probe")
    ap.add_argument("--frames", type=int, help="video frame count; default from the video's probe")
    ap.add_argument("--out", type=Path, help="write here instead of into the store")
    a = ap.parse_args()

    size, poses = load_labels(a.labels)
    store = None
    if a.size and a.frames is not None:
        w, h = map(int, a.size.lower().split("x"))
        frames = a.frames
    else:
        from rallyiq.store import Store
        store = Store(a.data)
        meta = store.get_video_meta(a.video_id)
        if meta is None:
            sys.exit(f"no probe for {a.video_id}; pass --size and --frames")
        w, h, frames = meta.width, meta.height, meta.frame_count
    cal = build(a.video_id, size, poses, (w, h), frames, source=a.labels.name)
    if a.out:
        a.out.write_text(cal.model_dump_json(indent=1))
        print(f"wrote {a.out}")
    else:
        if store is None:
            from rallyiq.store import Store
            store = Store(a.data)
        from rallyiq.pipeline.runner import MANUAL_CALIBRATION
        print(f"wrote {store.save_artifact(a.video_id, MANUAL_CALIBRATION, cal)}")
    for s in cal.segments:
        print(f"  frames {s.start_frame}-{s.end_frame}: {s.notes[0]}")


if __name__ == "__main__":
    main()
