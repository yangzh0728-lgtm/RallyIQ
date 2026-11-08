"""Heel and toe keypoints for an analysed video from the command line.

    python -m rallyiq.perception.feet match.mp4 players.json [--events events.json] --out feet.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ...contracts import EventsResult, FootLandmarks, PlayerTracks
from .model import MODEL_NAME, FootLandmarker
from .stage import detect_feet, iter_selected, wanted_frames


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("players")
    ap.add_argument("--events", help="events.json: only run on the rallies")
    ap.add_argument("--model", help="feet.tflite (default: $RALLYIQ_FOOT_MODEL, then weights/feet.tflite)")
    ap.add_argument("--fps", type=float, default=0, help="video fps (default: from the file)")
    ap.add_argument("--out", default="feet.json")
    args = ap.parse_args(argv)
    import cv2

    players = PlayerTracks.model_validate_json(Path(args.players).read_text())
    cap = cv2.VideoCapture(args.video)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    spans = []
    if args.events:
        ev = EventsResult.model_validate_json(Path(args.events).read_text())
        spans = [(r.start_frame, r.end_frame) for r in ev.rallies]
    wanted = wanted_frames(spans, fps, n)
    lm = FootLandmarker(args.model)
    t0 = time.time()
    found = detect_feet(iter_selected(args.video, wanted), players, lm, wanted)
    res = FootLandmarks(video_id=players.video_id, model=f"{MODEL_NAME}:{Path(lm.path).name}", frames=found)
    Path(args.out).write_text(res.model_dump_json())
    print(f"{len(found)} frames with feet in {time.time() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
