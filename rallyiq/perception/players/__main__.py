"""Run player tracking on one video from the command line.

    python -m rallyiq.perception.players match.mp4 --homography H.json --out players.json

``H.json`` holds a 3x3 image->court homography (metres) as a nested list, or
an object with a ``"homography"`` key. Without it, output stays in pixels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .pose import RTMOOnnx
from .runner import PlayerTrackingConfig, iter_video, track_players, video_fps
from .stage import MODEL_NAME, build_tracks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--homography", help="JSON file with a 3x3 image->court homography")
    ap.add_argument("--model", help="RTMO ONNX path (default: $RALLYIQ_POSE_MODEL)")
    ap.add_argument("--out", default="players.json")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--format", choices=["auto", "singles", "doubles"], default="auto",
                    help="match format; auto decides from the first seconds")
    ap.add_argument("--far-boost", action="store_true", help="second pass on the far half for small players")
    args = ap.parse_args(argv)

    H = None
    if args.homography:
        data = json.loads(Path(args.homography).read_text())
        H = np.asarray(data["homography"] if isinstance(data, dict) else data, dtype=np.float64)

    cfg = PlayerTrackingConfig(stride=args.stride, match_format=args.format)
    cfg.tracker.fps = video_fps(args.video)
    est = RTMOOnnx(args.model, tile_far_half=args.far_boost)
    est_path = args.model or os.environ.get("RALLYIQ_POSE_MODEL", "")
    t0 = time.time()
    results = track_players(iter_video(args.video, args.stride), est, (lambda f: H) if H is not None else None, cfg)
    dt = time.time() - t0
    tracks = build_tracks(Path(args.video).stem, results, f"{MODEL_NAME}:{Path(est_path).name}")
    Path(args.out).write_text(tracks.model_dump_json())
    n = len(results)
    seen = sum(len(r.players) for r in results)
    print(f"{n} frames in {dt:.1f}s ({n / max(dt, 1e-6):.1f} fps); {seen / max(n, 1):.2f} players/frame, {tracks.match_format} -> {args.out}")


if __name__ == "__main__":
    main()
