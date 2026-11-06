"""Run shuttle tracking on one video without the job queue.

    python -m rallyiq.perception.shuttle clip.mp4 --out shuttle.json
    python -m rallyiq.perception.shuttle clip.mp4 --benchmark-clip-id ae26_ms_f \\
        --benchmark-start-ms 900000 --benchmark-out predictions/ae26_ms_f.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rallyiq.perception.shuttle", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", help="checkpoint file or directory")
    ap.add_argument("--calibration", type=Path, help="CourtCalibration JSON (for per-segment background and --roi auto)")
    ap.add_argument("--roi", choices=["off", "auto"], default="off")
    ap.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--peak-threshold", type=float, default=0.2)
    ap.add_argument("--tracker", action="append", default=[], metavar="KEY=VALUE",
                    help="TrackerConfig override, repeatable (e.g. --tracker weak_min_travel_diag_per_s=0.1)")
    ap.add_argument("--from-candidates", type=Path,
                    help="skip the network: re-track a saved candidates file (fast settings sweeps)")
    ap.add_argument("--court-lock", choices=["auto", "off"], default="auto",
                    help="with --calibration: keep only trajectories over the target court")
    ap.add_argument("--out", type=Path, default=Path("shuttle.json"), help="ShuttleTrack JSON")
    ap.add_argument("--candidates", type=Path, help="also write raw per-frame candidates here")
    ap.add_argument("--benchmark-clip-id")
    ap.add_argument("--benchmark-start-ms", type=int, default=0, help="clip start in the source video")
    ap.add_argument("--benchmark-out", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from ...contracts import CourtCalibration
    from ...pipeline.probe import probe_video
    from .detector import DetectorConfig, HeatmapShuttleDetector, find_weights
    from .stage import build_track, court_lock, segments_for, tracker_config

    overrides = {}
    for kv in args.tracker:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = v.strip()
    tcfg = tracker_config(overrides, args.peak_threshold)
    meta = probe_video(args.video, video_id=args.video.stem)
    calib = CourtCalibration.model_validate_json(args.calibration.read_text()) if args.calibration else None
    region = support = None
    if calib is not None and args.court_lock == "auto":
        region, support, _mode = court_lock(calib, None)

    if args.from_candidates:
        doc = json.loads(args.from_candidates.read_text())
        size = tuple(doc["frame_size"])
        cands = {int(f): [tuple(c) for c in v["candidates"] if c[2] >= args.peak_threshold]
                 for f, v in doc["frames"].items()}
        name = doc.get("detector") or "unknown"
        track, selected = build_track(meta, cands, size[0], size[1], name, tcfg, region, support)
        return _finish(args, meta, track, cands, selected, name, size, elapsed=None, device="(cached)")

    weights = find_weights(args.weights)
    if weights is None:
        print("no weights found: see rallyiq/perception/shuttle/README.md", file=sys.stderr)
        return 2
    det = HeatmapShuttleDetector(weights, DetectorConfig(device=args.device, batch_size=args.batch_size,
                                                         peak_threshold=args.peak_threshold))
    t0 = time.monotonic()
    last = [0.0]

    def progress(p: float) -> None:
        if p - last[0] >= 0.1 or p >= 1.0:
            last[0] = p
            print(f"  {p:4.0%}  {time.monotonic() - t0:6.1f}s", file=sys.stderr)

    cands, size = det.detect(args.video, meta.fps, meta.frame_count, segments_for(meta, calib, args.roi), progress)
    elapsed = time.monotonic() - t0
    track, selected = build_track(meta, cands, size[0], size[1], det.name, tcfg, region, support)
    return _finish(args, meta, track, cands, selected, det.name, size, elapsed, str(det.device))


def _finish(args, meta, track, cands, selected, name, size, elapsed, device) -> int:
    from .stage import save_candidates

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(track.model_dump_json(indent=1))
    if args.candidates:
        save_candidates(args.candidates, cands, selected, name, size)
    if args.benchmark_out:
        from .export import to_benchmark_prediction

        doc = to_benchmark_prediction(track, meta, args.benchmark_clip_id or args.video.stem, args.benchmark_start_ms)
        args.benchmark_out.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    n_det = sum(p.source == "detected" for p in track.points)
    n_int = len(track.points) - n_det
    frames = max(meta.frame_count, 1)
    speed = f"in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps) " if elapsed is not None else ""
    print(f"{args.video.name}: {frames} frames {speed}on {device}; detected {n_det} ({n_det / frames:.0%}), "
          f"interpolated {n_int}, gaps {len(track.gaps)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
