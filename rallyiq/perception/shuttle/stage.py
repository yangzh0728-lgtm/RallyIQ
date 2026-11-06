"""Stage ``shuttle``: per-frame shuttle detection + trajectory tracking.

``run(ctx) -> ShuttleTrack``. Settings come from ``ctx.config["shuttle"]`` (all optional):

* ``weights``: checkpoint file or directory (else ``RALLYIQ_SHUTTLE_WEIGHTS``,
  ``weights/shuttle/`` in the repo, ``~/.cache/rallyiq/weights/shuttle/``).
* ``require_weights``: raise instead of returning an empty track when no weights are found.
* ``device``, ``batch_size``, ``peak_threshold``, ``top_k``, ``target_fps``, ``ensemble``.
* ``roi``: ``"off"`` (default) or ``"auto"``: crop to the calibrated court plus the air
  above it, which enlarges small far-court shuttles. Experimental until measured.
* ``tracker``: overrides for ``TrackerConfig`` fields.

Besides the contract output, every raw candidate (including rejected ones) is written
to ``<work_dir>/shuttle_candidates.json`` so nothing the model saw is lost.

When no weights are installed the stage returns a track with ``detector=None`` and no
points: downstream stages must read that as "shuttle not analysed", not "never visible".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from ...contracts import CourtCalibration, FrameGap, ShuttlePoint, ShuttleTrack, VideoMeta
from ...pipeline.context import StageContext, demo_mode
from .tracker import Candidate, ShuttleTracker, TrackerConfig

log = logging.getLogger("rallyiq.shuttle")

CANDIDATES_FILE = "shuttle_candidates.json"


def tracker_config(overrides: dict | None) -> TrackerConfig:
    cfg = TrackerConfig()
    names = {f.name for f in fields(TrackerConfig)}
    for k, v in (overrides or {}).items():
        if k not in names:
            raise ValueError(f"unknown shuttle tracker setting {k!r}")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    return cfg


def build_track(
    meta: VideoMeta,
    candidates: dict[int, Sequence[Candidate]],
    width: int,
    height: int,
    detector: str | None,
    config: TrackerConfig | None = None,
) -> tuple[ShuttleTrack, dict[int, int]]:
    """Run the tracker and convert to the contract. Also returns the selected
    candidate index per frame (for the raw-candidates file)."""
    frame_count = max(meta.frame_count, (max(candidates) + 1) if candidates else 0)
    if abs(width / height - meta.width / meta.height) > 0.02:
        raise ValueError(
            f"decoded frames are {width}x{height} but the probe says {meta.width}x{meta.height}; "
            "the decoder did not apply the video rotation"
        )
    # Coordinates are in decoded-frame px; scale if the decoder's size differs from the probe.
    sx, sy = meta.width / width, meta.height / height
    tracker = ShuttleTracker(meta.fps, meta.width, meta.height, config)
    scaled = {f: [(x * sx, y * sy, s) for x, y, s in cs] for f, cs in candidates.items()}
    res = tracker.track(scaled, frame_count)
    points = [
        ShuttlePoint(
            frame=p.frame, t_ms=meta.t_ms(p.frame), x=round(p.x, 2), y=round(p.y, 2),
            confidence=round(p.confidence, 4), source=p.source,
        )
        for p in res.points
    ]
    gaps = [FrameGap(start_frame=g.start_frame, end_frame=g.end_frame, reason=g.reason) for g in res.gaps]
    return ShuttleTrack(video_id=meta.video_id, detector=detector, points=points, gaps=gaps), res.selected


def segments_for(meta: VideoMeta, calibration: CourtCalibration | None, roi_mode: str):
    """Camera segments (background median per segment) covering every frame, with optional ROI."""
    from .detector import Segment, court_roi

    last = max(meta.frame_count - 1, 0)
    segs = sorted(calibration.segments, key=lambda s: s.start_frame) if calibration else []
    if not segs:
        return [Segment(0, last)]
    out = []
    for i, s in enumerate(segs):
        start = 0 if i == 0 else s.start_frame
        end = last if i == len(segs) - 1 else segs[i + 1].start_frame - 1
        roi = None
        if roi_mode == "auto" and s.image_corners and s.status != "undeterminable":
            roi = court_roi(s.image_corners, meta.width, meta.height)
        out.append(Segment(start, max(start, end), roi))
    return out


def save_candidates(path: Path, candidates: dict[int, Sequence[Candidate]], selected: dict[int, int],
                    detector: str, size: tuple[int, int]) -> None:
    doc = {
        "detector": detector,
        "frame_size": list(size),
        "note": "raw heatmap peaks in decoded-frame px; 'selected' = index kept by the tracker or -1",
        "frames": {
            str(f): {"candidates": [[round(x, 2), round(y, 2), round(s, 4)] for x, y, s in cs],
                     "selected": selected.get(f, -1)}
            for f, cs in sorted(candidates.items()) if cs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))


def run(ctx: StageContext) -> ShuttleTrack:
    meta = ctx.require_meta()
    if demo_mode(ctx) and (players := ctx.players) is not None and players.frames:
        from ...pipeline.demo import demo_shuttle

        return demo_shuttle(meta, players)

    cfg = dict(ctx.config.get("shuttle") or {})
    from .detector import DetectorConfig, HeatmapShuttleDetector, find_weights

    weights = find_weights(cfg.get("weights"))
    if weights is None:
        msg = ("no shuttle model weights found; put the checkpoint in weights/shuttle/ "
               "or set RALLYIQ_SHUTTLE_WEIGHTS (see rallyiq/perception/shuttle/README.md)")
        if cfg.get("require_weights"):
            raise RuntimeError(msg)
        log.warning(msg)
        return ShuttleTrack(video_id=meta.video_id, detector=None, points=[])

    det_cfg = DetectorConfig()
    for k in ("device", "batch_size", "peak_threshold", "top_k", "target_fps", "ensemble"):
        if k in cfg:
            setattr(det_cfg, k, cfg[k])
    detector = HeatmapShuttleDetector(weights, det_cfg)
    segments = segments_for(meta, ctx.calibration, cfg.get("roi", "off"))
    candidates, size = detector.detect(
        ctx.video_path, meta.fps, meta.frame_count, segments, progress=lambda p: ctx.report(0.95 * p)
    )
    track, selected = build_track(meta, candidates, size[0], size[1], detector.name,
                                  tracker_config(cfg.get("tracker")))
    save_candidates(Path(ctx.work_dir) / CANDIDATES_FILE, candidates, selected, detector.name, size)
    ctx.report(1.0)
    return track
