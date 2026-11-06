"""End-to-end player tracking over a video, independent of the pipeline plumbing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ..target_court import CourtRegion

from .appearance import colour_histogram
from .filtering import FilterConfig, build_candidates
from .geometry import image_net_y
from .match_format import FORMATS, FormatConfig, FormatDecision, decide_format
from .models import PlayerFrameResult
from .pose import PoseEstimator
from .postprocess import drop_offcourt_runs, fill_short_gaps
from .tracker import DoublesTracker, PlayerTracker, SinglesTracker, TrackerConfig

HomographyFn = Callable[[int], "np.ndarray | None"]
RegionFn = Callable[[int], "CourtRegion | None"]


@dataclass
class PlayerTrackingConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    max_interp_gap: int = 6
    stride: int = 1  # run pose on every n-th frame (2 halves cost on 60 fps)
    # "auto" decides from the first ``format_probe_frames`` processed frames.
    match_format: str = "auto"
    format_probe_frames: int = 90
    format: FormatConfig = field(default_factory=FormatConfig)


class PlayerTrackingOutput(list):
    """``list[PlayerFrameResult]`` plus the singles/doubles decision."""

    match_format: str = "singles"
    format_confidence: float | None = None  # None when the format was given, not detected
    format_decision: FormatDecision | None = None


def iter_video(path: str, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_ms, image_bgr) from the original video timeline."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not t_ms and i:
                    t_ms = i * 1000.0 / fps
                yield i, float(t_ms), img
            i += 1
    finally:
        cap.release()


def video_fps(path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def make_tracker(match_format: str, cfg: TrackerConfig) -> PlayerTracker:
    return DoublesTracker(cfg) if match_format == "doubles" else SinglesTracker(cfg)


def track_players(
    frames: Iterable[tuple[int, float, np.ndarray]],
    estimator: PoseEstimator,
    homography_for: HomographyFn | None = None,
    cfg: PlayerTrackingConfig | None = None,
    region_for: RegionFn | None = None,
) -> PlayerTrackingOutput:
    """Detect, filter and track the players (2 in singles, 4 in doubles) on every given frame.

    ``homography_for(frame)`` returns the 3x3 image->court homography valid for
    that frame, or None when calibration is unavailable (e.g. right after a
    camera cut). Without it, output stays in pixels and ``foot_court`` /
    ``on_court`` are None. ``region_for(frame)`` adds the other courts in view so
    their players are rejected (see ``target_court``).

    With ``cfg.match_format == "auto"`` the first ``format_probe_frames``
    processed frames are buffered (candidates only, not images), the format is
    decided from them, and tracking then replays the buffer.
    """
    cfg = cfg or PlayerTrackingConfig()
    if cfg.match_format not in (*FORMATS, "auto"):
        raise ValueError(f"match_format must be auto, singles or doubles, not {cfg.match_format!r}")
    out = PlayerTrackingOutput()
    tracker: PlayerTracker | None = None
    pending: list[tuple[int, float, list, bool, float | None]] = []

    def start(fmt: str, decision: FormatDecision | None) -> PlayerTracker:
        out.match_format = fmt
        out.format_decision = decision
        out.format_confidence = decision.confidence if decision else None
        return make_tracker(fmt, cfg.tracker)

    if cfg.match_format != "auto":
        tracker = start(cfg.match_format, None)

    def flush() -> None:
        nonlocal tracker
        if tracker is None:
            tracker = start(*_decided(decide_format([p[2] for p in pending], cfg.format)))
        for frame_idx, t_ms, cands, calibrated, net_y in pending:
            players = tracker.update(frame_idx, cands, calibrated=calibrated, net_y_px=net_y)
            out.append(PlayerFrameResult(frame_idx, t_ms, players))
        pending.clear()

    for frame_idx, t_ms, img in frames:
        H = homography_for(frame_idx) if homography_for else None
        dets = estimator(img)
        region = region_for(frame_idx) if region_for else None
        cands = build_candidates(dets, H, (img.shape[1], img.shape[0]), cfg.filter, region)
        for c in cands:
            c.appearance = colour_histogram(img, c.det)
        net_y = image_net_y(H) if H is not None else None
        pending.append((frame_idx, t_ms, cands, H is not None, net_y))
        if tracker is not None or len(pending) >= cfg.format_probe_frames:
            flush()
    flush()
    drop_offcourt_runs(out, cfg.filter.play_area)
    gap = cfg.max_interp_gap // max(cfg.stride, 1)
    if gap > 0:
        fill_short_gaps(out, gap)
    return out


def _decided(d: FormatDecision) -> tuple[str, FormatDecision]:
    return d.match_format, d
