"""End-to-end player tracking over a video, independent of the pipeline plumbing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .appearance import colour_histogram
from .filtering import FilterConfig, build_candidates
from .geometry import image_net_y
from .models import PlayerFrameResult
from .pose import PoseEstimator
from .postprocess import fill_short_gaps
from .tracker import SinglesTracker, TrackerConfig

HomographyFn = Callable[[int], "np.ndarray | None"]


@dataclass
class PlayerTrackingConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    max_interp_gap: int = 6
    stride: int = 1  # run pose on every n-th frame (2 halves cost on 60 fps)


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


def track_players(
    frames: Iterable[tuple[int, float, np.ndarray]],
    estimator: PoseEstimator,
    homography_for: HomographyFn | None = None,
    cfg: PlayerTrackingConfig | None = None,
) -> list[PlayerFrameResult]:
    """Detect, filter and track the two singles players on every given frame.

    ``homography_for(frame)`` returns the 3x3 image->court homography valid for
    that frame, or None when calibration is unavailable (e.g. right after a
    camera cut). Without it, output stays in pixels and ``foot_court`` /
    ``on_court`` are None.
    """
    cfg = cfg or PlayerTrackingConfig()
    tracker = SinglesTracker(cfg.tracker)
    results: list[PlayerFrameResult] = []
    for frame_idx, t_ms, img in frames:
        H = homography_for(frame_idx) if homography_for else None
        dets = estimator(img)
        cands = build_candidates(dets, H, (img.shape[1], img.shape[0]), cfg.filter)
        for c in cands:
            c.appearance = colour_histogram(img, c.det)
        net_y = image_net_y(H) if H is not None else None
        players = tracker.update(frame_idx, cands, calibrated=H is not None, net_y_px=net_y)
        results.append(PlayerFrameResult(frame_idx, t_ms, players))
    gap = cfg.max_interp_gap // max(cfg.stride, 1)
    return fill_short_gaps(results, gap) if gap > 0 else results
