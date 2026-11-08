"""End-to-end player tracking over a video, independent of the pipeline plumbing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ..target_court import CourtRegion

from .appearance import colour_histogram
from .filtering import FilterConfig, build_candidates, is_seated
from .geometry import far_court_roi, image_net_y
from .match_format import FORMATS, FormatConfig, FormatDecision, decide_format
from .models import PersonDetection, PlayerFrameResult
from .pose import PoseEstimator
from .postprocess import apply_relabels, drop_offcourt_runs, fill_short_gaps
from .sides import SideConfig, stabilise_sides
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
    sides: SideConfig = field(default_factory=SideConfig)
    # Second pose pass on an upscaled crop of the far half of the court, for
    # estimators that support it: "auto" only when the far player would be
    # small in the model input, "on" on every calibrated frame, "off" never.
    far_boost: str = "auto"
    far_boost_max_px: float = 110.0  # "auto": boost when the far player is shorter than this in the model input
    # Without calibration (camera cut, or a calibration the players contradicted)
    # the far player is looked for in the floor band around the nearest standing
    # player (see ``far_roi_from_players``), only on frames where fewer players than
    # the format needs were found. People it adds must pass ``plausible_far_player``.
    far_ratio: float = 0.35  # expected far/near height, for the "auto" size check
    far_band_above: float = 0.6  # crop top, in near-player heights above their head
    far_feet_above: float = 0.25  # far player's feet at most this far above the near head
    far_max_height_ratio: float = 0.75


class PlayerTrackingOutput(list):
    """``list[PlayerFrameResult]`` plus the singles/doubles decision."""

    match_format: str = "singles"
    format_confidence: float | None = None  # None when the format was given, not detected
    format_decision: FormatDecision | None = None
    side_switches: list[int] | None = None  # frames where the teams really changed ends


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
        roi = None
        if H is not None and cfg.far_boost != "off" and getattr(estimator, "supports_far_roi", False):
            roi = far_court_roi(H, (img.shape[1], img.shape[0]))
            if roi is not None and cfg.far_boost == "auto":
                need = estimator.far_player_px(img.shape, roi)  # far player height in the model input
                roi = roi if need is not None and need < cfg.far_boost_max_px else None
        dets = estimator(img, far_roi=roi) if roi is not None else estimator(img)
        if H is None and cfg.far_boost != "off" and hasattr(estimator, "detect_roi"):
            want = 4 if (tracker is not None and out.match_format == "doubles") or cfg.match_format == "doubles" else 2
            found = far_roi_from_players(dets, (img.shape[1], img.shape[0]), want, cfg)
            roi, near = found if found else (None, None)
            if roi is not None and cfg.far_boost == "auto":
                need = estimator.far_player_px(img.shape, roi)
                roi = roi if need is not None and need < cfg.far_boost_max_px else None
            if roi is not None:
                before = {id(d) for d in dets}
                dets = [d for d in estimator.detect_roi(img, roi, dets)
                        if id(d) in before or plausible_far_player(d, near, roi, cfg)]
        region = region_for(frame_idx) if region_for else None
        cands = build_candidates(dets, H, (img.shape[1], img.shape[0]), cfg.filter, region)
        for c in cands:
            c.appearance = colour_histogram(img, c.det)
        net_y = image_net_y(H) if H is not None else None
        pending.append((frame_idx, t_ms, cands, H is not None, net_y))
        if tracker is not None or len(pending) >= cfg.format_probe_frames:
            flush()
    flush()
    if tracker is not None:
        apply_relabels(out, tracker.relabels)
    drop_offcourt_runs(out, cfg.filter.play_area)
    out.side_switches = stabilise_sides(out, cfg.sides)
    gap = cfg.max_interp_gap // max(cfg.stride, 1)
    if gap > 0:
        fill_short_gaps(out, gap)
    return out


def _standing(dets: list[PersonDetection], conf: float = 0.3, min_kps: int = 6) -> list[PersonDetection]:
    return [d for d in dets if d.score >= conf and (d.keypoints[:, 2] >= conf).sum() >= min_kps
            and not is_seated(d, conf)]


def far_roi_from_players(dets: list[PersonDetection], image_size: tuple[int, int], want: int,
                         cfg: PlayerTrackingConfig) -> tuple[tuple[int, int, int, int, float], PersonDetection] | None:
    """Box where the far player should be when there is no homography, in the
    ``far_court_roi`` format ``(x1, y1, x2, y2, far_player_px)``, and the near
    player it was placed from.

    The tallest standing person is taken as the near player. Their feet and head
    bound the floor band where the far player stands: on a low camera the far
    player's feet are level with the near player's upper body, on a raised one a
    little above their head. The box spans from ``far_band_above`` near-player
    heights above their head down to just below their feet, as wide as it is
    tall, centred on them. None when enough players were already found, nobody
    stands in view, or the box would not be smaller than the frame.
    """
    people = _standing(dets)
    if not people or len(people) >= want:
        return None
    near = max(people, key=lambda d: d.height)
    w, h = image_size
    hp = float(near.height)
    if hp <= 0:
        return None
    y1 = max(float(near.bbox[1]) - cfg.far_band_above * hp, 0.0)
    y2 = min(float(near.bbox[3]) + 0.1 * hp, float(h))
    side = y2 - y1
    if side >= 0.9 * min(w, h):
        return None  # no upscaling to gain
    cx = 0.5 * float(near.bbox[0] + near.bbox[2])
    x1 = min(max(cx - side / 2, 0.0), max(w - side, 0.0))
    x2 = min(x1 + side, float(w))
    return (int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2)), cfg.far_ratio * hp), near


def plausible_far_player(d: PersonDetection, near: PersonDetection, roi: tuple, cfg: PlayerTrackingConfig) -> bool:
    """A person the uncalibrated far pass found that can be the far player:
    not cut off by the crop's top edge (a curtain or light fitted as a tall box),
    clearly smaller than the near player, and feet in the floor band between the
    near player's head and feet."""
    hp = float(near.height)
    if float(d.bbox[1]) <= roi[1] + 2:
        return False
    if float(d.height) > cfg.far_max_height_ratio * hp:
        return False
    foot = float(d.bbox[3])
    return float(near.bbox[1]) - cfg.far_feet_above * hp <= foot <= float(near.bbox[3])


def _decided(d: FormatDecision) -> tuple[str, FormatDecision]:
    return d.match_format, d
