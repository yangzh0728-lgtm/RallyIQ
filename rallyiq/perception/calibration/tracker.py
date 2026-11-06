"""Video-level calibration: follow the court, detect camera moves and cuts, recalibrate.

The tracker looks at a sampled frame every ``check_every_ms``. While the
current calibration still lines up with the court lines it only refines it
(absorbing hand-held jitter). When the lines stop lining up, it estimates the
camera motion from image features and recalibrates, starting a new segment.
A scene cut (e.g. a broadcast replay or a close-up) also starts a new
segment; frames where no court can be found get an ``undeterminable``
segment, and the tracker keeps retrying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import cv2
import numpy as np

from . import court_model as cm
from .calibrate import (
    FrameCalibration,
    _failed,
    _metrics,
    _scale_H,
    _status,
    _tol,
    _work_image,
    calibrate_frame,
    confidence_from,
)
from .homography import METRIC_TOL, metric_error, project, refine, score_one
from .lines import line_evidence


@dataclass
class TrackerConfig:
    check_every_ms: float = 500.0
    # Corner shift (fraction of the image diagonal) that counts as a camera move.
    move_frac: float = 0.006
    # Re-locate when the inlier ratio falls below this share of the segment's reference.
    drop_ratio: float = 0.75
    # Grey-histogram correlation below this between checks means a scene cut.
    cut_correlation: float = 0.6
    min_feature_inliers: int = 25


@dataclass
class Segment:
    start_frame: int
    end_frame: int
    calibration: FrameCalibration
    reason: str  # initial | camera_moved | scene_cut | recovered | lost
    notes: list[str] = field(default_factory=list)


def _signature(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, None).flatten()


def _camera_motion(prev: np.ndarray, cur: np.ndarray, min_inliers: int) -> np.ndarray | None:
    """Image-to-image homography prev -> cur from ORB features (players move, the hall does not)."""
    orb = cv2.ORB_create(1500)
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < min_inliers or len(k1) < min_inliers:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(matches) < min_inliers:
        return None
    p0 = np.float32([k0[m.queryIdx].pt for m in matches])
    p1 = np.float32([k1[m.trainIdx].pt for m in matches])
    G, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if G is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    return G


def recheck(image: np.ndarray, H_full: np.ndarray, do_refine: bool = True) -> FrameCalibration:
    """Re-evaluate (and optionally refine) a known calibration on a new frame."""
    h0, w0 = image.shape[:2]
    work, s = _work_image(image)
    ev = line_evidence(work)
    tol = _tol(ev)
    S = np.diag([s, s, 1.0])
    Hw = S @ H_full
    Hw = Hw / Hw[2, 2]
    if do_refine:
        Hr = refine(Hw, ev, tol)
        wh = (work.shape[1], work.shape[0])
        if score_one(Hr, ev, tol).value >= score_one(Hw, ev, tol).value and metric_error(Hr[None], *wh)[0][0] < METRIC_TOL:
            Hw = Hr
    m, sc = _metrics(Hw, ev, tol, s)
    if sc.matched < 30:
        return _failed((w0, h0), "auto", "court lines no longer match", m)
    conf = confidence_from(m)
    return FrameCalibration(
        image_size=(w0, h0),
        image_from_court=_scale_H(Hw, s),
        confidence=conf,
        status=_status(conf),
        method="tracked",
        metrics=m,
    )


def _corner_shift(Ha: np.ndarray, Hb: np.ndarray, size: tuple[int, int]) -> float:
    """Largest movement (px) of court points that are inside the frame."""
    w, h = size
    pts = np.array([k.xy for k in cm.keypoints()])
    a, wa = project(Ha, pts)
    b, _ = project(Hb, pts)
    inside = (wa > 0) & (a[:, 0] >= 0) & (a[:, 0] < w) & (a[:, 1] >= 0) & (a[:, 1] < h)
    if not inside.any():
        inside = np.ones(len(pts), bool)
    return float(np.nanmax(np.linalg.norm(a[inside] - b[inside], axis=1)))


class CourtTracker:
    """Feed it sampled frames in order with :meth:`update`, then call :meth:`finish`."""

    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.segments: list[Segment] = []
        self._prev_frame: np.ndarray | None = None
        self._prev_sig: np.ndarray | None = None
        self._ref_inlier = 0.0
        self._last_good: np.ndarray | None = None  # last good image_from_court, for label continuity

    @property
    def current(self) -> Segment | None:
        return self.segments[-1] if self.segments else None

    def _start(self, frame_idx: int, cal: FrameCalibration, reason: str, notes: list[str] | None = None) -> None:
        if self.segments:
            self.segments[-1].end_frame = max(self.segments[-1].start_frame, frame_idx - 1)
        self.segments.append(Segment(frame_idx, frame_idx, cal, reason, list(notes or [])))
        if cal.ok:
            self._ref_inlier = cal.metrics.inlier_ratio
            self._last_good = cal.image_from_court

    def update(self, frame_idx: int, frame: np.ndarray) -> FrameCalibration:
        sig = _signature(frame)
        cut = self._prev_sig is not None and cv2.compareHist(self._prev_sig, sig, cv2.HISTCMP_CORREL) < self.cfg.cut_correlation
        cur = self.current
        try:
            if cur is None:
                self._start(frame_idx, calibrate_frame(frame), "initial")
            elif cut:
                # After a cut the old labelling may not apply; calibrate from scratch.
                cal = calibrate_frame(frame)
                if cal.ok and self._last_good is not None:
                    cal = calibrate_frame(frame, previous=self._last_good)
                self._start(frame_idx, cal, "scene_cut")
            elif not cur.calibration.ok:
                cal = calibrate_frame(frame, previous=self._last_good)
                if cal.ok:
                    self._start(frame_idx, cal, "recovered")
                else:
                    cur.end_frame = frame_idx
            else:
                self._follow(frame_idx, frame, cur)
        finally:
            self._prev_frame = frame
            self._prev_sig = sig
        return self.current.calibration

    def _follow(self, frame_idx: int, frame: np.ndarray, cur: Segment) -> None:
        H = cur.calibration.image_from_court
        w, h = cur.calibration.image_size
        move_px = self.cfg.move_frac * float(np.hypot(w, h))
        chk = recheck(frame, H)
        if chk.ok and chk.metrics.inlier_ratio >= self.cfg.drop_ratio * self._ref_inlier:
            if _corner_shift(H, chk.image_from_court, (w, h)) > move_px:
                chk.method = "auto"
                self._start(frame_idx, chk, "camera_moved", ["followed a gradual camera move"])
            else:
                cur.end_frame = frame_idx
            return

        # Lines no longer line up: the camera moved. Predict where the court went.
        extra = []
        G = _camera_motion(self._prev_frame, frame, self.cfg.min_feature_inliers) if self._prev_frame is not None else None
        if G is not None:
            Hp = G @ H
            extra.append(Hp / Hp[2, 2])
        cal = calibrate_frame(frame, previous=H, extra_hypotheses=extra)
        if cal.ok:
            self._start(frame_idx, cal, "camera_moved")
        else:
            self._start(frame_idx, cal, "lost", [cal.reason] if cal.reason else [])

    def finish(self, last_frame: int) -> list[Segment]:
        if self.segments:
            self.segments[-1].end_frame = max(self.segments[-1].start_frame, last_frame)
        return self.segments


def read_sampled_frames(path: str, every_ms: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, fps, frame) roughly every ``every_ms`` of video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_ms * fps / 1000.0)))
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def calibrate_video(
    path: str,
    config: TrackerConfig | None = None,
    frame_count: int | None = None,
    report: Callable[[float], None] = lambda p: None,
) -> list[Segment]:
    tracker = CourtTracker(config)
    last = 0
    for idx, _fps, frame in read_sampled_frames(path, tracker.cfg.check_every_ms):
        tracker.update(idx, frame)
        last = idx
        if frame_count:
            report(min(1.0, idx / frame_count))
    end = (frame_count - 1) if frame_count else last
    return tracker.finish(max(end, last))
