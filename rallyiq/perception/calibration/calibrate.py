"""Single-frame court calibration: automatic, and from user-clicked points."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Sequence

import cv2
import numpy as np

from . import court_model as cm
from .homography import (
    METRIC_TOL,
    Score,
    metric_error,
    canonicalize,
    hypotheses,
    project,
    refine,
    residual_rms,
    score_one,
    verified_keypoints,
)
from .lines import LineEvidence, detect_lines, line_evidence, sort_family, split_families

ALGORITHM_VERSION = "court-calib/0.1.0"
WORK_WIDTH = 1280

Status = Literal["ok", "needs_review", "undeterminable"]
Method = Literal["auto", "manual", "tracked"]


@dataclass
class CalibrationMetrics:
    inlier_ratio: float = 0.0  # visible model line samples lying on detected line pixels
    visible_fraction: float = 0.0  # share of the court's painted lines inside the frame
    verified_keypoints: int = 0  # intersections independently confirmed by the image
    visible_keypoints: int = 0
    residual_px: float = float("nan")  # RMS line residual, full-resolution pixels
    ambiguity: float = 0.0  # 0..1; near 1 means a different line assignment explains the image as well
    lines_detected: int = 0
    other_courts: int = 0  # neighbouring courts that also fit (hall footage)
    chance_ratio: float = 0.0  # inlier ratio a random placement would get (line-pixel density)


@dataclass
class FrameCalibration:
    """Calibration of one frame.

    image_from_court maps court metres (x, y, 1) to image pixels; court_from_image
    is its inverse (the project-wide homography direction).
    """

    image_size: tuple[int, int]  # (width, height)
    image_from_court: np.ndarray | None
    confidence: float
    status: Status
    method: Method
    metrics: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    symmetry: str = "identity"  # which court symmetry was applied to reach the canonical labelling
    reason: str = ""  # human-readable explanation when not ok
    # Other whole courts that fit (neighbouring courts in a hall), image_from_court each.
    other_courts: list = field(default_factory=list)
    version: str = ALGORITHM_VERSION

    @property
    def ok(self) -> bool:
        return self.image_from_court is not None and self.status != "undeterminable"

    @property
    def court_from_image(self) -> np.ndarray | None:
        if self.image_from_court is None:
            return None
        Hi = np.linalg.inv(self.image_from_court)
        return Hi / Hi[2, 2]

    def to_court(self, image_pts) -> np.ndarray:
        return project(self.court_from_image, np.asarray(image_pts, float).reshape(-1, 2))[0]

    def to_image(self, court_pts: np.ndarray) -> np.ndarray:
        return project(self.image_from_court, np.asarray(court_pts, float).reshape(-1, 2))[0]

    def other_court_corners(self) -> list[list[list[float]]]:
        """Image corners (near-left, near-right, far-right, far-left) of each neighbouring court."""
        corners = np.array(list(cm.CORNERS.values()))
        return [project(H, corners)[0].tolist() for H in self.other_courts]

    def image_keypoints(self) -> dict[str, list[float]]:
        """Projected court keypoints that fall inside the frame."""
        if self.image_from_court is None:
            return {}
        w, h = self.image_size
        out = {}
        for kp in cm.keypoints():
            (x, y), = self.to_image(np.array([kp.xy]))
            if 0 <= x < w and 0 <= y < h:
                out[kp.name] = [float(x), float(y)]
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_from_court"] = None if self.image_from_court is None else self.image_from_court.tolist()
        cfi = self.court_from_image
        d["court_from_image"] = None if cfi is None else cfi.tolist()
        d["other_courts"] = [H.tolist() for H in self.other_courts]
        return d


def _status(conf: float) -> Status:
    if conf >= 0.6:
        return "ok"
    if conf >= 0.3:
        return "needs_review"
    return "undeterminable"


def _smooth(x: float, lo: float, hi: float) -> float:
    t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3 - 2 * t)


def confidence_from(m: CalibrationMetrics) -> float:
    """Combine evidence into a 0..1 confidence.

    - how much of the visible model is backed by line pixels, beyond chance,
    - how many intersections were independently verified (beyond the four
      that seeded the hypothesis),
    - whether a different court placement explains the image almost as well,
    - how much of the court is in frame.
    """
    # Inlier ratio above what a random placement would get on this image, so
    # busy or noisy frames full of bright edges do not look like a good fit.
    lift = (m.inlier_ratio - m.chance_ratio) / max(1e-6, 1.0 - m.chance_ratio)
    # When most of the frame is near a "line" pixel, the evidence says nothing.
    fit = _smooth(lift, 0.4, 0.8) * (1.0 - _smooth(m.chance_ratio, 0.35, 0.6))
    kps = _smooth(m.verified_keypoints, 4, 10)
    unique = 1.0 - 0.7 * _smooth(m.ambiguity, 0.8, 1.0)
    seen = _smooth(m.visible_fraction, 0.15, 0.45)
    return float(np.clip(fit * (0.4 + 0.6 * kps) * unique * (0.5 + 0.5 * seen), 0.0, 1.0))


def _work_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    s = min(1.0, WORK_WIDTH / w)
    if s < 1.0:
        image = cv2.resize(image, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return image, s


def _tol(ev: LineEvidence) -> float:
    h, w = ev.shape
    return max(2.5, 0.004 * float(np.hypot(w, h)))


def _scale_H(H_work: np.ndarray, s: float) -> np.ndarray:
    S_inv = np.diag([1 / s, 1 / s, 1.0])
    H = S_inv @ H_work
    return H / H[2, 2]


def _metrics(H: np.ndarray, ev: LineEvidence, tol: float, s: float) -> tuple[CalibrationMetrics, Score]:
    sc = score_one(H, ev, tol)
    ver, vis = verified_keypoints(H, ev, tol)
    return (
        CalibrationMetrics(
            inlier_ratio=sc.inlier_ratio,
            visible_fraction=sc.visible_fraction,
            verified_keypoints=ver,
            visible_keypoints=vis,
            residual_px=residual_rms(H, ev, tol) / s,
            chance_ratio=float(np.mean(ev.dist <= tol)),
        ),
        sc,
    )


def _failed(image_size, method: Method, reason: str, metrics=None) -> FrameCalibration:
    return FrameCalibration(
        image_size=image_size,
        image_from_court=None,
        confidence=0.0,
        status="undeterminable",
        method=method,
        metrics=metrics or CalibrationMetrics(),
        reason=reason,
    )


def _rival_strength(best: np.ndarray, other: np.ndarray, ev: LineEvidence, tol: float) -> float:
    """How well a clearly different placement explains the image, relative to the best.

    Placements count as different when a good part of the visible court lands more
    than 4 x tol apart in the image under every court symmetry (smaller far
    court wobble is a precision issue, not a different interpretation). Both
    are then scored with a tight tolerance. Returns rival / best; near or
    above 1 means the image cannot tell them apart. 0 when not different.
    """
    pts, _ = cm.sample_points(0.25)
    pb, wb = project(best, pts)
    h, w = ev.shape
    inside = (wb > 0) & (pb[:, 0] >= 0) & (pb[:, 0] < w) & (pb[:, 1] >= 0) & (pb[:, 1] < h)
    if inside.sum() < 10:
        return 0.0
    pts, pb = pts[inside], pb[inside]
    frac = 1.0
    for S in cm.SYMMETRIES.values():
        po, _ = project(other @ S, pts)
        gap = np.nan_to_num(np.linalg.norm(pb - po, axis=1), nan=1e9)
        frac = min(frac, float(np.mean(gap > 4 * tol)))
    if frac < 0.1:
        return 0.0
    sb = score_one(best, ev, 0.5 * tol).value
    so = score_one(other, ev, 0.5 * tol).value
    return float(max(so, 0.0) / sb) if sb > 0 else 1.0


def _overlap(Ha: np.ndarray, Hb: np.ndarray) -> float:
    """IoU of two projected outer courts."""
    corners = np.array(list(cm.CORNERS.values()))
    qa = project(Ha, corners)[0].astype(np.float32)
    qb = project(Hb, corners)[0].astype(np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(qa, qb)
    except cv2.error:
        return 0.0
    aa, ab = abs(cv2.contourArea(qa)), abs(cv2.contourArea(qb))
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def calibrate_frame(
    image: np.ndarray,
    previous: np.ndarray | None = None,
    extra_hypotheses: Sequence[np.ndarray] = (),
    method: Method = "auto",
) -> FrameCalibration:
    """Automatically calibrate one BGR frame.

    previous: an earlier image_from_court (full resolution) used to keep the
    court labelling consistent and as an extra starting hypothesis.
    extra_hypotheses: more full-resolution image_from_court guesses to try,
    e.g. the previous calibration warped by estimated camera motion.
    """
    h0, w0 = image.shape[:2]
    work, s = _work_image(image)
    ev = line_evidence(work)
    tol = _tol(ev)
    S = np.diag([s, s, 1.0])

    lines = detect_lines(ev.mask)
    centre = np.array([work.shape[1] / 2, work.shape[0] / 2])
    fam_a, fam_b = split_families(lines)
    fam_a = sort_family(fam_a[:8], centre)
    fam_b = sort_family(fam_b[:8], centre)

    seeds = hypotheses(fam_a, fam_b, ev, tol)
    for Hx in ([previous] if previous is not None else []) + list(extra_hypotheses):
        seeds.insert(0, S @ Hx)
    if not seeds:
        return _failed((w0, h0), method, "court lines not found", CalibrationMetrics(lines_detected=len(lines)))

    refined = []
    wh = (work.shape[1], work.shape[0])
    for H in seeds:
        Hr = refine(H, ev, tol)
        s_raw, s_ref = score_one(H, ev, tol), score_one(Hr, ev, tol)
        use_ref = s_ref.value >= s_raw.value and metric_error(Hr[None], *wh)[0][0] < METRIC_TOL
        refined.append((Hr, s_ref) if use_ref else (H, s_raw))
    refined.sort(key=lambda t: -t[1].value)
    # value = matched cells - half the unmatched ones; at <= 0 the best placement
    # contradicts the image at least as much as it explains it (and the ratios
    # below would divide by zero).
    if refined[0][1].matched < 30 or refined[0][1].value <= 0:
        return _failed((w0, h0), method, "no court placement fits the detected lines")

    # Neighbouring courts in a hall fit just as well but do not overlap the
    # best one. That is a question of which court, not of this court's
    # geometry: take the most central one and do not count it as ambiguity.
    top_value = refined[0][1].value
    whole_courts = [refined[0]]
    for Hc, sc in refined[1:]:
        if sc.value < 0.75 * top_value:
            break
        if all(_overlap(Hc, Ho) < 0.2 for Ho, _ in whole_courts):
            whole_courts.append((Hc, sc))
    centre = np.array([wh[0] / 2, wh[1] / 2])
    diag = float(np.hypot(*wh))

    def off_centre(item):
        c = project(item[0], np.array([[cm.COURT_WIDTH / 2, cm.COURT_LENGTH / 2]]))[0][0]
        return float(np.linalg.norm(c - centre)) / diag - 0.25 * item[1].value / top_value

    best_H, best_s = min(whole_courts, key=off_centre)

    # Rivals: overlapping placements that claim different lines somewhere.
    # Compare them only where they disagree, with a tight tolerance; switch to
    # a rival that explains that part better, then report the strongest
    # remaining one as ambiguity.
    rivals = [(Hc, sc) for Hc, sc in refined if sc.value >= 0.6 * best_s.value and _overlap(Hc, best_H) >= 0.2]
    rival = 0.0
    for _ in range(3):
        strengths = [(_rival_strength(best_H, Hc, ev, tol), Hc, sc) for Hc, sc in rivals if Hc is not best_H]
        if not strengths:
            break
        strength, Hc, sc = max(strengths, key=lambda t: t[0])
        rival = strength
        if strength > 1.1 and sc.value >= 0.9 * best_s.value:
            best_H, best_s = Hc, sc
            continue
        break

    m, _ = _metrics(best_H, ev, tol, s)
    m.ambiguity = min(1.0, rival)
    m.lines_detected = len(lines)
    m.other_courts = len(whole_courts) - 1

    H_full = _scale_H(best_H, s)
    H_full, sym = canonicalize(H_full, previous)
    conf = confidence_from(m)
    status = _status(conf)
    reason = ""
    if status != "ok":
        reasons = []
        if m.inlier_ratio < 0.6:
            reasons.append("court lines poorly matched")
        if m.verified_keypoints < 6:
            reasons.append("few line intersections verified")
        if m.ambiguity > 0.85:
            reasons.append("another court placement fits almost as well")
        if m.visible_fraction < 0.3:
            reasons.append("little of the court is in frame")
        reason = "; ".join(reasons)
    return FrameCalibration(
        image_size=(w0, h0),
        image_from_court=H_full,
        confidence=conf,
        status=status,
        method=method,
        metrics=m,
        symmetry=sym,
        reason=reason,
        other_courts=[_scale_H(Ho, s) for Ho, _ in whole_courts if Ho is not best_H],
    )


def calibrate_from_points(
    image_points: Sequence[Sequence[float]],
    court_points: Sequence[Sequence[float]] | Sequence[str] = tuple(cm.CORNERS),
    image: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    refine_with_lines: bool = True,
) -> FrameCalibration:
    """Manual fallback: the user clicks >= 4 known court points.

    court_points are court coordinates in metres, or names: corner names from
    court_model.CORNERS (default order near_left, near_right, far_right,
    far_left) or keypoint names from court_model.keypoints().

    With an image, the clicked solution is snapped to the visible lines when
    that improves the fit without moving any clicked point by more than a
    small margin, and quality metrics are reported. A manual calibration is
    user-confirmed, so it is never marked undeterminable.
    """
    lookup = dict(cm.CORNERS) | {k.name: k.xy for k in cm.keypoints()}
    court = np.array([lookup[p] if isinstance(p, str) else p for p in court_points], float)
    img = np.asarray(image_points, float)
    if len(img) < 4 or len(img) != len(court):
        raise ValueError("need at least 4 matching image/court points")
    H, _ = cv2.findHomography(court, img, 0)
    if H is None:
        raise ValueError("points are degenerate (collinear or repeated)")
    H = H / H[2, 2]
    if image is None:
        if image_size is None:
            raise ValueError("pass image or image_size")
        return FrameCalibration(
            image_size=tuple(image_size), image_from_court=H, confidence=1.0, status="ok", method="manual"
        )

    h0, w0 = image.shape[:2]
    work, s = _work_image(image)
    ev = line_evidence(work)
    tol = _tol(ev)
    S = np.diag([s, s, 1.0])
    Hw = S @ H
    m, sc = _metrics(Hw, ev, tol, s)
    if refine_with_lines:
        Hr = refine(Hw, ev, tol)
        mr, scr = _metrics(Hr, ev, tol, s)
        moved = np.linalg.norm(project(Hr, court)[0] - project(Hw, court)[0], axis=1).max() / s
        if scr.value > sc.value and moved <= max(8.0, 0.01 * np.hypot(w0, h0)):
            Hw, m = Hr, mr
    conf = max(0.6, confidence_from(m))  # user-confirmed geometry
    return FrameCalibration(
        image_size=(w0, h0),
        image_from_court=_scale_H(Hw, s),
        confidence=conf,
        status="ok",
        method="manual",
        metrics=m,
    )
