"""Re-choose the target court from where the players actually stand.

Calibration picks the target court from court lines alone (most central of the
equally good whole courts). In a hall that can be the wrong one: the players we
care about are on a neighbouring court that calibration also found and listed in
``other_courts``. Everything downstream then filters our players out as
"off-court" while the calibration still says ``ok`` (seen on sam's DJI_0009).

The players stage fixes that before tracking: it runs the pose model on a few
frames spread over each camera segment and scores every candidate court (target
plus ``other_courts``) by how often it has someone standing on each half, which is
what a court with a match on it looks like. A clearly better candidate becomes
the target; the old target moves to ``other_courts`` so its people keep being
rejected. Manual calibrations are never changed.

Only courts that calibration found can be chosen. When the right court is not
among them but people are clearly playing while the calibrated court stays empty,
the segment is marked ``undeterminable`` instead (players are then tracked
without it, the shuttle court lock falls back to the players' span, and the UI
asks for the 4 corners).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..contracts.geometry import COURT_CORNERS_M, fit_homography
from ..contracts.models import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, CalibrationSegment, CourtCalibration
from .players.filtering import expected_pixel_height, is_seated
from .players.geometry import apply_homography, foot_point
from .players.models import PersonDetection

W, L = COURT_WIDTH_M, COURT_LENGTH_M


@dataclass
class CourtChoiceConfig:
    frames_per_segment: int = 12
    min_det_score: float = 0.3
    min_keypoint_conf: float = 0.3
    min_visible_keypoints: int = 6
    # A player may stand a little past the lines while still "on" this court.
    margin_m: float = 0.5
    height_ratio_range: tuple[float, float] = (0.5, 2.0)
    # Switch only when another court is clearly better: per-frame score is the number
    # of occupied halves (0..2), averaged over the sampled frames.
    min_score: float = 1.0
    min_gain: float = 0.5
    # Contradiction check: when at least two standing people are visible in this share
    # of the sampled frames but the chosen court is (almost) never occupied, the
    # calibration is wrong whatever its line fit says.
    people_frames_min: float = 0.5
    min_people_frames: int = 4
    empty_court_max: float = 0.25


@dataclass
class CourtScore:
    index: int  # 0 = calibration's target, i >= 1 = other_courts[i - 1]
    corners: list[list[float]] | None
    score: float


def _court_homographies(seg: CalibrationSegment) -> list[tuple[np.ndarray, list[list[float]] | None]]:
    out: list[tuple[np.ndarray, list[list[float]] | None]] = [(np.asarray(seg.homography, dtype=np.float64),
                                                                seg.image_corners)]
    for corners in seg.other_courts:
        try:
            out.append((np.asarray(fit_homography(corners, COURT_CORNERS_M)), [list(map(float, c)) for c in corners]))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return out


def _occupied_halves(H: np.ndarray, dets: Sequence[PersonDetection], cfg: CourtChoiceConfig) -> int:
    halves = set()
    m = cfg.margin_m
    for d in dets:
        if d.score < cfg.min_det_score:
            continue
        if (d.keypoints[:, 2] >= cfg.min_keypoint_conf).sum() < cfg.min_visible_keypoints:
            continue
        if is_seated(d, cfg.min_keypoint_conf):
            continue
        fp = foot_point(d, cfg.min_keypoint_conf)
        x, y = apply_homography(H, fp[None])[0]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (-m <= x <= W + m and -m <= y <= L + m):
            continue
        exp_h = expected_pixel_height(H, fp)
        if exp_h:
            lo, hi = cfg.height_ratio_range
            if not lo <= d.height / exp_h <= hi:
                continue
        halves.add("near" if y < NET_Y_M else "far")
    return len(halves)


def _standing_people(dets: Sequence[PersonDetection], cfg: CourtChoiceConfig) -> int:
    n = 0
    for d in dets:
        if d.score < cfg.min_det_score:
            continue
        if (d.keypoints[:, 2] >= cfg.min_keypoint_conf).sum() < cfg.min_visible_keypoints:
            continue
        if is_seated(d, cfg.min_keypoint_conf):
            continue
        n += 1
    return n


def contradicted(seg: CalibrationSegment, frames: Sequence[Sequence[PersonDetection]],
                 cfg: CourtChoiceConfig | None = None) -> CalibrationSegment | None:
    """``seg`` marked undeterminable when people are clearly playing but none of them
    stands on the calibrated court (sam's DJI_0009: a band of far-court lines was
    fitted with status ok). None when the calibration is consistent with the people."""
    cfg = cfg or CourtChoiceConfig()
    if seg.homography is None or seg.method == "manual" or seg.status == "undeterminable" or not frames:
        return None
    busy = [dets for dets in frames if _standing_people(dets, cfg) >= 2]
    if len(busy) < cfg.min_people_frames or len(busy) < cfg.people_frames_min * len(frames):
        return None
    H = np.asarray(seg.homography, dtype=np.float64)
    occupied = float(np.mean([_occupied_halves(H, dets, cfg) for dets in busy]))
    if occupied > cfg.empty_court_max:
        return None
    note = (f"calibration contradicted by the players: two or more people in {len(busy)}/{len(frames)} sampled "
            f"frames, but the calibrated court is empty (occupied halves {occupied:.2f}); the court lines were "
            f"probably fitted to the wrong lines. Pick the 4 corners by hand.")
    return seg.model_copy(update={"status": "undeterminable", "confidence": 0.0, "notes": [*seg.notes, note]})


def score_courts(seg: CalibrationSegment, frames: Sequence[Sequence[PersonDetection]],
                 cfg: CourtChoiceConfig | None = None) -> list[CourtScore]:
    cfg = cfg or CourtChoiceConfig()
    out = []
    for i, (H, corners) in enumerate(_court_homographies(seg)):
        per = [_occupied_halves(H, dets, cfg) for dets in frames]
        out.append(CourtScore(i, corners, float(np.mean(per)) if per else 0.0))
    return out


def rechoose_segment(seg: CalibrationSegment, frames: Sequence[Sequence[PersonDetection]],
                     cfg: CourtChoiceConfig | None = None) -> CalibrationSegment | None:
    """A copy of ``seg`` with the target swapped to the court the players are on, or
    None when the current target is kept."""
    cfg = cfg or CourtChoiceConfig()
    if seg.homography is None or not seg.other_courts or seg.method == "manual" or not frames:
        return None
    scores = score_courts(seg, frames, cfg)
    if len(scores) < 2:
        return None
    current = scores[0]
    best = max(scores[1:], key=lambda s: s.score)
    if best.score < cfg.min_score or best.score < current.score + cfg.min_gain or best.corners is None:
        return None
    others = [c for i, c in enumerate(seg.other_courts) if i + 1 != best.index]
    if seg.image_corners:
        others.append(seg.image_corners)
    H = fit_homography(best.corners, COURT_CORNERS_M)
    note = (f"target court re-chosen from player positions: players on another detected court "
            f"(occupied halves {best.score:.2f} vs {current.score:.2f} on the line-based choice, "
            f"{len(frames)} sampled frames); homography refit from its 4 corners")
    return seg.model_copy(update={
        "homography": H, "image_corners": best.corners, "other_courts": others, "keypoints": [],
        "reprojection_error_px": None, "status": "needs_review" if seg.status == "ok" else seg.status,
        "notes": [*seg.notes, note],
    })


def rechoose(cal: CourtCalibration, detect_frames, cfg: CourtChoiceConfig | None = None) -> CourtCalibration | None:
    """Apply :func:`rechoose_segment`, then :func:`contradicted`, to every automatic segment.

    ``detect_frames(frame_indices) -> list[list[PersonDetection]]`` runs the pose
    model on those frames. Returns the new calibration, or None if nothing changed.
    """
    cfg = cfg or CourtChoiceConfig()
    changed = False
    segs = []
    for seg in cal.segments:
        new = None
        if seg.homography is not None and seg.method != "manual" and seg.status != "undeterminable":
            n = min(cfg.frames_per_segment, seg.end_frame - seg.start_frame + 1)
            idx = sorted({int(round(f)) for f in np.linspace(seg.start_frame, seg.end_frame, n + 2)[1:-1]})
            frames = detect_frames(idx)
            new = rechoose_segment(seg, frames, cfg) or contradicted(seg, frames, cfg)
        segs.append(new or seg)
        changed |= new is not None
    return cal.model_copy(update={"segments": segs}) if changed else None
