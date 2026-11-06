"""Turn raw person detections into scored player candidates.

The goal is to keep the two active players and drop spectators, umpires,
line judges, people on neighbouring courts and passers-by, without assuming a
particular camera position. Each rule lowers ``player_score`` and records a
reason instead of silently deleting, so the review UI can explain decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .geometry import PlayArea, apply_homography, foot_point
from .models import KP, Candidate, PersonDetection

if TYPE_CHECKING:
    from ..target_court import CourtRegion

TYPICAL_PLAYER_HEIGHT_M = 1.72


@dataclass
class FilterConfig:
    min_det_score: float = 0.25
    min_keypoint_conf: float = 0.3
    min_visible_keypoints: int = 5
    play_area: PlayArea = field(default_factory=PlayArea)
    # Hard reject when the foot maps this far outside the play area (metres).
    reject_outside_m: float = 1.5
    # Accepted range of observed / expected pixel height before penalising.
    height_ratio_range: tuple[float, float] = (0.4, 2.5)
    # Keep only this many candidates per frame (singles: 2 players + slack).
    max_candidates: int = 6
    # Without calibration, drop boxes smaller than this fraction of the largest.
    uncalibrated_min_rel_height: float = 0.35


def _visible(kps: np.ndarray, conf: float) -> np.ndarray:
    return kps[:, 2] >= conf


def is_seated(det: PersonDetection, conf: float = 0.3) -> bool:
    """Hips roughly level with knees and thighs short relative to the torso."""
    k = det.keypoints
    pairs = [("left_hip", "left_knee"), ("right_hip", "right_knee")]
    shoulders = [k[KP["left_shoulder"]], k[KP["right_shoulder"]]]
    hips = [k[KP["left_hip"]], k[KP["right_hip"]]]
    if not all(p[2] >= conf for p in shoulders + hips):
        return False
    torso = abs(np.mean([h[1] for h in hips]) - np.mean([s[1] for s in shoulders]))
    if torso < 1:
        return False
    legs = [
        abs(k[KP[kn]][1] - k[KP[hp]][1]) < 0.3 * torso
        for hp, kn in pairs
        if k[KP[hp]][2] >= conf and k[KP[kn]][2] >= conf
    ]
    # A deep lunge flattens one thigh, so require both legs to agree.
    return len(legs) == 2 and all(legs)


def expected_pixel_height(H_img_to_court: np.ndarray, foot_px: np.ndarray) -> float | None:
    """Rough pixel height of a typical player standing at ``foot_px``.

    Uses the local ground-plane scale across the court (x direction), which is
    close to the vertical image scale for the usual raised, behind-the-baseline
    phone view. It is deliberately only used with a wide tolerance.
    """
    try:
        Hinv = np.linalg.inv(H_img_to_court)
    except np.linalg.LinAlgError:
        return None
    c = apply_homography(H_img_to_court, foot_px[None])[0]
    if not np.all(np.isfinite(c)):
        return None
    a = apply_homography(Hinv, np.array([[c[0] - 0.5, c[1]], [c[0] + 0.5, c[1]]]))
    if not np.all(np.isfinite(a)):
        return None
    px_per_m = float(np.linalg.norm(a[1] - a[0]))
    return px_per_m * TYPICAL_PLAYER_HEIGHT_M if px_per_m > 0 else None


def build_candidates(
    dets: list[PersonDetection],
    H_img_to_court: np.ndarray | None,
    image_size: tuple[int, int],
    cfg: FilterConfig | None = None,
    region: "CourtRegion | None" = None,
) -> list[Candidate]:
    """Score detections; returns candidates sorted best first (rejected ones removed).

    ``region`` (optional) knows the other courts calibration found; anyone whose
    feet are on one of them is a player of another match and is dropped.
    """
    cfg = cfg or FilterConfig()
    w, h = image_size
    out: list[Candidate] = []
    max_h = max((d.height for d in dets), default=0.0)

    for det in dets:
        if det.score < cfg.min_det_score:
            continue
        vis = _visible(det.keypoints, cfg.min_keypoint_conf)
        if vis.sum() < cfg.min_visible_keypoints:
            continue
        reasons: list[str] = []
        score = det.score * (0.5 + 0.5 * vis.mean())

        fp = foot_point(det, cfg.min_keypoint_conf)
        # Feet cut off by the bottom edge: the foot point is unreliable and the
        # person is usually someone standing right next to the camera.
        if det.bbox[3] >= h - 2 and not (
            det.keypoints[KP["left_ankle"], 2] >= cfg.min_keypoint_conf
            or det.keypoints[KP["right_ankle"], 2] >= cfg.min_keypoint_conf
        ):
            score *= 0.4
            reasons.append("truncated_at_frame_edge")

        if is_seated(det, cfg.min_keypoint_conf):
            score *= 0.3
            reasons.append("seated")

        foot_court = None
        on_court = None
        if H_img_to_court is not None:
            fc = apply_homography(H_img_to_court, fp[None])[0]
            if np.all(np.isfinite(fc)):
                foot_court = fc.astype(np.float32)
                outside = cfg.play_area.distance_outside(fc)
                on_court = outside == 0.0
                if outside > cfg.reject_outside_m:
                    continue
                if outside > 0:
                    score *= float(np.exp(-outside / 0.5))
                    reasons.append("outside_play_area")
            else:
                continue  # projects behind the camera / horizon
            if region is not None and region.on_other_court(fp):
                continue
            exp_h = expected_pixel_height(H_img_to_court, fp)
            if exp_h:
                ratio = det.height / exp_h
                lo, hi = cfg.height_ratio_range
                if ratio < lo or ratio > hi:
                    score *= 0.35
                    reasons.append("implausible_size")
        else:
            if max_h > 0 and det.height < cfg.uncalibrated_min_rel_height * max_h:
                score *= 0.3
                reasons.append("small_relative_to_others")
            # Mild preference for people near the horizontal centre.
            cx = (det.bbox[0] + det.bbox[2]) / 2
            score *= 1.0 - 0.3 * abs(cx / max(w, 1) - 0.5) * 2

        out.append(Candidate(det, fp, foot_court, on_court, float(score), reasons))

    out.sort(key=lambda c: c.player_score, reverse=True)
    return out[: cfg.max_candidates]
