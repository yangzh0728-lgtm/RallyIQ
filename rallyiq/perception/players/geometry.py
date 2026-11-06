"""Foot-point estimation and image -> court mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import COURT_LENGTH_M, COURT_WIDTH_M, KP, NET_Y_M, PersonDetection, Side


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N, 2) points with a 3x3 homography. Returns (N, 2); NaN where w ~ 0."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    w = hom[:, 2:3]
    out = np.full((len(pts), 2), np.nan)
    ok = np.abs(w[:, 0]) > 1e-9
    out[ok] = hom[ok, :2] / w[ok]
    return out


def foot_point(det: PersonDetection, min_conf: float = 0.3, ground_offset: float = 0.025) -> np.ndarray:
    """Image point where the player touches the floor.

    Uses the ankle midpoint (or the single visible ankle) nudged down by a small
    fraction of the box height, since ankles sit a few cm above the floor. Falls
    back to the bottom-centre of the box when no ankle is reliable, which is the
    common case for far-side players on low-resolution phone footage.
    """
    kps = det.keypoints
    ankles = [kps[KP["left_ankle"]], kps[KP["right_ankle"]]]
    good = [a for a in ankles if a[2] >= min_conf]
    x1, y1, x2, y2 = det.bbox
    if good:
        pt = np.mean([a[:2] for a in good], axis=0)
        pt = pt + np.array([0.0, ground_offset * (y2 - y1)])
        # Never place the foot below the box: keypoints can drift on blur.
        pt[1] = min(pt[1], y2)
        return pt.astype(np.float32)
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)


@dataclass(frozen=True)
class PlayArea:
    """Court rectangle plus the run-off area where active players can stand.

    Margins are in metres beyond the outer court lines. Players regularly
    retreat ~1-1.5 m behind the baseline and step slightly past the sidelines;
    neighbouring courts and spectator seating are usually further away.
    """

    margin_side: float = 1.0
    margin_end: float = 2.0

    def contains(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return (
            -self.margin_side <= x <= COURT_WIDTH_M + self.margin_side
            and -self.margin_end <= y <= COURT_LENGTH_M + self.margin_end
        )

    def distance_outside(self, xy: np.ndarray) -> float:
        """Metres outside the play area (0 when inside)."""
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("inf")
        dx = max(-self.margin_side - x, 0.0, x - (COURT_WIDTH_M + self.margin_side))
        dy = max(-self.margin_end - y, 0.0, y - (COURT_LENGTH_M + self.margin_end))
        return float(np.hypot(dx, dy))


def court_side(xy: np.ndarray | None) -> Side | None:
    if xy is None or not np.all(np.isfinite(xy)):
        return None
    return "near" if float(xy[1]) < NET_Y_M else "far"


def image_net_y(H_img_to_court: np.ndarray) -> float | None:
    """Image y of the net line at court centre, used to split halves without metres."""
    try:
        Hinv = np.linalg.inv(H_img_to_court)
    except np.linalg.LinAlgError:
        return None
    p = apply_homography(Hinv, np.array([[COURT_WIDTH_M / 2, NET_Y_M]]))[0]
    return float(p[1]) if np.all(np.isfinite(p)) else None
