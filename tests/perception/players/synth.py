"""Synthetic court camera and people for player-module tests."""

from __future__ import annotations

import numpy as np

from rallyiq.perception.players.geometry import apply_homography
from rallyiq.perception.players.models import KP, PersonDetection

# Court corners (metres) -> image pixels for a raised camera behind the near baseline.
COURT_PTS = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float64)
IMG_PTS = np.array([[360, 1000], [1560, 1000], [1240, 250], [680, 250]], dtype=np.float64)
IMAGE_SIZE = (1920, 1080)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


H_COURT_TO_IMG = _homography(COURT_PTS, IMG_PTS)
H_IMG_TO_COURT = np.linalg.inv(H_COURT_TO_IMG)


def person_at_court(x: float, y: float, score: float = 0.9, height_m: float = 1.75, seated: bool = False) -> PersonDetection:
    """A standing (or seated) person whose ankles sit at court point (x, y)."""
    foot = apply_homography(H_COURT_TO_IMG, np.array([[x, y]]))[0]
    side = apply_homography(H_COURT_TO_IMG, np.array([[x - 0.5, y], [x + 0.5, y]]))
    px_per_m = float(np.linalg.norm(side[1] - side[0]))
    return person_at_pixel(foot[0], foot[1], px_per_m * height_m, score, seated)


def person_at_pixel(fx: float, fy: float, h: float, score: float = 0.9, seated: bool = False) -> PersonDetection:
    w = 0.35 * h
    kps = np.zeros((17, 3), dtype=np.float32)
    # Fractions of height above the floor for a standing person.
    rows = {
        "nose": 0.93, "left_eye": 0.95, "right_eye": 0.95, "left_ear": 0.94, "right_ear": 0.94,
        "left_shoulder": 0.81, "right_shoulder": 0.81, "left_elbow": 0.63, "right_elbow": 0.63,
        "left_wrist": 0.48, "right_wrist": 0.48, "left_hip": 0.52, "right_hip": 0.52,
        "left_knee": 0.28, "right_knee": 0.28, "left_ankle": 0.03, "right_ankle": 0.03,
    }
    if seated:
        rows.update({"left_knee": 0.50, "right_knee": 0.50})
    for name, frac in rows.items():
        dx = (-0.12 if name.startswith("left") else 0.12 if name.startswith("right") else 0.0) * h
        kps[KP[name]] = (fx + dx, fy - frac * h, 0.9)
    bbox = np.array([fx - w / 2, fy - h, fx + w / 2, fy + 0.01 * h])
    return PersonDetection(bbox, kps, score)
