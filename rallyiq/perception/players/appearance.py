"""Lightweight clothing-colour descriptor used to re-identify the two players.

Within a rally the court half and motion continuity are enough to keep the two
singles players apart. Appearance matters when both disappear (camera cut,
break between games) and come back on swapped halves after a change of ends.
"""

from __future__ import annotations

import numpy as np

from .models import KP, PersonDetection

H_BINS, S_BINS, V_BINS = 12, 4, 3


def torso_box(det: PersonDetection, conf: float = 0.3) -> tuple[int, int, int, int]:
    """Shoulder-to-hip box, falling back to the upper-middle part of the bbox."""
    k = det.keypoints
    idx = [KP[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    pts = k[idx]
    x1, y1, x2, y2 = det.bbox
    if (pts[:, 2] >= conf).all():
        tx1, ty1 = pts[:, 0].min(), pts[:, 1].min()
        tx2, ty2 = pts[:, 0].max(), pts[:, 1].max()
        # Narrow torsos (side-on players) still need a few pixels of shirt.
        pad = max(0.15 * (y2 - y1) - (tx2 - tx1), 0) / 2
        tx1, tx2 = tx1 - pad, tx2 + pad
    else:
        bw, bh = x2 - x1, y2 - y1
        tx1, tx2 = x1 + 0.25 * bw, x2 - 0.25 * bw
        ty1, ty2 = y1 + 0.2 * bh, y1 + 0.55 * bh
    return int(tx1), int(ty1), int(np.ceil(tx2)), int(np.ceil(ty2))


def colour_histogram(image_bgr: np.ndarray, det: PersonDetection) -> np.ndarray | None:
    """Normalised HSV histogram of the torso region; None if too small."""
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = torso_box(det)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else None


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Bhattacharyya coefficient in [0, 1]; None when either side is missing."""
    if a is None or b is None:
        return None
    return float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
