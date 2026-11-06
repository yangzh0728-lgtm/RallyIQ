"""Draw a calibration on a frame for visual checking."""

from __future__ import annotations

import cv2
import numpy as np

from . import court_model as cm
from .homography import project

_STATUS_BGR = {"ok": (80, 220, 80), "needs_review": (0, 200, 255), "undeterminable": (60, 60, 255)}


def draw_calibration(image: np.ndarray, H_image_from_court: np.ndarray | None, label: str = "", status: str = "ok") -> np.ndarray:
    out = image.copy()
    colour = _STATUS_BGR.get(status, (255, 255, 255))
    thick = max(1, out.shape[1] // 640)
    if H_image_from_court is not None:
        for seg in cm.segments():
            pts = np.linspace(seg.p0, seg.p1, 40)
            xy, wz = project(H_image_from_court, pts)
            xy = xy[wz > 0]
            if len(xy) >= 2:
                cv2.polylines(out, [np.round(xy).astype(np.int32)], False, colour, thick, cv2.LINE_AA)
        near = project(H_image_from_court, np.array([[cm.COURT_WIDTH / 2, 0.0]]))[0][0]
        if np.all(np.isfinite(near)):
            cv2.putText(out, "y=0", tuple(int(v) for v in near), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, colour, thick)
    if label:
        cv2.putText(out, label, (10, 30 * thick), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * thick, colour, thick, cv2.LINE_AA)
    return out
