"""Heatmap -> candidate points (numpy only).

The network outputs one heatmap per frame (values 0..1, a Gaussian blob where the
shuttle is). The usual approach thresholds at 0.5 and keeps the largest blob,
which throws away faint far-court shuttles. Here we keep up to ``top_k`` local
peaks down to a lower threshold, with a sub-pixel centroid and the peak value as
score, and let the tracker decide using temporal consistency.
"""

from __future__ import annotations

import numpy as np


def _max_filter3(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="constant", constant_values=-np.inf)
    h, w = a.shape
    out = a.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            np.maximum(out, p[dy:dy + h, dx:dx + w], out=out)
    return out


def extract_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.2,
    top_k: int = 3,
    nms_radius: int = 6,
    centroid_radius: int = 3,
) -> list[tuple[float, float, float]]:
    """Return ``[(x, y, score), ...]`` in heatmap pixel coordinates, best first."""
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2 or hm.size == 0 or float(hm.max()) < threshold:
        return []
    is_peak = (hm >= threshold) & (hm >= _max_filter3(hm))
    ys, xs = np.nonzero(is_peak)
    if len(xs) == 0:
        return []
    order = np.argsort(-hm[ys, xs])
    h, w = hm.shape
    out: list[tuple[float, float, float]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any((x - ox) ** 2 + (y - oy) ** 2 <= nms_radius ** 2 for ox, oy, _ in out):
            continue
        peak = float(hm[y, x])
        y0, y1 = max(0, y - centroid_radius), min(h, y + centroid_radius + 1)
        x0, x1 = max(0, x - centroid_radius), min(w, x + centroid_radius + 1)
        patch = hm[y0:y1, x0:x1]
        wts = np.where(patch >= 0.5 * peak, patch, 0.0)
        tot = float(wts.sum())
        gy, gx = np.mgrid[y0:y1, x0:x1]
        cx = float((wts * gx).sum() / tot) if tot > 0 else float(x)
        cy = float((wts * gy).sum() / tot) if tot > 0 else float(y)
        out.append((cx, cy, peak))
        if len(out) >= top_k:
            break
    return out
