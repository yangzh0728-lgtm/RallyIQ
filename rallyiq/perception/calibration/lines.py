"""Court line evidence: a line mask, straight lines and their two families."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LineEvidence:
    """Per-frame pixel evidence used by hypothesis scoring and refinement."""

    mask: np.ndarray  # uint8, 255 on candidate line pixels
    response: np.ndarray  # float32 0..1, slightly blurred mask (for sub-pixel search)
    dist: np.ndarray  # float32, distance in px to the nearest line pixel

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape[:2]


@dataclass
class ImageLine:
    """An infinite image line n . p = c with |n| = 1, plus how much evidence backs it."""

    normal: np.ndarray
    c: float
    support: float  # total length in px of the segments merged into it
    points: list = field(default_factory=list, repr=False)
    vp: np.ndarray | None = field(default=None, repr=False)  # family vanishing point (homogeneous)

    @property
    def direction_angle(self) -> float:
        """Direction angle in [0, pi)."""
        nx, ny = self.normal
        return float(np.arctan2(nx, -ny) % np.pi)

    def homogeneous(self) -> np.ndarray:
        return np.array([self.normal[0], self.normal[1], -self.c])


def line_evidence(image_bgr: np.ndarray) -> LineEvidence:
    """Find thin bright structures (painted lines) robustly to floor colour and lighting.

    A morphological top-hat on the brightest channel keeps structures thinner
    than the kernel, so it works for white or yellow lines on green, blue, red
    or wooden floors, and ignores broad glare and lighting gradients.
    """
    h, w = image_bgr.shape[:2]
    value = image_bgr.max(axis=2) if image_bgr.ndim == 3 else image_bgr
    k = max(9, int(round(w / 50)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    otsu, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(22.0, float(otsu))
    mask = (tophat >= thr).astype(np.uint8) * 255

    # Drop small blobs: lines are long connected structures.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_extent = 0.03 * w
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= min_extent
    keep[0] = False
    mask = (keep[labels]).astype(np.uint8) * 255

    response = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    return LineEvidence(mask=mask, response=response, dist=dist)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _fit(points: np.ndarray) -> tuple[np.ndarray, float]:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    n = np.array([-vy, vx], dtype=np.float64)
    n /= np.linalg.norm(n)
    return n, float(n @ np.array([x0, y0]))


def detect_lines(mask: np.ndarray, max_lines: int = 24) -> list[ImageLine]:
    """Hough segments merged into infinite lines, strongest first."""
    h, w = mask.shape[:2]
    diag = float(np.hypot(w, h))
    segs = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360,
        threshold=max(20, int(w * 0.02)),
        minLineLength=max(15, int(w * 0.035)),
        maxLineGap=max(5, int(w * 0.012)),
    )
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)
    angle_tol = np.deg2rad(1.2)
    dist_tol = max(2.0, 0.0028 * diag)

    lines: list[ImageLine] = []
    for idx in order:
        x0, y0, x1, y1 = segs[idx]
        length = lengths[idx]
        ang = np.arctan2(y1 - y0, x1 - x0) % np.pi
        p0, p1 = np.array([x0, y0]), np.array([x1, y1])
        n_pts = max(2, int(length // 4))
        pts = p0 + np.linspace(0, 1, n_pts)[:, None] * (p1 - p0)
        for line in lines:
            if _angle_diff(ang, line.direction_angle) > angle_tol * 2:
                continue
            d0 = abs(line.normal @ p0 - line.c)
            d1 = abs(line.normal @ p1 - line.c)
            if max(d0, d1) <= dist_tol:
                line.points.append(pts)
                line.support += length
                line.normal, line.c = _fit(np.concatenate(line.points))
                break
        else:
            n = np.array([-(y1 - y0), x1 - x0]) / length
            lines.append(ImageLine(normal=n, c=float(n @ p0), support=length, points=[pts]))

    lines.sort(key=lambda l: -l.support)
    return lines[:max_lines]


def _midpoint(line: ImageLine) -> np.ndarray:
    pts = np.concatenate(line.points)
    # Project the mean of its pixels onto the fitted line.
    m = pts.mean(0)
    return m - (line.normal @ m - line.c) * line.normal


def _vp_consistency(lines: list[ImageLine], vp: np.ndarray) -> np.ndarray:
    """Angle (rad) between each line and the direction from its midpoint to vp (homogeneous)."""
    out = np.empty(len(lines))
    for i, l in enumerate(lines):
        d = np.array([-l.normal[1], l.normal[0]])
        m = _midpoint(l)
        to_vp = vp[:2] - vp[2] * m  # works for finite and infinite vanishing points
        nv = np.linalg.norm(to_vp)
        if nv < 1e-12:
            out[i] = 0.0
            continue
        cosang = abs(d @ to_vp) / nv
        out[i] = np.arccos(np.clip(cosang, 0.0, 1.0))
    return out


def _best_vp(lines: list[ImageLine], tol: float) -> tuple[np.ndarray | None, np.ndarray]:
    """Vanishing point with the most length-weighted line support (exhaustive over pairs)."""
    wts = np.array([l.support for l in lines])
    best, best_s, best_in = None, -1.0, np.zeros(len(lines), bool)
    H = [l.homogeneous() for l in lines]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = np.cross(H[i], H[j])
            n = np.linalg.norm(vp)
            if n < 1e-12:
                continue
            vp = vp / n
            inl = _vp_consistency(lines, vp) < tol
            s = wts[inl].sum()
            if s > best_s:
                best, best_s, best_in = vp, s, inl
    return best, best_in


def split_families(lines: list[ImageLine]) -> tuple[list[ImageLine], list[ImageLine]]:
    """Split lines into the two court directions by their vanishing points.

    Lines parallel on the floor meet at one vanishing point in the image (at
    infinity if they stay parallel), which works for any camera angle, unlike
    clustering raw orientations.
    """
    if len(lines) < 4:
        return lines, []
    tol = np.deg2rad(1.5)
    vp1, in1 = _best_vp(lines, tol)
    rest = [l for l, b in zip(lines, in1) if not b]
    if vp1 is None or len(rest) < 2:
        return lines, []
    vp2, _ = _best_vp(rest, tol)
    if vp2 is None:
        return lines, []
    # Re-assign every line against both vanishing points. A line consistent
    # with both (e.g. a far line lying on the horizon, which passes through the
    # sidelines' vanishing point) joins both families; hypotheses that use the
    # same line twice are degenerate and get filtered out later.
    e1 = _vp_consistency(lines, vp1)
    e2 = _vp_consistency(lines, vp2)
    fam_a = [l for l, a, b in zip(lines, e1, e2) if a < tol and (a <= b or b < tol)]
    fam_b = [l for l, a, b in zip(lines, e1, e2) if b < tol and (b < a or a < tol)]
    fam_a = sorted(fam_a, key=lambda l: -l.support)
    fam_b = sorted(fam_b, key=lambda l: -l.support)
    return [_with_vp(l, vp1) for l in fam_a], [_with_vp(l, vp2) for l in fam_b]


def _with_vp(line: ImageLine, vp: np.ndarray) -> ImageLine:
    return ImageLine(normal=line.normal, c=line.c, support=line.support, points=line.points, vp=vp)


def sort_family(lines: list[ImageLine], centre: np.ndarray) -> list[ImageLine]:
    """Order lines of one family across the image (fan order around their vanishing point)."""
    if len(lines) < 2:
        return list(lines)
    vp = getattr(lines[0], "vp", None)
    mids = np.array([_midpoint(l) for l in lines])
    if vp is None or abs(vp[2]) < 1e-9 * np.linalg.norm(vp[:2]) or np.linalg.norm(vp[:2] / vp[2] - centre) > 1e5:
        # Parallel in the image: order by offset across the common direction.
        d = np.array([-lines[0].normal[1], lines[0].normal[0]])
        across = np.array([-d[1], d[0]])
        key = mids @ across
    else:
        v = vp[:2] / vp[2]
        ang = np.arctan2(mids[:, 1] - v[1], mids[:, 0] - v[0])
        ref = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        key = (ang - ref + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(key)
    return [lines[i] for i in order]
