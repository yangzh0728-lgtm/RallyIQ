"""Court model and homography helpers shared by calibration, players, events and overlay."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M

W, L, NET = COURT_WIDTH_M, COURT_LENGTH_M, NET_Y_M
SINGLES_INSET = 0.46
SHORT_SERVICE = 1.98
DOUBLES_LONG_SERVICE = 0.76

# Outer corners in court metres, order: near-left, near-right, far-right, far-left.
COURT_CORNERS_M: list[list[float]] = [[0.0, 0.0], [W, 0.0], [W, L], [0.0, L]]

# Named court line segments in court metres (both singles and doubles lines).
COURT_LINES_M: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_baseline": ((0, 0), (W, 0)),
    "far_baseline": ((0, L), (W, L)),
    "left_doubles_sideline": ((0, 0), (0, L)),
    "right_doubles_sideline": ((W, 0), (W, L)),
    "left_singles_sideline": ((SINGLES_INSET, 0), (SINGLES_INSET, L)),
    "right_singles_sideline": ((W - SINGLES_INSET, 0), (W - SINGLES_INSET, L)),
    "near_doubles_long_service": ((0, DOUBLES_LONG_SERVICE), (W, DOUBLES_LONG_SERVICE)),
    "far_doubles_long_service": ((0, L - DOUBLES_LONG_SERVICE), (W, L - DOUBLES_LONG_SERVICE)),
    "near_short_service": ((0, NET - SHORT_SERVICE), (W, NET - SHORT_SERVICE)),
    "far_short_service": ((0, NET + SHORT_SERVICE), (W, NET + SHORT_SERVICE)),
    "near_center": ((W / 2, 0), (W / 2, NET - SHORT_SERVICE)),
    "far_center": ((W / 2, NET + SHORT_SERVICE), (W / 2, L)),
    "net": ((0, NET), (W, NET)),
}


def fit_homography(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> list[list[float]]:
    """Least-squares DLT homography mapping ``src`` -> ``dst`` (>= 4 point pairs)."""
    s = np.asarray(src, dtype=float)
    d = np.asarray(dst, dtype=float)
    if s.shape != d.shape or s.ndim != 2 or s.shape[1] != 2 or len(s) < 4:
        raise ValueError("need >= 4 matching 2D point pairs")

    def norm(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = p.mean(axis=0)
        scale = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        t = np.array([[scale, 0, -scale * c[0]], [0, scale, -scale * c[1]], [0, 0, 1]])
        ph = np.c_[p, np.ones(len(p))] @ t.T
        return ph[:, :2], t

    sn, ts = norm(s)
    dn, td = norm(d)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    h = h / h[2, 2]
    return h.tolist()


def apply_homography(h: Sequence[Sequence[float]], pts: Sequence[Sequence[float]]) -> list[list[float]]:
    hm = np.asarray(h, dtype=float)
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    ph = np.c_[p, np.ones(len(p))] @ hm.T
    return (ph[:, :2] / ph[:, 2:3]).tolist()


def invert_homography(h: Sequence[Sequence[float]]) -> list[list[float]]:
    inv = np.linalg.inv(np.asarray(h, dtype=float))
    return (inv / inv[2, 2]).tolist()


def image_to_court(h_img_to_court, pt) -> list[float]:
    return apply_homography(h_img_to_court, [pt])[0]


def court_to_image(h_img_to_court, pt) -> list[float]:
    return apply_homography(invert_homography(h_img_to_court), [pt])[0]


def reprojection_error_px(h_img_to_court, image_pts, court_pts) -> float:
    """Mean pixel error of projecting court points back into the image."""
    proj = np.asarray(apply_homography(invert_homography(h_img_to_court), court_pts))
    return float(np.mean(np.linalg.norm(proj - np.asarray(image_pts, dtype=float), axis=1)))


def homography_from_corners(image_corners) -> list[list[float]]:
    """Image->court homography from 4 outer corners (near-left, near-right, far-right, far-left)."""
    return fit_homography(image_corners, COURT_CORNERS_M)


def project_court_lines(h_img_to_court) -> list[list[list[float]]]:
    """All court line segments projected into image px."""
    inv = invert_homography(h_img_to_court)
    out = []
    for a, b in COURT_LINES_M.values():
        pa, pb = apply_homography(inv, [a, b])
        out.append([pa, pb])
    return out


def on_court(xy, margin_m: float = 0.5) -> bool:
    x, y = xy
    return -margin_m <= x <= W + margin_m and -margin_m <= y <= L + margin_m
