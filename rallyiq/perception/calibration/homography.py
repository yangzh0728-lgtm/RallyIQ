"""Homography hypotheses, scoring against line evidence, and refinement.

All homographies here map court metres -> image pixels ("image_from_court").
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from . import court_model as cm
from .lines import ImageLine, LineEvidence


def project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 2) points with one (3, 3) H. Returns (xy (N, 2), w (N,))."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    w = ph[:, 2]
    safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return ph[:, :2] / safe[:, None], w


def _square_to_quad(q: np.ndarray) -> np.ndarray:
    """Closed-form homographies mapping the unit square to quads.

    q: (..., 4, 2) image points for (0,0), (1,0), (1,1), (0,1).
    Returns (..., 3, 3); entries are nan where the quad is degenerate.
    """
    x0, y0 = q[..., 0, 0], q[..., 0, 1]
    x1, y1 = q[..., 1, 0], q[..., 1, 1]
    x2, y2 = q[..., 2, 0], q[..., 2, 1]
    x3, y3 = q[..., 3, 0], q[..., 3, 1]
    dx1, dx2, dx3 = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, dy3 = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    den = dx1 * dy2 - dx2 * dy1
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (dx3 * dy2 - dx2 * dy3) / den
        h = (dx1 * dy3 - dx3 * dy1) / den
    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    one = np.ones_like(x0)
    H = np.stack(
        [
            np.stack([a, b, x0], -1),
            np.stack([d, e, y0], -1),
            np.stack([g, h, one], -1),
        ],
        -2,
    )
    H[np.abs(den) < 1e-9] = np.nan
    return H


def _model_rect_pairs(min_area: float = 2.0):
    """Pairs of transverse and longitudinal model lines spanning a usable rectangle."""
    tp = np.array(list(combinations(range(len(cm.TRANSVERSE_Y)), 2)))
    lp = np.array(list(combinations(range(len(cm.LONGITUDINAL_X)), 2)))
    out = []
    for p, q in tp:
        for r, s in lp:
            y1, y2 = cm.TRANSVERSE_Y[p], cm.TRANSVERSE_Y[q]
            x1, x2 = cm.LONGITUDINAL_X[r], cm.LONGITUDINAL_X[s]
            if (y2 - y1) * (x2 - x1) < min_area:
                continue
            # S maps the model rectangle onto the unit square.
            S = np.array(
                [
                    [1 / (x2 - x1), 0, -x1 / (x2 - x1)],
                    [0, 1 / (y2 - y1), -y1 / (y2 - y1)],
                    [0, 0, 1],
                ]
            )
            out.append(S)
    return np.array(out)


_MODEL_RECTS = _model_rect_pairs()


def _unique_per_row(cells: np.ndarray) -> np.ndarray:
    """Number of distinct non-negative ids per row of a (B, P) int array."""
    s = np.sort(cells, axis=1)
    new = np.ones_like(s, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    return (new & (s >= 0)).sum(1)


def _cell_score(u, v, vis, matched, w: int, cell: float) -> np.ndarray:
    """Score in distinct image cells: matched cells minus half the unmatched ones.

    Counting cells instead of points stops degenerate homographies that squash
    the whole court onto one bright line from scoring well.
    """
    ncols = int(w // cell) + 1
    cid = (np.floor(v / cell) * ncols + np.floor(u / cell)).astype(np.int64)
    m = _unique_per_row(np.where(matched, cid, -1))
    um = _unique_per_row(np.where(vis & ~matched, cid, -1))
    return m - 0.5 * um


@dataclass
class Score:
    matched: int
    visible: int
    total: int
    value: float = 0.0

    @property
    def inlier_ratio(self) -> float:
        return self.matched / self.visible if self.visible else 0.0

    @property
    def visible_fraction(self) -> float:
        return self.visible / self.total if self.total else 0.0


LINE_WIDTH_M = 0.04
MIN_DETECTABLE_PX = 1.2


def _evaluate(Hs: np.ndarray, spacing: float, ev: LineEvidence, tol: float):
    """Score a batch of homographies (B, 3, 3) against the line evidence.

    Returns (value, matched, visible) per homography. Only model samples that
    land in the frame where the painted line would be at least ~1 px wide
    count, as matched or unmatched: thinner far-away lines often vanish in
    real video, and dense far clutter must not reward a squashed placement.
    """
    pts, tans = cm.sample_points(spacing)
    nrm = np.stack([-tans[:, 1], tans[:, 0]], 1)
    h, w = ev.shape
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    po = np.concatenate([pts + 0.1 * nrm, np.ones((len(pts), 1))], axis=1)
    proj = np.einsum("bij,pj->bpi", Hs, ph)
    proj_o = np.einsum("bij,pj->bpi", Hs, po)
    wz = proj[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = proj[..., 0] / wz, proj[..., 1] / wz
        uo, vo = proj_o[..., 0] / proj_o[..., 2], proj_o[..., 1] / proj_o[..., 2]
        width_px = np.hypot(uo - u, vo - v) * (LINE_WIDTH_M / 0.1)
    inside = (wz > 1e-9) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    ui = np.clip(np.nan_to_num(u, nan=0.0), 0, w - 1).astype(np.int32)
    vi = np.clip(np.nan_to_num(v, nan=0.0), 0, h - 1).astype(np.int32)
    matched = inside & (ev.dist[vi, ui] <= tol)
    vis = inside & (np.nan_to_num(width_px) >= MIN_DETECTABLE_PX)
    matched &= vis
    value = _cell_score(u, v, vis, matched, w, tol)
    return value, matched.sum(1), vis.sum(1)


def score_many(Hs: np.ndarray, ev: LineEvidence, tol: float, spacing: float = 0.5) -> np.ndarray:
    return _evaluate(Hs, spacing, ev, tol)[0]


def score_one(H: np.ndarray, ev: LineEvidence, tol: float, spacing: float = 0.1) -> Score:
    value, matched, visible = _evaluate(H[None], spacing, ev, tol)
    total = len(cm.sample_points(spacing)[0])
    return Score(int(matched[0]), int(visible[0]), total, float(value[0]))


def _plausible(Hs: np.ndarray, w: int, h: int, min_area_frac: float = 0.015) -> np.ndarray:
    """Reject placements that cannot be a real court in front of the camera.

    All four outer corners must be in front of the camera and the projected
    court must be a convex quad of reasonable size. This removes degenerate
    solutions that squash the court onto a single bright line.
    """
    corners = np.array(list(cm.CORNERS.values()))
    ph = np.concatenate([corners, np.ones((4, 1))], 1)
    q = np.einsum("bij,pj->bpi", Hs, ph)
    ok = (q[..., 2] > 1e-9).all(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[..., :2] / q[..., 2:3]
    e = np.roll(xy, -1, axis=1) - xy
    en = np.roll(e, -1, axis=1)
    cross = e[..., 0] * en[..., 1] - e[..., 1] * en[..., 0]
    ok &= np.all(cross > 0, 1) | np.all(cross < 0, 1)
    area = 0.5 * np.abs((xy[..., 0] * np.roll(xy[..., 1], -1, 1) - np.roll(xy[..., 0], -1, 1) * xy[..., 1]).sum(1))
    ok &= area > min_area_frac * w * h
    # Very thin courts are degenerate: shortest side vs longest side.
    sides = np.linalg.norm(e, axis=2)
    ok &= sides.min(1) > 0.02 * sides.max(1)
    ok &= metric_error(Hs, w, h)[0] < METRIC_TOL
    ok &= upright(Hs)
    return np.nan_to_num(ok, nan=False).astype(bool)


METRIC_TOL = 0.2


def upright(Hs: np.ndarray) -> np.ndarray:
    """True where the court lies below its horizon in the image, as it must for a
    camera above the floor that is not upside down.

    The horizon is the image of the floor's line at infinity: the line through
    the vanishing points of the two court axes (columns 0 and 1 of an
    image_from_court homography). A placement whose horizon runs *below* the
    court can match bright lines (a thin band of floor markings, an advertising
    board) and still pass the pinhole check, but no upright camera sees a court
    that way (sam's DJI_0009: a 2000 x 100 px band, horizon ~60 px below it).
    Horizons that are nearly vertical (camera rolled > ~70 degrees) or at
    infinity (camera looking straight down) are not judged.
    """
    Hs = np.asarray(Hs, dtype=np.float64).reshape(-1, 3, 3)
    corners = np.array(list(cm.CORNERS.values()))
    ph = np.concatenate([corners, np.ones((4, 1))], 1)
    q = np.einsum("bij,pj->bpi", Hs, ph)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[..., :2] / q[..., 2:3]
    line = np.cross(Hs[:, :, 0], Hs[:, :, 1])  # (B, 3): a x + b y + c = 0
    a, b, c = line[:, 0], line[:, 1], line[:, 2]
    norm = np.hypot(a, b)
    cx = np.nanmean(xy[..., 0], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        horizon_y = -(a * cx + c) / b
    span = np.nanmax(xy[..., 1], 1) - np.nanmin(xy[..., 1], 1)
    cy = np.nanmean(xy[..., 1], 1)
    extent = np.maximum(span, np.nanmax(xy[..., 0], 1) - np.nanmin(xy[..., 0], 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(a * cx + b * cy + c) / norm  # court centre to horizon, px
    # Far away (near top-down view) the horizon side is numerically meaningless.
    judged = (np.abs(b) > 0.35 * norm) & np.isfinite(horizon_y) & (dist < 50 * np.maximum(extent, 1.0))
    # "Below" means below the whole court, with a little slack for numerical noise.
    below = horizon_y > np.nanmax(xy[..., 1], 1) + 0.01 * np.maximum(span, 1.0)
    return ~(judged & below)
_F_GRID = np.geomspace(0.35, 5.0, 40)  # focal length / image width


def metric_error(Hs: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """How far each homography is from a real pinhole camera looking at a flat court.

    For a camera with square pixels and the principal point near the image
    centre, K^-1 H has two orthogonal, equal-length first columns (the floor
    axes). We search the focal length and return the smallest violation
    (combining |cos angle| and |log length ratio|) and the focal length that
    achieves it. A squashed or sheared court placement has a large error even
    if it happens to lie on bright lines.
    """
    T = np.array([[1.0, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    G = np.einsum("ij,bjk->bik", T, Hs)
    h1, h2 = G[:, :, 0], G[:, :, 1]
    f = _F_GRID[None, :] * w  # (1, F)
    shape = (len(Hs), f.shape[1])
    r1 = np.stack([h1[:, None, 0] / f, h1[:, None, 1] / f, np.broadcast_to(h1[:, None, 2], shape)], -1)
    r2 = np.stack([h2[:, None, 0] / f, h2[:, None, 1] / f, np.broadcast_to(h2[:, None, 2], shape)], -1)
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.abs((r1 * r2).sum(-1)) / (n1 * n2)
        asp = np.abs(np.log(n1 / n2))
    err = np.nan_to_num(np.hypot(cos, asp), nan=np.inf)
    k = np.argmin(err, axis=1)
    return err[np.arange(len(Hs)), k], _F_GRID[k] * w


def _intersections(fam_y: list[ImageLine], fam_x: list[ImageLine]) -> np.ndarray:
    ly = np.array([l.homogeneous() for l in fam_y])
    lx = np.array([l.homogeneous() for l in fam_x])
    P = np.cross(ly[:, None, :], lx[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = P[..., :2] / P[..., 2:3]
    pts[np.abs(P[..., 2]) < 1e-9] = np.nan
    return pts


def hypotheses(
    fam_a: list[ImageLine],
    fam_b: list[ImageLine],
    ev: LineEvidence,
    tol: float,
    top_k: int = 60,
) -> list[np.ndarray]:
    """Enumerate line-to-model correspondences and return the best-scoring homographies.

    Two image lines from each family plus two model lines of each kind fix a
    homography. Image lines are ordered within their family and matched to
    model lines in increasing order; the reversed orders are the court's own
    symmetries, which are resolved later.
    """
    h, w = ev.shape
    diag = np.hypot(w, h)
    centre = np.array([w / 2, h / 2])
    cand_H, cand_s = [], []

    for fam_y, fam_x in ((fam_a, fam_b), (fam_b, fam_a)):
        if len(fam_y) < 2 or len(fam_x) < 2:
            continue
        P = _intersections(fam_y, fam_x)
        py = list(combinations(range(len(fam_y)), 2))
        px = list(combinations(range(len(fam_x)), 2))
        quads = []
        for i, j in py:
            for k, l in px:
                quads.append([P[i, k], P[i, l], P[j, l], P[j, k]])
        quads = np.array(quads)  # (Q, 4, 2)
        ok = np.isfinite(quads).all(axis=(1, 2))
        ok &= (np.linalg.norm(quads - centre, axis=2) < 2.0 * diag).all(axis=1)
        # Convex, non-degenerate quads only.
        e = np.roll(quads, -1, axis=1) - quads
        cross = e[:, :, 0] * np.roll(e, -1, axis=1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, axis=1)[:, :, 0]
        ok &= (np.all(cross > 0, axis=1) | np.all(cross < 0, axis=1))
        area = 0.5 * np.abs(
            (quads[:, :, 0] * np.roll(quads[:, :, 1], -1, 1) - np.roll(quads[:, :, 0], -1, 1) * quads[:, :, 1]).sum(1)
        )
        ok &= area > 0.0005 * w * h
        quads = quads[ok]
        if len(quads) == 0:
            continue
        Hq = _square_to_quad(quads)  # (Q, 3, 3)
        Hs = np.einsum("qij,mjk->qmik", Hq, _MODEL_RECTS).reshape(-1, 3, 3)
        Hs = Hs[np.isfinite(Hs).all(axis=(1, 2))]
        Hs = Hs[_plausible(Hs, w, h)]
        for start in range(0, len(Hs), 8000):
            chunk = Hs[start : start + 8000]
            s = score_many(chunk, ev, tol * 1.5)
            keep = np.argsort(-s)[:top_k]
            cand_H.append(chunk[keep])
            cand_s.append(s[keep])

    if not cand_H:
        return []
    H_all = np.concatenate(cand_H)
    s_all = np.concatenate(cand_s)
    order = np.argsort(-s_all)
    return [H_all[i] for i in order[:top_k]]


def refine(H: np.ndarray, ev: LineEvidence, tol: float, iterations: int = 6) -> np.ndarray:
    """Snap projected model lines onto line pixels and re-fit H.

    Each model sample searches along its projected normal for the nearest line
    response peak; RANSAC on the resulting correspondences rejects occlusions
    (players, net) and lines that do not belong to this court.
    """
    h, w = ev.shape
    pts, tans = cm.sample_points(0.1)
    radii = [4 * tol, 3 * tol, 2 * tol] + [1.5 * tol] * max(0, iterations - 3)
    best = H
    for r in radii[:iterations]:
        xy, wz = project(best, pts)
        xy2, _ = project(best, pts + 0.05 * tans)
        t_img = xy2 - xy
        nrm = np.stack([-t_img[:, 1], t_img[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        vis = (wz > 1e-9) & (xy[:, 0] >= 1) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 1) & (xy[:, 1] <= h - 2)
        if vis.sum() < 12:
            break
        P, N = xy[vis], nrm[vis]
        offs = np.arange(-int(np.ceil(r)), int(np.ceil(r)) + 1, dtype=np.float32)
        sx = (P[:, None, 0] + offs[None] * N[:, None, 0]).astype(np.float32)
        sy = (P[:, None, 1] + offs[None] * N[:, None, 1]).astype(np.float32)
        prof = cv2.remap(ev.response, sx, sy, cv2.INTER_LINEAR, borderValue=0)
        prior = np.exp(-0.5 * (offs / (0.6 * r)) ** 2)
        peak = np.argmax(prof * prior[None], axis=1)
        peak_val = prof[np.arange(len(P)), peak]
        # Local centroid around the peak for sub-pixel position.
        win = np.clip(peak[:, None] + np.arange(-2, 3)[None], 0, len(offs) - 1)
        wv = np.take_along_axis(prof, win, axis=1)
        t_star = (wv * offs[win]).sum(1) / (wv.sum(1) + 1e-9)
        good = peak_val > 0.35
        if good.sum() < 12:
            break
        src = pts[vis][good].astype(np.float32)
        dst = (P[good] + t_star[good, None] * N[good]).astype(np.float32)
        Hn, inl = cv2.findHomography(src, dst, cv2.RANSAC, max(1.5, tol * 0.75), maxIters=500)
        if Hn is None or inl is None or inl.sum() < 12:
            break
        best = Hn
    return best


def orientation_preserving(H: np.ndarray) -> bool:
    """True if H is consistent with a camera above the floor looking down at it.

    Seen from above, the right-handed court frame (x across, y away) appears in
    the image with x to the right and y going up the image, which flips sign
    in image coordinates (v points down): the local Jacobian has det < 0.
    """
    c = np.array([cm.COURT_WIDTH / 2, cm.COURT_LENGTH / 2, 1.0])
    wz = H[2] @ c
    return np.sign(np.linalg.det(H)) * np.sign(wz) < 0


# Relative near/far baseline length difference above which perspective alone
# says which end is nearer the camera (broadcast and elevated views: ~0.3-0.6).
PERSPECTIVE_CUE = 0.1


def canonicalize(H: np.ndarray, previous: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Pick one of the four symmetric labellings of the court.

    Only labellings a camera above the floor can produce are considered (x to
    the right, y away). Of those, y = 0 goes at the end nearer the camera, whose
    baseline appears longer. Only when perspective cannot tell the ends apart
    (both baselines about as long, e.g. a camera level with the net) does the
    previous calibration decide, to keep player sides stable.

    The previous calibration must not override a clear perspective cue: after a
    scene cut it comes from another shot, and matching corners across shots
    flipped whole stretches of broadcast video to the 180-degree or mirrored
    labelling.
    Returns (H, symmetry name applied).
    """
    corners = np.array(list(cm.CORNERS.values()))
    variants = {name: H @ S for name, S in cm.SYMMETRIES.items()}
    candidates = {n: v for n, v in variants.items() if orientation_preserving(v)} or variants

    def baselines(Hv):
        xy, _ = project(Hv, corners)
        return np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[3])  # near, far

    def far_longer(Hv):
        near, far = baselines(Hv)
        return far - near

    name = min(candidates, key=lambda n: far_longer(candidates[n]))
    near, far = baselines(candidates[name])
    cue = (near - far) / max(near, far, 1e-9)
    if previous is not None and not cue > PERSPECTIVE_CUE:
        prev_xy, _ = project(previous, corners)

        def dist(n):
            xy, _ = project(candidates[n], corners)
            return float(np.nanmax(np.linalg.norm(xy - prev_xy, axis=1)))

        name = min(candidates, key=dist)
    return candidates[name], name


def verified_keypoints(H: np.ndarray, ev: LineEvidence, tol: float, arm: float = 0.3) -> tuple[int, int]:
    """Count line intersections confirmed by the image.

    A keypoint counts as verified when it projects inside the image and the
    line pixels are present at the point and along every painted arm leaving
    it. Returns (verified, visible).
    """
    h, w = ev.shape
    verified = visible = 0
    for kp in cm.keypoints():
        pts = [kp.xy] + [(kp.xy[0] + arm * dx, kp.xy[1] + arm * dy) for dx, dy in kp.arms]
        xy, wz = project(H, np.array(pts))
        inside = (wz > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
        if not inside[0]:
            continue
        visible += 1
        if not inside.all():
            continue
        d = ev.dist[xy[:, 1].astype(int), xy[:, 0].astype(int)]
        if (d <= tol).all():
            verified += 1
    return verified, visible


def residual_rms(H: np.ndarray, ev: LineEvidence, tol: float) -> float:
    """RMS distance (px) from projected model samples to the nearest line pixel, inliers only."""
    pts, _ = cm.sample_points(0.1)
    h, w = ev.shape
    xy, wz = project(H, pts)
    vis = (wz > 1e-9) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
    d = ev.dist[xy[vis, 1].astype(int), xy[vis, 0].astype(int)]
    d = d[d <= tol]
    return float(np.sqrt(np.mean(d**2))) if len(d) else float("nan")
