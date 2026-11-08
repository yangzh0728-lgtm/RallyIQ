"""Fit a badminton court to one frame from the two net-post feet plus the visible floor lines.

    python scripts/fit_court_posts.py FRAME.jpg --left 935 585 --right 1860 690 \\
        --out court.json --check check.jpg [--video clip.mp4]

Automatic calibration often locks on to the right white lines with the wrong identities
(a service line taken for a sideline, a neighbouring court's lines, a 90 degree flip):
every line sits on paint, so the line residual alone cannot tell. The net posts can: they
stand on the doubles sidelines at the net, which pins court points (0, 6.7) and (6.1, 6.7).

1. Detect floor line candidates (white, low-saturation pixels; Hough segments merged
   into infinite lines).
2. Lines through a post are taken as that doubles sideline (x = 0 / 6.1). Two points and
   two lines leave a one-parameter family of homographies (a homology), so each hypothesis
   uses 2 post points + 3 line correspondences: both post sidelines + 1 other line, or one
   post sideline + 2 other lines, for every identity of the other lines (short and long
   service lines, baselines, singles sidelines, centre line), solved in least squares.
3. Score each hypothesis by how well ALL projected court lines land on white paint
   (capped chamfer distance), with a penalty for court lines that land on nothing and
   for courts mostly out of frame. A rotated or neighbour-court identity leaves many
   court lines on bare floor and loses.
4. Refine the best by snapping sampled court-line points to the nearest paint and
   refitting (the posts are kept as heavily weighted points).

Writes court_points in the own_clips.yaml naming (only points inside the frame), the
image_from_court homography, residuals and post errors; ``--check`` draws the court with
the net (y = 6.7) in yellow. With ``--video`` the residual is also measured at 1/4, 1/2
and 3/4 of the video (a moved camera fails there).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

COURT_W, COURT_L, NET_Y = 6.1, 13.4, 6.7
X_LINES = {"left_doubles_sideline": 0.0, "left_singles_sideline": 0.46, "centre_line": 3.05,
           "right_singles_sideline": 5.64, "right_doubles_sideline": 6.1}
Y_LINES = {"near_baseline": 0.0, "near_doubles_long_service": 0.76, "near_short_service": 4.72,
           "far_short_service": 8.68, "far_doubles_long_service": 12.64, "far_baseline": 13.4}
# identities a detected line may take (x = const or y = const); the net is not painted
IDENTITIES = [("x", v) for v in X_LINES.values()] + [("y", v) for v in Y_LINES.values()]


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segs = [((x, 0.0), (x, COURT_L)) for n, x in X_LINES.items() if n != "centre_line"]
    segs += [((3.05, 0.0), (3.05, 4.72)), ((3.05, 8.68), (3.05, COURT_L))]
    segs += [((0.0, y), (COURT_W, y)) for y in Y_LINES.values()]
    return segs


def sample_court(step: float = 0.1) -> np.ndarray:
    pts = []
    for (x0, y0), (x1, y1) in court_segments():
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
    return np.concatenate(pts)


COURT_SAMPLES = sample_court()


def project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    w = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / w[:, None]
    return xy, w


def white_mask(img: np.ndarray, sat: int = 60, val: int = 160) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < sat) & (hsv[..., 2] > val)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def detect_lines(mask: np.ndarray, floor_y: float, max_lines: int = 24) -> list[np.ndarray]:
    """Infinite image lines (a, b, c), a^2 + b^2 = 1, from Hough segments below ``floor_y``."""
    m = mask.copy()
    m[: max(0, int(floor_y)), :] = 0
    segs = cv2.HoughLinesP(m, 1, np.pi / 360, 60, minLineLength=60, maxLineGap=12)
    if segs is None:
        return []
    found: list[list] = []  # [theta, rho, total_length]
    for x0, y0, x1, y1 in segs.reshape(-1, 4):
        length = float(np.hypot(x1 - x0, y1 - y0))
        theta = np.arctan2(x1 - x0, -(y1 - y0)) % np.pi  # normal angle
        a, b = np.cos(theta), np.sin(theta)
        rho = a * x0 + b * y0
        for f in found:
            dth = abs(f[0] - theta)
            if min(dth, np.pi - dth) < np.radians(1.5) and abs(f[1] - rho * (1 if dth < np.pi / 2 else -1)) < 8:
                f[2] += length
                break
        else:
            found.append([theta, rho, length])
    found.sort(key=lambda f: -f[2])
    return [np.array([np.cos(t), np.sin(t), -r]) for t, r, _ in found[:max_lines]]


def solve(posts: np.ndarray, lines: list[np.ndarray], ids: list[tuple[str, float]], scale: float) -> np.ndarray | None:
    """Homography court -> image from the post points and >= 3 (image line, court line) pairs."""
    S = np.diag([1 / scale, 1 / scale, 1.0])  # condition the image side
    rows = []
    for (X, Y), (u, v) in zip([(0.0, NET_Y), (COURT_W, NET_Y)], posts / scale):
        P = np.array([X, Y, 1.0])
        # the posts are the anchors: weight them so the least squares honours them exactly
        rows.append(100.0 * np.r_[P, np.zeros(3), -u * P])
        rows.append(100.0 * np.r_[np.zeros(3), P, -v * P])
    for l, (kind, c) in zip(lines, ids):
        ln = np.linalg.inv(S).T @ l  # line in the scaled image frame
        d, p = (np.array([0.0, 1.0, 0.0]), np.array([c, 0.0, 1.0])) if kind == "x" else \
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, c, 1.0]))
        rows.append(np.kron(ln, d))
        rows.append(np.kron(ln, p))
    _, sv, vt = np.linalg.svd(np.array(rows))
    if len(rows) < 9 or sv[7] < 1e-9 * sv[0]:  # under-determined (e.g. only 2 lines)
        return None
    Hs = vt[-1].reshape(3, 3)
    H = np.linalg.inv(S) @ Hs
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def plausible(H: np.ndarray, posts: np.ndarray, min_area: float = 0.04) -> bool:
    """A real court: the outline is a convex quad, not collapsed, in front of the camera."""
    quad, wq = project(H, np.array([[0, 0], [COURT_W, 0], [COURT_W, NET_Y], [0, NET_Y]], float))
    if np.any(wq <= 0) or not np.isfinite(quad).all():
        return False
    cross = [_cross2(quad[(k + 1) % 4] - quad[k], quad[(k + 2) % 4] - quad[(k + 1) % 4]) for k in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False
    area = 0.5 * abs(sum(_cross2(quad[k], quad[(k + 1) % 4]) for k in range(4)))
    # the near half reaches from the net toward the camera: a sizeable part of the frame,
    # not a sliver squeezed between the posts (min_area as a fraction of the posts' span squared)
    span = float(np.linalg.norm(posts[1] - posts[0]))
    return area > min_area * span * span * 4


def score(H: np.ndarray, dist: np.ndarray, posts: np.ndarray, cap: float = 15.0,
          min_points: int = 150, post_tol: float = 6.0) -> float:
    h, w = dist.shape
    xy, wq = project(H, COURT_SAMPLES)
    if np.any(wq <= 0) and np.mean(wq <= 0) > 0.5:
        return np.inf
    ok = (wq > 0) & np.isfinite(xy).all(1)
    xy = xy[ok]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)
    pin = np.unique(xy[inside].astype(int), axis=0)
    if len(pin) < 20 or not plausible(H, posts):
        return np.inf
    pe, _ = project(H, np.array([[0.0, NET_Y], [COURT_W, NET_Y]]))
    if np.max(np.linalg.norm(pe - posts, axis=1)) > post_tol:
        return np.inf
    d = np.minimum(dist[pin[:, 1], pin[:, 0]], cap)
    # the near half (y < net) faces the camera: it projects below the net in the image
    near, _ = project(H, np.array([[3.05, 2.0]]))
    net_mid = posts.mean(0)
    if not np.isfinite(near).all() or near[0, 1] <= net_mid[1]:
        return np.inf
    shortfall = max(0, min_points - len(pin)) / min_points
    return float(d.mean() + cap * shortfall)


def snap_points(H: np.ndarray, mask: np.ndarray, radius: int, step: float = 0.05):
    """Sampled court-line points and the nearest paint along each line's normal."""
    h, w = mask.shape
    src, dst, off = [], [], []
    for (x0, y0), (x1, y1) in court_segments():
        a, _ = project(H, np.array([[x0, y0]]))
        b, _ = project(H, np.array([[x1, y1]]))
        n = b[0] - a[0]
        L = np.linalg.norm(n)
        if not np.isfinite(L) or L < 1:
            continue
        nrm = np.array([-n[1], n[0]]) / L
        k = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        for t in np.linspace(0.02, 0.98, k):
            cp = np.array([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
            p, wq = project(H, cp[None])
            p = p[0]
            if wq[0] <= 0 or not (3 < p[0] < w - 3 and 3 < p[1] < h - 3):
                continue
            best = None
            for s in range(-radius, radius + 1):
                q = (p + s * nrm).astype(int)
                if 0 <= q[0] < w and 0 <= q[1] < h and mask[q[1], q[0]] and (best is None or abs(s) < abs(best)):
                    best = s
            if best is not None:
                src.append(cp)
                dst.append(p + best * nrm)
                off.append(abs(best))
    return np.array(src), np.array(dst), np.array(off, float)


def refine(H: np.ndarray, mask: np.ndarray, posts: np.ndarray, post_weight: int = 30) -> np.ndarray:
    post_court = np.array([[0.0, NET_Y], [COURT_W, NET_Y]])
    for radius in (10, 6, 4, 3):
        s, d, _ = snap_points(H, mask, radius)
        if len(s) < 20:
            break
        src = np.r_[s, np.repeat(post_court, post_weight, 0)]
        dst = np.r_[d, np.repeat(posts, post_weight, 0)]
        Hn, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        if Hn is None or not plausible(Hn / Hn[2, 2], posts):
            break
        pe, _ = project(Hn / Hn[2, 2], post_court)
        if np.max(np.linalg.norm(pe - posts, axis=1)) > 10:  # the snap pulled the court off its posts
            break
        H = Hn / Hn[2, 2]
    return H


def residual(H: np.ndarray, mask: np.ndarray, radius: int = 10) -> dict:
    _, _, off = snap_points(H, mask, radius)
    if len(off) == 0:
        return {"points": 0, "median_px": None, "p90_px": None}
    return {"points": int(len(off)), "median_px": round(float(np.median(off)), 2),
            "p90_px": round(float(np.percentile(off, 90)), 2)}


def fit(img: np.ndarray, left: tuple[float, float], right: tuple[float, float], max_lines: int = 22,
        post_tol: float = 30.0) -> dict:
    posts = np.array([left, right], float)
    mask = white_mask(img)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    floor_y = min(left[1], right[1]) - 40
    lines = detect_lines(mask, floor_y, max_lines)
    scale = float(max(img.shape[:2]))
    def through(p):
        return [k for k, l in enumerate(lines) if abs(l @ np.r_[p, 1.0]) < post_tol]

    on_left, on_right = through(posts[0]), through(posts[1])
    others = [k for k in range(len(lines)) if k not in on_left and k not in on_right]
    other_ids = [i for i in IDENTITIES if i not in (("x", 0.0), ("x", COURT_W))]
    hyps = []
    for a in on_left:
        for b in on_right:
            for k in others:
                hyps += [((a, ("x", 0.0)), (b, ("x", COURT_W)), (k, i)) for i in other_ids]
    for side, ident in ((on_left, ("x", 0.0)), (on_right, ("x", COURT_W))):
        for a in side:
            for k, m in itertools.combinations(others, 2):
                hyps += [((a, ident), (k, i), (m, j)) for i in other_ids for j in other_ids if i != j]
    hyps = [h for h in hyps if any(i[0] == "y" for _, i in h)]  # x-lines only: degenerate
    best = (np.inf, None, None)
    for h in hyps:
        H = solve(posts, [lines[k] for k, _ in h], [i for _, i in h], scale)
        if H is None:
            continue
        s = score(H, dist, posts)
        if s < best[0]:
            best = (s, H, h)
    if best[1] is None:
        raise RuntimeError("no court hypothesis fits")
    H = refine(best[1], mask, posts)
    pe, _ = project(H, np.array([[0.0, NET_Y], [COURT_W, NET_Y]]))
    return {"image_from_court": H.tolist(), "score": round(best[0], 3), "lines_detected": len(lines),
            "hypothesis": [list(i) for _, i in best[2]], "hypotheses_tried": len(hyps),
            "post_error_px": [round(float(np.linalg.norm(pe[k] - posts[k])), 2) for k in range(2)],
            "residual": residual(H, mask)}


def court_points(H: np.ndarray, size: tuple[int, int], margin: int = 5) -> dict[str, list[float]]:
    w, h = size
    out = {}
    for yn, y in Y_LINES.items():
        for xn, x in X_LINES.items():
            p, wq = project(H, np.array([[x, y]]))
            if wq[0] > 0 and margin <= p[0, 0] < w - margin and margin <= p[0, 1] < h - margin:
                out[f"{yn}/{xn}"] = [round(float(p[0, 0]), 1), round(float(p[0, 1]), 1)]
    return out


def draw(img: np.ndarray, H: np.ndarray, posts: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    for (x0, y0), (x1, y1) in court_segments() + [((0.0, NET_Y), (COURT_W, NET_Y))]:
        t = np.linspace(0, 1, 40)
        xy, wq = project(H, np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))
        net = y0 == y1 == NET_Y
        for p, q, wp, wqq in zip(xy, xy[1:], wq, wq[1:]):
            if wp > 0 and wqq > 0 and np.isfinite(p).all() and np.isfinite(q).all():
                cv2.line(out, tuple(np.clip(p, -1e4, 1e4).astype(int)), tuple(np.clip(q, -1e4, 1e4).astype(int)),
                         (0, 255, 255) if net else (255, 0, 255), 4 if net else 2)
    if posts is not None:
        for p in posts:
            cv2.circle(out, tuple(np.asarray(p, int)), 9, (0, 0, 255), 2)
    return out


def frames_at(video: Path, fractions=(0.25, 0.5, 0.75)) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fractions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, im = cap.read()
        if ok:
            out.append((int(n * f), im))
    cap.release()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", type=Path)
    ap.add_argument("--left", type=float, nargs=2, required=True, metavar=("X", "Y"), help="left net-post foot (px)")
    ap.add_argument("--right", type=float, nargs=2, required=True, metavar=("X", "Y"), help="right net-post foot (px)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--video", type=Path, help="also measure the residual at 1/4, 1/2, 3/4 of this video")
    ap.add_argument("--max-lines", type=int, default=22)
    a = ap.parse_args(argv)
    img = cv2.imread(str(a.frame))
    if img is None:
        raise SystemExit(f"cannot read {a.frame}")
    res = fit(img, tuple(a.left), tuple(a.right), a.max_lines)
    H = np.array(res["image_from_court"])
    res["image_size"] = [img.shape[1], img.shape[0]]
    res["court_points"] = court_points(H, (img.shape[1], img.shape[0]))
    posts = np.array([a.left, a.right])
    checks = []
    if a.video:
        res["video_check"] = {}
        for f, im in frames_at(a.video):
            res["video_check"][str(f)] = residual(H, white_mask(im))
            checks.append(draw(im, H, posts))
    a.out.write_text(json.dumps(res, indent=1))
    if a.check:
        tiles = [draw(img, H, posts)] + checks
        tiles = [cv2.resize(t, (960, 540)) for t in tiles]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(a.check), cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    print(json.dumps({k: v for k, v in res.items() if k not in ("court_points", "image_from_court")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
