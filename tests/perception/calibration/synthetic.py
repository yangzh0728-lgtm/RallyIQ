"""Synthetic court frames with known ground truth, for tests and tuning."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from rallyiq.perception.calibration import court_model as cm

PX_PER_M = 80
FLOOR_X = (-9.0, 15.1)
FLOOR_Y = (-6.0, 19.4)


@dataclass
class Camera:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float] = (cm.COURT_WIDTH / 2, cm.COURT_LENGTH / 2, 0.0)
    focal: float = 1000.0
    size: tuple[int, int] = (1280, 720)
    roll_deg: float = 0.0

    def projection(self) -> np.ndarray:
        """3x4 projection matrix P = K [R | t]."""
        c = np.array(self.position, float)
        f = np.array(self.look_at, float) - c
        f /= np.linalg.norm(f)
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        R = np.stack([right, down, f])
        a = np.deg2rad(self.roll_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        R = Rz @ R
        w, h = self.size
        K = np.array([[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]])
        return K @ np.hstack([R, (-R @ c)[:, None]])

    def homography(self) -> np.ndarray:
        """image_from_court for the floor plane z = 0."""
        P = self.projection()
        H = P[:, [0, 1, 3]]
        return H / H[2, 2]


@dataclass
class Scene:
    camera: Camera
    floor_bgr: tuple[int, int, int] = (60, 120, 40)  # green mat
    surround_bgr: tuple[int, int, int] = (70, 90, 110)
    wall_bgr: tuple[int, int, int] = (150, 150, 140)
    line_bgr: tuple[int, int, int] = (235, 235, 235)
    adjacent_courts: bool = False
    net: bool = True
    players: int = 2
    noise: float = 4.0
    blur: float = 0.0
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _draw_court(tex: np.ndarray, x_off: float, colour, width_m: float = 0.04):
    def to_px(x, y):
        return (
            int(round((x + x_off - FLOOR_X[0]) * PX_PER_M)),
            int(round((y - FLOOR_Y[0]) * PX_PER_M)),
        )

    th = max(1, int(round(width_m * PX_PER_M)))
    for seg in cm.segments():
        cv2.line(tex, to_px(*seg.p0), to_px(*seg.p1), colour, th, cv2.LINE_AA)


def render(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR image, ground-truth image_from_court)."""
    rng = np.random.default_rng(scene.seed)
    cam = scene.camera
    w, h = cam.size
    tw = int((FLOOR_X[1] - FLOOR_X[0]) * PX_PER_M)
    th = int((FLOOR_Y[1] - FLOOR_Y[0]) * PX_PER_M)
    tex = np.full((th, tw, 3), scene.surround_bgr, np.uint8)
    # Court mat a bit larger than the court.
    m = 1.5
    p0 = (int((-m - FLOOR_X[0]) * PX_PER_M), int((-m - FLOOR_Y[0]) * PX_PER_M))
    p1 = (int((cm.COURT_WIDTH + m - FLOOR_X[0]) * PX_PER_M), int((cm.COURT_LENGTH + m - FLOOR_Y[0]) * PX_PER_M))
    cv2.rectangle(tex, p0, p1, scene.floor_bgr, -1)
    if scene.adjacent_courts:
        for off in (-(cm.COURT_WIDTH + 2.0), cm.COURT_WIDTH + 2.0):
            q0 = (p0[0] + int(off * PX_PER_M), p0[1])
            q1 = (p1[0] + int(off * PX_PER_M), p1[1])
            cv2.rectangle(tex, q0, q1, scene.floor_bgr, -1)
            _draw_court(tex, off, scene.line_bgr)
    _draw_court(tex, 0.0, scene.line_bgr)
    # Floor texture / lighting variation.
    grad = np.linspace(0.85, 1.1, tw)[None, :, None]
    tex = np.clip(tex.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    H = cam.homography()
    T = np.array([[1 / PX_PER_M, 0, FLOOR_X[0]], [0, 1 / PX_PER_M, FLOOR_Y[0]], [0, 0, 1]])  # court_from_tex
    H_img_tex = H @ T
    img = cv2.warpPerspective(
        tex, H_img_tex, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=scene.wall_bgr
    )
    # Anything at or above the horizon is wall, not floor.
    hl = H[2]  # horizon: w = 0 in court coords -> image line H^-T [0,0,1]
    horizon = np.linalg.inv(H).T @ np.array([0, 0, 1.0])
    ys, xs = np.mgrid[0:h, 0:w]
    side = horizon[0] * xs + horizon[1] * ys + horizon[2]
    ref = horizon @ (H @ np.array([cm.COURT_WIDTH / 2, cm.COURT_LENGTH / 2, 1.0]))
    img[np.sign(side) != np.sign(ref)] = scene.wall_bgr
    del hl

    P = cam.projection()

    def proj3(X):
        X = np.asarray(X, float)
        x = P @ np.append(X, 1.0)
        return (int(round(x[0] / x[2])), int(round(x[1] / x[2]))), x[2]

    if scene.net:
        a, _ = proj3((-0.0, cm.NET_Y, 0))
        b, _ = proj3((cm.COURT_WIDTH, cm.NET_Y, 0))
        a_top, _ = proj3((0.0, cm.NET_Y, 1.55))
        b_top, _ = proj3((cm.COURT_WIDTH, cm.NET_Y, 1.55))
        a_mid, _ = proj3((0.0, cm.NET_Y, 0.79))
        b_mid, _ = proj3((cm.COURT_WIDTH, cm.NET_Y, 0.79))
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, np.array([a_mid, b_mid, b_top, a_top]), (30, 30, 30))
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.line(img, a_top, b_top, (245, 245, 245), 3, cv2.LINE_AA)  # white net tape: a distractor line
        cv2.line(img, a, a_top, (40, 40, 40), 3)
        cv2.line(img, b, b_top, (40, 40, 40), 3)

    for i in range(scene.players):
        x = rng.uniform(0.8, cm.COURT_WIDTH - 0.8)
        y = rng.uniform(1.0, 5.5) if i % 2 == 0 else rng.uniform(7.9, 12.4)
        feet, z = proj3((x, y, 0))
        head, _ = proj3((x, y, 1.75))
        if z <= 0:
            continue
        height = max(4, feet[1] - head[1])
        width = max(3, int(height * 0.3))
        shirt = (240, 240, 240) if i == 0 else (40, 40, 200)
        cv2.rectangle(img, (feet[0] - width // 2, head[1] + height // 6), (feet[0] + width // 2, feet[1] - height // 2), shirt, -1)
        cv2.rectangle(img, (feet[0] - width // 2, feet[1] - height // 2), (feet[0] + width // 2, feet[1]), (20, 20, 20), -1)
        cv2.circle(img, (feet[0], head[1] + height // 12), max(2, height // 12), (150, 180, 220), -1)

    if scene.blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), scene.blur)
    if scene.noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, scene.noise, img.shape), 0, 255).astype(np.uint8)
    return img, H


def keypoint_error(H_est: np.ndarray, H_true: np.ndarray, size: tuple[int, int]) -> float:
    """Max px error over court keypoints visible in frame, up to court symmetry."""
    w, h = size
    pts = np.array([k.xy for k in cm.keypoints()] + list(cm.CORNERS.values()))

    def proj(H, p):
        q = np.concatenate([p, np.ones((len(p), 1))], 1) @ H.T
        return q[:, :2] / q[:, 2:3], q[:, 2]

    gt, wz = proj(H_true, pts)
    vis = (wz > 0) & (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
    best = np.inf
    for S in cm.SYMMETRIES.values():
        est, _ = proj(H_est @ S, pts)
        best = min(best, float(np.max(np.linalg.norm(est[vis] - gt[vis], axis=1))))
    return best
