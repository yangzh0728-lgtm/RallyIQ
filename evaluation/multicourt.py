"""Synthetic multi-court benchmark for the target-court lock.

Real side-court footage (``ko26_court2/3`` in ``clips.yaml``) has no labels yet,
so this builds controlled scenes: our singles rally on the target court and a
second rally on a neighbouring court, with the neighbour's shuttle deliberately
*more* visible (higher detector score). It measures how much of the shuttle
track ends up on the wrong court with and without the court lock.

    python -m evaluation.multicourt          # prints the table

The image model is approximate (a point at height z is drawn ``z`` metres above
its floor point at the local ground scale), but it is the geometry that matters:
side-by-side courts, and courts end to end behind the far baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rallyiq.contracts.geometry import apply_homography, homography_from_corners, invert_homography
from rallyiq.contracts.models import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    CalibrationSegment,
    CourtCalibration,
    PlayerDetection,
    PlayerFrame,
    PlayerTracks,
    VideoMeta,
)

W, L = COURT_WIDTH_M, COURT_LENGTH_M
IMAGE = (1920, 1080)
FPS = 30.0

CAMERAS = {
    # Image corners of the target court: near-left, near-right, far-right, far-left.
    "behind_raised": [[620, 980], [1300, 980], [1110, 420], [810, 420]],
    "broadcast": [[560, 900], [1360, 900], [1180, 330], [740, 330]],
}
# Offset of the neighbouring court in target-court metres (BWF: >= 2 m between courts).
LAYOUTS = {"side_by_side": (W + 2.0, 0.0), "end_to_end": (0.0, L + 3.0)}


@dataclass
class Scene:
    meta: VideoMeta
    calibration: CourtCalibration
    players: PlayerTracks
    candidates: dict[int, list[tuple[float, float, float]]]
    ours: dict[int, tuple[float, float]]  # ground truth, target-court shuttle
    theirs: dict[int, tuple[float, float]]  # neighbour-court shuttle


class _Cam:
    def __init__(self, corners):
        self.H = homography_from_corners(corners)  # image -> court
        self.Hinv = invert_homography(self.H)

    def ground(self, x: float, y: float) -> np.ndarray:
        return np.asarray(apply_homography(self.Hinv, [[x, y]])[0])

    def scale(self, x: float, y: float) -> float:
        a, b = apply_homography(self.Hinv, [[x - 0.5, y], [x + 0.5, y]])
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))

    def air(self, x: float, y: float, z: float) -> tuple[float, float]:
        g = self.ground(x, y)
        return float(g[0]), float(g[1] - self.scale(x, y) * z)


def _rally(frames: int, ox: float, oy: float, phase: int, seed: int):
    """Shuttle (x, y, z) per frame and the two players' floor positions, in target-court metres."""
    rng = np.random.default_rng(seed)
    near = np.array([W / 2 + ox, 2.0 + oy])
    far = np.array([W / 2 + ox, L - 2.0 + oy])
    flight = int(FPS * 1.0)
    shuttle, feet = {}, {}
    a, b = near.copy(), far.copy()
    for f in range(frames):
        k, t = divmod(f + phase, flight)
        if t == 0:
            # Each player moves to where they hit from.
            a = near + rng.uniform([-1.5, -1.0], [1.5, 1.5])
            b = far + rng.uniform([-1.5, -1.5], [1.5, 1.0])
        src, dst = (a, b) if k % 2 == 0 else (b, a)
        u = t / flight
        xy = src + (dst - src) * u
        z = 2.4 + 4.0 * math.sin(math.pi * u)
        shuttle[f] = (float(xy[0]), float(xy[1]), z)
        feet[f] = (a.copy(), b.copy())
    return shuttle, feet


def make_scene(camera: str, layout: str, frames: int = 300, seed: int = 0,
               ours_score: float = 0.6, theirs_score: float = 0.85, report_other_court: bool = True) -> Scene:
    cam = _Cam(CAMERAS[camera])
    ox, oy = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    meta = VideoMeta(video_id=f"sim_{camera}_{layout}", filename="sim.mp4", fps=FPS, frame_count=frames,
                     width=IMAGE[0], height=IMAGE[1], duration_ms=int(frames / FPS * 1000))
    other = [cam.ground(x + ox, y + oy).tolist() for x, y in [[0, 0], [W, 0], [W, L], [0, L]]]
    seg = CalibrationSegment(start_frame=0, end_frame=frames - 1, homography=cam.H,
                             image_corners=[cam.ground(x, y).tolist() for x, y in [[0, 0], [W, 0], [W, L], [0, L]]],
                             other_courts=[other] if report_other_court else [], confidence=0.9, status="ok")
    calibration = CourtCalibration(video_id=meta.video_id, segments=[seg])

    s_ours, f_ours = _rally(frames, 0.0, 0.0, 0, seed + 1)
    s_theirs, _ = _rally(frames, ox, oy, 13, seed + 2)

    def box(x, y):
        fx, fy = cam.ground(x, y)
        h = cam.scale(x, y) * 1.75
        return [fx - 0.18 * h, fy - h, fx + 0.18 * h, fy]

    pframes = []
    for f in range(frames):
        dets = []
        for pid, (x, y) in zip(("P1", "P2"), f_ours[f]):
            b = box(x, y)
            dets.append(PlayerDetection(player_id=pid, track_id=0 if pid == "P1" else 1, bbox=b,
                                        keypoints=[[b[0], b[1], 0.9]] * 17, foot_px=[(b[0] + b[2]) / 2, b[3]],
                                        foot_court=[float(x), float(y)], on_court=True, confidence=0.9))
        pframes.append(PlayerFrame(frame=f, t_ms=meta.t_ms(f), players=dets))
    players = PlayerTracks(video_id=meta.video_id, model="sim", frames=pframes)

    cands: dict[int, list[tuple[float, float, float]]] = {}
    ours, theirs = {}, {}
    for f in range(frames):
        cs = []
        for truth, store, score, miss in ((s_ours, ours, ours_score, 0.15), (s_theirs, theirs, theirs_score, 0.05)):
            x, y = cam.air(*truth[f])
            if not (0 <= x < IMAGE[0] and 0 <= y < IMAGE[1]):
                continue
            store[f] = (x, y)
            if rng.random() > miss:
                cs.append((x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), float(score)))
        if rng.random() < 0.1:  # background clutter
            cs.append((float(rng.uniform(0, IMAGE[0])), float(rng.uniform(0, IMAGE[1])), 0.55))
        cands[f] = cs
    return Scene(meta, calibration, players, cands, ours, theirs)


def score_track(scene: Scene, points, tol_px: float = 8.0) -> dict[str, float]:
    det = [p for p in points if p.source == "detected"]

    def hits(truth, p):
        t = truth.get(p.frame)
        return t is not None and math.hypot(p.x - t[0], p.y - t[1]) <= tol_px

    ours = sum(hits(scene.ours, p) for p in det)
    theirs = sum(hits(scene.theirs, p) for p in det)
    return {
        "precision": ours / len(det) if det else 0.0,
        "recall": ours / max(len(scene.ours), 1),
        "foreign_rate": theirs / len(det) if det else 0.0,
    }


def run(camera: str, layout: str, lock: bool, seed: int = 0, report_other_court: bool = True,
        use_calibration: bool = True) -> dict[str, float]:
    from rallyiq.perception.shuttle.stage import build_track, court_lock

    scene = make_scene(camera, layout, seed=seed, report_other_court=report_other_court)
    region = support = None
    if lock:
        region, support, _ = court_lock(scene.calibration if use_calibration else None, scene.players)
    track, _ = build_track(scene.meta, scene.candidates, IMAGE[0], IMAGE[1], "sim", None, region, support)
    return score_track(scene, track.points)


def main() -> None:
    rows = []
    variants = [
        ("off", dict(lock=False)),
        ("on", dict(lock=True)),
        ("on, other court not reported", dict(lock=True, report_other_court=False)),
        ("on, no calibration", dict(lock=True, use_calibration=False)),
    ]
    print(f"{'camera':15} {'layout':13} {'court lock':30} {'precision':>9} {'recall':>7} {'foreign':>8}")
    for camera in CAMERAS:
        for layout in LAYOUTS:
            for name, kw in variants:
                ms = [run(camera, layout, seed=s, **kw) for s in range(5)]
                m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
                rows.append((camera, layout, name, m))
                print(f"{camera:15} {layout:13} {name:30} {m['precision']:9.2f} {m['recall']:7.2f} "
                      f"{m['foreign_rate']:8.2f}")


if __name__ == "__main__":
    main()
