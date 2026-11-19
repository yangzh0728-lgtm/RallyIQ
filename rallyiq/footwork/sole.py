"""Heel and toe of each planted foot (足底): which way the foot points, and which part landed first.

Needs the ``feet`` stage (heel and toe keypoints from the foot landmark model). Each model foot is
paired with a tracked ankle by the model's own ankle keypoint, so the pose models' left/right swaps do
not mix the feet up.

* **Direction.** While a foot is flat on the floor, heel and toe both lie on the court plane: their
  projections give the foot's direction and length. A projected length far from a real shoe
  (``FLAT_LENGTH_M``) means the projection cannot be trusted for that foot; a player whose typical
  length is off (the far player on a low end-on camera) gets no foot directions at all.
* **Landing.** Around the moment the foot settles (``TOUCHDOWN_S`` either side), the toe is compared
  with where it would be on a flat foot with the heel down. The difference, in metres at the foot's image scale,
  is the foot's pitch: toe up = heel strike (脚跟先着地, right for a lunge), heel up = forefoot landing
  (前脚掌着地, right for moving and split steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import NET_Y_M, FootLandmarks
from .tracks import Run, apply_h

MIN_CONF = 0.5  # heel and toe visibility from the model
MATCH_M = 0.25  # the model's ankle must be this close to the tracked ankle (metres at the foot's scale)
FLAT_LENGTH_M = (0.14, 0.42)  # plausible projected heel-to-toe length of a flat shoe
PLAYER_LENGTH_M = (0.17, 0.36)  # a player's median projected length must be in here to use directions
TOUCHDOWN_S = 0.1
LIFT_MIN_M = 0.04  # toe (or heel) this far above a flat foot at touchdown
MIN_FLAT_FRAMES = 3


def attach_feet(runs: list[Run], feet: FootLandmarks | None) -> None:
    """Fill ``run.heel`` / ``run.toe`` from the foot model, paired with the tracked ankles."""
    if feet is None or not feet.frames:
        return
    idx = {(f.frame, p.player_id): p for f in feet.frames for p in f.players if p.player_id}
    for run in runs:
        n = len(run.frames)
        heel = np.full((n, 2, 2), np.nan)
        toe = np.full((n, 2, 2), np.nan)
        for i, fr in enumerate(run.frames):
            pf = idx.get((int(fr), run.pid))
            if pf is None:
                continue
            a = np.asarray(pf.ankle, dtype=float)
            h = np.asarray(pf.heel, dtype=float)
            t = np.asarray(pf.toe, dtype=float)
            tr = run.ankle[i]
            keep = np.linalg.norm(a[0, :2] - tr[0]) + np.linalg.norm(a[1, :2] - tr[1])
            cross = np.linalg.norm(a[0, :2] - tr[1]) + np.linalg.norm(a[1, :2] - tr[0])
            order = (0, 1) if keep <= cross else (1, 0)
            for j in range(2):
                m = order[j]
                if np.linalg.norm(a[m, :2] - tr[j]) / max(run.scale[i, j], 1e-6) > MATCH_M:
                    continue
                if h[m, 2] < MIN_CONF or t[m, 2] < MIN_CONF:
                    continue
                heel[i, j] = h[m, :2]
                toe[i, j] = t[m, :2]
        run.heel, run.toe = heel, toe


@dataclass
class Sole:
    heel_image: np.ndarray  # px, median while flat
    toe_image: np.ndarray
    heel_court: np.ndarray
    toe_court: np.ndarray
    length_m: float
    direction: np.ndarray  # unit heel -> toe on the court
    landing: str | None  # "heel" | "forefoot" | "flat" | None (not judged)


def _pitch(run: Run, j: int, ii: np.ndarray, length: float, u: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    """How far the toe is above where it would be on a flat foot with the heel down (m) at indices ``ii``.

    Positive = toe up (heel strike), negative = heel up (forefoot). With only two points per foot the
    "heel up" reading is the same number with the sign flipped, so one value covers both."""
    heel = run.heel[ii, j]
    toe = run.toe[ii, j]
    toe_floor = apply_h(Hinv, apply_h(run.H, heel) + length * u)
    return (toe_floor[:, 1] - toe[:, 1]) / run.scale[ii, j]


def plant_sole(run: Run, j: int, a: int, b: int, fps: float) -> Sole | None:
    """Heel / toe of foot ``j`` planted over run indices [a, b]; None without usable keypoints."""
    if run.heel is None or run.toe is None:
        return None
    k = max(2, int(round(TOUCHDOWN_S * fps)))
    lo = a + k if b - a + 1 >= 2 * k + MIN_FLAT_FRAMES else a
    ii = np.arange(lo, b + 1)
    ok = np.isfinite(run.heel[ii, j]).all(axis=1) & np.isfinite(run.toe[ii, j]).all(axis=1)
    ii = ii[ok]
    if len(ii) < MIN_FLAT_FRAMES:
        return None
    hc = apply_h(run.H, run.heel[ii, j])
    tc = apply_h(run.H, run.toe[ii, j])
    h_med, t_med = np.median(hc, axis=0), np.median(tc, axis=0)
    v = t_med - h_med
    length = float(np.linalg.norm(v))
    if not FLAT_LENGTH_M[0] <= length <= FLAT_LENGTH_M[1]:
        return None
    u = v / length
    sole = Sole(np.median(run.heel[ii, j], axis=0), np.median(run.toe[ii, j], axis=0), h_med, t_med, length, u, None)
    # landing: only for a real touchdown (the foot was in the air just before), with a flat reference.
    # The ankle counts as planted around the moment the foot settles, so a heel strike can show just
    # before or just after that: the foot is judged over ``TOUCHDOWN_S`` either side. The second most
    # extreme reading is used so that one bad keypoint frame cannot decide it.
    if a < k or lo == a:
        return sole
    td = np.arange(a - k, a + k)
    td = td[np.isfinite(run.heel[td, j]).all(axis=1) & np.isfinite(run.toe[td, j]).all(axis=1)]
    if len(td) < 3:
        return sole
    Hinv = np.linalg.inv(run.H)
    p = np.sort(_pitch(run, j, td, length, u, Hinv))
    toe_up, heel_up = float(p[-2]), float(-p[1])
    noise = float(np.median(np.abs(_pitch(run, j, ii, length, u, Hinv))))
    thr = max(LIFT_MIN_M, 2.5 * noise)
    if toe_up >= thr and heel_up < thr:
        sole.landing = "heel"
    elif heel_up >= thr and toe_up < thr:
        sole.landing = "forefoot"
    else:
        sole.landing = "flat"
    return sole


def toe_angle_deg(direction: np.ndarray, at_y: float) -> float:
    """Foot direction in the player's own half: 0 = towards the net, +90 = to the player's right."""
    dx, dy = float(direction[0]), float(direction[1])
    if at_y > NET_Y_M:  # far half: the player faces the other way
        dx, dy = -dx, -dy
    return float(np.degrees(np.arctan2(dx, dy)))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / (nu * nv), -1.0, 1.0))))
