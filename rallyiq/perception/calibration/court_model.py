"""Standard badminton court geometry (BWF), in metres.

World frame (shared project convention):
    origin at a court corner on the floor, x across the court 0..6.10,
    y along the court 0..13.40, y = 0 is the baseline nearer the camera,
    the net is at y = 6.70.

Only lines painted on the floor are modelled. The net tape is not a floor
line and is treated as clutter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

COURT_WIDTH = 6.10
COURT_LENGTH = 13.40
NET_Y = COURT_LENGTH / 2
SINGLES_INSET = 0.46  # singles sideline is 0.46 m inside the doubles sideline
SHORT_SERVICE_FROM_NET = 1.98
DOUBLES_LONG_SERVICE_FROM_BASE = 0.76

# Lines of constant y (parallel to the net), ordered by y.
TRANSVERSE_Y = np.array(
    [
        0.0,
        DOUBLES_LONG_SERVICE_FROM_BASE,
        NET_Y - SHORT_SERVICE_FROM_NET,
        NET_Y + SHORT_SERVICE_FROM_NET,
        COURT_LENGTH - DOUBLES_LONG_SERVICE_FROM_BASE,
        COURT_LENGTH,
    ]
)
TRANSVERSE_NAMES = (
    "near_baseline",
    "near_doubles_long_service",
    "near_short_service",
    "far_short_service",
    "far_doubles_long_service",
    "far_baseline",
)

# Lines of constant x (parallel to the sidelines), ordered by x.
LONGITUDINAL_X = np.array(
    [
        0.0,
        SINGLES_INSET,
        COURT_WIDTH / 2,
        COURT_WIDTH - SINGLES_INSET,
        COURT_WIDTH,
    ]
)
LONGITUDINAL_NAMES = (
    "left_doubles_sideline",
    "left_singles_sideline",
    "centre_line",
    "right_singles_sideline",
    "right_doubles_sideline",
)


@dataclass(frozen=True)
class Segment:
    """A painted line segment on the floor."""

    name: str
    p0: tuple[float, float]
    p1: tuple[float, float]


def _longitudinal_spans(i: int) -> list[tuple[float, float]]:
    if LONGITUDINAL_NAMES[i] == "centre_line":
        # The centre line runs from each short service line to the baseline.
        return [(0.0, TRANSVERSE_Y[2]), (TRANSVERSE_Y[3], COURT_LENGTH)]
    return [(0.0, COURT_LENGTH)]


@lru_cache(maxsize=1)
def segments() -> tuple[Segment, ...]:
    out: list[Segment] = []
    for name, y in zip(TRANSVERSE_NAMES, TRANSVERSE_Y):
        out.append(Segment(name, (0.0, float(y)), (COURT_WIDTH, float(y))))
    for i, (name, x) in enumerate(zip(LONGITUDINAL_NAMES, LONGITUDINAL_X)):
        for k, (y0, y1) in enumerate(_longitudinal_spans(i)):
            suffix = "" if len(_longitudinal_spans(i)) == 1 else ("_near" if k == 0 else "_far")
            out.append(Segment(name + suffix, (float(x), float(y0)), (float(x), float(y1))))
    return tuple(out)


@lru_cache(maxsize=8)
def sample_points(spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points sampled along every painted line.

    Returns (points (N, 2), unit tangents (N, 2)) in court metres.
    """
    pts, tans = [], []
    for seg in segments():
        p0, p1 = np.asarray(seg.p0), np.asarray(seg.p1)
        length = float(np.linalg.norm(p1 - p0))
        n = max(2, int(round(length / spacing)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(p0 + t * (p1 - p0))
        tans.append(np.repeat(((p1 - p0) / length)[None], n, axis=0))
    return np.concatenate(pts), np.concatenate(tans)


@dataclass(frozen=True)
class Keypoint:
    name: str
    xy: tuple[float, float]
    # Directions (in court metres) in which a painted line leaves the point.
    arms: tuple[tuple[float, float], ...]


@lru_cache(maxsize=1)
def keypoints() -> tuple[Keypoint, ...]:
    """All 30 floor-line intersections, each with the directions lines leave it."""
    out = []
    for ti, (tname, y) in enumerate(zip(TRANSVERSE_NAMES, TRANSVERSE_Y)):
        for li, (lname, x) in enumerate(zip(LONGITUDINAL_NAMES, LONGITUDINAL_X)):
            spans = _longitudinal_spans(li)
            if not any(a - 1e-9 <= y <= b + 1e-9 for a, b in spans):
                continue
            arms = []
            if x > 1e-9:
                arms.append((-1.0, 0.0))
            if x < COURT_WIDTH - 1e-9:
                arms.append((1.0, 0.0))
            if any(a - 1e-9 <= y - 1e-3 <= b for a, b in spans):
                arms.append((0.0, -1.0))
            if any(a <= y + 1e-3 <= b + 1e-9 for a, b in spans):
                arms.append((0.0, 1.0))
            out.append(Keypoint(f"{tname}/{lname}", (float(x), float(y)), tuple(arms)))
    return tuple(out)


# Outer (doubles) corners, the default set for manual clicking.
CORNERS = {
    "near_left": (0.0, 0.0),
    "near_right": (COURT_WIDTH, 0.0),
    "far_right": (COURT_WIDTH, COURT_LENGTH),
    "far_left": (0.0, COURT_LENGTH),
}

# The court looks the same under these flips, so line geometry alone fixes the
# homography only up to this group. Each maps court coords -> court coords.
SYMMETRIES = {
    "identity": np.eye(3),
    "flip_x": np.array([[-1.0, 0, COURT_WIDTH], [0, 1, 0], [0, 0, 1]]),
    "flip_y": np.array([[1.0, 0, 0], [0, -1, COURT_LENGTH], [0, 0, 1]]),
    "rotate_180": np.array([[-1.0, 0, COURT_WIDTH], [0, -1, COURT_LENGTH], [0, 0, 1]]),
}
