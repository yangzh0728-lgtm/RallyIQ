"""Lock analysis onto one target court when several courts are visible.

Calibration picks the target court (``CalibrationSegment.homography``) and lists
the other detected courts (``CalibrationSegment.other_courts``). This module
turns that into regions every perception stage can use to reject people and
shuttles that belong to a neighbouring match:

* **Floor**: the target court plus run-off where active players can stand
  (players lunge and retreat past the lines, so the margin is generous).
  A foot point standing on another detected court is rejected outright.
* **Air**: shuttles fly, so their image position cannot be mapped to the floor.
  We use the image footprint of the box above the (enlarged) target court, up
  to ``air_height_m``. Its vertical extent comes from the local ground scale,
  which is approximate but only used with a soft fall-off.

Without calibration, :class:`PlayerSpanRegion` gives a weaker fallback: the
horizontal band spanned by the two tracked players.

All of this is numpy only (no OpenCV), and decisions are soft weights in
``[0, 1]`` so trajectory-level logic, not single frames, makes the final call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..contracts.geometry import COURT_CORNERS_M, fit_homography
from ..contracts.models import COURT_LENGTH_M, COURT_WIDTH_M, CourtCalibration, PlayerTracks

W, L = COURT_WIDTH_M, COURT_LENGTH_M


@dataclass(frozen=True)
class RegionConfig:
    # Floor run-off for players (metres beyond the outer lines). Same as players.PlayArea.
    floor_margin_side_m: float = 1.0
    floor_margin_end_m: float = 2.0
    # Footprint of the air box for shuttles. Wider than the floor: shuttles that land
    # out still belong to this rally, and clears go high above the baseline.
    air_margin_side_m: float = 1.5
    air_margin_end_m: float = 2.5
    air_height_m: float = 10.0
    # Height profile: a shuttle is high only over the middle of the court. Above a
    # baseline it is at most ``air_height_end_m`` (falling into the back corner or
    # just hit), rising by ``air_height_slope`` per metre inside; past a sideline it
    # drops by the same slope. Never below ``air_height_min_m``.
    air_height_end_m: float = 4.5
    air_height_slope: float = 1.5
    air_height_min_m: float = 3.0
    # Outside the air box the weight falls off with this length (metres at the
    # local ground scale) and is 0 beyond ``air_reject_m``.
    air_decay_m: float = 1.0
    air_reject_m: float = 3.0
    # Inside our air box but also inside another court's (perspective makes the boxes
    # overlap in the image): ambiguous, so the trajectory and player evidence decide.
    air_shared_weight: float = 0.6
    # A foot point counts as standing on another court when it is inside that
    # court's lines plus this margin.
    other_court_margin_m: float = 0.3


# --------------------------------------------------------------------- geometry


def _project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map (N, 2) points; returns (xy, w) with the raw homogeneous w."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    w = hom[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = hom[:, :2] / w[:, None]
    return xy, w


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull, counter-clockwise in a y-up frame. (M, 2)."""
    p = sorted({(float(x), float(y)) for x, y in np.asarray(pts).reshape(-1, 2)})
    if len(p) <= 2:
        return np.asarray(p, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def distance_to_polygon(poly: np.ndarray, x: float, y: float) -> float:
    """0 inside, else distance to the nearest edge (px)."""
    if len(poly) >= 3 and point_in_polygon(poly, x, y):
        return 0.0
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom == 0 else max(0.0, min(1.0, float((np.array([x, y]) - a) @ ab) / denom))
        q = a + t * ab
        best = min(best, math.hypot(x - q[0], y - q[1]))
    return best


def _rect(margin_side: float, margin_end: float) -> np.ndarray:
    return np.array([
        [-margin_side, -margin_end], [W + margin_side, -margin_end],
        [W + margin_side, L + margin_end], [-margin_side, L + margin_end],
    ])


def _densify(rect: np.ndarray, n: int = 4) -> np.ndarray:
    out = []
    for i in range(4):
        a, b = rect[i], rect[(i + 1) % 4]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(a + t * (b - a))
    return np.asarray(out)


def _distance_outside_rect(xy: np.ndarray, ms: float, me: float) -> float:
    x, y = float(xy[0]), float(xy[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.inf
    dx = max(-ms - x, 0.0, x - (W + ms))
    dy = max(-me - y, 0.0, y - (L + me))
    return math.hypot(dx, dy)


class _Court:
    """One court seen through an image->court homography."""

    def __init__(self, H_img_to_court: np.ndarray):
        self.H = np.asarray(H_img_to_court, dtype=np.float64)
        self.Hinv = np.linalg.inv(self.H)
        # The overall sign of a homography is arbitrary; points in front of the
        # camera share the sign of w at the court centre.
        _, wc = _project(self.Hinv, np.array([[W / 2, L / 2]]))
        self.front = 1.0 if wc[0] >= 0 else -1.0

    def to_court(self, px: Sequence[float]) -> np.ndarray:
        xy, w = _project(self.H, np.asarray(px)[None])
        return xy[0]

    def to_image(self, court_pts: np.ndarray) -> np.ndarray:
        """Image points of court points in front of the camera (others dropped)."""
        xy, w = _project(self.Hinv, court_pts)
        ok = (w * self.front > 1e-9) & np.all(np.isfinite(xy), axis=1)
        return xy[ok]

    def px_per_m(self, court_pt: np.ndarray) -> float | None:
        a = self.to_image(np.array([[court_pt[0] - 0.5, court_pt[1]], [court_pt[0] + 0.5, court_pt[1]]]))
        if len(a) != 2:
            return None
        v = float(np.linalg.norm(a[1] - a[0]))
        return v if v > 0 else None

    def scales(self, court_pts: np.ndarray) -> np.ndarray:
        """Local ground scale (px per metre across the court) at each court point; NaN if behind."""
        p = np.asarray(court_pts, dtype=np.float64).reshape(-1, 2)
        a, wa = _project(self.Hinv, p - [0.5, 0.0])
        b, wb = _project(self.Hinv, p + [0.5, 0.0])
        s = np.linalg.norm(b - a, axis=1)
        s[(wa * self.front <= 1e-9) | (wb * self.front <= 1e-9)] = np.nan
        return s

    def height_above(self, x: float, y: float, margin_side: float, margin_end: float, zmax,
                     max_lift_px: float, steps: int = 64) -> float | None:
        """Lowest height (m, <= ``zmax``, a number or a function of court points) at which a point drawn at image (x, y) can
        hang above the enlarged court, or None.

        Walks down the image column below (x, y): every pixel there that lands on
        the court footprint is a floor point the shuttle could be above, at a height
        of (pixels walked) / (local px per metre). The camera is assumed upright
        (vertical lines stay vertical), as everywhere else in this module.
        """
        dv = np.linspace(0.0, max_lift_px, steps)
        pts = np.stack([np.full_like(dv, x), y + dv], axis=1)
        court, w = _project(self.H, pts)
        ok = np.all(np.isfinite(court), axis=1)
        ok &= (court[:, 0] >= -margin_side) & (court[:, 0] <= W + margin_side)
        ok &= (court[:, 1] >= -margin_end) & (court[:, 1] <= L + margin_end)
        if not ok.any():
            return None
        sc = self.scales(court[ok])
        z = dv[ok] / sc
        lim = zmax(court[ok]) if callable(zmax) else zmax
        z = z[np.isfinite(z) & (z <= lim)]
        return float(z.min()) if len(z) else None

    def air_polygon(self, margin_side: float, margin_end: float, height_m: float) -> np.ndarray | None:
        ground = _densify(_rect(margin_side, margin_end))
        pts = []
        for g in ground:
            img = self.to_image(g[None])
            if not len(img):
                continue
            s = self.px_per_m(g)
            pts.append(img[0])
            if s:
                pts.append(img[0] + np.array([0.0, -s * height_m]))
        if len(pts) < 3:
            return None
        return convex_hull(np.asarray(pts))


# ----------------------------------------------------------------------- regions


class CourtRegion:
    """Target court (and known neighbours) for one camera segment."""

    def __init__(
        self,
        H_img_to_court: Sequence[Sequence[float]],
        other_courts: Sequence[Sequence[Sequence[float]]] = (),
        cfg: RegionConfig | None = None,
    ):
        self.cfg = cfg or RegionConfig()
        c = self.cfg
        self.target = _Court(np.asarray(H_img_to_court, dtype=np.float64))
        self.others: list[_Court] = []
        for corners in other_courts or ():
            try:
                self.others.append(_Court(np.asarray(fit_homography(corners, COURT_CORNERS_M))))
            except (ValueError, np.linalg.LinAlgError):
                continue
        self.air = self.target.air_polygon(c.air_margin_side_m, c.air_margin_end_m, c.air_height_m)
        # Neighbour air boxes use the lines only, so they never eat into our run-off.
        # Convex hulls are a cheap pre-test; the exact test walks the image column.
        self.other_air_hulls = [o.air_polygon(0.0, 0.0, c.air_height_m) for o in self.others]
        self.other_air = [p for p in self.other_air_hulls if p is not None]
        centre_scale = self.target.px_per_m(np.array([W / 2, L / 2]))
        self.scale_px_per_m = centre_scale or 1.0
        self._max_lift = [self._lift(o) for o in [self.target, *self.others]]

    def _lift(self, court: _Court) -> float:
        c = self.cfg
        s = court.scales(_densify(_rect(c.air_margin_side_m, c.air_margin_end_m)))
        s = s[np.isfinite(s)]
        return float(s.max() * c.air_height_m) if len(s) else 0.0

    def _in_air(self, i: int, x: float, y: float, margin_side: float, margin_end: float) -> bool:
        court = self.target if i == 0 else self.others[i - 1]
        return court.height_above(x, y, margin_side, margin_end, self.max_height, self._max_lift[i]) is not None

    def max_height(self, court_pts: np.ndarray) -> np.ndarray:
        """Highest plausible shuttle height (m) above each court floor point."""
        c = self.cfg
        x, y = court_pts[:, 0], court_pts[:, 1]
        inside_end = np.minimum(y, L - y)
        past_side = np.maximum(np.maximum(-x, x - W), 0.0)
        z = c.air_height_end_m + c.air_height_slope * (inside_end - past_side)
        return np.clip(z, c.air_height_min_m, c.air_height_m)

    # -- floor (players) --------------------------------------------------------
    def floor_outside_m(self, foot_px: Sequence[float]) -> float:
        """Metres outside the target play area (0 inside, inf if unmappable)."""
        return _distance_outside_rect(self.target.to_court(foot_px), self.cfg.floor_margin_side_m,
                                      self.cfg.floor_margin_end_m)

    def on_other_court(self, foot_px: Sequence[float]) -> bool:
        """True when the foot stands on a neighbouring court and not in our play area."""
        if not self.others or self.floor_outside_m(foot_px) == 0.0:
            return False
        m = self.cfg.other_court_margin_m
        return any(_distance_outside_rect(o.to_court(foot_px), m, m) == 0.0 for o in self.others)

    # -- air (shuttle) ----------------------------------------------------------
    def air_weight(self, x: float, y: float) -> float:
        """1 inside the air box over the target court, decaying outside it; 0 when
        the point is clearly in the air over another detected court."""
        c = self.cfg
        if self.air is None:
            return 1.0
        in_other = any(
            self._in_air(i + 1, x, y, 0.0, 0.0)
            for i, poly in enumerate(self.other_air_hulls)
            if poly is not None and point_in_polygon(poly, x, y)
        )
        if self._in_air(0, x, y, c.air_margin_side_m, c.air_margin_end_m):
            return c.air_shared_weight if in_other else 1.0
        if in_other:
            return 0.0
        # Outside: fall off with the sideways distance to our air space, probing left
        # and right in steps of a quarter metre at the court-centre scale.
        step = 0.25 * self.scale_px_per_m
        for k in range(1, int(c.air_reject_m / 0.25) + 1):
            for sgn in (-1.0, 1.0):
                if self._in_air(0, x + sgn * k * step, y, c.air_margin_side_m, c.air_margin_end_m):
                    return float(math.exp(-(k * 0.25) / c.air_decay_m))
        return 0.0

    def floor_polygon_px(self) -> np.ndarray:
        c = self.cfg
        return self.target.to_image(_rect(c.floor_margin_side_m, c.floor_margin_end_m))


class PlayerSpanRegion:
    """Fallback air region without calibration: the horizontal band covered by
    the two tracked players, padded by a couple of body heights.

    Only constrains x: it is enough to drop shuttles from the courts beside us in
    the common behind-the-baseline view, and never rejects high clears.
    """

    def __init__(self, x_lo: float, x_hi: float, body_px: float, pad_bodies: float = 1.5,
                 decay_bodies: float = 1.0, reject_bodies: float = 3.0):
        self.x_lo, self.x_hi = x_lo - pad_bodies * body_px, x_hi + pad_bodies * body_px
        self.body = max(body_px, 1.0)
        self.decay, self.reject = decay_bodies, reject_bodies

    def air_weight(self, x: float, y: float) -> float:
        d = max(self.x_lo - x, 0.0, x - self.x_hi) / self.body
        if d == 0.0:
            return 1.0
        if d >= self.reject:
            return 0.0
        return float(math.exp(-d / self.decay))

    @classmethod
    def from_players(cls, players: PlayerTracks | None, min_frames: int = 15) -> PlayerSpanRegion | None:
        if players is None:
            return None
        xs, hs = [], []
        for f in players.frames:
            for p in f.players:
                if p.player_id not in ("P1", "P2") or p.interpolated:
                    continue
                x1, y1, x2, y2 = p.bbox
                xs.append(p.foot_px[0] if p.foot_px else (x1 + x2) / 2)
                hs.append(y2 - y1)
        if len(xs) < min_frames:
            return None
        lo, hi = np.percentile(xs, [2, 98])
        return cls(float(lo), float(hi), float(np.median(hs)))


# ------------------------------------------------------------------ per-frame lookup


class SegmentRegions:
    """``CourtRegion`` per calibration segment, looked up by frame."""

    def __init__(self, calibration: CourtCalibration | None, cfg: RegionConfig | None = None):
        self._segs: list[tuple[int, int, CourtRegion]] = []
        for s in (calibration.segments if calibration else []):
            if s.homography is None or s.status == "undeterminable":
                continue
            try:
                r = CourtRegion(s.homography, getattr(s, "other_courts", None) or (), cfg)
            except np.linalg.LinAlgError:
                continue
            self._segs.append((s.start_frame, s.end_frame, r))

    def __bool__(self) -> bool:
        return bool(self._segs)

    def for_frame(self, frame: int) -> CourtRegion | None:
        for a, b, r in self._segs:
            if a <= frame <= b:
                return r
        return None


# -------------------------------------------------------------- shuttle support


class PlayerProximity:
    """How close an image point is to one of the two target players' reach.

    The shuttle is hit near a player's upper body; a trajectory that passes close
    to our players is very likely ours. Distances are in body heights.
    """

    def __init__(self, players: PlayerTracks | None, reach_bodies: float = 0.8):
        self.reach = reach_bodies
        self.by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        for f in (players.frames if players else []):
            boxes = [tuple(p.bbox) for p in f.players if p.player_id in ("P1", "P2")]
            if boxes:
                self.by_frame[f.frame] = boxes
        self._frames = np.array(sorted(self.by_frame), dtype=np.int64)

    def _boxes_near(self, frame: int, window: int = 3):
        if frame in self.by_frame:
            return self.by_frame[frame]
        if not len(self._frames):
            return []
        i = int(np.searchsorted(self._frames, frame))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self._frames) and abs(int(self._frames[j]) - frame) <= window:
                if best is None or abs(int(self._frames[j]) - frame) < abs(best - frame):
                    best = int(self._frames[j])
        return self.by_frame.get(best, []) if best is not None else []

    def near(self, frame: int, x: float, y: float) -> bool:
        for x1, y1, x2, y2 in self._boxes_near(frame):
            h = max(y2 - y1, 1.0)
            # Racket reach: a band around the box, extending well above the head.
            if (x1 - self.reach * h <= x <= x2 + self.reach * h
                    and y1 - self.reach * h <= y <= y1 + 0.6 * h):
                return True
        return False

    def __bool__(self) -> bool:
        return bool(self.by_frame)
