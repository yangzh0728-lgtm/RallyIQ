"""Per-foot ground tracks from the ankle keypoints, and when each foot is planted.

For every player the two ankle keypoints (COCO-17 ``left_ankle`` / ``right_ankle``) are turned into
floor points: the ankle sits about ``ANKLE_HEIGHT_M`` above the floor, so the point is moved down by
that height at the image scale of the player's position. The floor points are smoothed in the image
and projected onto the court with the calibration homography.

A foot is *planted* while its image speed, expressed in metres per second at the player's distance
(pixels / pixels-per-metre), stays low; it is *in the air* (swinging, lifted) otherwise. Measuring
speed in the image rather than on the court keeps the threshold the same for the near and the far
player: on a low end-on camera a far player's step along the court moves the foot by only a few
pixels in depth, but lifting the foot still moves it by its full height in the image.

Pixel noise is turned into a per-player ``jitter_m`` (how much a planted foot wanders on the court)
so later steps can widen their thresholds where the projection is coarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import CourtCalibration, PlayerTracks

L_ANKLE, R_ANKLE = 15, 16
MIN_ANKLE_CONF = 0.3
ANKLE_HEIGHT_M = 0.08  # ankle keypoint above the floor
MAX_FILL_S = 0.2  # bridge missing frames up to this long by interpolation
MIN_RUN_S = 0.5  # shorter runs are dropped
SMOOTH_S = 0.1  # moving average over the image floor points
PLANT_ON_MPS = 0.55  # a foot becomes planted below this image speed (m/s at the player's distance)
PLANT_OFF_MPS = 1.0  # ... and is lifted again above this
MIN_PLANT_S = 0.06  # shorter "plants" are part of the swing
MIN_LIFT_M = 0.05  # a swing must move the foot this far (image, metres) to count as a lift
APEX_RISE_M = 0.08  # a still foot this far above the floor line is in the air (top of a jump) ...
APEX_MAX_S = 0.15  # ... for at most this long ...
MAX_FLIGHT_S = 0.9  # ... inside a flight no longer than this


@dataclass
class Run:
    """One continuous stretch of one player's foot track (fixed homography, no long gaps)."""

    pid: str
    frames: np.ndarray  # (n,) int
    t: np.ndarray  # (n,) seconds
    img: np.ndarray  # (n, 2, 2) smoothed floor point per foot (0 = left, 1 = right), px
    court: np.ndarray  # (n, 2, 2) court metres
    scale: np.ndarray  # (n, 2) px per metre (lateral) at each foot
    depth_scale: np.ndarray  # (n, 2) px per metre along the court at each foot
    speed: np.ndarray  # (n, 2) image speed of each foot in m/s at the player's distance
    planted: np.ndarray  # (n, 2) bool
    filled: np.ndarray  # (n,) bool, interpolated frame
    H: np.ndarray  # image -> court
    ankle: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))  # (n, 2, 2) tracked ankles (L/R fixed), px
    heel: np.ndarray | None = None  # (n, 2, 2) heel px from the foot model, NaN where missing (see ``sole``)
    toe: np.ndarray | None = None  # (n, 2, 2) toe px
    center: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (n, 2) smoothed mid-feet, court m
    center_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n,) m/s
    center_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (n, 2) m/s

    def index(self, frame: int) -> int | None:
        i = int(np.searchsorted(self.frames, frame))
        if i < len(self.frames) and self.frames[i] == frame:
            return i
        return None

    def nearest(self, frame: int, max_gap: int) -> int | None:
        if not len(self.frames):
            return None
        i = int(np.clip(np.searchsorted(self.frames, frame), 0, len(self.frames) - 1))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.frames)), key=lambda j: abs(self.frames[j] - frame))
        return best if abs(int(self.frames[best]) - frame) <= max_gap else None


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    shp = pts.shape
    p = pts.reshape(-1, 2)
    h = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    w = h[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return (h[:, :2] / w).reshape(shp)


def _odd(n: float) -> int:
    k = max(1, int(round(n)))
    return k if k % 2 else k + 1


def smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Median-of-3 (outliers) then centred moving average of width ``w`` along axis 0, edges held."""
    if len(a) < 3:
        return a.copy()
    pad = np.concatenate([a[:1], a, a[-1:]])
    a = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
    if w < 3 or len(a) < w:
        return a
    h = w // 2
    pad = np.concatenate([np.repeat(a[:1], h, 0), a, np.repeat(a[-1:], h, 0)])
    c = np.cumsum(np.concatenate([np.zeros((1,) + a.shape[1:]), pad]), axis=0)
    return (c[w:] - c[:-w]) / w


def _scales(H: np.ndarray, Hinv: np.ndarray, court: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """px per metre across (x) and along (y) the court at each court point."""
    a = apply_h(Hinv, court)
    bx = apply_h(Hinv, court + np.array([0.25, 0.0]))
    by = apply_h(Hinv, court + np.array([0.0, 0.25]))
    return np.linalg.norm(bx - a, axis=-1) * 4, np.linalg.norm(by - a, axis=-1) * 4


def _hysteresis(speed: np.ndarray, on: float, off: float) -> np.ndarray:
    out = np.zeros(len(speed), dtype=bool)
    state = bool(speed[0] < on) if len(speed) else False
    for i, v in enumerate(speed):
        if state and v > off:
            state = False
        elif not state and v < on:
            state = True
        out[i] = state
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] index pairs (inclusive) of True runs."""
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def _clean_plants(planted: np.ndarray, img: np.ndarray, scale: np.ndarray, fps: float) -> np.ndarray:
    """Drop too-short plants, and lifts that hardly moved the foot (keypoint jitter)."""
    p = planted.copy()
    min_plant = max(2, int(round(MIN_PLANT_S * fps)))
    for a, b in _segments(p):
        if b - a + 1 < min_plant and a > 0 and b < len(p) - 1:
            p[a:b + 1] = False
    # a "plant" high above the contacts on either side is the top of a jump (the foot is briefly still in the air)
    segs = _segments(p)
    for k in range(1, len(segs) - 1):
        a, b = segs[k]
        prev_end, next_start = segs[k - 1][1], segs[k + 1][0]
        if (b - a + 1) > APEX_MAX_S * fps or (next_start - prev_end) > MAX_FLIGHT_S * fps:
            continue
        y0, y1 = img[prev_end, 1], img[next_start, 1]
        u = (np.arange(a, b + 1) - prev_end) / max(next_start - prev_end, 1)
        rise = (y0 + (y1 - y0) * u - img[a:b + 1, 1]) / scale[a:b + 1]
        if np.median(rise) > APEX_RISE_M:
            p[a:b + 1] = False
    for a, b in _segments(~p):
        if a == 0 or b == len(p) - 1:
            continue
        ref = img[a - 1]
        moved = np.max(np.linalg.norm(img[a:b + 2] - ref, axis=-1) / scale[a:b + 2])
        if moved < MIN_LIFT_M:
            p[a:b + 1] = True
    return p


def _fix_swaps(feet: np.ndarray) -> np.ndarray:
    """Keep each foot on its own track when the pose model swaps left and right ankles.

    Frame to frame, the two ankles are matched to the two tracks by distance; each track is then named
    by the label the pose model gave it most often."""
    out = feet.copy()
    prev = np.full((2, 2), np.nan)
    kept_left = 0
    seen = 0
    for i in range(len(out)):
        cur = out[i]
        ok = np.isfinite(cur).all(axis=1)
        pk = np.isfinite(prev).all(axis=1)
        swap = False
        if ok.all() and pk.all():
            keep = np.linalg.norm(cur[0] - prev[0]) + np.linalg.norm(cur[1] - prev[1])
            cross = np.linalg.norm(cur[0] - prev[1]) + np.linalg.norm(cur[1] - prev[0])
            swap = cross < 0.7 * keep
        elif ok.sum() == 1 and pk.all():
            j = int(np.where(ok)[0][0])
            swap = np.linalg.norm(cur[j] - prev[1 - j]) < np.linalg.norm(cur[j] - prev[j])
        if swap:
            out[i] = cur[::-1].copy()
        if ok.any():
            seen += 1
            kept_left += 0 if swap else 1
        for j in range(2):
            if np.isfinite(out[i, j]).all():
                prev[j] = out[i, j]
    if seen and kept_left < seen / 2:  # track 0 was mostly labelled "right" by the model
        out = out[:, ::-1].copy()
    return out


def build_runs(players: PlayerTracks, calibration: CourtCalibration | None, fps: float,
               player_ids: tuple[str, ...] = ("P1", "P2", "P3", "P4")) -> list[Run]:
    """Foot tracks of every identified player, split where tracking or the camera breaks."""
    if calibration is None:
        return []
    per: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for pf in players.frames:
        for p in pf.players:
            if p.player_id not in player_ids or p.interpolated or len(p.keypoints) < 17 or p.on_court is False:
                continue
            k = np.asarray(p.keypoints, dtype=float)
            feet = k[[L_ANKLE, R_ANKLE], :2].copy()
            feet[k[[L_ANKLE, R_ANKLE], 2] < MIN_ANKLE_CONF] = np.nan
            per.setdefault(p.player_id, []).append((pf.frame, feet, np.asarray(p.bbox, dtype=float)))
    runs: list[Run] = []
    max_fill = max(1, int(round(MAX_FILL_S * fps)))
    min_run = max(5, int(round(MIN_RUN_S * fps)))
    for pid, rows in per.items():
        rows.sort(key=lambda r: r[0])
        # split on long gaps and calibration segment changes
        chunks: list[list] = []
        cur: list = []
        seg_of = None
        for r in rows:
            seg = calibration.segment_for(r[0])
            if seg is None or seg.homography is None:
                continue
            if cur and (r[0] - cur[-1][0] > max_fill + 1 or seg is not seg_of):
                chunks.append(cur)
                cur = []
            seg_of = seg
            cur.append(r)
        if cur:
            chunks.append(cur)
        for ch in chunks:
            run = _make_run(pid, ch, calibration, fps, max_fill)
            runs.extend(r for r in run if len(r.frames) >= min_run)
    return runs


def _make_run(pid: str, rows: list, calibration: CourtCalibration, fps: float, max_fill: int) -> list[Run]:
    f0, f1 = rows[0][0], rows[-1][0]
    n = f1 - f0 + 1
    feet = np.full((n, 2, 2), np.nan)
    for fr, ft, _ in rows:
        feet[fr - f0] = ft
    filled = np.ones(n, dtype=bool)
    filled[[r[0] - f0 for r in rows]] = False
    feet = _fix_swaps(feet)
    # interpolate short holes per foot; long holes split the run
    ok = np.all(np.isfinite(feet), axis=(1, 2))
    for j in range(2):
        for c in range(2):
            v = feet[:, j, c]
            good = np.isfinite(v)
            if good.sum() >= 2:
                feet[:, j, c] = np.interp(np.arange(n), np.where(good)[0], v[good])
    bad = ~ok
    holes = [(a, b) for a, b in _segments(bad) if b - a + 1 > max_fill]
    cuts = [0]
    for a, b in holes:
        cuts += [a, b + 1]
    cuts.append(n)
    H = np.asarray(calibration.homography_for(f0), dtype=float)
    Hinv = np.linalg.inv(H)
    w = _odd(SMOOTH_S * fps)
    out = []
    for a, b in zip(cuts[0::2], cuts[1::2]):
        if b - a < 3:
            continue
        ft = feet[a:b]
        if not np.all(np.isfinite(ft)):
            continue
        # floor point: ankle moved down by its height at the local image scale
        c0 = apply_h(H, ft)
        sx, _ = _scales(H, Hinv, c0)
        floor = ft + np.stack([np.zeros_like(sx), ANKLE_HEIGHT_M * sx], axis=-1)
        img = smooth(floor, w)
        court = apply_h(H, img)
        scale, dscale = _scales(H, Hinv, court)
        vel = np.gradient(img, axis=0) * fps
        speed = np.linalg.norm(vel, axis=-1) / scale
        planted = np.stack([_clean_plants(_hysteresis(speed[:, j], PLANT_ON_MPS, PLANT_OFF_MPS), img[:, j],
                                          scale[:, j], fps) for j in range(2)], axis=1)
        frames = np.arange(f0 + a, f0 + b)
        run = Run(pid=pid, frames=frames, t=frames / fps, img=img, court=court, scale=scale, depth_scale=dscale,
                  speed=speed, planted=planted, filled=filled[a:b], H=H, ankle=ft)
        _center(run, fps)
        out.append(run)
    return out


CENTER_SMOOTH_S = 0.2


def _center(run: Run, fps: float) -> None:
    """Mid-feet court position, smoothed more along the court where the projection is coarse."""
    mid = run.court.mean(axis=1)
    w = _odd(CENTER_SMOOTH_S * fps)
    c = smooth(mid, w)
    # far side: depth is ~10x coarser than lateral; smooth depth again
    if np.median(run.depth_scale) < 20:
        c[:, 1] = smooth(c[:, 1:2], _odd(2 * CENTER_SMOOTH_S * fps))[:, 0]
    run.center = c
    run.center_vel = np.gradient(c, axis=0) * fps
    run.center_speed = np.linalg.norm(run.center_vel, axis=1)
