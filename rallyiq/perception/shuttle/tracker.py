"""Turn per-frame shuttle candidates into one trajectory (numpy only, no torch).

Input: for every frame, zero or more candidates ``(x, y, score)`` in original-video
pixels, as produced by the heatmap detector. Output: ``ShuttlePoint``s and
``FrameGap``s following the shared contract.

Design rules (from the project README):

* Raw detections are never moved. A frame either keeps one of its own candidates
  (``source="detected"``) or, if it had none, may receive a filled point
  (``source="interpolated"``) — only inside short gaps, only when the fill is
  physically plausible, and always with a lower confidence.
* Long gaps stay gaps. We do not invent trajectory when the shuttle is out of frame
  or simply was not seen.
* All thresholds are expressed per second and as fractions of the image diagonal,
  so the same settings work for 25/30/60/120 fps and 720p/1080p/4K.

Pipeline:

1. **Link** candidates frame to frame into tracklets with a constant-velocity
   prediction and a speed-dependent gate (allows a few skipped frames).
2. **Select** tracklets greedily by total score; a weaker tracklet keeps only
   frames no stronger one claimed (the shuttle is a single object, so at most one
   point per frame). Weak candidates (faint far-court shuttles) survive only when
   they form a consistent tracklet; strong single-frame blips are dropped.
3. **Clean** isolated spikes (a point far from both neighbours, while the
   neighbours agree with each other). Direction changes at hits are kept.
4. **Fill** short gaps with a quadratic fit (falls back to linear), check speed.
5. **Report** remaining gaps, with ``out_of_frame`` when the shuttle left the
   image through a border.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

Candidate = tuple[float, float, float]  # x px, y px, score 0..1
# (frame, x, y) -> weight in [0, 1]: how plausible a shuttle there is for the target court.
RegionFn = Callable[[int, float, float], float]
# (frame, x, y) -> True when the point is within reach of one of the target players.
SupportFn = Callable[[int, float, float], bool]
# (frame, x, y) -> True where a shuttle can lie or be held still: on our court's floor
# (with run-off) or inside one of our players' boxes.
RestFn = Callable[[int, float, float], bool]


@dataclass
class TrackerConfig:
    # Candidate scores. ``strong`` is the usual heatmap threshold; weaker candidates
    # are only kept if they belong to a tracklet with enough support.
    strong_score: float = 0.5
    weak_score: float = 0.2  # follows the detector peak_threshold (stage.tracker_config)
    weak_min_len: int = 3
    # A weak-only tracklet (no point >= strong_score) must also travel: its bounding box
    # must grow by at least this many image diagonals per second. Hands at a handshake,
    # a mop, a caption barely move; a shuttle in flight, even far away, does.
    weak_min_travel_diag_per_s: float = 0.03
    # A short weak-only tracklet must travel faster. On footage between rallies most
    # false shuttles are short (0.1-0.3 s) and walking-paced: a shuttle held in a hand,
    # carried in a bag, picked up from the floor, a white patch on clothing. A shuttle in
    # play covers more ground than that even in a short glimpse, and longer weak flights
    # (a slow far-court arc) are still judged by ``weak_min_travel_diag_per_s`` alone.
    weak_short_s: float = 0.5
    weak_short_min_travel_diag_per_s: float = 0.07
    min_len: int = 3
    # A fixed bright spot (a chink of light in a curtain) looks like a shuttle at any
    # score but never moves. A stretch counts as held still when a tracklet stays inside
    # this radius (fraction of diagonal) for ``static_min_s``. Once one place has held
    # still for ``static_total_s`` in all, every candidate there is dropped before
    # tracklets are linked, so it cannot claim frames from the real shuttle or flicker
    # back in short pieces. The total is long so that a shuttle briefly lying on the
    # floor at the end of a rally stays: that is real rally-end evidence.
    static_min_s: float = 0.5
    static_total_s: float = 3.0
    static_radius_frac: float = 0.003
    # Fastest plausible image-space speed, in image diagonals per second. A smash
    # close to the camera crosses ~half the frame in 0.15 s.
    max_speed_diag_per_s: float = 4.5
    # Base association radius (fraction of diagonal) for a tracklet with known velocity.
    gate_frac: float = 0.025
    # Extra radius proportional to the current speed (drag slows the shuttle quickly).
    gate_speed_frac: float = 0.35
    # Frames a tracklet may skip while linking, in seconds (at least 1 frame).
    link_skip_s: float = 0.1
    # Spike removal: deviation from the neighbours' midpoint, fraction of diagonal.
    spike_frac: float = 0.04
    # Gap filling: longest gap filled, in seconds, and the frame cap.
    max_fill_s: float = 0.3
    max_fill_frames: int = 12
    # Gaps longer than this are only filled when a quadratic fits both sides.
    linear_fill_s: float = 0.1
    # Quadratic fit residual tolerance (fraction of diagonal) for a fill to count as smooth.
    fit_tol_frac: float = 0.01
    # Border margin (fraction of width/height) used to decide ``out_of_frame``.
    border_frac: float = 0.06
    # Target-court lock (only used when the tracker gets a ``region``). A tracklet
    # whose mean region weight is below this is another court's shuttle and is dropped
    # as a whole; points with weight 0 are dropped from kept tracklets.
    region_min_mean: float = 0.5
    # Below this mean weight a tracklet is ambiguous (mostly where our air space and
    # another court's overlap in the image) and is kept only if it reaches our players.
    region_ambiguous_mean: float = 0.75
    # Weight multiplier for a tracklet that passes within reach of a target player.
    player_support_bonus: float = 0.5
    # A tracklet that barely moves (same travel test as weak-only ones) is a shuttle at
    # rest or a still white thing. With a ``rest`` function it is kept only where a
    # shuttle can rest: on our floor or in a player's hand (a duct tip, a poster, a pile
    # on a neighbour's floor are dropped, strong scores or not). Without one, it is
    # dropped when it is not wholly inside our own air space.
    drop_idle_shared: bool = True
    idle_rest_min_frac: float = 0.5  # share of its points that must pass ``rest``


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    confidence: float
    source: str  # "detected" | "interpolated"
    score: float = 0.0  # raw heatmap score for detected points


@dataclass
class Gap:
    start_frame: int
    end_frame: int
    reason: str


@dataclass
class TrackResult:
    points: list[TrackPoint]
    gaps: list[Gap]
    # Per frame: index into that frame's candidate list that was selected, or -1.
    selected: dict[int, int] = field(default_factory=dict)


@dataclass
class _Tracklet:
    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    cand_idx: list[int] = field(default_factory=list)
    # Target-court plausibility of each point (1.0 when no region is used).
    region: list[float] = field(default_factory=list)
    supported: bool = False

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def velocity(self) -> tuple[float, float] | None:
        """Pixels per frame from the last two points."""
        if len(self.frames) < 2:
            return None
        df = self.frames[-1] - self.frames[-2]
        (x0, y0), (x1, y1) = self.xy[-2], self.xy[-1]
        return (x1 - x0) / df, (y1 - y0) / df

    def weight(self, cfg: TrackerConfig) -> float:
        n = len(self.frames)
        need = cfg.min_len if max(self.scores) >= cfg.strong_score else cfg.weak_min_len
        dense = n >= 0.5 * (self.end - self.start + 1)  # mostly consecutive frames
        if not (n >= need and dense):
            return 0.0
        if not self.region:
            return float(sum(self.scores))
        w = float(sum(s * r for s, r in zip(self.scores, self.region)))
        return w * (1.0 + cfg.player_support_bonus) if self.supported else w

    def mean_region(self) -> float:
        return float(np.mean(self.region)) if self.region else 1.0

    def subset(self, keep: list[int]) -> "_Tracklet":
        return _Tracklet(
            [self.frames[i] for i in keep], [self.xy[i] for i in keep],
            [self.scores[i] for i in keep], [self.cand_idx[i] for i in keep],
            [self.region[i] for i in keep] if self.region else [], self.supported,
        )


class ShuttleTracker:
    def __init__(self, fps: float, width: int, height: int, config: TrackerConfig | None = None,
                 region: RegionFn | None = None, support: SupportFn | None = None,
                 rest: RestFn | None = None):
        """``region`` / ``support`` lock the tracker onto the target court when
        several matches are in view (see ``rallyiq.perception.target_court``)."""
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self.cfg = config or TrackerConfig()
        self.region, self.support, self.rest = region, support, rest
        self.rejected_foreign = 0  # tracklets dropped as another court's shuttle
        # (start, end, n, mean region, supported, verdict) per linked tracklet, for diagnosis
        self.log: list[tuple[int, int, int, float | None, bool, str]] = []
        self.diag = math.hypot(self.width, self.height)
        self.max_step = self.cfg.max_speed_diag_per_s * self.diag / self.fps  # px per frame
        self.link_skip = max(1, int(round(self.cfg.link_skip_s * self.fps)))
        self.max_fill = max(1, min(self.cfg.max_fill_frames, int(round(self.cfg.max_fill_s * self.fps))))
        self.linear_fill = max(1, int(round(self.cfg.linear_fill_s * self.fps)))

    # ------------------------------------------------------------------ public

    def track(self, candidates: dict[int, Sequence[Candidate]], frame_count: int) -> TrackResult:
        cands = {
            f: [c for c in cs if c[2] >= self.cfg.weak_score]
            for f, cs in candidates.items()
        }
        # Keep original candidate indices so the caller can mark what was selected.
        index_map = {
            f: [i for i, c in enumerate(candidates[f]) if c[2] >= self.cfg.weak_score]
            for f in candidates
        }
        tracklets = self._link(cands, index_map)
        spots = self._static_spots(tracklets)
        if spots:
            cands, index_map = self._without_spots(cands, index_map, spots)
            tracklets = self._link(cands, index_map)
        tracklets = self._target_court(tracklets)
        chosen = self._select(tracklets)

        det: dict[int, TrackPoint] = {}
        selected: dict[int, int] = {}
        for t in chosen:
            support = min(1.0, len(t.frames) / 4.0) ** 0.5
            for f, (x, y), s, ci in zip(t.frames, t.xy, t.scores, t.cand_idx):
                det[f] = TrackPoint(f, x, y, _clip01(s * (0.5 + 0.5 * support)), "detected", s)
                selected[f] = ci

        for f in self._spikes(det):
            det.pop(f)
            selected.pop(f, None)

        points = self._fill(det)
        gaps = self._gaps(points, frame_count)
        return TrackResult(points=points, gaps=gaps, selected=selected)

    # ------------------------------------------------------------ static spots

    def _static_spots(self, tracklets: list[_Tracklet]) -> list[tuple[float, float]]:
        """Places that held still for ``static_total_s`` in all, over the whole video."""
        radius = 1.5 * self.cfg.static_radius_frac * self.diag
        held: list[tuple[float, float, float]] = []  # x, y, seconds
        for t in tracklets:
            held.extend(self._held_stretches(t))
        spots: list[tuple[float, float]] = []
        for x, y, _ in held:
            total = sum(s for hx, hy, s in held if math.hypot(hx - x, hy - y) <= radius)
            if total >= self.cfg.static_total_s:
                spots.append((x, y))
        return spots

    def _held_stretches(self, t: _Tracklet) -> list[tuple[float, float, float]]:
        radius = self.cfg.static_radius_frac * self.diag
        out: list[tuple[float, float, float]] = []
        i = 0
        while i < len(t.frames):
            j = i
            while j + 1 < len(t.frames) and math.dist(t.xy[j + 1], t.xy[i]) <= radius:
                j += 1
            seconds = (t.frames[j] - t.frames[i]) / self.fps
            if seconds >= self.cfg.static_min_s:
                pts = t.xy[i : j + 1]
                out.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), seconds))
                i = j + 1
            else:
                i += 1
        return out

    def _without_spots(
        self, cands: dict[int, list[Candidate]], index_map: dict[int, list[int]], spots: list[tuple[float, float]]
    ) -> tuple[dict[int, list[Candidate]], dict[int, list[int]]]:
        """Drop candidates at a static spot, keeping both maps aligned."""
        radius = 1.5 * self.cfg.static_radius_frac * self.diag
        centres = np.asarray(spots)
        kept_c: dict[int, list[Candidate]] = {}
        kept_i: dict[int, list[int]] = {}
        for f, cs in cands.items():
            keep = [k for k, c in enumerate(cs) if np.hypot(*(centres - c[:2]).T).min() > radius]
            kept_c[f] = [cs[k] for k in keep]
            kept_i[f] = [index_map[f][k] for k in keep]
        return kept_c, kept_i

    # ----------------------------------------------------------------- linking

    def _link(self, cands: dict[int, list[Candidate]], index_map: dict[int, list[int]]) -> list[_Tracklet]:
        active: list[_Tracklet] = []
        done: list[_Tracklet] = []
        for f in sorted(cands):
            # Close tracklets that have skipped too many frames.
            still = []
            for t in active:
                (still if f - t.end <= self.link_skip + 1 else done).append(t)
            active = still

            cs = cands[f]
            if not cs:
                continue
            pairs = []
            for ti, t in enumerate(active):
                dt = f - t.end
                px, py = t.xy[-1]
                v = t.velocity()
                if v is None:
                    if dt > 2:  # without a velocity, a long jump is too ambiguous
                        continue
                    gate = self.max_step * dt
                    ex, ey = px, py
                else:
                    ex, ey = px + v[0] * dt, py + v[1] * dt
                    speed = math.hypot(*v)
                    gate = self.cfg.gate_frac * self.diag * math.sqrt(dt) + self.cfg.gate_speed_frac * speed * dt
                for ci, (x, y, _s) in enumerate(cs):
                    if v is not None and math.hypot(x - px, y - py) > self.max_step * dt:
                        continue
                    d = math.hypot(x - ex, y - ey)
                    if d <= gate:
                        pairs.append((d / max(gate, 1e-6), ti, ci))
            pairs.sort()
            used_t, used_c = set(), set()
            for _cost, ti, ci in pairs:
                if ti in used_t or ci in used_c:
                    continue
                used_t.add(ti)
                used_c.add(ci)
                t = active[ti]
                x, y, s = cs[ci]
                t.frames.append(f)
                t.xy.append((x, y))
                t.scores.append(s)
                t.cand_idx.append(index_map[f][ci])
            for ci, (x, y, s) in enumerate(cs):
                if ci not in used_c:
                    active.append(_Tracklet([f], [(x, y)], [s], [index_map[f][ci]]))
        return done + active

    # ------------------------------------------------------------ target court

    def _target_court(self, tracklets: list[_Tracklet]) -> list[_Tracklet]:
        """Trajectory-level court lock: judge each linked tracklet as a whole, so one
        frame drifting over a line neither keeps a foreign trajectory nor cuts ours."""
        if self.region is None and self.support is None:
            return tracklets
        out = []
        for t in tracklets:
            if self.region is not None:
                t.region = [self.region(f, x, y) for f, (x, y) in zip(t.frames, t.xy)]
            if self.support is not None:
                t.supported = any(self.support(f, x, y) for f, (x, y) in zip(t.frames, t.xy))
            if t.region:
                m = t.mean_region()
                foreign = m < self.cfg.region_min_mean and not (t.supported and m > 0)
                ambiguous = m < self.cfg.region_ambiguous_mean and not t.supported and self.support is not None
                if foreign or ambiguous:
                    if len(t.frames) >= self.cfg.min_len:
                        self.rejected_foreign += 1
                    self._note(t, "foreign" if foreign else "ambiguous")
                    continue
                keep = [i for i, r in enumerate(t.region) if r > 0.0]
                if len(keep) < len(t.frames):
                    if not keep:
                        continue
                    t = t.subset(keep)
            out.append(t)
        return out

    # --------------------------------------------------------------- selection

    def _select(self, tracklets: list[_Tracklet]) -> list[_Tracklet]:
        """Greedy by weight: the strongest tracklet claims its frames first; weaker ones
        keep only frames nobody claimed yet (the shuttle is a single object)."""
        claimed: set[int] = set()
        chosen: list[_Tracklet] = []
        for t in sorted(tracklets, key=lambda t: t.weight(self.cfg), reverse=True):
            keep = [i for i, f in enumerate(t.frames) if f not in claimed]
            if not keep:
                continue
            sub = t.subset(keep)
            if sub.weight(self.cfg) <= 0:
                continue
            if self._idle_weak(sub) or self._idle_shared(sub):
                self._note(sub, "idle")
                continue
            claimed.update(sub.frames)
            chosen.append(sub)
            self._note(sub, "kept")
        chosen.sort(key=lambda t: t.start)
        return chosen

    def _note(self, t: _Tracklet, verdict: str) -> None:
        if len(t.frames) >= self.cfg.min_len:
            self.log.append((t.start, t.end, len(t.frames),
                             round(t.mean_region(), 3) if t.region else None, t.supported, verdict))

    def _idle_weak(self, t: _Tracklet) -> bool:
        """True for a weak-only tracklet that hardly moves (likely not a shuttle)."""
        if max(t.scores) >= self.cfg.strong_score:
            return False
        return self._idle(t)

    def _idle_shared(self, t: _Tracklet) -> bool:
        """True for a still tracklet where no shuttle of ours can be resting."""
        if not self.cfg.drop_idle_shared:
            return False
        if self.rest is not None:
            if not self._idle(t):
                return False
            ok = sum(1 for f, (x, y) in zip(t.frames, t.xy) if self.rest(f, x, y))
            return ok < self.cfg.idle_rest_min_frac * len(t.frames)
        if not t.region or min(t.region) >= 1.0:
            return False
        return self._idle(t)

    def _idle(self, t: _Tracklet) -> bool:
        xs = [p[0] for p in t.xy]
        ys = [p[1] for p in t.xy]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        duration_s = max(t.end - t.start, 1) / self.fps
        need = self.cfg.weak_min_travel_diag_per_s
        if duration_s < self.cfg.weak_short_s:
            need = max(need, self.cfg.weak_short_min_travel_diag_per_s)
        return extent / self.diag / duration_s < need

    # ------------------------------------------------------------------ spikes

    def _spikes(self, det: dict[int, TrackPoint]) -> list[int]:
        frames = sorted(det)
        thr = self.cfg.spike_frac * self.diag
        out = []
        for a, b, c in zip(frames, frames[1:], frames[2:]):
            if c - a > 2 * (self.link_skip + 1):
                continue
            pa, pb, pc = det[a], det[b], det[c]
            u = (b - a) / (c - a)
            mx, my = pa.x + (pc.x - pa.x) * u, pa.y + (pc.y - pa.y) * u
            off = math.hypot(pb.x - mx, pb.y - my)
            span = math.hypot(pc.x - pa.x, pc.y - pa.y)
            # A hit makes b far from the a-c midpoint too, but then a and c are both
            # far from b in the same direction; a spike is far from both a and c
            # while a and c are close to each other.
            if off > thr and span < 0.5 * off and pb.score < max(pa.score, pc.score):
                out.append(b)
        return out

    # -------------------------------------------------------------- gap filling

    def _fill(self, det: dict[int, TrackPoint]) -> list[TrackPoint]:
        frames = sorted(det)
        out: list[TrackPoint] = []
        for a, b in zip(frames, frames[1:]):
            out.append(det[a])
            g = b - a - 1
            if g < 1 or g > self.max_fill:
                continue
            pa, pb = det[a], det[b]
            if math.hypot(pb.x - pa.x, pb.y - pa.y) > self.max_step * (b - a):
                continue
            filled = self._fit_gap(det, frames, a, b)
            if filled is None:
                continue
            base = min(pa.confidence, pb.confidence) * (1.0 - g / (self.max_fill + 1))
            for f, (x, y, smooth) in filled.items():
                if not (0 <= x < self.width and 0 <= y < self.height):
                    break  # leaving the image: do not invent points outside
                conf = base * (0.6 if smooth else 0.4)
                out.append(TrackPoint(f, float(x), float(y), _clip01(conf), "interpolated"))
        if frames:
            out.append(det[frames[-1]])
        out.sort(key=lambda p: p.frame)
        return out

    def _fit_gap(self, det, frames, a, b) -> dict[int, tuple[float, float, bool]] | None:
        ia, ib = frames.index(a), frames.index(b)
        window = self.max_fill + 2
        left = [f for f in frames[max(0, ia - 3): ia + 1] if a - f <= window]
        right = [f for f in frames[ib: ib + 4] if f - b <= window]
        missing = range(a + 1, b)
        ts = np.array(left + right, dtype=float)
        xs = np.array([det[f].x for f in left + right])
        ys = np.array([det[f].y for f in left + right])
        if len(ts) >= 4:
            cx = np.polyfit(ts, xs, 2)
            cy = np.polyfit(ts, ys, 2)
            res = np.hypot(np.polyval(cx, ts) - xs, np.polyval(cy, ts) - ys).max()
            if res <= self.cfg.fit_tol_frac * self.diag:
                return {f: (float(np.polyval(cx, f)), float(np.polyval(cy, f)), True) for f in missing}
        # Linear fallback between the two neighbours (e.g. a hit inside the gap), only
        # for short gaps: over longer ones a straight line is not a believable flight.
        if b - a - 1 > self.linear_fill:
            return None
        pa, pb = det[a], det[b]
        return {
            f: (pa.x + (pb.x - pa.x) * (f - a) / (b - a), pa.y + (pb.y - pa.y) * (f - a) / (b - a), False)
            for f in missing
        }

    # -------------------------------------------------------------------- gaps

    def _gaps(self, points: list[TrackPoint], frame_count: int) -> list[Gap]:
        have = {p.frame for p in points}
        by_frame = {p.frame: p for p in points}
        gaps: list[Gap] = []
        f = 0
        while f < frame_count:
            if f in have:
                f += 1
                continue
            s = f
            while f < frame_count and f not in have:
                f += 1
            e = f - 1
            before = by_frame.get(s - 1)
            after = by_frame.get(e + 1)
            gaps.append(Gap(s, e, self._gap_reason(before, after, by_frame)))
        return gaps

    def _gap_reason(self, before: TrackPoint | None, after: TrackPoint | None, by_frame) -> str:
        if before is None and after is None:
            return "not_detected"
        if (before is not None and self._exiting(before, by_frame, -1)) or (
            after is not None and self._exiting(after, by_frame, +1)
        ):
            return "out_of_frame"
        return "not_detected"

    def _exiting(self, p: TrackPoint, by_frame, direction: int) -> bool:
        """True if ``p`` is near a border and moving out (direction -1: exits after p,
        +1: entered just before p)."""
        mx, my = self.cfg.border_frac * self.width, self.cfg.border_frac * self.height
        near = {
            "left": p.x < mx, "right": p.x > self.width - mx,
            "top": p.y < my, "bottom": p.y > self.height - my,
        }
        if not any(near.values()):
            return False
        q = by_frame.get(p.frame + direction)  # the neighbour inside the track
        if q is None:
            return True
        # Exit after p: motion q -> p points outward. Entry before p: motion p -> q
        # points inward. Either way the vector p - q points towards the border.
        vx, vy = p.x - q.x, p.y - q.y
        return (
            (near["left"] and vx < 0) or (near["right"] and vx > 0)
            or (near["top"] and vy < 0) or (near["bottom"] and vy > 0)
        )


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))
