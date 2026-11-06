"""Per-shot evidence for stroke-type classification.

Everything is measured from data the pipeline already has: the hit (``ShotEvent``,
with the hitter's feet in court metres when calibration worked), the next hit in
the rally (where and when the opponent met the shuttle), the hitter's COCO-17 pose
at contact and the shuttle track between the two hits. A feature that cannot be
measured is ``None`` and carries a reason code; it is never filled with a guess.

Features
--------
contact_h        shuttle height at contact above the hitter's ankles, in ankle-to-nose
                 spans (1.0 = nose height; ~1.6 = overhead racket contact; < 0.6 = low)
hitter_depth_m   hitter's distance from the net (court metres)
receiver_depth_m where the opponent met the shuttle: distance from the net
lateral_m        sideways distance between hitter and receiver (court metres)
flight_ms        time to the next hit (or to the landing for the last shot)
court_speed_mps  mean horizontal speed hitter -> receiver (court distance / flight time)
rise             how far the flight rose above its higher end point, in heights of the
                 player at that end (the far player is the reference, so camera
                 perspective cancels out); clears and lifts rise, drives and smashes do not
left_frame       the shuttle left the top of the frame during the flight (a very high shot)
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from ..contracts import NET_Y_M, EventsResult, PlayerDetection, PlayerTracks, Rally, ShotEvent, ShuttleTrack

KP_NOSE, KP_EYES, KP_EARS = 0, (1, 2), (3, 4)
KP_ANKLES = (15, 16)

# Reason codes (stable strings for the review UI).
R_NO_HITTER = "hitter_unknown"
R_NO_POSE = "no_hitter_pose"
R_NO_SHUTTLE_AT_HIT = "no_shuttle_at_hit"
R_NO_COURT = "no_court_position"
R_NO_NEXT = "last_shot_of_rally"
R_NEXT_SAME_SIDE = "next_hit_same_side"
R_FLIGHT_UNSEEN = "flight_not_observed"
R_FIRST_NOT_SERVE = "first_hit_not_serve_like"


@dataclass
class StrokeFeatures:
    shot_id: str
    index_in_rally: int
    first_in_rally: bool
    last_in_rally: bool
    contact_h: float | None = None
    hitter_depth_m: float | None = None
    receiver_depth_m: float | None = None
    lateral_m: float | None = None
    flight_ms: float | None = None
    court_speed_mps: float | None = None
    rise: float | None = None
    left_frame: bool = False
    reasons: list[str] = field(default_factory=list)

    def evidence(self) -> dict[str, float | None]:
        r = lambda v, n=3: None if v is None else round(float(v), n)  # noqa: E731
        return {
            "contact_h": r(self.contact_h),
            "hitter_depth_m": r(self.hitter_depth_m, 2),
            "receiver_depth_m": r(self.receiver_depth_m, 2),
            "lateral_m": r(self.lateral_m, 2),
            "flight_ms": r(self.flight_ms, 0),
            "court_speed_mps": r(self.court_speed_mps, 2),
            "rise": r(self.rise),
            "left_frame": 1.0 if self.left_frame else 0.0,
        }


class _Players:
    def __init__(self, tracks: PlayerTracks | None, search: int):
        self.by_frame = {pf.frame: pf for pf in tracks.frames} if tracks else {}
        self.search = search

    def find(self, frame: int, player_id: str | None, *, other_than: str | None = None) -> PlayerDetection | None:
        for d in range(self.search + 1):
            for f in (frame - d, frame + d):
                pf = self.by_frame.get(f)
                if pf is None:
                    continue
                for p in pf.players:
                    if p.on_court is False:
                        continue
                    if player_id is not None and p.player_id == player_id:
                        return p
                    if other_than is not None and p.player_id not in (None, other_than):
                        return p
        return None


def _kp(p: PlayerDetection, idx: int, min_conf: float) -> tuple[float, float] | None:
    if len(p.keypoints) <= idx:
        return None
    x, y, c = p.keypoints[idx]
    return (x, y) if c >= min_conf else None


def body_span(p: PlayerDetection, min_conf: float) -> tuple[float, float] | None:
    """(ankle_y, ankle_y - head_y) in px, from keypoints, falling back to the box bottom."""
    head = _kp(p, KP_NOSE, min_conf)
    if head is None:
        pts = [q for i in (*KP_EYES, *KP_EARS) if (q := _kp(p, i, min_conf))]
        head = (0.0, sum(q[1] for q in pts) / len(pts)) if pts else None
    if head is None:
        return None
    ankles = [q for i in KP_ANKLES if (q := _kp(p, i, min_conf))]
    ankle_y = max(q[1] for q in ankles) if ankles else p.bbox[3]
    span = ankle_y - head[1]
    return (ankle_y, span) if span > 4.0 else None


class _Shuttle:
    def __init__(self, track: ShuttleTrack | None, min_conf: float):
        pts = sorted((p for p in (track.points if track else []) if p.confidence >= min_conf), key=lambda p: p.frame)
        self.frames = [p.frame for p in pts]
        self.points = pts
        self.out_gaps = [(g.start_frame, g.end_frame) for g in (track.gaps if track else []) if g.reason == "out_of_frame"]

    def at(self, frame: int, search: int) -> tuple[float, float] | None:
        i = bisect_left(self.frames, frame - search)
        best = None
        for p in self.points[i:]:
            if p.frame > frame + search:
                break
            if best is None or abs(p.frame - frame) < abs(best.frame - frame):
                best = p
        return (best.x, best.y) if best else None

    def between(self, f0: int, f1: int):
        return self.points[bisect_right(self.frames, f0):bisect_left(self.frames, f1)]

    def left_frame(self, f0: int, f1: int) -> bool:
        return any(a <= f1 and b >= f0 for a, b in self.out_gaps)


@dataclass(frozen=True)
class FeatureConfig:
    min_kp_conf: float = 0.3
    min_shuttle_conf: float = 0.2
    pose_search_frames: int = 3
    shuttle_search_frames: int = 2
    last_shot_window_ms: float = 2500.0  # how far to follow the last shot of a rally
    min_flight_points: int = 3


def _landing(flight, scale_px: float):
    """First point after which the shuttle no longer moves (lying on the floor)."""
    tol = max(3.0, 0.03 * scale_px)
    last = flight[-1]
    i = len(flight) - 1
    while i > 0 and abs(flight[i - 1].x - last.x) <= tol and abs(flight[i - 1].y - last.y) <= tol:
        i -= 1
    return flight[i]


def _depth(court) -> float | None:
    return abs(court[1] - NET_Y_M) if court is not None else None


def _side(court) -> int | None:
    return None if court is None else (1 if court[1] > NET_Y_M else -1)


def shot_features(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    fps: float,
    cfg: FeatureConfig | None = None,
) -> list[StrokeFeatures]:
    cfg = cfg or FeatureConfig()
    pl = _Players(players, cfg.pose_search_frames)
    sh = _Shuttle(shuttle, cfg.min_shuttle_conf)
    rallies: dict[str, Rally] = {r.rally_id: r for r in events.rallies}
    by_rally: dict[str | None, list[ShotEvent]] = {}
    for s in sorted(events.shots, key=lambda s: s.frame):
        by_rally.setdefault(s.rally_id, []).append(s)

    out: list[StrokeFeatures] = []
    for rid, shots in by_rally.items():
        rally = rallies.get(rid) if rid else None
        for k, s in enumerate(shots):
            nxt = shots[k + 1] if k + 1 < len(shots) and rally is not None else None
            out.append(_one(s, k, nxt, rally, pl, sh, fps, cfg))
    order = {s.shot_id: i for i, s in enumerate(events.shots)}
    out.sort(key=lambda f: order[f.shot_id])
    return out


def _one(s: ShotEvent, k: int, nxt: ShotEvent | None, rally: Rally | None, pl: _Players, sh: _Shuttle,
         fps: float, cfg: FeatureConfig) -> StrokeFeatures:
    f = StrokeFeatures(s.shot_id, k, first_in_rally=(k == 0 and rally is not None), last_in_rally=nxt is None)
    reasons = f.reasons
    if s.hitter is None:
        reasons.append(R_NO_HITTER)

    # --- contact height on the hitter's body ---
    hitter = pl.find(s.frame, s.hitter) if s.hitter else None
    contact = tuple(s.shuttle_px) if s.shuttle_px else sh.at(s.frame, cfg.shuttle_search_frames)
    if contact is None:
        reasons.append(R_NO_SHUTTLE_AT_HIT)
    span_h = body_span(hitter, cfg.min_kp_conf) if hitter else None
    if span_h is None and s.hitter is not None:
        reasons.append(R_NO_POSE)
    if contact is not None and span_h is not None:
        f.contact_h = (span_h[0] - contact[1]) / span_h[1]

    # --- court positions ---
    f.hitter_depth_m = _depth(s.hitter_court)
    if s.hitter_court is None:
        reasons.append(R_NO_COURT)
    end_frame: int
    end_xy: tuple[float, float] | None = None
    end_span: tuple[float, float] | None = None
    if nxt is not None:
        f.flight_ms = float(nxt.t_ms - s.t_ms)
        end_frame = nxt.frame
        if s.hitter_court is not None and nxt.hitter_court is not None:
            if _side(nxt.hitter_court) == _side(s.hitter_court):
                reasons.append(R_NEXT_SAME_SIDE)
            else:
                f.receiver_depth_m = _depth(nxt.hitter_court)
                f.lateral_m = abs(nxt.hitter_court[0] - s.hitter_court[0])
                dx = nxt.hitter_court[0] - s.hitter_court[0]
                dy = nxt.hitter_court[1] - s.hitter_court[1]
                if f.flight_ms > 0:
                    f.court_speed_mps = (dx * dx + dy * dy) ** 0.5 / (f.flight_ms / 1000.0)
        end_xy = tuple(nxt.shuttle_px) if nxt.shuttle_px else sh.at(nxt.frame, cfg.shuttle_search_frames)
        recv = pl.find(nxt.frame, nxt.hitter) if nxt.hitter else None
        end_span = body_span(recv, cfg.min_kp_conf) if recv else None
    else:
        reasons.append(R_NO_NEXT)
        horizon = s.frame + int(round(cfg.last_shot_window_ms * fps / 1000.0))
        end_frame = min(rally.end_frame, horizon) if rally else horizon
        flight = sh.between(s.frame, end_frame + 1)
        if flight:
            landed = _landing(flight, span_h[1] if span_h else 100.0)
            end_xy = (landed.x, landed.y)
            end_frame = landed.frame + 1
            if rally is not None and rally.end_reason == "shuttle_static":
                f.flight_ms = float(landed.t_ms - s.t_ms)  # to the landing
        opp = pl.find(end_frame, None, other_than=s.hitter) if s.hitter else None
        end_span = body_span(opp, cfg.min_kp_conf) if opp else None

    # --- flight shape: rise above the higher end point (in that player's heights) ---
    f.left_frame = sh.left_frame(s.frame + 1, end_frame - 1)
    flight = sh.between(s.frame, end_frame)
    if contact is not None and end_xy is not None and len(flight) >= cfg.min_flight_points:
        top_y = min(p.y for p in flight)
        if contact[1] <= end_xy[1]:
            ref_y, ref = contact[1], span_h
        else:
            ref_y, ref = end_xy[1], end_span
        if ref is not None:
            f.rise = max(-0.5, (ref_y - top_y) / ref[1])
    elif not f.left_frame:
        reasons.append(R_FLIGHT_UNSEEN)
    return f
