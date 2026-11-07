"""Pose lookups: player scale, racket-head proxies and shuttle proximity."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional, Sequence

from .config import EventsConfig
from .trajectory import Vec, dist
from .types import (
    KP_L_ELBOW,
    KP_L_SHOULDER,
    KP_L_WRIST,
    KP_R_ELBOW,
    KP_R_SHOULDER,
    KP_R_WRIST,
    PlayerPose,
    PoseFrame,
)


def player_height(p: PlayerPose, min_conf: float) -> Optional[float]:
    ys = [k.y for k in p.keypoints if k.confidence >= min_conf]
    if len(ys) < 4:
        return None
    h = max(ys) - min(ys)
    return h if h > 1.0 else None


@dataclass(frozen=True)
class Proximity:
    player_id: str
    score: float  # 1 = shuttle at the racket proxy, 0 = beyond max distance
    distance_rel: float  # distance in player heights
    used_wrist: bool  # False when only the shoulders were visible
    pose: PlayerPose


class PoseIndex:
    def __init__(self, frames: Sequence[PoseFrame], cfg: EventsConfig):
        self.cfg = cfg
        self.by_frame = {f.frame: f for f in frames}
        heights: dict[str, list[float]] = {}
        for f in frames:
            for p in f.players:
                h = player_height(p, cfg.min_kp_conf)
                if h is not None:
                    heights.setdefault(p.player_id, []).append(h)
        self.median_height = {pid: statistics.median(hs) for pid, hs in heights.items()}
        all_h = [h for hs in heights.values() for h in hs]
        self.global_height = statistics.median(all_h) if all_h else cfg.default_player_height_px

    @property
    def empty(self) -> bool:
        return not any(f.players for f in self.by_frame.values())

    def frame_near(self, frame: int, search: Optional[int] = None) -> Optional[PoseFrame]:
        for d in range((self.cfg.pose_search_frames if search is None else search) + 1):
            for f in (frame - d, frame + d):
                pf = self.by_frame.get(f)
                if pf is not None and pf.players:
                    return pf
        return None

    def height_of(self, p: PlayerPose) -> float:
        h = player_height(p, self.cfg.min_kp_conf)
        med = self.median_height.get(p.player_id, self.global_height)
        # A crouching / partly detected player looks short; do not let that shrink the scale much.
        return max(h or med, 0.7 * med)

    def racket_points(self, p: PlayerPose) -> tuple[list[Vec], bool]:
        """Wrists plus extrapolated racket heads; shoulders if no wrist is visible."""
        kps = p.keypoints
        c = self.cfg.min_kp_conf
        pts: list[Vec] = []
        for w, e in ((KP_L_WRIST, KP_L_ELBOW), (KP_R_WRIST, KP_R_ELBOW)):
            if len(kps) <= max(w, e) or kps[w].confidence < c:
                continue
            wx, wy = kps[w].x, kps[w].y
            pts.append((wx, wy))
            if kps[e].confidence >= c:
                k = self.cfg.racket_extension
                ex, ey = kps[e].x, kps[e].y
                pts.append((wx + k * (wx - ex), wy + k * (wy - ey)))
                pts.append((wx + 2 * k * (wx - ex), wy + 2 * k * (wy - ey)))
        if pts:
            return pts, True
        for s in (KP_L_SHOULDER, KP_R_SHOULDER):
            if len(kps) > s and kps[s].confidence >= c:
                pts.append((kps[s].x, kps[s].y))
        return pts, False

    def scale_at(self, frame: int, xy: Vec) -> float:
        """Height (px) of the player nearest to ``xy``: the local pixel scale.

        The far player can be half the size of the near one, so speeds and
        distances near them must be judged on their own scale.
        """
        pf = self.frame_near(frame)
        if pf is None:
            return self.global_height
        best, best_d = None, float("inf")
        for p in pf.players:
            pts = [(k.x, k.y) for k in p.keypoints if k.confidence >= self.cfg.min_kp_conf]
            if not pts:
                continue
            cx = sum(q[0] for q in pts) / len(pts)
            cy = sum(q[1] for q in pts) / len(pts)
            d = dist((cx, cy), xy)
            if d < best_d:
                best, best_d = p, d
        return self.height_of(best) if best is not None else self.global_height

    def proximities(self, frame: int, shuttle_xy: Vec, search: Optional[int] = None) -> list[Proximity]:
        """Every player's proximity to the shuttle at ``frame``, best first.

        ``search``: how many frames around ``frame`` to look for a pose (default ``cfg.pose_search_frames``).
        """
        pf = self.frame_near(frame, search)
        if pf is None:
            return []
        out = []
        local = self.scale_at(frame, shuttle_xy) if self.cfg.hitter_scale == "mixed" else None
        for p in pf.players:
            pts, used_wrist = self.racket_points(p)
            if not pts:
                continue
            h = self.height_of(p)
            if local is not None:
                h = (h * local) ** 0.5
            d_rel = min(dist(q, shuttle_xy) for q in pts) / h
            score = max(0.0, 1.0 - d_rel / self.cfg.max_hitter_dist_rel)
            if not used_wrist:
                score *= 0.6
            out.append(Proximity(p.player_id, score, d_rel, used_wrist, p))
        out.sort(key=lambda x: -x.score)
        return out
