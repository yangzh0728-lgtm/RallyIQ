"""Synthetic singles or doubles footage (shuttle track + COCO-17 poses) with ground truth.

Used to develop and regression-test the rules before real perception output
exists. The camera sits behind the near baseline: the near player is large
and low in the image, the far player small and high.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .types import Keypoint, PlayerPose, PoseFrame, ShuttleObs

# COCO-17 template, in player heights relative to the feet midpoint (y up is negative).
_NEUTRAL = [
    (0.0, -0.93),  # nose
    (-0.03, -0.95), (0.03, -0.95),  # eyes
    (-0.06, -0.94), (0.06, -0.94),  # ears
    (-0.12, -0.80), (0.12, -0.80),  # shoulders
    (-0.17, -0.63), (0.17, -0.63),  # elbows
    (-0.18, -0.48), (0.18, -0.48),  # wrists
    (-0.08, -0.50), (0.08, -0.50),  # hips
    (-0.08, -0.27), (0.08, -0.27),  # knees
    (-0.08, -0.03), (0.08, -0.03),  # ankles
]
# Right arm at contact (overhead or side), elbow and wrist offsets.
_HIT_ARMS = [((0.20, -0.95), (0.24, -1.15)), ((0.30, -0.70), (0.45, -0.75)), ((0.25, -0.60), (0.40, -0.45))]
_R_ELBOW, _R_WRIST = 8, 10


@dataclass
class PlayerSpec:
    player_id: str
    side: str
    height_px: float
    contact_box: tuple[float, float, float, float]  # x0, y0, x1, y1 where hits happen
    team: Optional[str] = None  # doubles only


NEAR = PlayerSpec("P1", "near", 330.0, (500, 600, 1400, 880))
FAR = PlayerSpec("P2", "far", 150.0, (700, 230, 1200, 360))

# Doubles: partners side by side, each covering half of their end.
DOUBLES = {
    "A": [PlayerSpec("P1", "near", 330.0, (500, 600, 950, 880), "A"),
          PlayerSpec("P3", "near", 330.0, (950, 600, 1400, 880), "A")],
    "B": [PlayerSpec("P2", "far", 150.0, (950, 230, 1200, 360), "B"),
          PlayerSpec("P4", "far", 150.0, (700, 230, 950, 360), "B")],
}


@dataclass
class SynthRally:
    hits: list[tuple[float, str]]  # (t_ms, player_id)
    start_ms: float
    end_ms: float
    end: str  # "static" | "lost"


@dataclass
class SynthVideo:
    shuttle: list[ShuttleObs]
    poses: list[PoseFrame]
    rallies: list[SynthRally]
    fps: float
    gt_hits: list[tuple[int, str]] = field(default_factory=list)  # (frame, player_id)


def _lerp(a, b, s):
    return a + (b - a) * s


def make_video(
    n_rallies: int = 3,
    hits_per_rally: tuple[int, int] = (4, 10),
    fps: float = 30.0,
    seed: int = 0,
    px_noise: float = 1.5,
    dropout: float = 0.0,
    outlier_rate: float = 0.0,
    occlude_hits: float = 0.0,  # probability a hit's ±2 frames are unseen
    interpolate_gaps: bool = False,
    kp_noise: float = 2.0,
    end_modes: Optional[list[str]] = None,
    with_poses: bool = True,
    trailing_ms: float = 2000.0,
    doubles: bool = False,
) -> SynthVideo:
    rng = random.Random(seed)
    if doubles:
        teams = DOUBLES
    else:
        teams = {"A": [NEAR], "B": [FAR]}
    players = {p.player_id: p for ps in teams.values() for p in ps}
    frame_ms = 1000.0 / fps

    # --- script: hits (time, player, contact point, arm) and shuttle path pieces ---
    t = 1000.0
    rallies: list[SynthRally] = []
    shuttle_fn = []  # (t0, t1, fn(t) -> (x, y) or None)
    hit_script: list[tuple[float, str, tuple[float, float], int]] = []
    for r in range(n_rallies):
        serve_team = "A" if r % 2 == 0 else "B"
        recv_team = "B" if serve_team == "A" else "A"
        n_hits = rng.randint(*hits_per_rally)
        order = [
            (teams[serve_team] if k % 2 == 0 else teams[recv_team])[0]
            if not doubles
            else rng.choice(teams[serve_team] if k % 2 == 0 else teams[recv_team])
            for k in range(n_hits)
        ]
        contacts = []
        for p in order:
            x0, y0, x1, y1 = p.contact_box
            contacts.append((rng.uniform(x0, x1), rng.uniform(y0, y1)))
        # serve: shuttle held still near the server's wrist, then struck
        hold = 900.0
        c0 = contacts[0]
        shuttle_fn.append((t, t + hold, lambda _t, c=c0: (c[0] - 2, c[1] + 4)))
        t += hold
        start = t
        hits = []
        for k, p in enumerate(order):
            arm = rng.randrange(len(_HIT_ARMS))
            hit_script.append((t, p.player_id, contacts[k], arm))
            hits.append((t, p.player_id))
            dur = rng.uniform(700, 1300)
            a = contacts[k]
            if k + 1 < n_hits:
                b = contacts[k + 1]
            else:  # last shot lands on the opponent's floor
                opp_team = "B" if p in teams["A"] else "A"
                opp = teams[opp_team][0] if not doubles else rng.choice(teams[opp_team])
                x0, y0, x1, y1 = opp.contact_box
                b = (rng.uniform(x0, x1), y1 + opp.height_px * rng.uniform(0.6, 0.9))
            h = rng.uniform(30, 250)
            drag = rng.uniform(0.8, 2.0)

            def fn(_t, t0=t, d=dur, a=a, b=b, h=h, drag=drag):
                tau = (_t - t0) / d
                s = (1 - math.exp(-drag * tau)) / (1 - math.exp(-drag))
                return (_lerp(a[0], b[0], s), _lerp(a[1], b[1], s) - h * 4 * s * (1 - s))

            shuttle_fn.append((t, t + dur, fn))
            t += dur
            last_b = b
        modes = end_modes or ["static", "lost"]
        mode = modes[r % len(modes)]
        if mode == "static":
            shuttle_fn.append((t, t + 1500.0, lambda _t, b=last_b: b))
            end_ms = t
            t += 1500.0
        else:
            end_ms = t
        rallies.append(SynthRally(hits, start, end_ms, mode))
        t += 3000.0  # dead time between rallies, shuttle unseen
    t_end = t - 3000.0 + trailing_ms

    # --- render shuttle ---
    n_frames = int(t_end / frame_ms) + 1
    hit_frames = {round(h[0] / frame_ms) for h in hit_script}
    occluded = set()
    for hf in hit_frames:
        if rng.random() < occlude_hits:
            occluded.update(range(hf - 2, hf + 3))
    shuttle: list[ShuttleObs] = []
    piece = 0
    for f in range(n_frames):
        tf = f * frame_ms
        while piece < len(shuttle_fn) and shuttle_fn[piece][1] < tf:
            piece += 1
        pos = None
        if piece < len(shuttle_fn) and shuttle_fn[piece][0] <= tf <= shuttle_fn[piece][1]:
            pos = shuttle_fn[piece][2](tf)
        if pos is None or f in occluded or rng.random() < dropout:
            shuttle.append(ShuttleObs(f, tf))
            continue
        if rng.random() < outlier_rate:
            pos = (rng.uniform(0, 1920), rng.uniform(0, 1080))
        x = pos[0] + rng.gauss(0, px_noise)
        y = pos[1] + rng.gauss(0, px_noise)
        shuttle.append(ShuttleObs(f, tf, x, y, True, rng.uniform(0.6, 0.99)))
    if interpolate_gaps:
        shuttle = _interpolate_short_gaps(shuttle, max_gap=5)

    # --- render poses: feet move linearly between the positions each hit needs ---
    poses: list[PoseFrame] = []
    if with_poses:
        keyframes: dict[str, list[tuple[float, tuple[float, float], int]]] = {pid: [] for pid in players}
        for th, pid, c, arm in hit_script:
            hgt = players[pid].height_px
            wx, wy = _HIT_ARMS[arm][1]
            keyframes[pid].append((th, (c[0] - wx * hgt, c[1] - wy * hgt), arm))
        for f in range(n_frames):
            tf = f * frame_ms
            ps = []
            for pid, spec in players.items():
                ps.append(_render_player(spec, keyframes[pid], tf, rng, kp_noise))
            poses.append(PoseFrame(f, tf, ps))

    gt = [(round(th / frame_ms), pid) for th, pid, _, _ in hit_script]
    return SynthVideo(shuttle, poses, rallies, fps, gt)


def _render_player(spec, keys, tf, rng, kp_noise) -> PlayerPose:
    hgt = spec.height_px
    if not keys:
        cx = 960.0 if spec.team is None else (spec.contact_box[0] + spec.contact_box[2]) / 2
        foot, arm, w = ((cx, spec.contact_box[3] + hgt), 0, 0.0)
    else:
        prev = max((k for k in keys if k[0] <= tf), default=None, key=lambda k: k[0])
        nxt = min((k for k in keys if k[0] > tf), default=None, key=lambda k: k[0])
        if prev is None:
            foot, arm = nxt[1], nxt[2]
        elif nxt is None:
            foot, arm = prev[1], prev[2]
        else:
            s = (tf - prev[0]) / (nxt[0] - prev[0])
            foot = (_lerp(prev[1][0], nxt[1][0], s), _lerp(prev[1][1], nxt[1][1], s))
            arm = prev[2] if s < 0.5 else nxt[2]
        nearest = min(keys, key=lambda k: abs(k[0] - tf))
        w = max(0.0, 1.0 - abs(nearest[0] - tf) / 200.0)
        arm = nearest[2]
    offs = list(_NEUTRAL)
    (ex, ey), (wx, wy) = _HIT_ARMS[arm]
    offs[_R_ELBOW] = (_lerp(offs[_R_ELBOW][0], ex, w), _lerp(offs[_R_ELBOW][1], ey, w))
    offs[_R_WRIST] = (_lerp(offs[_R_WRIST][0], wx, w), _lerp(offs[_R_WRIST][1], wy, w))
    kps = [
        Keypoint(foot[0] + dx * hgt + rng.gauss(0, kp_noise), foot[1] + dy * hgt + rng.gauss(0, kp_noise), rng.uniform(0.6, 0.95))
        for dx, dy in offs
    ]
    return PlayerPose(spec.player_id, kps, spec.side, team=spec.team)


def _interpolate_short_gaps(obs: list[ShuttleObs], max_gap: int) -> list[ShuttleObs]:
    out = list(obs)
    vis = [i for i, o in enumerate(obs) if o.usable]
    for a, b in zip(vis, vis[1:]):
        if 1 < b - a <= max_gap + 1:
            oa, ob = obs[a], obs[b]
            for i in range(a + 1, b):
                s = (i - a) / (b - a)
                out[i] = ShuttleObs(
                    obs[i].frame, obs[i].t_ms, _lerp(oa.x, ob.x, s), _lerp(oa.y, ob.y, s), True, 0.5, True
                )
    return out
