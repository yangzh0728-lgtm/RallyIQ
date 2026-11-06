"""Synthetic singles rallies with stroke-type ground truth (3D physics + pinhole camera).

Each stroke launches the shuttle with a type-specific elevation; the launch speed
is solved so the shuttle reaches a type-specific depth on the other side at the
height where the next stroke meets it. The shuttle flies with quadratic drag
(terminal speed 6.8 m/s). Players are 1.75 m COCO-17 skeletons standing where
they hit. A camera behind and above the near baseline renders everything.

Like ``rallyiq.events.synthetic`` this only shows that the rules behave as
designed. It says nothing about accuracy on real video.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..contracts import (
    NET_Y_M,
    EventsResult,
    FrameGap,
    PlayerDetection,
    PlayerFrame,
    PlayerTracks,
    Rally,
    ShotEvent,
    ShuttlePoint,
    ShuttleTrack,
)

G = 9.81
V_TERMINAL = 6.8
NET_H = 1.55
W, L = 6.10, 13.40

# type: (elevation deg range, hitter contact z range, target depth from the net range)
SPEC: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "short_serve": ((5, 15), (1.0, 1.1), (2.0, 2.7)),
    "long_serve": ((45, 60), (0.8, 1.0), (5.4, 6.3)),
    "clear": ((35, 50), (2.5, 2.8), (5.0, 6.2)),
    "smash": ((-15, -6), (2.6, 2.9), (2.5, 5.0)),
    "drop": ((-5, 12), (2.4, 2.8), (0.6, 2.2)),
    "net_shot": ((40, 70), (0.8, 1.3), (0.4, 1.5)),
    "cross_net": ((30, 50), (0.8, 1.3), (0.6, 1.8)),
    "push": ((-3, 6), (1.3, 1.8), (3.3, 5.0)),
    "net_kill": ((-35, -20), (1.8, 2.3), (1.0, 4.0)),
    "drive": ((0, 8), (1.1, 1.6), (3.0, 5.5)),
    "lift": ((40, 60), (0.4, 0.9), (4.8, 6.2)),
    "block": ((10, 35), (0.5, 1.0), (0.6, 2.0)),
}

NEXT: dict[str, list[str]] = {
    "short_serve": ["net_shot", "push", "lift", "cross_net"],
    "long_serve": ["clear", "smash", "drop"],
    "clear": ["clear", "drop", "smash"],
    "smash": ["block", "lift", "drive"],
    "drop": ["net_shot", "lift", "cross_net", "push"],
    "net_shot": ["lift", "net_shot", "net_kill", "cross_net"],
    "cross_net": ["lift", "net_shot", "net_kill"],
    "push": ["drive", "block", "lift"],
    "drive": ["drive", "push", "block"],
    "lift": ["smash", "clear", "drop"],
    "block": ["net_shot", "lift", "net_kill", "cross_net"],
    "net_kill": [],
}

# COCO-17 template for a 1.75 m player: (right-of-body m, height m, forward m).
_TEMPLATE = [
    (0.0, 1.62, 0.05), (-0.03, 1.66, 0.04), (0.03, 1.66, 0.04), (-0.07, 1.63, 0.0), (0.07, 1.63, 0.0),
    (-0.19, 1.45, 0.0), (0.19, 1.45, 0.0), (-0.24, 1.15, 0.0), (0.24, 1.15, 0.0), (-0.26, 0.88, 0.05),
    (0.26, 0.88, 0.05), (-0.11, 0.95, 0.0), (0.11, 0.95, 0.0), (-0.11, 0.5, 0.05), (0.11, 0.5, 0.05),
    (-0.12, 0.08, 0.0), (0.12, 0.08, 0.0),
]
_R_ELBOW, _R_WRIST = 8, 10


@dataclass
class Camera:
    pos: tuple[float, float, float] = (3.05, -6.0, 3.5)
    look_at: tuple[float, float, float] = (3.05, 7.0, 0.0)
    width: int = 1920
    height: int = 1080
    hfov_deg: float = 58.0

    def __post_init__(self):
        d = [b - a for a, b in zip(self.pos, self.look_at)]
        n = math.sqrt(sum(v * v for v in d))
        self.fwd = [v / n for v in d]
        r = (self.fwd[1] * 1 - 0, 0 - self.fwd[0] * 1, 0.0)  # fwd x (0, 0, 1)
        rn = math.hypot(r[0], r[1])
        self.right = (r[0] / rn, r[1] / rn, 0.0)
        f, rt = self.fwd, self.right
        self.up = (rt[1] * f[2] - rt[2] * f[1], rt[2] * f[0] - rt[0] * f[2], rt[0] * f[1] - rt[1] * f[0])
        self.f = (self.width / 2) / math.tan(math.radians(self.hfov_deg / 2))

    def project(self, p) -> tuple[float, float]:
        q = [a - b for a, b in zip(p, self.pos)]
        zc = sum(a * b for a, b in zip(q, self.fwd))
        xc = sum(a * b for a, b in zip(q, self.right))
        yc = -sum(a * b for a, b in zip(q, self.up))
        return (self.width / 2 + self.f * xc / zc, self.height / 2 + self.f * yc / zc)

    def inside(self, uv) -> bool:
        return 0 <= uv[0] < self.width and 0 <= uv[1] < self.height


def fly(p0, v0, dt: float, stop_z: float, receiver_far: bool, t_max: float = 4.0):
    """Integrate with drag. Stops when descending through ``stop_z`` beyond the net (or at the floor)."""
    k = G / V_TERMINAL ** 2
    x, y, z = p0
    vx, vy, vz = v0
    pts = [(0.0, x, y, z)]
    t = 0.0
    net_ok = True
    crossed = False
    sub = 8
    h = dt / sub
    while t < t_max:
        for _ in range(sub):
            sp = math.sqrt(vx * vx + vy * vy + vz * vz)
            ax, ay, az = -k * sp * vx, -k * sp * vy, -G - k * sp * vz
            py = y
            x, y, z = x + vx * h, y + vy * h, z + vz * h
            vx, vy, vz = vx + ax * h, vy + ay * h, vz + az * h
            if not crossed and (py - NET_Y_M) * (y - NET_Y_M) <= 0:
                crossed = True
                net_ok = z > NET_H
        t += dt
        pts.append((t, x, y, z))
        beyond = (y > NET_Y_M) if receiver_far else (y < NET_Y_M)
        if z <= 0 or (beyond and vz < 0 and z <= stop_z):
            break
    return pts, net_ok and crossed


def _solve(rng, kind, contact, receiver_far, stop_z, dt, target_x=None):
    el_r, _, depth_r = SPEC[kind]
    for _ in range(40):
        el = math.radians(rng.uniform(*el_r))
        depth = rng.uniform(*depth_r)
        ty = NET_Y_M + depth if receiver_far else NET_Y_M - depth
        tx = target_x if target_x is not None else rng.uniform(0.7, W - 0.7)
        dx, dy = tx - contact[0], ty - contact[1]
        dn = math.hypot(dx, dy)
        dx, dy = dx / dn, dy / dn
        lo, hi = 1.0, 140.0
        best = None
        for _ in range(40):
            sp = 0.5 * (lo + hi)
            v0 = (sp * math.cos(el) * dx, sp * math.cos(el) * dy, sp * math.sin(el))
            pts, ok = fly(contact, v0, dt, stop_z, receiver_far)
            end = pts[-1]
            d_end = (end[2] - NET_Y_M) if receiver_far else (NET_Y_M - end[2])
            if d_end < depth:
                lo = sp
            else:
                hi = sp
            best = (pts, ok, d_end)
        pts, ok, d_end = best
        if ok and abs(d_end - depth) < 0.3 and 0.2 < pts[-1][1] < W - 0.2 and pts[-1][3] >= stop_z - 0.4:
            return pts
    return None


@dataclass
class SynthShot:
    t_ms: float
    hitter: str  # P1 near, P2 far
    kind: str
    contact: tuple[float, float, float]


@dataclass
class StrokeVideo:
    fps: float
    events: EventsResult
    shuttle: ShuttleTrack
    players: PlayerTracks
    truth: dict[str, str] = field(default_factory=dict)  # shot_id -> type
    camera: Camera = field(default_factory=Camera)


def _feet_for(contact, hitter_far: bool):
    """Where the player stands so their racket meets ``contact``."""
    x, y, z = contact
    fwd = -1.0 if hitter_far else 1.0
    right = -1.0 if hitter_far else 1.0
    lat, ahead = (0.35, 0.15) if z > 2.0 else (0.8, 0.3) if z > 1.0 else (0.6, 0.6)
    return (x - right * lat, y - fwd * ahead)


def _pose(cam: Camera, feet, far: bool, contact=None, w: float = 0.0):
    fwd = -1.0 if far else 1.0
    right = -1.0 if far else 1.0
    pts = []
    for i, (r, h, a) in enumerate(_TEMPLATE):
        if contact is not None and w > 0 and i in (_R_ELBOW, _R_WRIST):
            # pull the racket arm toward the contact point (wrist ~ 0.55 m short of it)
            cx, cy, cz = contact
            sx, sy, sz = feet[0] + right * 0.19, feet[1], 1.45
            vx, vy, vz = cx - sx, cy - sy, cz - sz
            n = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
            reach = 0.3 if i == _R_ELBOW else 0.6
            tx, ty, tz = sx + vx / n * reach, sy + vy / n * reach, sz + vz / n * reach
            bx, by, bz = feet[0] + right * r, feet[1] + fwd * a, h
            p = (bx + (tx - bx) * w, by + (ty - by) * w, bz + (tz - bz) * w)
        else:
            p = (feet[0] + right * r, feet[1] + fwd * a, h)
        pts.append(p)
    return [cam.project(p) for p in pts]


def make_rallies(
    n_rallies: int = 6,
    fps: float = 30.0,
    seed: int = 0,
    px_noise: float = 1.0,
    dropout: float = 0.0,
    court_known: bool = True,
    court_noise_m: float = 0.15,
    frame_jitter: int = 0,
    max_shots: int = 12,
    video_id: str = "synth",
) -> StrokeVideo:
    rng = random.Random(seed)
    cam = Camera()
    dt = 1.0 / fps
    t = 1.0
    flights: list[tuple[float, list]] = []  # (t0, pts)
    script: list[list[SynthShot]] = []
    for r in range(n_rallies):
        server_far = r % 2 == 1
        sx = rng.choice([1.6, 4.5])
        sy = NET_Y_M + 2.4 if server_far else NET_Y_M - 2.4
        kind = rng.choice(["short_serve", "long_serve"])
        contact = (sx, sy, rng.uniform(*SPEC[kind][1]))
        hitter_far = server_far
        shots: list[SynthShot] = []
        while True:
            options = NEXT[kind]
            end = not options or len(shots) + 1 >= max_shots or (len(shots) >= 3 and rng.random() < 0.15)
            nxt = None if end else rng.choice(options)
            stop_z = 0.0 if nxt is None else rng.uniform(*SPEC[nxt][1])
            target_x = None
            if kind == "cross_net":
                target_x = W - 0.9 if contact[0] < W / 2 else 0.9
            elif kind == "net_shot":
                target_x = min(W - 0.7, max(0.7, contact[0] + rng.uniform(-0.5, 0.5)))
            pts = _solve(rng, kind, contact, not hitter_far, stop_z, dt, target_x)
            if pts is None:
                break  # could not realise this stroke: end the rally here
            shots.append(SynthShot(t * 1000.0, "P2" if hitter_far else "P1", kind, contact))
            flights.append((t, pts))
            t += pts[-1][0]
            if nxt is None:
                flights.append((t, [(0.0, pts[-1][1], pts[-1][2], 0.0), (1.0, pts[-1][1], pts[-1][2], 0.0)]))
                t += 1.0
                break
            contact = (pts[-1][1], pts[-1][2], pts[-1][3])
            kind, hitter_far = nxt, not hitter_far
        if shots:
            script.append(shots)
        t += 3.0
    t_end = t

    # --- shuttle samples per frame ---
    n_frames = int(t_end * fps) + 1
    samples: dict[int, tuple[float, float, float]] = {}
    for t0, pts in flights:
        for tt, x, y, z in pts:
            samples[int(round((t0 + tt) * fps))] = (x, y, z)
    points: list[ShuttlePoint] = []
    gaps: list[FrameGap] = []
    gap_start, gap_reason = None, None
    for f in range(n_frames):
        p3 = samples.get(f)
        uv = cam.project(p3) if p3 else None
        visible = uv is not None and cam.inside(uv) and rng.random() >= dropout
        if visible:
            if gap_start is not None:
                gaps.append(FrameGap(start_frame=gap_start, end_frame=f - 1, reason=gap_reason))
                gap_start = None
            points.append(ShuttlePoint(frame=f, t_ms=round(f * 1000 / fps), x=round(uv[0] + rng.gauss(0, px_noise), 1),
                                       y=round(uv[1] + rng.gauss(0, px_noise), 1), confidence=round(rng.uniform(0.6, 0.99), 2)))
        elif p3 is not None:
            reason = "out_of_frame" if not cam.inside(uv) else "not_detected"
            if gap_start is None or gap_reason != reason:
                if gap_start is not None:
                    gaps.append(FrameGap(start_frame=gap_start, end_frame=f - 1, reason=gap_reason))
                gap_start, gap_reason = f, reason
    shuttle = ShuttleTrack(video_id=video_id, detector="synthetic", points=points, gaps=gaps)

    # --- players: feet keyframes at their own contacts ---
    keys: dict[str, list[tuple[float, tuple[float, float], tuple | None]]] = {"P1": [], "P2": []}
    for shots in script:
        for s in shots:
            keys[s.hitter].append((s.t_ms, _feet_for(s.contact, s.hitter == "P2"), s.contact))
    home = {"P1": (3.05, 3.5), "P2": (3.05, L - 3.5)}
    frames: list[PlayerFrame] = []
    for f in range(n_frames):
        tm = f * 1000.0 / fps
        dets = []
        for pid in ("P1", "P2"):
            ks = keys[pid]
            prev = next((k for k in reversed(ks) if k[0] <= tm), None)
            nxt = next((k for k in ks if k[0] > tm), None)
            if prev and nxt:
                s = (tm - prev[0]) / (nxt[0] - prev[0])
                feet = (prev[1][0] + (nxt[1][0] - prev[1][0]) * s, prev[1][1] + (nxt[1][1] - prev[1][1]) * s)
            else:
                feet = (prev or nxt)[1] if (prev or nxt) else home[pid]
            near_k = min(ks, key=lambda k: abs(k[0] - tm), default=None)
            w = max(0.0, 1.0 - abs(near_k[0] - tm) / 180.0) if near_k else 0.0
            if w > 0:
                feet = near_k[1]
            uv = _pose(cam, feet, pid == "P2", near_k[2] if near_k else None, w)
            kps = [[round(u + rng.gauss(0, px_noise), 1), round(v + rng.gauss(0, px_noise), 1), round(rng.uniform(0.6, 0.95), 2)]
                   for u, v in uv]
            xs, ys = [k[0] for k in kps], [k[1] for k in kps]
            fc = [round(feet[0] + rng.gauss(0, court_noise_m), 2), round(feet[1] + rng.gauss(0, court_noise_m), 2)]
            dets.append(PlayerDetection(
                player_id=pid, track_id=1 if pid == "P1" else 2, side="far" if feet[1] > NET_Y_M else "near",
                bbox=[min(xs) - 5, min(ys) - 10, max(xs) + 5, max(ys) + 5], keypoints=kps,
                foot_px=list(cam.project((feet[0], feet[1], 0.0))), foot_court=fc if court_known else None,
                on_court=True, confidence=0.9,
            ))
        frames.append(PlayerFrame(frame=f, t_ms=round(tm), players=dets))
    players = PlayerTracks(video_id=video_id, model="synthetic", frames=frames)

    # --- ground-truth hits as ShotEvents ---
    shots_out: list[ShotEvent] = []
    rallies: list[Rally] = []
    truth: dict[str, str] = {}
    for n, shots in enumerate(script, start=1):
        rid = f"r{n:03d}"
        ids = []
        for s in shots:
            f = int(round(s.t_ms * fps / 1000.0)) + (rng.randint(-frame_jitter, frame_jitter) if frame_jitter else 0)
            uv = cam.project(s.contact)
            feet = _feet_for(s.contact, s.hitter == "P2")
            sid = f"s{len(shots_out) + 1:04d}"
            shots_out.append(ShotEvent(
                shot_id=sid, rally_id=rid, frame=f, t_ms=round(f * 1000 / fps), hitter=s.hitter, hitter_confidence=0.9,
                shuttle_px=[round(uv[0], 1), round(uv[1], 1)] if cam.inside(uv) else None,
                hitter_court=[round(feet[0] + rng.gauss(0, court_noise_m), 2), round(feet[1] + rng.gauss(0, court_noise_m), 2)]
                if court_known else None,
                confidence=0.9, status="ok",
            ))
            truth[sid] = s.kind
            ids.append(sid)
        last = shots[-1]
        ef = min(n_frames - 1, int((last.t_ms / 1000.0 + 2.5) * fps))
        sf = max(0, int(round(shots[0].t_ms * fps / 1000.0)) - 5)
        rallies.append(Rally(rally_id=rid, start_frame=sf, end_frame=ef, start_ms=round(sf * 1000 / fps),
                             end_ms=round(ef * 1000 / fps), shot_ids=ids, end_reason="shuttle_static",
                             confidence=0.9, status="ok"))
    events = EventsResult(video_id=video_id, rallies=rallies, shots=shots_out)
    return StrokeVideo(fps, events, shuttle, players, truth, cam)
