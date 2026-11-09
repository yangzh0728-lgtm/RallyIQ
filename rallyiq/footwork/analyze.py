"""Footwork from the foot tracks: contacts, steps, split steps, jumps, lunges, step patterns and recovery.

Everything is measured on the court plane from the two ankle keypoints (see ``tracks``) and linked to
the *effective* hits (after human corrections), so it is a pure function like ``report.stats``:

* **Contacts and steps.** A contact is a stretch where one foot is planted; a step is the foot moving
  from one contact to the next by at least ``STEP_MIN_M`` (more where the projection is coarse).
* **Flights.** Both feet off the floor at once. A short flight that ends with both feet landing
  together, close to the opponent's hit, is a **split step** (启动步); a longer, higher one is a
  **jump** (起跳). Split-step timing is the landing time minus the opponent's hit time.
* **Reaction.** Opponent's hit to the first sustained movement of the mid-feet point towards where
  the player hits next.
* **Pattern.** The steps between the opponent's hit and the player's own hit, judged along the
  direction of travel: the trailing foot closing up to the leading foot without passing it is a
  **chassé** (并步); passing it while moving sideways or back is a **cross step** (交叉步); passing it
  while moving forward is **running** (跑动).
* **Finish.** A wide stance (≥ ``LUNGE_STANCE_M``) at the hit after a long last step is a **lunge**
  (弓步); a jump that lands just after the hit is a jump hit.
* **Recovery.** After the hit, the time until the mid-feet point is back within ``BASE_RADIUS_M`` of
  the base (centre of the own half), and how far from the base the player still was when the opponent
  hit next.

Doubles: contacts, steps and jumps are computed for every player; shot-linked measures need one
opponent and are singles only for now.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from ..contracts import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    CourtCalibration,
    EventsResult,
    FootContact,
    FootLandmarks,
    FootStep,
    FootworkEvent,
    FootworkResult,
    PlayerFootworkQuality,
    PlayerTracks,
    ShotEvent,
    ShotFootwork,
)
from .sole import PLAYER_LENGTH_M, Sole, angle_between, attach_feet, plant_sole, toe_angle_deg
from .tracks import Run, _odd, _segments, build_runs, smooth

STEP_MIN_M = 0.18
BASE = (COURT_WIDTH_M / 2, 3.6)  # own-half frame: centre line, ~1 m behind the short service line
BASE_RADIUS_M = 1.0
SPLIT_WINDOW_S = (-0.45, 0.45)  # split-step landing relative to the opponent's hit
SPLIT_ON_TIME_S = (-0.1, 0.3)  # landing just after the opponent's contact (pros: about +0.1..+0.25 s)
SPLIT_MAX_MOVE_MPS = 2.5  # mid-feet speed during the hop: faster is a running stride
HOP_UP_MPS = 0.2  # both ankles rise at least this fast (image, m/s at the player's distance) ...
HOP_DOWN_MPS = 0.2  # ... then come down at least this fast
HOP_MAX_S = 0.35  # from the rise to the start of the descent
HOP_DETREND_S = 0.5  # vertical motion slower than this is walking towards/away from the camera
JUMP_MIN_FLIGHT_S = 0.2
JUMP_MAX_FLIGHT_S = 0.9
JUMP_MIN_HEIGHT_M = 0.12
MOVE_ONSET_MPS = 1.0  # mid-feet speed towards the next hit that counts as "moving"
MOVE_ONSET_HOLD_S = 0.1
MAX_REACTION_S = 1.0
LUNGE_STANCE_M = 1.0
LUNGE_LAST_STEP_M = 0.5
NEAR_HIT_S = 0.3  # actions this close to a hit belong to it
SOLE_MAX_JITTER_M = 0.08  # foot directions / landings only for players located this well
LUNGE_TOE_OFF_DEG = 45  # front foot pointing further than this from the lunge direction
LUNGE_BACKWARD = 0.5  # a lunge step going back more than this share of its length: toe direction not judged
LUNGE_TOE_FLIP_DEG = 120  # a toe pointing back along the step is a heel / toe mix-up, not a technique
MAX_RECOVERY_S = 2.5  # judge recovery only when the opponent's next hit comes within this
HIT_SLACK_S = 0.07  # a logged hit frame can be this far off the moment the feet show (2 frames at 30 fps)
JUMP_HIT_SLACK_S = 0.1  # a hit this close to a jump's take-off / landing belongs to the jump
JITTER_MIN_S = 0.13  # contacts at least this long measure a player's foot jitter ...
JITTER_MIN_RUN_S = 1.0  # ... in runs at least this long


def own_half(x: float, y: float) -> tuple[float, float]:
    """Court metres -> (lateral from own left sideline, depth from own baseline)."""
    if y <= NET_Y_M:
        return x, y
    return COURT_WIDTH_M - x, COURT_LENGTH_M - y


def base_court(y_side: float) -> np.ndarray:
    """Base position in court metres for a player on the half containing ``y_side``."""
    if y_side <= NET_Y_M:
        return np.array(BASE)
    return np.array([COURT_WIDTH_M - BASE[0], COURT_LENGTH_M - BASE[1]])


# --------------------------------------------------------------------------- per run


@dataclass
class Flight:
    run: Run
    i0: int  # first index with both feet up
    i1: int  # last index with both feet up
    land: tuple[int, int]  # landing index of each foot (first planted index after the flight)
    height_m: float
    travel_m: float

    @property
    def landing(self) -> int:
        return max(self.land)

    @property
    def landing_gap(self) -> int:
        return abs(self.land[0] - self.land[1])


@dataclass
class Plant:
    run: Run
    foot: int
    a: int
    b: int
    court: np.ndarray
    image: np.ndarray
    sole: Sole | None = None


def _plants(run: Run) -> list[Plant]:
    out = []
    for j in range(2):
        for a, b in _segments(run.planted[:, j]):
            out.append(Plant(run, j, a, b, np.median(run.court[a:b + 1, j], axis=0),
                             np.median(run.img[a:b + 1, j], axis=0)))
    out.sort(key=lambda p: (p.a, p.foot))
    return out


def jitter_m(run: Run, fps: float) -> float:
    """Typical planted-foot wander on the court (median absolute deviation within contacts)."""
    devs = []
    min_len = max(4, int(round(JITTER_MIN_S * fps)))
    for j in range(2):
        for a, b in _segments(run.planted[:, j]):
            if b - a >= min_len:
                c = run.court[a:b + 1, j]
                devs.append(float(np.median(np.linalg.norm(c - np.median(c, axis=0), axis=1))))
    return float(np.median(devs)) if devs else float("nan")


def _flights(run: Run) -> list[Flight]:
    both_up = ~run.planted[:, 0] & ~run.planted[:, 1]
    out = []
    n = len(run.frames)
    for a, b in _segments(both_up):
        if a == 0 or b >= n - 1:
            continue
        land = []
        for j in range(2):
            k = b + 1
            while k < n and not run.planted[k, j]:
                k += 1
            land.append(min(k, n - 1))
        # height: how far each ankle rose above the straight line between take-off and landing (image)
        heights = []
        travel = []
        for j in range(2):
            i_off, i_on = a - 1, land[j]
            p0, p1 = run.img[i_off, j], run.img[i_on, j]
            seg = run.img[i_off:i_on + 1, j]
            lin = p0 + np.outer(np.linspace(0, 1, len(seg)), p1 - p0)
            rise = (lin[:, 1] - seg[:, 1]) / run.scale[i_off:i_on + 1, j]
            heights.append(float(np.max(rise)))
            travel.append(float(np.linalg.norm(run.court[i_on, j] - run.court[i_off, j])))
        out.append(Flight(run, a, b, (land[0], land[1]), min(heights), max(travel)))
    return out


@dataclass
class Hop:
    """Both ankles rising and coming down together: the small hop of a split step (启动步)."""

    run: Run
    i0: int  # rise starts
    land: int  # descent ends (landing)
    rise_m: float


def _hops(run: Run, fps: float) -> list[Hop]:
    vy = -np.gradient(run.img[:, :, 1], axis=0) * fps / run.scale  # up positive, m/s at the player's distance
    vy = vy - smooth(vy, _odd(HOP_DETREND_S * fps))
    up, down = (vy > 0).all(axis=1), (vy < 0).all(axis=1)
    b = np.where(up, vy.min(axis=1), np.where(down, vy.max(axis=1), 0.0))  # what both feet agree on
    out = []
    n = len(b)
    for a, e in _segments(b > 0.05):
        if b[a:e + 1].max() < HOP_UP_MPS:
            continue
        lim = min(n - 1, e + int(HOP_MAX_S * fps))
        k = e + 1
        while k <= lim and b[k] > -0.05:
            k += 1
        if k > lim:
            continue
        k2 = k
        while k2 < n and b[k2] < -0.05:
            k2 += 1
        if b[k:k2].min() > -HOP_DOWN_MPS:
            continue
        out.append(Hop(run, a, min(k2, n - 1), float(b[a:e + 1].sum() / fps)))
    return out


# --------------------------------------------------------------------------- result builder


@dataclass
class _Hit:
    shot: ShotEvent
    t: float


def fingerprint(events: EventsResult, players: PlayerTracks | None, calibration: CourtCalibration | None,
                feet: FootLandmarks | None = None) -> str:
    h = hashlib.sha256()
    h.update(events.model_dump_json(include={"rallies", "shots"}).encode())
    if players is not None:
        h.update(f"{players.model}|{len(players.frames)}|{players.side_switches}".encode())
        if players.frames:
            h.update(players.frames[0].model_dump_json().encode())
            h.update(players.frames[-1].model_dump_json().encode())
    if calibration is not None:
        h.update(calibration.model_dump_json(include={"segments"}).encode())
    if feet is not None and feet.frames:
        h.update(f"{feet.model}|{len(feet.frames)}".encode())
        h.update(feet.frames[-1].model_dump_json().encode())
    return h.hexdigest()[:16]


class Footwork:
    """Footwork of one video. ``result()`` gives the contract; the helpers serve stats and moments."""

    def __init__(self, events: EventsResult, players: PlayerTracks | None, calibration: CourtCalibration | None,
                 fps: float = 30.0, feet: FootLandmarks | None = None):
        self.events = events
        self.fps = fps
        self.runs = build_runs(players, calibration, fps) if players is not None else []
        attach_feet(self.runs, feet)
        self.by_pid: dict[str, list[Run]] = {}
        for r in self.runs:
            self.by_pid.setdefault(r.pid, []).append(r)
        self.plants: dict[str, list[Plant]] = {pid: [p for r in rs for p in _plants(r)] for pid, rs in self.by_pid.items()}
        self._plants_of: dict[tuple[int, int], list[Plant]] = {}
        for ps in self.plants.values():
            for p in ps:
                self._plants_of.setdefault((id(p.run), p.foot), []).append(p)
        self.flights: dict[str, list[Flight]] = {pid: [f for r in rs for f in _flights(r)] for pid, rs in self.by_pid.items()}
        self.hops: dict[str, list[Hop]] = {pid: [h for r in rs for h in _hops(r, fps)] for pid, rs in self.by_pid.items()}
        self._split: dict[str, tuple[str, str, int | None]] = {}  # opponent shot id -> (player, timing, offset ms)
        self.jitter: dict[str, float] = {}
        self.slack = max(2, int(round(HIT_SLACK_S * fps)))  # frames
        for pid, rs in self.by_pid.items():
            js = [jitter_m(r, fps) for r in rs if len(r.frames) > JITTER_MIN_RUN_S * fps]
            js = [j for j in js if np.isfinite(j)]
            self.jitter[pid] = float(np.median(js)) if js else float("nan")
        self.sole_ok: dict[str, bool] = {pid: self._attach_soles(pid) for pid in self.plants}
        self.fingerprint = fingerprint(events, players, calibration, feet)
        self._events: list[FootworkEvent] = []
        self._shots: dict[str, ShotFootwork] = {}
        self._contacts: list[FootContact] = []
        self._steps: list[FootStep] = []
        self._build()

    # ------------------------------------------------------------------ lookup

    def run_at(self, pid: str, frame: int) -> Run | None:
        for r in self.by_pid.get(pid, []):
            if r.frames[0] <= frame <= r.frames[-1]:
                return r
        return None

    def center_at(self, pid: str, frame: int) -> np.ndarray | None:
        r = self.run_at(pid, frame)
        if r is None:
            return None
        i = r.index(frame)
        return None if i is None else r.center[i]

    def steps_between(self, pid: str, f0: int, f1: int) -> list[FootStep]:
        """Steps of ``pid`` landing in (f0, f1]."""
        frames = self._step_frames.get(pid, [])
        sts = self._steps_by_pid.get(pid, [])
        return sts[bisect_left(frames, f0 + 1):bisect_left(frames, f1 + 1)]

    def step_min(self, pid: str) -> float:
        j = self.jitter.get(pid, float("nan"))
        return max(STEP_MIN_M, 3 * j) if np.isfinite(j) else STEP_MIN_M

    def _attach_soles(self, pid: str) -> bool:
        """Heel / toe of every contact, when this player's feet are located well enough to trust them."""
        j = self.jitter.get(pid, float("nan"))
        if not np.isfinite(j) or j > SOLE_MAX_JITTER_M:
            return False
        soles = [(p, plant_sole(p.run, p.foot, p.a, p.b, self.fps)) for p in self.plants.get(pid, [])]
        lengths = [s.length_m for _, s in soles if s is not None]
        if not lengths or (len(lengths) >= 5 and not PLAYER_LENGTH_M[0] <= float(np.median(lengths)) <= PLAYER_LENGTH_M[1]):
            return False  # the projection does not keep a shoe's shape for this player
        for p, s in soles:
            p.sole = s
        return True

    def _ms(self, frame: int) -> int:
        return int(round(frame * 1000 / self.fps))

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        for pid, plants in self.plants.items():
            smin = self.step_min(pid)
            last: dict[tuple[int, int], Plant] = {}
            for p in plants:
                r = p.run
                so = p.sole
                self._contacts.append(FootContact(
                    player_id=pid, foot="left" if p.foot == 0 else "right", start_frame=int(r.frames[p.a]),
                    end_frame=int(r.frames[p.b]), t_ms=self._ms(int(r.frames[p.a])), end_ms=self._ms(int(r.frames[p.b])),
                    image=[round(float(v), 1) for v in p.image], court=[round(float(v), 3) for v in p.court],
                    heel_image=None if so is None else [round(float(v), 1) for v in so.heel_image],
                    toe_image=None if so is None else [round(float(v), 1) for v in so.toe_image],
                    toe_dir_deg=None if so is None else round(toe_angle_deg(so.direction, float(p.court[1])), 0),
                    landing=None if so is None else so.landing))
                prev = last.get((id(r), p.foot))
                last[(id(r), p.foot)] = p
                if prev is None:
                    continue
                length = float(np.linalg.norm(p.court - prev.court))
                if length >= smin:
                    self._steps.append(FootStep(
                        player_id=pid, foot="left" if p.foot == 0 else "right", start_frame=int(r.frames[prev.b + 1]),
                        end_frame=int(r.frames[p.a]), t_ms=self._ms(int(r.frames[p.a])), length_m=round(length, 2),
                        from_court=[round(float(v), 3) for v in prev.court], to_court=[round(float(v), 3) for v in p.court],
                        landing=None if p.sole is None else p.sole.landing))
        self._steps.sort(key=lambda s: (s.end_frame, s.player_id))
        self._steps_by_pid: dict[str, list[FootStep]] = {}
        for st in self._steps:
            self._steps_by_pid.setdefault(st.player_id, []).append(st)
        self._step_frames = {pid: [st.end_frame for st in sts] for pid, sts in self._steps_by_pid.items()}
        self._contacts.sort(key=lambda c: (c.start_frame, c.player_id, c.foot))

        # jumps (any time); split steps are judged against the opponent's hits below
        for pid, fls in self.flights.items():
            for f in fls:
                dur = (f.i1 - f.i0 + 1) / self.fps
                if JUMP_MIN_FLIGHT_S <= dur <= JUMP_MAX_FLIGHT_S and f.height_m >= JUMP_MIN_HEIGHT_M:
                    r = f.run
                    first = min(f.land)
                    self._add_event(pid, "jump", int(r.frames[first]), int(r.frames[f.i0]), int(r.frames[first]),
                                    r.center[first], round(f.height_m, 2), f"起跳（腾空约 {dur:.2f} 秒）",
                                    self._conf(pid, 0.7))
        if self.events.match_format == "singles":
            self._link_shots()
        # a jump that covers one of the player's own hits belongs to it (jump smash / jump clear)
        hits = sorted((s.frame, s.shot_id, s.hitter) for s in self.events.shots if s.hitter)
        js = max(3, int(round(JUMP_HIT_SLACK_S * self.fps)))
        for e in self._events:
            if e.kind == "jump" and e.shot_id is None:
                near = [(abs(f - (e.start_frame + e.frame) / 2), sid) for f, sid, h in hits
                        if h == e.player_id and e.start_frame - js <= f <= e.frame + js]
                if near:
                    e.shot_id = min(near)[1]
        self._events.sort(key=lambda e: (e.t_ms, e.player_id))
        for k, e in enumerate(self._events, 1):
            e.event_id = f"fw{k}"

    def _conf(self, pid: str, base: float) -> float:
        """Lower confidence where the projection is coarse (far player on an end-on camera)."""
        j = self.jitter.get(pid, float("nan"))
        if not np.isfinite(j):
            return round(base * 0.5, 2)
        return round(float(base * np.clip(1.2 - j / 0.15, 0.3, 1.0)), 2)

    def _add_event(self, pid: str, kind: str, frame: int, f0: int, f1: int, court, value, label: str, conf: float,
                   shot_id: str | None = None) -> FootworkEvent:
        e = FootworkEvent(event_id="", player_id=pid, kind=kind, frame=frame, t_ms=self._ms(frame), start_frame=f0,
                          end_frame=f1, court=None if court is None else [round(float(v), 2) for v in court],
                          value=None if value is None else float(value), shot_id=shot_id, label=label, confidence=conf)
        self._events.append(e)
        return e

    # ------------------------------------------------------------------ shot-linked measures

    def _link_shots(self) -> None:
        shots_by_id = {s.shot_id: s for s in self.events.shots}
        for rally in sorted(self.events.rallies, key=lambda r: r.start_frame):
            seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
            for s in seq:
                if s.hitter in ("P1", "P2"):
                    self._split_step("P2" if s.hitter == "P1" else "P1", s)
            for k, s in enumerate(seq):
                if s.hitter not in ("P1", "P2"):
                    continue
                opp = "P2" if s.hitter == "P1" else "P1"
                prev = next((p for p in reversed(seq[:k]) if p.hitter == opp), None) if k else None
                if prev is not None and seq[k - 1].hitter != opp:
                    prev = None  # the hit before this one was not the opponent's: no clean opponent-to-me phase
                nxt = seq[k + 1] if k + 1 < len(seq) and seq[k + 1].hitter == opp else None
                end_frame = nxt.frame if nxt is not None else rally.end_frame
                sf = self._shot_footwork(s, prev, end_frame, nxt is not None)
                if sf is not None:
                    self._shots[s.shot_id] = sf

    def _split_step(self, pid: str, opp_shot: ShotEvent) -> None:
        """Did ``pid`` split-step for the opponent's hit ``opp_shot``, and when did they land?"""
        run = self.run_at(pid, opp_shot.frame)
        if run is None or run.nearest(opp_shot.frame, self.slack) is None:
            return
        t_opp = opp_shot.frame / self.fps
        best = None
        for h in self.hops.get(pid, []):
            if h.run is not run:
                continue
            off = run.frames[h.land] / self.fps - t_opp
            if float(np.mean(run.center_speed[h.i0:h.land + 1])) > SPLIT_MAX_MOVE_MPS:
                continue  # a running stride, not a hop in place
            if SPLIT_WINDOW_S[0] <= off <= SPLIT_WINDOW_S[1] and (best is None or abs(off - 0.15) < abs(best[0] - 0.15)):
                best = (off, h)
        if best is None:
            self._split[opp_shot.shot_id] = (pid, "none", None)
            return
        off, h = best
        ms = int(round(off * 1000))
        timing = "early" if off < SPLIT_ON_TIME_S[0] else "late" if off > SPLIT_ON_TIME_S[1] else "on_time"
        self._split[opp_shot.shot_id] = (pid, timing, ms)
        label = {"early": "启动步偏早", "late": "启动步偏晚", "on_time": "启动步"}[timing]
        self._add_event(pid, "split_step", int(run.frames[h.land]), int(run.frames[h.i0]), int(run.frames[h.land]),
                        run.center[h.land], ms, f"{label}（{_rel(ms)}落地）", self._conf(pid, 0.7), opp_shot.shot_id)

    def _drop_split(self, opp_shot_id: str) -> None:
        pid = self._split[opp_shot_id][0]
        self._split[opp_shot_id] = (pid, "none", None)
        self._events = [e for e in self._events if not (e.kind == "split_step" and e.shot_id == opp_shot_id)]

    def _shot_footwork(self, s: ShotEvent, prev: ShotEvent | None, end_frame: int, opp_hits_next: bool
                       ) -> ShotFootwork | None:
        pid = s.hitter
        run = self.run_at(pid, s.frame)
        if run is None:
            return None
        fps = self.fps
        i_hit = run.nearest(s.frame, self.slack)
        if i_hit is None:
            return None
        conf = self._conf(pid, 0.8)
        out = ShotFootwork(shot_id=s.shot_id, player_id=pid, t_ms=s.t_ms, confidence=conf)
        f_hit = int(run.frames[i_hit])
        hit_pos = run.center[i_hit]

        # ---- split step + reaction (needs the opponent's hit just before, inside this run)
        if prev is not None and run.frames[0] <= prev.frame <= run.frames[-1] - self.slack:
            i_opp = run.nearest(prev.frame, self.slack)
            got = self._split.get(prev.shot_id)
            if got is not None and got[0] == pid:
                out.split_step, out.split_offset_ms = got[1], got[2]
            # reaction: first sustained movement towards where I hit
            if i_opp is not None:
                start = run.center[i_opp]
                d = hit_pos - start
                dist = float(np.linalg.norm(d))
                if dist >= 0.4:
                    u = d / dist
                    v = run.center_vel @ u
                    hold = max(1, int(round(MOVE_ONSET_HOLD_S * fps)))
                    lo = i_opp
                    hi = min(i_hit, i_opp + int(MAX_REACTION_S * fps))
                    for i in range(lo, hi):
                        if np.all(v[i:i + hold] > MOVE_ONSET_MPS):
                            out.reaction_ms = int(round((run.frames[i] - prev.frame) * 1000 / fps))
                            break
                # a "hop" that lands well after the player already set off is a running stride, not a split step
                if out.reaction_ms and out.split_offset_ms is not None and \
                        out.split_offset_ms > out.reaction_ms + 150:
                    self._drop_split(prev.shot_id)
                    out.split_step, out.split_offset_ms = "none", None
                # steps, distance, speed from the opponent's hit to mine
                steps = self.steps_between(pid, prev.frame, s.frame + self.slack)
                out.steps = len(steps)
                seg = run.center[i_opp:i_hit + 1]
                out.move_m = round(float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))), 2) if len(seg) > 1 else 0.0
                out.peak_speed_mps = round(float(np.max(run.center_speed[i_opp:i_hit + 1])), 1)
                out.pattern = self._pattern(pid, run, steps, start, hit_pos, s)

        # ---- finish at the hit: lunge / jump / planted
        out.stance_m, finish = self._finish(pid, run, i_hit, s)
        out.finish = finish
        if finish == "lunge":
            self._lunge_sole(pid, run, s, out)
            self._add_event(pid, "lunge", f_hit, f_hit, f_hit, hit_pos, out.stance_m,
                            f"弓步击球（两脚相距 {out.stance_m:.1f} 米{_lunge_form(out)}）", conf, s.shot_id)
            if "lunge" not in out.pattern:
                out.pattern.append("lunge")
        elif finish == "jump" and "jump" not in out.pattern:
            out.pattern.append("jump")

        # ---- recovery after the hit
        base = base_court(float(hit_pos[1]))
        j_end = run.nearest(min(end_frame, int(run.frames[-1])), self.slack)
        if j_end is not None and j_end > i_hit:
            dists = np.linalg.norm(run.center[i_hit:j_end + 1] - base, axis=1)
            inside = np.where(dists <= BASE_RADIUS_M)[0]
            if len(inside):
                k = int(inside[0])
                out.recovery_ms = int(round(k * 1000 / fps))
                if k > 0:
                    self._add_event(pid, "recovery", int(run.frames[i_hit + k]), f_hit, int(run.frames[i_hit + k]),
                                    run.center[i_hit + k], out.recovery_ms, f"回到中心（击球后 {k / fps:.1f} 秒）",
                                    conf, s.shot_id)
            if opp_hits_next and int(run.frames[j_end]) >= end_frame - self.slack and \
                    (end_frame - f_hit) / fps <= MAX_RECOVERY_S:
                out.recovery_gap_m = round(float(dists[-1]), 2)
        out.summary = summarize(out)
        return out

    def _finish(self, pid: str, run: Run, i_hit: int, s: ShotEvent) -> tuple[float | None, str]:
        fps = self.fps
        w = int(round(NEAR_HIT_S * fps))
        lo, hi = max(0, i_hit - w), min(len(run.frames) - 1, i_hit + w)
        # a jump whose flight covers the hit
        for f in self.flights.get(pid, []):
            if f.run is run and f.i0 - self.slack <= i_hit <= f.landing + self.slack and \
                    JUMP_MIN_FLIGHT_S <= (f.i1 - f.i0 + 1) / fps <= JUMP_MAX_FLIGHT_S and f.height_m >= JUMP_MIN_HEIGHT_M:
                return None, "jump"
        both = run.planted[lo:hi + 1].all(axis=1)
        if not both.any():
            return None, "unknown"
        idx = np.where(both)[0] + lo
        k = idx[np.argmin(np.abs(idx - i_hit))]
        stance = float(np.linalg.norm(run.court[k, 0] - run.court[k, 1]))
        jit = self.jitter.get(pid, 0.0)
        jit = jit if np.isfinite(jit) else 0.0
        if stance >= LUNGE_STANCE_M + 2 * jit:
            # the leading foot (the one nearer the net or the side the player moved to) took a long last step
            last = self.steps_between(pid, s.frame - 2 * w - 1, s.frame + w)
            if any(st.length_m >= LUNGE_LAST_STEP_M for st in last):
                return round(stance, 2), "lunge"
        return round(stance, 2), "planted"

    def _lunge_sole(self, pid: str, run: Run, s: ShotEvent, out: ShotFootwork) -> None:
        """Front foot of a lunge: how it landed and where its toe points relative to the lunge step."""
        w = int(round(NEAR_HIT_S * self.fps))
        last = [st for st in self.steps_between(pid, s.frame - 2 * w - 1, s.frame + w) if st.length_m >= LUNGE_LAST_STEP_M]
        if not last:
            return
        st = max(last, key=lambda x: x.length_m)
        foot = 0 if st.foot == "left" else 1
        plant = next((p for p in self._plants_of.get((id(run), foot), []) if int(run.frames[p.a]) == st.end_frame), None)
        if plant is None or plant.sole is None:
            return
        out.lunge_landing = plant.sole.landing
        # toe direction is judged for forward and sideways lunges only: stepping back into a wide stance the
        # front foot is not expected to follow the step
        fr = np.asarray(own_half(*st.from_court))
        to = np.asarray(own_half(*st.to_court))
        d = to - fr
        if d[1] < -LUNGE_BACKWARD * float(np.linalg.norm(d)):
            return
        off = angle_between(plant.sole.direction, np.asarray(st.to_court) - np.asarray(st.from_court))
        out.lunge_toe_off_deg = None if not np.isfinite(off) or off > LUNGE_TOE_FLIP_DEG else round(off, 0)

    def _pattern(self, pid: str, run: Run, steps: list[FootStep], start: np.ndarray, end: np.ndarray,
                 s: ShotEvent) -> list[str]:
        d = end - start
        dist = float(np.linalg.norm(d))
        if dist < 0.6 or len(steps) < 2:
            return []
        u = d / dist
        lat, dep = own_half(*end)
        lat0, dep0 = own_half(*start)
        sideways = abs(lat - lat0) > abs(dep - dep0)
        backward = dep < dep0 - 0.3
        kinds: list[tuple[str, FootStep]] = []
        for st in steps:
            foot = 0 if st.foot == "left" else 1
            other = self._plant_before(pid, run, 1 - foot, st.end_frame)
            if other is None:
                continue
            a = (np.asarray(st.from_court) - other) @ u
            b = (np.asarray(st.to_court) - other) @ u
            if a < -0.1 and b > 0.1:  # the trailing foot passed the other one
                kinds.append(("cross_step" if sideways or backward else "run", st))
            elif a < -0.1:  # the trailing foot closed up without passing
                kinds.append(("chasse", st))
        # group consecutive steps of the same kind into one action
        groups: list[list] = []
        for k, st in kinds:
            if groups and groups[-1][0] == k:
                groups[-1][1].append(st)
            else:
                groups.append([k, [st]])
        out: list[str] = []
        for k, sts in groups:
            f0, f1 = sts[0].start_frame, sts[-1].end_frame
            i0, i1 = run.nearest(f0, self.slack), run.nearest(f1, self.slack)
            if i0 is None or i1 is None:
                continue
            seg = run.center[min(i0, i1):max(i0, i1) + 1]
            moved = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))) if len(seg) > 1 else 0.0
            if moved < 0.3:
                continue
            out.append(k)
            label = {"chasse": "并步", "cross_step": "交叉步", "run": "跑动"}[k]
            self._add_event(pid, k, f1, f0, f1, run.center[i1], round(moved, 2),
                            f"{label} {len(sts)} 步，移动 {moved:.1f} 米", self._conf(pid, 0.6), s.shot_id)
        return out

    def _plant_before(self, pid: str, run: Run, foot: int, frame: int) -> np.ndarray | None:
        plants = [p for p in self._plants_of.get((id(run), foot), [])]
        starts = [int(run.frames[p.a]) for p in plants]
        k = bisect_left(starts, frame + 1) - 1
        return plants[k].court if k >= 0 else None

    # ------------------------------------------------------------------ outputs

    base = BASE
    base_radius = BASE_RADIUS_M

    def split_judgements(self) -> dict[str, tuple[str, str, int | None]]:
        """Opponent shot id -> (player, timing, landing offset ms) for every judged opponent hit."""
        return dict(self._split)

    def steps_of(self, pid: str) -> list[FootStep]:
        return self._steps_by_pid.get(pid, [])

    def speeds(self, pid: str, spans: list[tuple[int, int]]) -> np.ndarray:
        """Mid-feet speeds (m/s) of ``pid`` inside the given frame spans."""
        out = []
        for r in self.by_pid.get(pid, []):
            for a, b in spans:
                lo, hi = np.searchsorted(r.frames, a), np.searchsorted(r.frames, b, side="right")
                if hi > lo:
                    out.append(r.center_speed[lo:hi])
        return np.concatenate(out) if out else np.zeros(0)

    def shot(self, shot_id: str) -> ShotFootwork | None:
        return self._shots.get(shot_id)

    @property
    def shots(self) -> list[ShotFootwork]:
        return sorted(self._shots.values(), key=lambda x: x.t_ms)

    @property
    def footwork_events(self) -> list[FootworkEvent]:
        return self._events

    def quality(self, spans: list[tuple[int, int]]) -> list[PlayerFootworkQuality]:
        """Per player: share of rally frames with both feet tracked, and planted-foot noise."""
        rally_frames = sum(b - a + 1 for a, b in spans)
        out = []
        for pid in sorted(self.by_pid):
            n = 0
            for r in self.by_pid[pid]:
                for a, b in spans:
                    n += int(np.searchsorted(r.frames, b, side="right") - np.searchsorted(r.frames, a))
            tracked = 100.0 * n / rally_frames if rally_frames else None
            j = self.jitter.get(pid)
            ok = j is not None and np.isfinite(j)
            status = ("ok" if ok and j < 0.06 and (tracked is None or tracked >= 50) else
                      "needs_review" if ok and j < 0.2 else "undeterminable")
            out.append(PlayerFootworkQuality(player_id=pid, tracked_pct=None if tracked is None else round(tracked, 1),
                                             jitter_m=round(j, 3) if ok else None, status=status))
        return out

    def result(self) -> FootworkResult:
        caveats = []
        if not self.runs:
            caveats.append("缺少球员骨架或球场标定，无法分析脚步。")
        q = self.quality([(r.start_frame, r.end_frame) for r in self.events.rallies])
        for x in q:
            if x.status != "ok":
                if x.jitter_m is not None and x.jitter_m >= 0.06:
                    caveats.append(f"{x.player_id} 脚部定位误差较大（约 {x.jitter_m:.2f} 米），步伐细节只作参考。")
                elif x.tracked_pct is not None:
                    caveats.append(f"{x.player_id} 回合中只有 {x.tracked_pct:.0f}% 的帧能看清双脚，脚步数据不完整。")
                else:
                    caveats.append(f"{x.player_id} 脚部跟踪不足。")
        if self.runs and not any(r.heel is not None for r in self.runs):
            caveats.append("没有脚跟 / 脚尖关键点（脚部模型未运行），只用踝点分析，脚尖朝向和着地方式无法判断。")
        else:
            for pid in sorted(p for p, ok in self.sole_ok.items() if not ok):
                caveats.append(f"{pid} 的脚跟 / 脚尖在画面里定位不够准，不判断脚尖朝向和着地方式。")
        if self.events.match_format != "singles":
            caveats.append("双打暂只给出触地、步伐和起跳，不做启动步与回位分析。")
        caveats.append("启动步时间以识别出的对手击球时刻为准；击球时刻或击球者识别错误会影响结果。")
        return FootworkResult(
            video_id=self.events.video_id, fingerprint=self.fingerprint, base=[BASE[0], BASE[1]],
            base_radius_m=BASE_RADIUS_M, contacts=self._contacts, steps=self._steps, events=self._events,
            shots=self.shots, quality=q, caveats=caveats)


PATTERN_ZH = {"split_step": "启动步", "lunge": "弓步", "jump": "起跳", "chasse": "并步", "cross_step": "交叉步",
              "run": "跑动", "recovery": "回位"}


LANDING_ZH = {"heel": "脚跟先着地", "forefoot": "前脚掌先着地", "flat": "全脚掌着地"}


def _lunge_form(f: ShotFootwork) -> str:
    parts = []
    if f.lunge_landing is not None:
        parts.append(LANDING_ZH[f.lunge_landing])
    if f.lunge_toe_off_deg is not None:
        parts.append(f"脚尖偏离跨步方向 {f.lunge_toe_off_deg:.0f}°" if f.lunge_toe_off_deg > LUNGE_TOE_OFF_DEG
                     else "脚尖对准跨步方向")
    return "，" + "，".join(parts) if parts else ""


def _rel(ms: int) -> str:
    return f"对手击球后 {ms} ms " if ms >= 0 else f"对手击球前 {-ms} ms "


def summarize(f: ShotFootwork) -> str:
    """One Chinese sentence describing the footwork into and out of a shot."""
    parts = []
    if f.split_step == "on_time":
        parts.append(f"启动步及时（{_rel(f.split_offset_ms)}落地）")
    elif f.split_step == "early":
        parts.append(f"启动步偏早（比对手击球早 {-f.split_offset_ms} ms 落地）")
    elif f.split_step == "late":
        parts.append(f"启动步偏晚（对手击球后 {f.split_offset_ms} ms 才落地）")
    elif f.split_step == "none":
        parts.append("对手击球时没有看到启动步（双脚小跳）")
    if f.reaction_ms == 0:
        parts.append("对手击球时已经在朝来球方向移动")
    elif f.reaction_ms is not None:
        parts.append(f"对手击球后 {f.reaction_ms} ms 启动")
    if f.steps is not None and f.move_m is not None and f.move_m >= 0.3:
        how = "、".join(PATTERN_ZH[p] for p in f.pattern if p in ("chasse", "cross_step", "run"))
        steps = f"{f.steps} 步" if f.steps else ""
        parts.append(f"{steps}移动 {f.move_m:.1f} 米" + (f"（{how}）" if how else ""))
    if f.finish == "lunge":
        parts.append(f"弓步击球（两脚相距 {f.stance_m:.1f} 米{_lunge_form(f)}）")
    elif f.finish == "jump":
        parts.append("起跳击球")
    if f.recovery_gap_m is not None:
        if f.recovery_gap_m <= BASE_RADIUS_M:
            parts.append("对手出手前已回到中心")
        else:
            parts.append(f"对手出手时离中心还有 {f.recovery_gap_m:.1f} 米")
    return "，".join(parts)


def analyze_footwork(events: EventsResult, players: PlayerTracks | None, calibration: CourtCalibration | None,
                     fps: float = 30.0, feet: FootLandmarks | None = None) -> FootworkResult:
    return Footwork(events, players, calibration, fps, feet).result()


__all__ = ["Footwork", "analyze_footwork", "summarize", "BASE", "BASE_RADIUS_M", "PATTERN_ZH", "LANDING_ZH"]
