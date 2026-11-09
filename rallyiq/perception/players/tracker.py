"""Player identity tracker for singles (2 slots) and doubles (4 slots).

Each frame the scored candidates are assigned to persistent slots: ``P1`` /
``P2`` in singles, ``P1``..``P4`` in doubles where team A = P1 + P3 and team
B = P2 + P4 (team A is the pair first seen on the near half). Assignment
combines

* motion continuity (court metres when calibrated, else pixels scaled by body
  height) with a speed gate,
* torso colour similarity,
* the candidate's player-likeness score, and
* half-court priors: opponents on opposite halves, and in doubles partners on
  the same half (in doubles only for slots being (re-)acquired).

Two guards keep a slot from being wasted on the wrong person (calibrated only):

* a slot that stands still outside the court lines for ``bystander_s`` seconds
  holds a bystander (umpire, spectator by the post): it is released and that
  spot is ignored from then on, so a real player can take the slot;
* in doubles, when one team has a free slot and an opponent has shared its
  tracked player's half for ``team_repair_s`` seconds, the opponent was given
  the wrong team: it moves into the free slot (``relabels`` records the
  track, so the frames already emitted can be renamed afterwards).

``player_id`` stays with the person across changes of ends; ``side`` is
recomputed every frame from the foot position. ``track_id`` changes whenever a
slot is re-acquired after being lost, so downstream code can tell a continuous
track from a re-identification.

All joint assignments are scored at once with numpy (at most 9^4 options in
doubles), so the priors that couple slots stay exact.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .appearance import similarity
from .geometry import PlayArea, court_side
from .models import COURT_WIDTH_M, NET_Y_M, Candidate, PlayerState, Side

SLOT_IDS = ("P1", "P2")
DOUBLES_SLOT_IDS = ("P1", "P2", "P3", "P4")
TEAM_OF = {"P1": "A", "P3": "A", "P2": "B", "P4": "B"}


@dataclass
class TrackerConfig:
    fps: float = 30.0
    max_speed_mps: float = 7.0  # elite lateral bursts peak ~6 m/s
    gate_base_m: float = 0.8  # slack for foot-point jitter (lunges, blur)
    gate_px_rel: float = 0.6  # uncalibrated gate, in body heights per frame-gap step
    lost_after_frames: int = 45  # after this many misses a slot is re-acquired freely
    w_motion: float = 1.0
    w_appearance: float = 0.8
    w_score: float = 0.6
    same_half_penalty: float = 0.8
    appearance_momentum: float = 0.9
    min_candidate_score: float = 0.15
    # Leaving a slot empty must cost more than a re-acquisition, so real
    # players get picked up, but less than a far-fetched match.
    empty_slot_cost: float = 0.8
    reacquire_cost: float = 0.5
    # Doubles: partners standing on different halves (never happens in play).
    partner_split_penalty: float = 0.8
    first_sighting_penalty: float = 0.5
    # Re-acquired slots prefer the half they were last seen on: camera cuts
    # happen far more often than changes of ends. Distinct clothing colours
    # (appearance term, up to w_appearance per slot) still override it.
    side_memory_penalty: float = 0.5
    # Candidates considered per frame in doubles (keeps the joint search small).
    max_doubles_candidates: int = 8
    # Bystanders: still (within static_extent_m) and at least offcourt_m outside
    # the court lines for this long. Players are never off court that long.
    bystander_s: float = 6.0
    offcourt_m: float = 0.3
    static_extent_m: float = 1.0
    # The stillness test runs on the position smoothed over this long: near the net a few pixels of box change
    # move the foot point ~0.5 m along the court, so the raw position of someone standing still keeps leaving
    # the static_extent_m box and the guard never fired on the real clip (2026-03-02 doubles).
    still_smooth_s: float = 0.5
    # A free slot (first sighting, re-acquisition) is not given to someone standing further than this outside a
    # sideline (calibrated only). Spectators, umpires and line judges stand there; players only lunge past it for
    # a moment. On the 2026-03-02 doubles clip the opening frames seeded slots with a spectator by the left post
    # and a man walking outside the right sideline, so the near man ended up in team B and the far man never got
    # a slot. Behind the baselines is not restricted: far players often stand a metre behind theirs.
    acquire_max_outside_side_m: float = 0.4
    bystander_radius_m: float = 1.0
    # Doubles: opponents sharing a half (feet away from the net) for this long.
    team_repair_s: float = 3.0
    net_margin_m: float = 0.6


@dataclass
class _Slot:
    player_id: str
    track_id: int = -1
    pos: np.ndarray | None = None  # court metres or image px
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0  # body height px (uncalibrated mode)
    last_frame: int = -(10**9)
    appearance: np.ndarray | None = None
    side: Side | None = None
    # Start frame and position of the current still, off-court stretch (calibrated only).
    still_since: int | None = None
    still_at: np.ndarray | None = None
    still_ema: np.ndarray | None = None  # smoothed position for the stillness test

    def missed(self, frame: int) -> int:
        return frame - self.last_frame

    def predict(self, frame: int) -> np.ndarray | None:
        if self.pos is None:
            return None
        dt = min(self.missed(frame), 10)  # don't extrapolate far
        return self.pos + self.vel * dt


def _cand_pos(c: Candidate, calibrated: bool) -> np.ndarray:
    return c.foot_court.astype(np.float64) if calibrated else c.foot_px.astype(np.float64)


def _outside_side(c: Candidate) -> float:
    """Metres the candidate's feet are outside the nearer sideline (0 inside, or without calibration)."""
    if c.foot_court is None:
        return 0.0
    x = float(c.foot_court[0])
    return max(-x, x - COURT_WIDTH_M, 0.0)


def _cand_side(c: Candidate, net_y_px: float | None) -> Side | None:
    if c.foot_court is not None:
        return court_side(c.foot_court)
    if net_y_px is not None:
        return "near" if float(c.foot_px[1]) > net_y_px else "far"
    return None


@lru_cache(maxsize=64)
def _assignments(n_slots: int, n_cands: int) -> np.ndarray:
    """Every slot -> candidate assignment as rows; value ``n_cands`` = slot left empty."""
    rows = [
        a
        for a in itertools.product(range(n_cands + 1), repeat=n_slots)
        if len({x for x in a if x < n_cands}) == sum(1 for x in a if x < n_cands)
    ]
    return np.asarray(rows, dtype=np.int64).reshape(-1, n_slots)


class PlayerTracker:
    """Assigns candidates to ``len(slot_ids)`` persistent player slots."""

    def __init__(self, cfg: TrackerConfig | None = None, slot_ids: tuple[str, ...] = SLOT_IDS):
        self.cfg = cfg or TrackerConfig()
        self.slots = [_Slot(pid) for pid in slot_ids]
        self._next_track_id = 0
        teams = [TEAM_OF[pid] for pid in slot_ids]
        self._team_a = np.array([t == "A" for t in teams])
        # Slot pairs on opposite teams / the same team.
        n = len(slot_ids)
        self._opp_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if teams[i] != teams[j]]
        self._mate_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if teams[i] == teams[j]]
        self._team_size = max(n // 2, 1)
        self.bystander_spots: list[np.ndarray] = []
        self.relabels: dict[int, str] = {}  # track_id -> player_id it was moved to
        self._clash: dict[tuple[int, int], int] = {}  # (slot, opponent slot) -> first frame on one half

    @property
    def doubles(self) -> bool:
        return len(self.slots) == 4

    # -- cost -----------------------------------------------------------------
    def _motion_cost(self, slot: _Slot, c: Candidate, frame: int, calibrated: bool) -> float | None:
        """Normalised distance to prediction (<=1 inside gate), None if gated out
        or the slot is free for re-acquisition."""
        cfg = self.cfg
        if slot.pos is None or slot.missed(frame) > cfg.lost_after_frames:
            return None
        pred = slot.predict(frame)
        d = float(np.linalg.norm(_cand_pos(c, calibrated) - pred))
        n = max(slot.missed(frame), 1)
        if calibrated:
            # A tracked player is followed past a sideline step by step, but a slot that lost its player does
            # not jump to someone standing outside it (a spectator by the post took over the far man's slot
            # while he was hidden, on the 2026-03-02 doubles clip).
            if d > cfg.gate_base_m and _outside_side(c) > cfg.acquire_max_outside_side_m:
                return float("inf")
            gate = cfg.gate_base_m + cfg.max_speed_mps * n / cfg.fps
        else:
            gate = slot.scale * (0.25 + cfg.gate_px_rel * min(n, 10) / 3)
        if d > gate:
            return float("inf")
        return d / gate

    def _pair_cost(self, slot: _Slot, c: Candidate, frame: int, calibrated: bool) -> float:
        cfg = self.cfg
        m = self._motion_cost(slot, c, frame, calibrated)
        if m == float("inf"):
            return float("inf")
        if m is None and calibrated and _outside_side(c) > cfg.acquire_max_outside_side_m:
            return float("inf")
        cost = -cfg.w_score * c.player_score
        if m is not None:
            cost += cfg.w_motion * m
        else:
            cost += cfg.w_motion * cfg.reacquire_cost  # allowed but not free
        sim = similarity(slot.appearance, c.appearance)
        if sim is not None:
            cost += cfg.w_appearance * (1.0 - sim)
        return cost

    # -- main -----------------------------------------------------------------
    def update(
        self,
        frame: int,
        candidates: list[Candidate],
        calibrated: bool,
        net_y_px: float | None = None,
    ) -> list[PlayerState]:
        cfg = self.cfg
        cands = [c for c in candidates if c.player_score >= cfg.min_candidate_score]
        if calibrated:
            cands = [c for c in cands if c.foot_court is not None and not self._at_bystander_spot(c.foot_court)]
        sides = [_cand_side(c, net_y_px) for c in cands]

        never_seen = all(s.pos is None for s in self.slots)
        assignment = self._best_assignment(cands, sides, frame, calibrated, never_seen)

        out: list[PlayerState] = []
        for slot, ci in zip(self.slots, assignment):
            if ci is None:
                continue
            c = cands[ci]
            self._commit(slot, c, frame, calibrated, sides[ci])
            out.append(
                PlayerState(
                    player_id=slot.player_id,
                    track_id=slot.track_id,
                    team=TEAM_OF[slot.player_id],
                    side=sides[ci],
                    bbox=c.det.bbox,
                    keypoints=c.det.keypoints,
                    foot_px=c.foot_px,
                    foot_court=c.foot_court,
                    on_court=c.on_court,
                    confidence=float(np.clip(c.player_score, 0.0, 1.0)),
                    appearance=c.appearance,
                )
            )
        if calibrated:
            for slot in self.slots:
                if slot.last_frame == frame:
                    self._check_bystander(slot, frame)
            if self.doubles:
                self._repair_teams(frame)
        return out

    # -- guards ---------------------------------------------------------------
    _LINES = PlayArea(margin_side=0.0, margin_end=0.0)

    def _at_bystander_spot(self, xy: np.ndarray) -> bool:
        if not self.bystander_spots or self._LINES.distance_outside(xy) < self.cfg.offcourt_m:
            return False
        return any(float(np.linalg.norm(xy - s)) <= self.cfg.bystander_radius_m for s in self.bystander_spots)

    def _check_bystander(self, slot: _Slot, frame: int) -> None:
        cfg = self.cfg
        a = 1.0 / max(cfg.still_smooth_s * cfg.fps, 1.0)
        slot.still_ema = slot.pos.copy() if slot.still_ema is None else (1 - a) * slot.still_ema + a * slot.pos
        pos = slot.still_ema
        off = self._LINES.distance_outside(pos) >= cfg.offcourt_m
        if not off:
            slot.still_since, slot.still_at = None, None
            return
        if slot.still_at is None or np.max(np.abs(pos - slot.still_at)) > cfg.static_extent_m / 2:
            slot.still_since, slot.still_at = frame, pos.copy()
            return
        if frame - slot.still_since >= cfg.bystander_s * cfg.fps:
            self.bystander_spots.append(slot.still_at.copy())
            self._release(slot)

    def _release(self, slot: _Slot) -> None:
        fresh = _Slot(slot.player_id)
        fresh.last_frame = slot.last_frame
        self.slots[self.slots.index(slot)] = fresh

    def _repair_teams(self, frame: int) -> None:
        """Move an opponent into a team's free slot when it keeps sharing that team's half."""
        cfg = self.cfg
        lost = cfg.lost_after_frames

        def half(s: _Slot) -> Side | None:
            if s.pos is None or s.last_frame != frame or abs(float(s.pos[1]) - NET_Y_M) < cfg.net_margin_m:
                return None
            return s.side

        def tracked(s: _Slot) -> bool:
            return s.pos is not None and s.missed(frame) <= lost

        clash: dict[tuple[int, int], int] = {}
        for a, b in self._mate_pairs:
            for i, free in ((a, b), (b, a)):
                if not tracked(self.slots[i]) or tracked(self.slots[free]) or half(self.slots[i]) is None:
                    continue
                h = half(self.slots[i])
                for o in range(len(self.slots)):
                    if TEAM_OF[self.slots[o].player_id] == TEAM_OF[self.slots[i].player_id] or half(self.slots[o]) != h:
                        continue
                    mate = next(j for pair in self._mate_pairs if o in pair for j in pair if j != o)
                    if tracked(self.slots[mate]) and half(self.slots[mate]) == h:
                        continue  # the opponent's own partner is here too: i is the odd one out
                    clash[(i, o)] = self._clash.get((i, o), frame)
                    if frame - clash[(i, o)] >= cfg.team_repair_s * cfg.fps:
                        self._move(o, free)
                        self._clash = {}
                        return
        self._clash = clash

    def _move(self, src: int, dst: int) -> None:
        s = self.slots[src]
        moved = _Slot(self.slots[dst].player_id, s.track_id, s.pos, s.vel, s.scale, s.last_frame, s.appearance,
                      s.side, s.still_since, s.still_at, s.still_ema)
        self.slots[dst] = moved
        self.relabels[s.track_id] = moved.player_id
        self._release(s)

    def _best_assignment(
        self, cands: list[Candidate], sides: list[Side | None], frame: int, calibrated: bool, never_seen: bool
    ) -> list[int | None]:
        cfg = self.cfg
        n_slots = len(self.slots)
        if self.doubles and len(cands) > cfg.max_doubles_candidates:
            cands = cands[: cfg.max_doubles_candidates]  # sorted best first upstream
        m = len(cands)
        # cost[s, c]; the extra column is "leave the slot empty".
        cost = np.full((n_slots, m + 1), cfg.empty_slot_cost)
        for si, slot in enumerate(self.slots):
            for ci, c in enumerate(cands):
                cost[si, ci] = self._pair_cost(slot, c, frame, calibrated)
        A = _assignments(n_slots, m)
        total = cost[np.arange(n_slots), A].sum(axis=1)

        # Side codes per candidate (+ empty): 0 near, 1 far, -1 unknown/empty.
        code = np.array([{"near": 0, "far": 1}.get(s, -1) for s in sides[:m]] + [-1])
        S = code[A]
        known = S >= 0
        # In doubles the half-court priors only steer slots without motion
        # continuity (first sighting, re-acquisition). Tracked players cross
        # the net at a change of ends, and four people moving at once would
        # otherwise let a prior override clear motion evidence.
        fresh = [s.pos is None or s.missed(frame) > cfg.lost_after_frames for s in self.slots]

        def gated(i: int, j: int) -> bool:
            return not self.doubles or fresh[i] or fresh[j]

        for i, j in self._opp_pairs:
            if gated(i, j):
                same = known[:, i] & known[:, j] & (S[:, i] == S[:, j])
                total = total + same * (cfg.same_half_penalty / self._team_size)
        for i, j in self._mate_pairs:
            if gated(i, j):
                split = known[:, i] & known[:, j] & (S[:, i] != S[:, j])
                total = total + split * cfg.partner_split_penalty
        if not never_seen and cfg.side_memory_penalty:
            for i, slot in enumerate(self.slots):
                # In singles each half holds one player, so two slots remembering the same half means the
                # memory came from a bad seeding (spectators, a wrong court before calibration settled).
                # Following it would keep the other half's player untracked for the rest of the clip.
                contradicted = not self.doubles and any(
                    o.side == slot.side for j, o in enumerate(self.slots) if j != i and o.pos is not None)
                if fresh[i] and slot.side is not None and not contradicted:
                    want = 0 if slot.side == "near" else 1
                    total = total + (known[:, i] & (S[:, i] != want)) * cfg.side_memory_penalty
        if never_seen:
            # First sighting: team A (P1, P3) starts on the near half by convention.
            assigned = A < m
            has_a = (assigned & self._team_a).any(axis=1)
            has_b = (assigned & ~self._team_a).any(axis=1)
            wrong = known & np.where(self._team_a, S == 1, S == 0)
            # All of one pair on the wrong halves costs exactly the penalty.
            total = total + (has_a & has_b) * cfg.first_sighting_penalty * wrong.sum(axis=1) / (2 * self._team_size)
        best = A[int(np.argmin(total))]
        # When every slot was lost (camera cut, change of ends) the motion term is
        # absent, so appearance in _pair_cost decides who is who.
        return [int(ci) if ci < m else None for ci in best]

    def _commit(self, slot: _Slot, c: Candidate, frame: int, calibrated: bool, side: Side | None) -> None:
        cfg = self.cfg
        pos = _cand_pos(c, calibrated)
        if slot.pos is None or slot.missed(frame) > cfg.lost_after_frames:
            slot.track_id = self._next_track_id
            self._next_track_id += 1
            slot.vel = np.zeros(2)
            slot.still_since, slot.still_at, slot.still_ema = None, None, None
        else:
            n = max(slot.missed(frame), 1)
            v = (pos - slot.pos) / n
            slot.vel = 0.6 * slot.vel + 0.4 * v
        slot.pos = pos
        slot.scale = max(c.det.height, 1.0)
        slot.last_frame = frame
        slot.side = side
        if c.appearance is not None:
            if slot.appearance is None:
                slot.appearance = c.appearance.copy()
            else:
                m = cfg.appearance_momentum
                slot.appearance = m * slot.appearance + (1 - m) * c.appearance


class SinglesTracker(PlayerTracker):
    def __init__(self, cfg: TrackerConfig | None = None):
        super().__init__(cfg, SLOT_IDS)


class DoublesTracker(PlayerTracker):
    def __init__(self, cfg: TrackerConfig | None = None):
        super().__init__(cfg, DOUBLES_SLOT_IDS)
