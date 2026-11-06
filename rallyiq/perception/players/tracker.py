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

``player_id`` stays with the person across changes of ends; ``side`` is
recomputed every frame from the foot position. ``track_id`` changes whenever a
slot is re-acquired after being lost, so downstream code can tell a continuous
track from a re-identification.

All joint assignments are scored at once with numpy (at most 7^4 options in
doubles), so the priors that couple slots stay exact.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .appearance import similarity
from .geometry import court_side
from .models import Candidate, PlayerState, Side

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
    # Candidates considered per frame in doubles (keeps the joint search small).
    max_doubles_candidates: int = 6


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

    def missed(self, frame: int) -> int:
        return frame - self.last_frame

    def predict(self, frame: int) -> np.ndarray | None:
        if self.pos is None:
            return None
        dt = min(self.missed(frame), 10)  # don't extrapolate far
        return self.pos + self.vel * dt


def _cand_pos(c: Candidate, calibrated: bool) -> np.ndarray:
    return c.foot_court.astype(np.float64) if calibrated else c.foot_px.astype(np.float64)


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
            cands = [c for c in cands if c.foot_court is not None]
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
                )
            )
        return out

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
