"""Two-player (singles) identity tracker.

Each frame the scored candidates are assigned to two persistent slots, ``P1``
and ``P2``. Assignment combines

* motion continuity (court metres when calibrated, else pixels scaled by body
  height) with a speed gate,
* torso colour similarity,
* the candidate's player-likeness score, and
* a soft "one player per half" prior.

``player_id`` stays with the person across changes of ends; ``side`` is
recomputed every frame from the foot position. ``track_id`` changes whenever a
slot is re-acquired after being lost, so downstream code can tell a continuous
track from a re-identification.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .appearance import similarity
from .geometry import court_side
from .models import Candidate, PlayerState, Side

SLOT_IDS = ("P1", "P2")


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


class SinglesTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.slots = [_Slot(pid) for pid in SLOT_IDS]
        self._next_track_id = 0

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

        best: tuple[float, tuple[int | None, int | None]] = (float("inf"), (None, None))
        options = list(range(len(cands))) + [None]
        for a, b in itertools.product(options, options):
            if a is not None and a == b:
                continue
            cost = 0.0
            for slot, ci in zip(self.slots, (a, b)):
                if ci is None:
                    cost += cfg.empty_slot_cost
                    continue
                pc = self._pair_cost(slot, cands[ci], frame, calibrated)
                cost += pc
            if a is not None and b is not None and sides[a] is not None and sides[a] == sides[b]:
                cost += cfg.same_half_penalty
            if never_seen and a is not None and b is not None:
                # First sighting: P1 starts on the near half by convention.
                if sides[a] == "far" and sides[b] == "near":
                    cost += 0.5
            if cost < best[0]:
                best = (cost, (a, b))

        # When both players were lost (camera cut, change of ends) the motion
        # term is absent, so appearance in _pair_cost decides who is who.
        assignment = best[1]

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
