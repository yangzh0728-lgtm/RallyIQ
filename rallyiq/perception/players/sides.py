"""Offline side consistency: keep identities stable and count real changes of ends.

The frame-by-frame tracker re-identifies players from scratch after every camera
cut (broadcast close-ups, replays). When two players look alike it can hand the
two team labels out the wrong way round, and a naive "P1 is now on the other
half" check then reports a change of ends after nearly every cut. A player
stepping on the net line produces the same false alarm.

Real matches change ends only between games (and once at 11 in the deciding
game), so a change of ends must be

* sustained: the teams stay on their new halves for ``min_hold_s`` seconds,
* rare: at least ``min_game_s`` after the previous one, and
* either seen happening (the same tracks walk to the other end) or, after a
  re-identification, backed by clothing colour that matches the old labels.

Everything else that puts a team on the "wrong" half is a labelling error and
is fixed by swapping the team labels (P1<->P2, P3<->P4) for that stretch, so the
ids follow the people.

The work is done on *pieces*: an identity epoch (the frames between two full
re-identifications, where every raw track is new) is one piece, split further
where the same tracks visibly change halves for good. Labels inside a piece are
continuous, so each piece is either kept or swapped as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .appearance import similarity
from .models import NET_Y_M, PlayerFrameResult, PlayerState

SWAP_ID = {"P1": "P2", "P2": "P1", "P3": "P4", "P4": "P3"}
TEAM_OF = {"P1": "A", "P3": "A", "P2": "B", "P4": "B"}


@dataclass
class SideConfig:
    # Feet this close to the net say nothing about the half (net play, homography error).
    net_margin_m: float = 0.6
    # A new half must hold this long, with this share of the decided frames agreeing.
    min_hold_s: float = 15.0
    min_hold_agree: float = 0.8
    # ... and with decided frames covering at least this share of that window.
    min_hold_coverage: float = 0.15
    # Shortest plausible game: changes of ends closer than this are rejected.
    min_game_s: float = 180.0
    # Colour evidence needed to accept a change of ends after a re-identification.
    appearance_margin: float = 0.1
    # Weight of the newest piece when updating the reference colours.
    ref_update: float = 0.3


@dataclass
class _Piece:
    start: int  # index into the sorted frames
    end: int  # exclusive
    continuous: bool  # labels carry over from the previous piece by motion
    swap: bool = False
    hist: dict[str, np.ndarray] = field(default_factory=dict)  # team -> mean colour histogram


def _vote(p: PlayerState, margin: float) -> int:
    """+1 when this observation says team A is on the near half, -1 far, 0 unknown."""
    if p.interpolated or p.player_id not in TEAM_OF:
        return 0
    if p.foot_court is not None:
        y = float(p.foot_court[1])
        if not np.isfinite(y) or abs(y - NET_Y_M) < margin:
            return 0
        near = y < NET_Y_M
    elif p.side is not None:
        near = p.side == "near"
    else:
        return 0
    a = TEAM_OF[p.player_id] == "A"
    return 1 if near == a else -1


def _epochs(frames: list[PlayerFrameResult]) -> list[int]:
    """Start indices of identity epochs: a new one begins when every raw track in
    a frame is unseen in the current epoch (all players re-acquired)."""
    starts: list[int] = []
    current: set[int] = set()
    for i, f in enumerate(frames):
        tracks = {p.track_id for p in f.players if not p.interpolated and p.player_id in TEAM_OF}
        if not tracks:
            continue
        if not starts or not (tracks & current):
            starts.append(i)
            current = set()
        current |= tracks
    return starts or [0]


class _Votes:
    """Prefix sums of per-frame votes for O(log n) window queries over time."""

    def __init__(self, frames: list[PlayerFrameResult], cfg: SideConfig):
        self.t = np.array([f.t_ms / 1000.0 for f in frames], dtype=np.float64)
        v = np.array([np.sign(sum(_vote(p, cfg.net_margin_m) for p in f.players)) for f in frames], dtype=np.int64)
        self.v = v
        self.pos = np.concatenate([[0], np.cumsum(v > 0)])
        self.neg = np.concatenate([[0], np.cumsum(v < 0)])
        self.cfg = cfg
        # Typical frame spacing, to turn a time window into an expected frame count.
        dt = np.diff(self.t)
        self.dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1 / 30

    def counts(self, i: int, j: int) -> tuple[int, int]:
        return int(self.pos[j] - self.pos[i]), int(self.neg[j] - self.neg[i])

    def sign(self, i: int, j: int) -> int:
        p, n = self.counts(i, j)
        return 0 if p == n else (1 if p > n else -1)

    def sustained(self, i: int, sign: int) -> bool:
        """Does ``sign`` hold from frame ``i`` for ``min_hold_s`` seconds?"""
        cfg = self.cfg
        t_end = self.t[i] + cfg.min_hold_s
        if self.t[-1] < t_end - 1e-6:
            return False  # not enough video left to tell
        j = int(np.searchsorted(self.t, t_end, side="right"))
        p, n = self.counts(i, j)
        agree, decided = (p if sign > 0 else n), p + n
        expected = max(cfg.min_hold_s / self.dt, 1.0)
        return decided >= cfg.min_hold_coverage * expected and agree >= cfg.min_hold_agree * decided


def _pieces(frames: list[PlayerFrameResult], votes: _Votes) -> list[_Piece]:
    starts = _epochs(frames)
    bounds = starts[1:] + [len(frames)]
    out: list[_Piece] = []
    for s, e in zip(starts, bounds):
        cur = s
        first = True
        # The epoch starts on the half of its first decided frame; a flicker
        # there is harmless because only sustained flips split the epoch.
        nz = np.flatnonzero(votes.v[s:e])
        sign = int(votes.v[s + nz[0]]) if len(nz) else 0
        for i in range(s, e):
            vi = int(votes.v[i])
            if sign != 0 and vi == -sign and votes.sustained(i, vi):
                if i > cur:
                    out.append(_Piece(cur, i, continuous=not first))
                    first = False
                cur, sign = i, vi
        out.append(_Piece(cur, e, continuous=not first))
    return out


def _team_hist(frames: list[PlayerFrameResult], piece: _Piece) -> dict[str, np.ndarray]:
    acc: dict[str, list[np.ndarray]] = {}
    for f in frames[piece.start:piece.end]:
        for p in f.players:
            if not p.interpolated and p.appearance is not None and p.player_id in TEAM_OF:
                acc.setdefault(TEAM_OF[p.player_id], []).append(p.appearance)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def _swapped(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {("B" if k == "A" else "A"): v for k, v in h.items()}


def _appearance_score(h: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> float | None:
    """> 0 when the piece's labels match the reference colours, < 0 when swapped."""
    if "A" not in ref or "B" not in ref:
        return None
    scores = []
    for team, other in (("A", "B"), ("B", "A")):
        if team in h:
            keep, swap = similarity(h[team], ref[team]), similarity(h[team], ref[other])
            if keep is not None and swap is not None:
                scores.append(keep - swap)
    return float(np.mean(scores)) if scores else None


def _swap_labels(f: PlayerFrameResult) -> None:
    for p in f.players:
        if p.player_id in SWAP_ID:
            p.player_id = SWAP_ID[p.player_id]
            p.team = TEAM_OF[p.player_id]


def stabilise_sides(
    frames: list[PlayerFrameResult], cfg: SideConfig | None = None, apply: bool = True
) -> list[int]:
    """Fix team labels in place (when ``apply``) and return the frames where the
    teams really changed ends. ``frames`` must be sorted by frame index."""
    cfg = cfg or SideConfig()
    frames = sorted(frames, key=lambda f: f.frame)
    if not frames:
        return []
    votes = _Votes(frames, cfg)
    pieces = _pieces(frames, votes)

    state = 0  # +1 team A near, -1 team A far, 0 not known yet
    last_switch_t: float | None = None
    ref: dict[str, np.ndarray] = {}
    switches: list[int] = []
    prev_swap = False
    for pc in pieces:
        base = prev_swap if pc.continuous else False
        raw = votes.sign(pc.start, pc.end)
        s = -raw if base else raw
        hist = _team_hist(frames, pc)
        if base:
            hist = _swapped(hist)
        swap = base
        if state == 0:
            state = s
        elif s != 0 and s != state:
            t0 = float(votes.t[pc.start])
            app = _appearance_score(hist, ref)
            # Continuous pieces start at a sustained flip by construction; a
            # re-identified piece must show it too.
            held = pc.continuous or votes.sustained(pc.start, raw)
            spaced = last_switch_t is None or t0 - last_switch_t >= cfg.min_game_s
            if pc.continuous:
                # The same tracks walked over: motion is the evidence; colour may veto.
                genuine = held and spaced and (app is None or app > -cfg.appearance_margin)
            else:
                genuine = held and spaced and app is not None and app > cfg.appearance_margin
            if genuine:
                state = s
                last_switch_t = t0
                switches.append(frames[pc.start].frame)
            else:
                swap = not base
                hist = _swapped(hist)
        pc.swap = swap
        prev_swap = swap
        if s != 0 and hist:
            for team, h in hist.items():
                ref[team] = h if team not in ref else (1 - cfg.ref_update) * ref[team] + cfg.ref_update * h
    if apply:
        for pc in pieces:
            if pc.swap:
                for f in frames[pc.start:pc.end]:
                    _swap_labels(f)
        for f in frames:
            f.players.sort(key=lambda p: p.player_id or "")
    return switches
