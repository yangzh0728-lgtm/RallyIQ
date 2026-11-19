"""Singles: decide who hit each confident candidate jointly over the rally (two-state Viterbi).

Why: the hitter is the player nearest the shuttle in the image. On a camera behind one baseline the far
player is small and often has no pose at the contact frame, so the near player wins the proximity contest.
On DJI_0009 13 of 26 far-player hits at real hit times went to the near player; the alternation rules then
saw "the same player twice" and dropped real hits (``self_feed``) or flagged them.

In singles the hitting side alternates. This step treats the side (near / far) of every confident candidate
as hidden: each candidate votes by proximity (``gain * (far score - near score)``, a missing pose votes 0),
and consecutive candidates prefer different sides (``same_p`` for the same side, ``same_p_long`` after a long
gap, where a missed hit in between is plausible). A candidate whose decoded side differs from its nearest
player is handed to the best player of the decoded side (looked up a few frames around the contact when
that player has no pose there) and keeps ``hitter_from_sequence``. Weak candidates are left alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EventsConfig
    from .hits import Candidate
    from .players import PoseIndex, Proximity
    from .trajectory import Track

SIDES = ("near", "far")


def _side_score(c: Candidate, side: str) -> float:
    return max((p.score for p in c.prox if p.pose.side == side), default=0.0)


def _logit_far(c: Candidate, cfg: EventsConfig) -> float:
    """Proximity vote for "far". When one side has no pose at all near the contact, the other side's
    score is weaker evidence (the missing player may simply be undetected): ``side_decode_gain_missing``."""
    have = {p.pose.side for p in c.prox}
    gain = cfg.side_decode_gain if {"near", "far"} <= have else cfg.side_decode_gain_missing
    return gain * (_side_score(c, "far") - _side_score(c, "near"))


def is_singles(cands: Sequence[Candidate]) -> bool:
    keys: dict[str, set[str]] = {}
    for c in cands:
        for p in c.prox:
            keys.setdefault(p.pose.team_key, set()).add(p.player_id)
    return bool(keys) and all(len(v) == 1 for v in keys.values())


def viterbi_sides(logit_far: Sequence[float], gaps_ms: Sequence[float], same_p: float, same_p_long: float,
                  long_gap_ms: float) -> list[str]:
    """Most likely near/far sequence. ``gaps_ms[k]``: time from candidate k to k+1."""
    n = len(logit_far)
    if n == 0:
        return []

    def emit(k: int) -> tuple[float, float]:  # log p(near), log p(far) from the logit
        lf = logit_far[k]
        return -math.log1p(math.exp(lf)), -math.log1p(math.exp(-lf))

    score = list(emit(0))
    back: list[tuple[int, int]] = []
    for k in range(1, n):
        p = same_p_long if gaps_ms[k - 1] > long_gap_ms else same_p
        ls, ld = math.log(p), math.log(1.0 - p)
        e = emit(k)
        new, bp = [0.0, 0.0], [0, 0]
        for s in (0, 1):
            stay, switch = score[s] + ls, score[1 - s] + ld
            new[s], bp[s] = (stay, s) if stay >= switch else (switch, 1 - s)
            new[s] += e[s]
        score, back = new, back + [tuple(bp)]
    s = 0 if score[0] >= score[1] else 1
    path = [s]
    for bp in reversed(back):
        s = bp[s]
        path.append(s)
    return [SIDES[s] for s in reversed(path)]


def _player_of_side(c: Candidate, side: str, track: Track, poses: PoseIndex, search: int) -> Proximity | None:
    own = [p for p in c.prox if p.pose.side == side]
    if own:
        return own[0]
    pt = track.points[c.idx]
    for d in range(search + 1):
        for f in (pt.frame - d, pt.frame + d):
            pf = poses.by_frame.get(f)
            if pf is None or not any(p.side == side for p in pf.players):
                continue
            got = [p for p in poses.proximities(f, pt.xy, 0) if p.pose.side == side]
            if got:
                return got[0]
    return None


def decode_sides(cands: list[Candidate], track: Track, poses: PoseIndex, cfg: EventsConfig) -> int:
    """Re-assign hitters of confident candidates in place. Returns how many were handed over."""
    from .hits import R_HITTER_AMBIGUOUS, R_HITTER_FROM_SEQUENCE, R_TEAMMATE_AMBIGUOUS

    chain = [c for c in cands if c.confidence >= cfg.side_decode_min_conf and c.prox
             and c.best.pose.side in SIDES]
    if len(chain) < 3 or not is_singles(cands):
        return 0
    logits = [_logit_far(c, cfg) for c in chain]
    gaps = [b.t_ms - a.t_ms for a, b in zip(chain, chain[1:])]
    sides = viterbi_sides(logits, gaps, cfg.side_decode_same_p, cfg.side_decode_same_p_long,
                          cfg.side_decode_long_gap_ms)
    moved = 0
    for c, side in zip(chain, sides):
        if c.best.pose.side == side:
            continue
        alt = _player_of_side(c, side, track, poses, cfg.side_decode_search_frames)
        if alt is None:
            continue
        c.prox = [alt] + [p for p in c.prox if p is not alt]
        c.reasons = [r for r in c.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)]
        if R_HITTER_FROM_SEQUENCE not in c.reasons:
            c.reasons.append(R_HITTER_FROM_SEQUENCE)
        moved += 1
    return moved
