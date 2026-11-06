"""Hit-candidate detection inside a rally.

A hit shows up as a sharp change of the shuttle's direction or speed close to
a player's racket. Every candidate gets:

- ``turn``: how sharply the direction changes (0..1);
- ``speed``: how much the speed changes (0..1);
- ``sharpness``: whether the change is instantaneous (hit) rather than a
  smooth curve such as the top of a lob (0..1);
- ``proximity``: how close the shuttle is to the best player's wrist /
  extrapolated racket head, in player heights (0..1);
- ``track_quality``: how much of the trajectory around it is real, confident
  detection rather than gaps or interpolation (0..1).

The candidate's confidence is a weighted mix; anything below
``ok_threshold`` or carrying a warning reason is marked ``needs_review``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .config import EventsConfig
from .learned import features, get_scorer
from .players import PoseIndex, Proximity
from .rallies import RallySpan
from .trajectory import Track, angle_deg, norm
from .types import RallyEndReason

# Reason codes (stable strings; the review UI can translate them).
R_NO_PLAYER_NEAR = "no_player_near"
R_NO_POSE = "no_pose_data"
R_HITTER_AMBIGUOUS = "hitter_ambiguous"
R_WRIST_NOT_VISIBLE = "wrist_not_visible"
R_OCCLUDED = "shuttle_occluded_at_hit"
R_INTERPOLATED = "interpolated_shuttle_at_hit"
R_LOW_TRACK_QUALITY = "low_track_quality"
R_WEAK_CHANGE = "weak_direction_change"
R_SAME_HITTER = "same_hitter_consecutive"  # singles: same player twice; doubles: same team twice
R_TEAMMATE_AMBIGUOUS = "teammate_ambiguous"  # doubles: both partners equally close to the shuttle
R_HITTER_FROM_ALTERNATION = "hitter_from_alternation"  # nearest player broke alternation; next-nearest used
R_INFERRED_SERVE = "inferred_from_rally_start"

WEIGHTS = {"change": 0.35, "proximity": 0.35, "sharpness": 0.15, "track_quality": 0.15}


@dataclass
class Candidate:
    idx: int  # index into Track.points
    turn_deg: float
    change: float  # max(turn, speed) score
    sharpness: float
    track_quality: float
    impulse: float = 0.0
    prox: list[Proximity] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def best(self) -> Optional[Proximity]:
        return self.prox[0] if self.prox else None


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _change_at(track: Track, i: int, cfg: EventsConfig, window_ms: float, min_speed: float):
    """(turn angle, speed-up ratio, |v_out - v_in|) around point i, or None.

    Only a speed *increase* counts as a speed change: drag slows the shuttle
    smoothly all the time, while a racket either speeds it up or (for a block)
    turns it around, which the angle already captures.
    """
    vin = track.incoming(i, window_ms, cfg.max_window_ms)
    vout = track.outgoing(i, window_ms, cfg.max_window_ms)
    if vin is None or vout is None:
        return None
    sin, sout = norm(vin), norm(vout)
    # Direction is meaningless while the shuttle is (nearly) still: pixel noise
    # alone produces any angle. Starting from rest is a pure speed change.
    ang = angle_deg(vin, vout) if min(sin, sout) >= min_speed else None
    ratio = (sout + 1e-6) / (sin + 1e-6)
    impulse = norm((vout[0] - vin[0], vout[1] - vin[1]))
    return (ang or 0.0), ratio, impulse


def _score_point(
    track: Track, i: int, cfg: EventsConfig, fps_ms: float, poses: PoseIndex
) -> Optional[Candidate]:
    pt = track.points[i]
    scale_px = poses.scale_at(pt.frame, pt.xy)
    min_speed = 2 * cfg.static_speed_rel * scale_px
    full = _change_at(track, i, cfg, cfg.window_ms, min_speed)
    if full is None:
        return None
    ang, ratio, _ = full
    turn = _clip((ang - cfg.min_turn_deg) / (cfg.full_turn_deg - cfg.min_turn_deg))
    speed = _clip(math.log(ratio) / math.log(cfg.speed_jump_ratio * 2)) if ratio >= cfg.speed_jump_ratio else 0.0
    if turn <= 0 and speed <= 0:
        return None
    # Sharpness: an impact changes direction at once, so half the window shows
    # most of the turn; a smooth curve shows only about half of it.
    half = _change_at(track, i, cfg, cfg.window_ms / 2, min_speed)
    if half is None:
        return None
    if ang < 1e-6:
        sharp = 1.0 if speed > 0 else 0.5
    else:
        sharp = _clip((half[0] / ang - 0.4) / 0.5)
    q = track.quality(i, cfg.window_ms, fps_ms)
    # Impulse |v_out - v_in| (in player heights / s) localises the contact
    # frame: it peaks where the racket changes the shuttle's velocity.
    return Candidate(i, ang, max(turn, speed), sharp, q, impulse=full[2] / scale_px)


def detect_hits(
    track: Track,
    span: RallySpan,
    poses: PoseIndex,
    cfg: EventsConfig,
    fps_ms: float,
    collect: Optional[list] = None,
) -> list[Candidate]:
    """Hit candidates of one rally.

    ``collect``: if given, (features, candidate) of every candidate after non-maximum suppression is
    appended to it, before any filtering (training data for ``learned.py``).
    """
    pts = track.points
    raw: list[Candidate] = []
    for i in range(span.start, span.end + 1):
        c = _score_point(track, i, cfg, fps_ms, poses)
        if c is None:
            continue
        c.prox = poses.proximities(pts[i].frame, pts[i].xy)
        raw.append(c)

    # Non-maximum suppression: keep the most hit-like point, drop its
    # neighbours within min_hit_gap_ms, repeat.
    def merit(c: Candidate) -> float:
        p = c.best.score if c.best else 0.0
        return c.impulse * (0.5 + p)

    picked: list[Candidate] = []
    for c in sorted(raw, key=merit, reverse=True):
        t = pts[c.idx].t_ms
        if all(abs(t - pts[k.idx].t_ms) >= cfg.min_hit_gap_ms for k in picked):
            picked.append(c)
    picked.sort(key=lambda c: c.idx)

    scorer = get_scorer(cfg.hit_scorer)
    no_pose = poses.empty
    out: list[Candidate] = []
    for c in picked:
        rule_conf, landing, rejected = _decide(c, track, span, cfg, no_pose)
        if scorer is not None or collect is not None:
            x = features(c, track, span, poses, cfg, rule_conf, landing, no_pose)
            if collect is not None:
                collect.append((x, c))
            if scorer is not None:
                rejected = False
                c.confidence = round(float(scorer.predict([x])[0]), 3)
        if not rejected and c.confidence >= cfg.min_candidate_conf:
            out.append(c)

    serve = None
    if collect is not None or not out or pts[out[0].idx].t_ms - pts[span.start].t_ms > cfg.min_hit_gap_ms:
        serve = _inferred_serve(track, span, poses, cfg, fps_ms)
    if serve is not None:
        if scorer is not None or collect is not None:
            x = features(serve, track, span, poses, cfg, serve.confidence, False, no_pose)
            if collect is not None:
                collect.append((x, serve))
            if scorer is not None:
                serve.confidence = round(float(scorer.predict([x])[0]), 3)
        needed = not out or pts[out[0].idx].t_ms - pts[span.start].t_ms > cfg.min_hit_gap_ms
        if needed and (scorer is None or serve.confidence >= cfg.min_candidate_conf):
            out.insert(0, serve)

    out = _alternating_subset(out, cfg)
    _reattribute(out, cfg)
    _check_alternation(out)
    return out


def _decide(c: Candidate, track: Track, span: RallySpan, cfg: EventsConfig, no_pose: bool) -> tuple[float, bool, bool]:
    """Set reasons and the rule confidence of a candidate.

    Returns (rule confidence before rejection, is a landing, rejected by the rules). A rejected
    candidate gets confidence 0; the learned scorer sees the same facts as features instead.
    """
    pts = track.points
    best = c.best
    p = best.score if best else 0.0
    rejected = False

    if no_pose or not c.prox:
        # No pose around this frame: only the trajectory speaks. Keep strong,
        # sharp changes for review; the hitter stays unknown.
        if c.change * c.sharpness < 0.5:
            rejected = True
        c.reasons.append(R_NO_POSE)
    elif p <= 0.0:
        # Players are visible but none is near: floor landing, net cord, lob
        # apex or a false detection. Not a hit.
        rejected = True

    # A change right where the rally stops on the floor is the landing.
    is_last = pts[span.end].t_ms - pts[c.idx].t_ms <= cfg.landing_window_ms
    landing = is_last and span.end_reason == RallyEndReason.SHUTTLE_STATIC
    if landing and p < 0.7:
        rejected = True

    if best is not None and not best.used_wrist:
        c.reasons.append(R_WRIST_NOT_VISIBLE)
    if len(c.prox) > 1 and c.prox[1].score > 0 and best.score - c.prox[1].score < cfg.hitter_margin:
        same_team = best.pose.team is not None and best.pose.team == c.prox[1].pose.team
        c.reasons.append(R_TEAMMATE_AMBIGUOUS if same_team else R_HITTER_AMBIGUOUS)
    if track.gap_around(c.idx) > cfg.occlusion_gap_ms:
        c.reasons.append(R_OCCLUDED)
    if pts[c.idx].interpolated:
        c.reasons.append(R_INTERPOLATED)
    if c.track_quality < cfg.low_track_quality:
        c.reasons.append(R_LOW_TRACK_QUALITY)
    if c.change < 0.3:
        c.reasons.append(R_WEAK_CHANGE)

    w = WEIGHTS
    conf = round(
        w["change"] * c.change
        + w["proximity"] * p
        + w["sharpness"] * c.sharpness
        + w["track_quality"] * c.track_quality,
        3,
    )
    c.confidence = 0.0 if rejected else conf
    return conf, landing, rejected


def _inferred_serve(
    track: Track, span: RallySpan, poses: PoseIndex, cfg: EventsConfig, fps_ms: float
) -> Optional[Candidate]:
    """The serve is often missed (shuttle unseen before it is struck). If the
    rally's first point is close to a player, propose a hit there."""
    i = span.start
    pt = track.points[i]
    prox = poses.proximities(pt.frame, pt.xy)
    if not prox or prox[0].score <= 0.3:
        return None
    q = track.quality(i, cfg.window_ms, fps_ms, before=False)
    c = Candidate(i, 0.0, 0.0, 0.0, q, prox=prox, reasons=[R_INFERRED_SERVE])
    # Proximity is the only evidence, so this always goes to review.
    c.confidence = round(0.5 * prox[0].score + 0.1 * q, 3)
    return c


def _hitter(c: Candidate) -> Optional[str]:
    return c.best.player_id if c.best and c.best.score > 0 else None


def _side_key(c: Candidate) -> Optional[str]:
    """What must alternate between consecutive hits: player (singles) or team (doubles)."""
    return c.best.pose.team_key if c.best and c.best.score > 0 else None


def _alternating_subset(cands: list[Candidate], cfg: EventsConfig) -> list[Candidate]:
    """Pick the subset of candidates that best explains the rally.

    The hitting side must alternate: the player in singles, the team in doubles. Dynamic programming chooses the
    candidates maximising sum(confidence - keep_threshold) under that rule,
    so a weak hit survives when it is needed between two strong hits by the
    other player, and a weak spurious hit is dropped when it would break the
    alternation. Strong candidates that do not fit are kept but flagged.
    """
    n = len(cands)
    if n < 2:
        return cands
    w = [c.confidence - cfg.keep_threshold for c in cands]
    best = [0.0] * n
    prev: list[Optional[int]] = [None] * n
    for i in range(n):
        hi = _side_key(cands[i])
        best[i], prev[i] = w[i], None
        for j in range(i):
            hj = _side_key(cands[j])
            if hi is not None and hi == hj:
                continue
            if best[j] + w[i] > best[i]:
                best[i], prev[i] = best[j] + w[i], j
    end = max(range(n), key=lambda i: best[i])
    chosen = set()
    k: Optional[int] = end
    while k is not None:
        chosen.add(k)
        k = prev[k]
    return [
        c
        for i, c in enumerate(cands)
        if i in chosen or c.confidence >= cfg.ok_threshold
    ]


def _reattribute(hits: list[Candidate], cfg: EventsConfig) -> None:
    """Fix a hitter that breaks alternation when another player of the right side is nearly as close.

    Partners and opponents often overlap on screen near the net or in a
    doubles front/back rotation. When the nearest player belongs to the side
    that just hit, and a player of the other side is within ``2 * hitter_margin``
    of the same proximity, the other side's player is the more plausible hitter.
    """
    for a, b in zip(hits, hits[1:]):
        ka, kb = _side_key(a), _side_key(b)
        if ka is None or kb is None or ka != kb:
            continue
        best = b.best
        for k, alt in enumerate(b.prox[1:], start=1):
            if alt.score > 0 and alt.pose.team_key != ka and best.score - alt.score < 2 * cfg.hitter_margin:
                b.prox.insert(0, b.prox.pop(k))
                b.reasons = [r for r in b.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)]
                b.reasons.append(R_HITTER_FROM_ALTERNATION)
                break


def _check_alternation(hits: list[Candidate]) -> None:
    """Consecutive hits by the same side (player in singles, team in doubles) mean a missed or spurious hit."""
    for a, b in zip(hits, hits[1:]):
        ka, kb = _side_key(a), _side_key(b)
        if ka is not None and ka == kb:
            for c in (a, b):
                if R_SAME_HITTER not in c.reasons:
                    c.reasons.append(R_SAME_HITTER)
