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
from .trajectory import Track, Vec, angle_deg, norm
from .types import KP_L_ANKLE, KP_L_HIP, KP_L_SHOULDER, KP_L_WRIST, KP_R_ANKLE, KP_R_HIP, KP_R_SHOULDER, KP_R_WRIST
from .types import PlayerPose, RallyEndReason

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
R_HITTER_FROM_TRAJECTORY = "hitter_from_trajectory"  # learned side model overruled the nearest player
R_HITTER_FROM_SEQUENCE = "hitter_from_sequence"  # no player near the shuttle: hitter taken from alternation
R_FLOOR_CONTACT = "floor_contact"  # shuttle at the hitter's feet: bounce, pickup or floor push (rejected)
R_OUT_OF_REACH = "out_of_reach"  # shuttle beyond any racket reach of the nearest wrist (rejected)

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
    t_ms: float = 0.0  # time of the track point (sequence rules need the gap between hits)
    # Players around the hit from a wider pose search, for a hit without a usable pose at the contact frame
    # (``cfg.hitter_pose_fallback_frames``). Only the hitter fill (``_fill_hitters``) uses it.
    prox_wide: list[Proximity] = field(default_factory=list)

    @property
    def best(self) -> Optional[Proximity]:
        return self.prox[0] if self.prox else None

    @property
    def hitter_known(self) -> bool:
        b = self.best
        return b is not None and (
            b.score > 0 or R_HITTER_FROM_TRAJECTORY in self.reasons or R_HITTER_FROM_SEQUENCE in self.reasons
        )


def apply_side_model(c: "Candidate", p_near: float, threshold: float) -> None:
    """When the learned side model is confident the hit came from the other end of the court than the
    nearest player's, hand the hit to the best player on that end (trajectory direction and contact height
    say which end hit it even when two players overlap on screen)."""
    b = c.best
    if b is None or b.pose.side not in ("near", "far"):
        return
    want = "near" if p_near >= 0.5 else "far"
    if b.pose.side == want or max(p_near, 1 - p_near) < threshold:
        return
    for k, alt in enumerate(c.prox[1:], start=1):
        if alt.pose.side == want:
            c.prox.insert(0, c.prox.pop(k))
            c.reasons = [r for r in c.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)]
            c.reasons.append(R_HITTER_FROM_TRAJECTORY)
            return


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
    return Candidate(i, ang, max(turn, speed), sharp, q, impulse=full[2] / scale_px, t_ms=pt.t_ms)


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
        if cfg.hitter_fill and not c.prox and cfg.hitter_pose_fallback_frames > cfg.pose_search_frames:
            c.prox_wide = poses.proximities(pts[i].frame, pts[i].xy, cfg.hitter_pose_fallback_frames)
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
        if scorer is None and _contact_gate(c, track, poses, cfg):
            # Rules only: the learned scorer already sees contact height and incoming speed (y_rel,
            # log_speed_in), and on the ShuttleSet test clips the gate on top of it lost a true hit.
            rejected = True
            c.confidence = 0.0
        if scorer is not None or collect is not None:
            x = features(c, track, span, poses, cfg, rule_conf, landing, no_pose)
            if collect is not None:
                collect.append((x, c))
            if scorer is not None:
                rejected = False
                c.confidence = round(float(scorer.predict([x])[0]), 3)
                if scorer.side is not None:
                    apply_side_model(c, float(scorer.side.predict([x])[0]), cfg.side_threshold)
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
                if scorer.side is not None:
                    apply_side_model(serve, float(scorer.side.predict([x])[0]), cfg.side_threshold)
        needed = not out or pts[out[0].idx].t_ms - pts[span.start].t_ms > cfg.min_hit_gap_ms
        if needed and (scorer is None or serve.confidence >= cfg.min_candidate_conf):
            out.insert(0, serve)

    return resolve_sequence(out, cfg)


def resolve_sequence(cands: list[Candidate], cfg: EventsConfig) -> list[Candidate]:
    """Rally-level rules on the time-ordered candidates: alternation, quick repeats, re-attribution."""
    out = _alternating_subset(cands, cfg)
    out = _resolve_quick_repeats(out, cfg)
    if not _choose_teams(out, cfg):
        _reattribute(out, cfg)
    if cfg.hitter_fill:
        _fill_hitters(out)
    if cfg.self_feed_max_gap_ms > 0:
        out = _drop_self_feeds(out, cfg)
    _check_alternation(out)
    return out


def _drop_self_feeds(hits: list[Candidate], cfg: EventsConfig) -> list[Candidate]:
    """Drop hits inside a run of 3+ consecutive hits by one side with short gaps (see ``self_feed_max_gap_ms``)."""
    gap = cfg.self_feed_max_gap_ms
    drop: set[int] = set()
    for i in range(1, len(hits) - 1):
        a, b, c = hits[i - 1], hits[i], hits[i + 1]
        k = _side_key(b)
        if k is None or _side_key(a) != k or _side_key(c) != k:
            continue
        if b.t_ms - a.t_ms > gap or c.t_ms - b.t_ms > gap:
            continue
        drop.add(i)
        if cfg.self_feed_run:
            drop.update((i - 1, i + 1))
    return [h for i, h in enumerate(hits) if i not in drop]


def _kp(p: PlayerPose, i: int, min_conf: float):
    if len(p.keypoints) <= i or p.keypoints[i].confidence < min_conf:
        return None
    return p.keypoints[i].x, p.keypoints[i].y


def contact_height(p: PlayerPose, xy: Vec, min_conf: float) -> Optional[float]:
    """Height of ``xy`` above the player's lower ankle, in that player's (ankle-to-top keypoint) heights.

    None without a visible ankle: an occluded or cut-off lower body says nothing about the floor.
    """
    ankles = [a for a in (_kp(p, KP_L_ANKLE, min_conf), _kp(p, KP_R_ANKLE, min_conf)) if a]
    ys = [k.y for k in p.keypoints if k.confidence >= min_conf]
    if not ankles or len(ys) < 4:
        return None
    foot = max(a[1] for a in ankles)
    h = foot - min(ys)
    return (foot - xy[1]) / h if h > 1.0 else None


def reach_torso(p: PlayerPose, xy: Vec, min_conf: float) -> Optional[float]:
    """Distance from the closest visible wrist to ``xy``, in torso lengths (shoulder centre to hip centre)."""
    sh = [q for q in (_kp(p, KP_L_SHOULDER, min_conf), _kp(p, KP_R_SHOULDER, min_conf)) if q]
    hp = [q for q in (_kp(p, KP_L_HIP, min_conf), _kp(p, KP_R_HIP, min_conf)) if q]
    wr = [q for q in (_kp(p, KP_L_WRIST, min_conf), _kp(p, KP_R_WRIST, min_conf)) if q]
    if not (sh and hp and wr):
        return None
    mid = lambda qs: (sum(q[0] for q in qs) / len(qs), sum(q[1] for q in qs) / len(qs))  # noqa: E731
    torso = math.hypot(*(a - b for a, b in zip(mid(sh), mid(hp))))
    if torso <= 1.0:
        return None
    return min(math.hypot(w[0] - xy[0], w[1] - xy[1]) for w in wr) / torso


def _contact_gate(c: Candidate, track: Track, poses: PoseIndex, cfg: EventsConfig) -> bool:
    """Measured-contact checks that hold for any scorer. True (and a reason code) when the candidate cannot be
    a stroke: the shuttle arrives slowly at the hitter's feet, is out of any racket's reach, or (optionally)
    was not measured at the contact."""
    pt = track.points[c.idx]
    if cfg.require_observed_contact and pt.interpolated:
        return True
    b = c.best
    if b is None or b.score <= 0:
        return False
    if cfg.floor_contact_h > 0:
        h = contact_height(b.pose, pt.xy, cfg.min_kp_conf)
        vin = track.incoming(c.idx, cfg.window_ms, cfg.max_window_ms)
        slow = vin is None or norm(vin) / max(poses.scale_at(pt.frame, pt.xy), 1.0) < cfg.floor_max_speed_in
        if h is not None and h < cfg.floor_contact_h and slow:
            if R_FLOOR_CONTACT not in c.reasons:
                c.reasons.append(R_FLOOR_CONTACT)
            return True
    if cfg.max_reach_torso > 0:
        r = reach_torso(b.pose, pt.xy, cfg.min_kp_conf)
        if r is not None and r > cfg.max_reach_torso:
            if R_OUT_OF_REACH not in c.reasons:
                c.reasons.append(R_OUT_OF_REACH)
            return True
    return False


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
    c = Candidate(i, 0.0, 0.0, 0.0, q, prox=prox, reasons=[R_INFERRED_SERVE], t_ms=pt.t_ms)
    # Proximity is the only evidence, so this always goes to review.
    c.confidence = round(0.5 * prox[0].score + 0.1 * q, 3)
    return c


def _hitter(c: Candidate) -> Optional[str]:
    return c.best.player_id if c.hitter_known else None


def _side_key(c: Candidate) -> Optional[str]:
    """What must alternate between consecutive hits: player (singles) or team (doubles)."""
    return c.best.pose.team_key if c.hitter_known else None


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


def _other_side_alt(c: Candidate, key: str) -> Optional[int]:
    """Index in ``c.prox`` of the closest player of a side other than ``key`` who is near the shuttle at all."""
    for k, alt in enumerate(c.prox[1:], start=1):
        if alt.score > 0 and alt.pose.team_key != key:
            return k
    return None


def _hand_over(c: Candidate, k: int) -> None:
    c.prox.insert(0, c.prox.pop(k))
    c.reasons = [r for r in c.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)]
    c.reasons.append(R_HITTER_FROM_ALTERNATION)


def _resolve_quick_repeats(hits: list[Candidate], cfg: EventsConfig) -> list[Candidate]:
    """Two hits by the same side closer than ``same_side_min_gap_ms`` cannot both be right: a missed
    opponent hit between them would need the shuttle to cross the net twice in that time. Either one of
    them was the opponent's (a small far player loses the proximity contest to a large near player), or
    one is not a hit at all (a wobble of the track around a single contact).

    The pair is fixed by handing one hit to a player of the other side who is near the shuttle at all,
    preferring the hand-over that does not clash with the neighbouring hits; with nobody there, the
    weaker hit is dropped.
    """
    gap = cfg.same_side_min_gap_ms
    if gap <= 0 or len(hits) < 2:
        return hits
    out = list(hits)
    i = 0
    while i + 1 < len(out):
        a, b = out[i], out[i + 1]
        key = _side_key(a)
        if key is None or key != _side_key(b) or b.t_ms - a.t_ms >= gap:
            i += 1
            continue
        options = []
        for c, nb in ((a, out[i - 1] if i > 0 else None), (b, out[i + 2] if i + 2 < len(out) else None)):
            k = _other_side_alt(c, key)
            if k is None:
                continue
            clash = nb is not None and _side_key(nb) == c.prox[k].pose.team_key
            options.append((not clash, c.prox[k].score, k, c))
        if options:
            _, _, k, c = max(options, key=lambda o: (o[0], o[1]))
            _hand_over(c, k)
            i += 1
            continue
        out.remove(a if a.confidence < b.confidence else b)
        i = max(i - 1, 0)
    return out


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
        k = _other_side_alt(b, ka)
        if k is not None and b.best.score - b.prox[k].score < 2 * cfg.hitter_margin:
            _hand_over(b, k)


def _choose_teams(hits: list[Candidate], cfg: EventsConfig) -> bool:
    """Doubles only: pick the hitting team of every hit in the rally jointly (see
    ``EventsConfig.doubles_alternation_penalty``) and hand each hit to the closest player of its team.

    Returns False (nothing done) for singles, with the rule disabled, or when the rally does not show two teams.
    """
    pen = cfg.doubles_alternation_penalty
    if pen <= 0 or len(hits) < 2:
        return False
    teams = sorted({p.pose.team for c in hits for p in c.prox if p.pose.team})
    if len(teams) != 2:
        return False
    n = len(hits)
    # e[i][t]: proximity of team t's best player at hit i (0 when nobody of that team is near).
    e = [
        [max((p.score for p in c.prox if p.pose.team == t), default=0.0) for t in teams]
        for c in hits
    ]
    score = [list(e[0])]
    back: list[list[int]] = [[0, 0]]
    for i in range(1, n):
        close = hits[i].t_ms - hits[i - 1].t_ms < cfg.doubles_alternation_max_gap_ms
        row, arg = [], []
        for t in range(2):
            opts = [
                score[i - 1][u] - (pen if close and u == t else 0.0) for u in range(2)
            ]
            u = max(range(2), key=lambda k: opts[k])
            row.append(opts[u] + e[i][t])
            arg.append(u)
        score.append(row)
        back.append(arg)
    t = max(range(2), key=lambda k: score[-1][k])
    chosen = [0] * n
    for i in range(n - 1, -1, -1):
        chosen[i] = t
        t = back[i][t]
    for c, ti in zip(hits, chosen):
        team = teams[ti]
        if c.best is not None and c.best.pose.team == team and c.hitter_known:
            continue
        mates = [k for k, p in enumerate(c.prox) if p.pose.team == team]
        if not mates:
            continue
        k = min(mates, key=lambda k: (-c.prox[k].score, c.prox[k].distance_rel))
        had = c.best is not None and c.best.pose.team != team and c.hitter_known
        c.prox.insert(0, c.prox.pop(k))
        c.reasons = [
            r for r in c.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)
        ]
        if c.prox[0].score > 0:
            c.reasons.append(
                R_HITTER_FROM_ALTERNATION if had else R_HITTER_FROM_SEQUENCE
            )
        else:
            c.reasons.append(R_HITTER_FROM_SEQUENCE)
    return True


def _fill_hitters(hits: list[Candidate]) -> None:
    """Give a hit whose hitter is unknown (no player near the shuttle, or no pose at the contact frame) the
    player the alternation calls for: someone of a side other than the previous known hit's (and the next
    one's), the closest such player at the contact frame or, without a pose there, a few frames around it.

    On ShuttleSet about 7% of correctly timed hits had no hitter at all, mostly at the start of a rally
    when pose frames are empty for 5-20 frames; alternation names the right end for nearly all of them.
    The hit keeps ``hitter_from_sequence`` and goes to review.
    """
    known = [i for i, c in enumerate(hits) if c.hitter_known]
    if not known:
        return
    for i, c in enumerate(hits):
        if c.hitter_known:
            continue
        pool = c.prox or c.prox_wide
        if not pool:
            continue
        prev = next((hits[j] for j in reversed(known) if j < i), None)
        nxt = next((hits[j] for j in known if j > i), None)
        avoid = {_side_key(h) for h in (prev, nxt) if h is not None}
        options = [k for k, p in enumerate(pool) if p.pose.team_key not in avoid]
        if not options:
            # previous and next known hits are by different sides (a missed hit somewhere): follow the previous
            avoid = {_side_key(prev or nxt)}
            options = [k for k, p in enumerate(pool) if p.pose.team_key not in avoid]
        if not options:
            continue
        k = min(options, key=lambda k: pool[k].distance_rel)
        c.prox = [pool[k]] + [p for j, p in enumerate(pool) if j != k]
        c.reasons = [r for r in c.reasons if r not in (R_HITTER_AMBIGUOUS, R_TEAMMATE_AMBIGUOUS)]
        c.reasons.append(R_HITTER_FROM_SEQUENCE)


def _check_alternation(hits: list[Candidate]) -> None:
    """Consecutive hits by the same side (player in singles, team in doubles) mean a missed or spurious hit."""
    for a, b in zip(hits, hits[1:]):
        ka, kb = _side_key(a), _side_key(b)
        if ka is not None and ka == kb:
            for c in (a, b):
                if R_SAME_HITTER not in c.reasons:
                    c.reasons.append(R_SAME_HITTER)
