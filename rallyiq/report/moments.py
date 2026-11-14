"""Shot-by-shot review: which shots should change, judged from stroke type + both players' positions.

For every singles shot with a known hitter and stroke type this module looks up where the
hitter and the opponent stood at the moment of the hit (player tracks, court metres), how far
the opponent then had to move to reach the shot (their next hit position) and what happened
next (the opponent's reply, the rally result). A few tactical rules flag shots that gave the
opponent an easy ball, and for each one propose a better shot and a target: the corner of the
opponent's half farthest from where the opponent stood, restricted to shots that can be played
from where the hitter stood.

Rules (``issue`` codes):

* ``to_opponent``: the shot went (almost) straight to the opponent: they moved under
  ``REACH_EASY_M`` to hit it and then attacked or won the rally.
* ``clear_to_rear``: clear / lift while the opponent was already in their rear court.
* ``short_to_front``: drop / net shot / cross-court net while the opponent was already at the net.
* ``lift_from_front``: lift from the front court that the opponent smashed or killed, when a net
  shot or push was possible (the hitter was in the front court).
* ``attack_off_balance``: smash / kill / drive from a stretched position (rear corner) that lost
  the rally: a high clear to the deep middle buys time to recover.

Footwork rules (need ``footwork``, see ``rallyiq.footwork``), used when no shot-choice rule fired:

* ``slow_recovery``: after the shot the hitter was still far from the base when the opponent hit
  next, and the opponent then attacked or won the rally.
* ``late_start``: no split step (or a late one) on the opponent's previous hit and a slow first
  move, and this shot then got punished.

Every moment also carries the hitter's ``footwork`` for that shot (split step, steps, lunge, recovery)
when it is known, so the advice can say how the hitter got there.

Doubles: the "opponent" is the one who replied (else whichever opponent was tracked); targets
use the doubles sidelines and the open spot is the one farthest from the nearer of the two
opponents, with the gap between them as an extra candidate. Outcomes are judged by team.

Nothing here calls a model: the coach provider only rewrites the explanation. Without stroke
types or player tracks the list is empty.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from ..contracts import (
    COURT_WIDTH_M,
    NET_Y_M,
    EventsResult,
    PlayerTracks,
    ShotEvent,
    ShotMoment,
    team_of,
)
from ..contracts.geometry import SINGLES_INSET
from ..footwork import BASE_RADIUS_M, Footwork
from ..strokes.detail import short_label

ZH = {
    "short_serve": "发短球", "long_serve": "发高远球", "clear": "高远球", "smash": "杀球", "drop": "吊球",
    "net_shot": "放网", "cross_net": "勾对角", "push": "推球", "net_kill": "扑球", "drive": "平抽",
    "lift": "挑球", "block": "挡网",
}
ATTACKS = {"smash", "net_kill"}
LONG = {"clear", "lift"}
SERVES = {"short_serve", "long_serve"}
SHORT = {"drop", "net_shot", "cross_net", "block"}
POWER = {"smash", "net_kill", "drive"}

REACH_EASY_M = 1.0  # opponent moved less than this to reach the shot
REAR_M = 2.2  # own-half depth bands (same as stats.DEPTH_ZONES)
FRONT_M = 4.47
POS_WINDOW_S = 0.2  # look this far around the hit frame for a foot position
MAX_PER_PLAYER = 12
OPPONENTS = {"P1": ("P2", "P4"), "P3": ("P2", "P4"), "P2": ("P1", "P3"), "P4": ("P1", "P3")}

SINGLES_LEFT = SINGLES_INSET
SINGLES_RIGHT = COURT_WIDTH_M - SINGLES_INSET
SINGLES_W = SINGLES_RIGHT - SINGLES_LEFT


def own_half(x: float, y: float) -> tuple[float, float]:
    """Court metres -> (lateral from own left sideline, depth from own baseline)."""
    if y <= NET_Y_M:
        return x, y
    return COURT_WIDTH_M - x, 2 * NET_Y_M - y


def depth_name(depth: float) -> str:
    return "后场" if depth < REAR_M else "中场" if depth < FRONT_M else "网前"


def side_name(lateral: float) -> str:
    """Lateral (from the viewer's left sideline) -> 左侧/中路/右侧 of the singles court."""
    third = SINGLES_W / 3
    if lateral < SINGLES_LEFT + third:
        return "左侧"
    if lateral < SINGLES_LEFT + 2 * third:
        return "中路"
    return "右侧"


def zone(lateral: float, depth: float) -> str:
    return depth_name(depth) + side_name(lateral)


def timestamp(t_ms: int) -> str:
    s = t_ms // 1000
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# --------------------------------------------------------------------------- positions


class _Positions:
    """Foot positions per player, looked up near a frame (median of detections in a small window)."""

    def __init__(self, players: PlayerTracks, fps: float):
        self.win = max(1, round(POS_WINDOW_S * fps))
        self.by_pid: dict[str, tuple[list[int], list[tuple[float, float]]]] = {}
        for pf in players.frames:
            for p in pf.players:
                if p.player_id is None or p.foot_court is None or p.on_court is False:
                    continue
                frames, xy = self.by_pid.setdefault(p.player_id, ([], []))
                frames.append(pf.frame)
                xy.append((float(p.foot_court[0]), float(p.foot_court[1])))

    def at(self, pid: str, frame: int) -> tuple[float, float] | None:
        got = self.by_pid.get(pid)
        if not got:
            return None
        frames, xy = got
        i = bisect_left(frames, frame - self.win)
        near = []
        while i < len(frames) and frames[i] <= frame + self.win:
            near.append(xy[i])
            i += 1
        if not near:
            return None
        xs, ys = sorted(p[0] for p in near), sorted(p[1] for p in near)
        return xs[len(xs) // 2], ys[len(ys) // 2]


# --------------------------------------------------------------------------- targets


@dataclass
class _Target:
    shot: str
    zone: str  # opponent's half, hitter's view


MIDDLE = "中场中路（两人之间）"


def _corners(doubles: bool = False) -> list[tuple[str, float, float]]:
    """(name, lateral from hitter's left, depth from the opponent's baseline) of the target spots:
    four corners (doubles sidelines in doubles), plus the gap between the two opponents in doubles."""
    lo, hi = (0.5, COURT_WIDTH_M - 0.5) if doubles else (SINGLES_LEFT + 0.5, SINGLES_RIGHT - 0.5)
    out = [("网前左侧", lo, NET_Y_M - 1.0), ("网前右侧", hi, NET_Y_M - 1.0),
           ("后场左侧", lo, 0.6), ("后场右侧", hi, 0.6)]
    if doubles:
        out.append((MIDDLE, COURT_WIDTH_M / 2, 3.3))
    return out


def _target(hitter_depth: float, opp_view_lat: float, opp_depth: float, exclude_short: bool = False,
            exclude_long: bool = False, others: list[tuple[float, float]] | None = None) -> _Target | None:
    """Best open spot and a shot that reaches it from where the hitter stands. ``others`` holds the other
    opponents' (view lateral, depth) in doubles: the spot farthest from the nearest opponent wins."""
    opps = [(opp_view_lat, opp_depth), *(others or [])]
    best = None
    for name, lat, dep in _corners(doubles=others is not None):
        middle = name == MIDDLE
        short = name.startswith("网前")
        if not middle and (short and exclude_short or not short and exclude_long):
            continue
        d = min(((lat - ol) ** 2 + (dep - od) ** 2) ** 0.5 for ol, od in opps)
        if best is None or d > best[0]:
            best = (d, name, short, middle)
    if best is None:
        return None
    _, name, short, middle = best
    hz = depth_name(hitter_depth)
    if middle:
        shot = "push" if hz == "网前" else "smash" if hz == "后场" else "drive"
    elif short:
        shot = "net_shot" if hz == "网前" else "drop"
    else:
        shot = "lift" if hz == "网前" else "clear" if hz == "后场" else "push"
    return _Target(shot, name)


# --------------------------------------------------------------------------- rules


def _outcome(winner_side: str | None, hitter: str, last_idx: int, i: int) -> str:
    if winner_side is None:
        return "unknown"
    if last_idx - i > 1:
        return "continued"
    return "won" if winner_side == hitter else "lost"


def _conf(shot: ShotEvent, metric_ok: bool) -> str:
    st = shot.stroke
    if st is not None and (st.source == "human" or st.status == "ok"):
        c = "high"
    elif st is None or st.confidence >= 0.5:
        c = "medium"
    else:
        c = "low"
    if not metric_ok:
        c = {"high": "medium", "medium": "low", "low": "low"}[c]
    return c


SLOW_RECOVERY_M = 1.5  # still this far from base when the opponent hit
SLOW_REACTION_MS = 350


def shot_moments(events: EventsResult, players: PlayerTracks | None, fps: float = 30.0,
                 metric_ok: bool = True, footwork: Footwork | None = None) -> list[ShotMoment]:
    """Shots to change, at most ``MAX_PER_PLAYER`` per player (most severe), in video order."""
    if players is None or not any(s.shot_type for s in events.shots):
        return []
    doubles = events.match_format == "doubles"
    pids = ("P1", "P2", "P3", "P4") if doubles else ("P1", "P2")
    pos = _Positions(players, fps)
    shots_by_id = {s.shot_id: s for s in events.shots}
    rallies = sorted(events.rallies, key=lambda r: r.start_frame)
    found: list[ShotMoment] = []
    for rno, rally in enumerate(rallies, 1):
        seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        last = len(seq) - 1
        for i, s in enumerate(seq):
            if s.hitter not in pids or not s.shot_type or s.shot_type in SERVES or i == 0:
                continue  # serves are a separate choice (short vs long), not judged here
            nxt = seq[i + 1] if i < last else None
            others: list[tuple[float, float]] | None = None
            if doubles:
                # the opponent who replied, else whichever opponent was tracked; the other one also counts
                # when picking the open spot
                opps = OPPONENTS[s.hitter]
                first = nxt.hitter if nxt is not None and nxt.hitter in opps else None
                order = [first] + [o for o in opps if o != first] if first else list(opps)
                at = {o: pos.at(o, s.frame) for o in order}
                tracked = [o for o in order if at[o] is not None]
                if not tracked:
                    continue
                opp = tracked[0]
                others = [(COURT_WIDTH_M - own_half(*at[o])[0], own_half(*at[o])[1]) for o in tracked[1:]]
            else:
                opp = "P2" if s.hitter == "P1" else "P1"
            hpos = tuple(s.hitter_court) if s.hitter_court is not None else pos.at(s.hitter, s.frame)
            opos = pos.at(opp, s.frame)
            if hpos is None or opos is None:
                continue
            h_lat, h_dep = own_half(*hpos)
            o_lat, o_dep = own_half(*opos)
            o_view_lat = COURT_WIDTH_M - o_lat  # opponent's left is the hitter's right
            reach = None
            if nxt is not None and nxt.hitter == opp:
                npos = tuple(nxt.hitter_court) if nxt.hitter_court is not None else pos.at(opp, nxt.frame)
                if npos is not None:
                    reach = ((npos[0] - opos[0]) ** 2 + (npos[1] - opos[1]) ** 2) ** 0.5
            next_type = nxt.shot_type if nxt is not None and nxt.hitter == opp else None
            if next_type in SERVES:  # a serve cannot answer a shot: mis-split rally or stroke error
                next_type = None
            outcome = _outcome(rally.winner, team_of(s.hitter) if doubles else s.hitter, last, i)
            punished = outcome == "lost" or next_type in ATTACKS
            fw = footwork.shot(s.shot_id) if footwork is not None else None
            m = _rule(s, h_dep, o_view_lat, o_dep, reach, next_type, outcome, punished, others)
            if m is None and fw is not None:
                m = _footwork_rule(s, fw, h_dep, o_view_lat, o_dep, next_type, outcome, punished)
            if m is None:
                continue
            issue, problem, target, why, severity = m
            hz, oz = zone(h_lat, h_dep), zone(o_view_lat, o_dep)
            det = s.stroke.detail if s.stroke else None
            if issue in FOOTWORK_ISSUES:
                sugg = FOOTWORK_SUGGESTION[issue]
            else:
                sugg = (f"改打{ZH[target.shot]}到对方{target.zone}" if target else "先打高远球到对方后场中路，争取回位时间")
            found.append(ShotMoment(
                moment_id="", shot_id=s.shot_id, rally_id=rally.rally_id, rally_no=rno, shot_no=i + 1, t_ms=s.t_ms,
                player_id=s.hitter, shot_type=s.shot_type, hitter_zone=hz,
                shot_label_zh=short_label(s.shot_type, det) if det else None,
                shot_detail_zh=(det.description_zh or None) if det else None, hitter_pos=[round(h_lat, 2), round(h_dep, 2)],
                opponent_id=opp, opponent_zone=oz, opponent_pos=[round(o_lat, 2), round(o_dep, 2)],
                opponent_reach_m=None if reach is None else round(reach, 1), next_shot_type=next_type,
                outcome=outcome, issue=issue, problem=problem,
                suggested_shot=(target.shot if target else None if issue in FOOTWORK_ISSUES else "clear"),
                target_zone=target.zone if target else None if issue in FOOTWORK_ISSUES else "后场中路",
                suggestion=sugg, why=why, severity=round(severity, 2),
                confidence=_conf(s, metric_ok) if issue not in FOOTWORK_ISSUES else _fw_conf(s, metric_ok, fw),
                footwork=fw,
            ))
    keep: list[ShotMoment] = []
    for pid in pids:
        mine = sorted((m for m in found if m.player_id == pid), key=lambda m: (-m.severity, m.t_ms))
        keep += mine[:MAX_PER_PLAYER]
    keep.sort(key=lambda m: m.t_ms)
    return [m.model_copy(update={"moment_id": f"m{k}"}) for k, m in enumerate(keep, 1)]


def _reply(next_type: str | None, outcome: str) -> str:
    parts = []
    if next_type:
        parts.append(f"对手回了{ZH.get(next_type, next_type)}")
    if outcome == "lost":
        parts.append("这一分丢了")
    return "，".join(parts)


def _rule(s: ShotEvent, h_dep: float, o_view_lat: float, o_dep: float, reach: float | None, next_type: str | None,
          outcome: str, punished: bool, others: list[tuple[float, float]] | None = None):
    """(issue, problem, target, why, severity) or None."""
    t = s.shot_type
    me = ZH.get(t, t)
    opp_zone = zone(o_view_lat, o_dep)
    reply = _reply(next_type, outcome)
    tail = f"，{reply}" if reply else ""
    bonus = (1.0 if outcome == "lost" else 0.0) + (0.5 if next_type in ATTACKS else 0.0)
    easy = reach is not None and reach < 1.5  # the opponent barely had to move

    # 1. attacking from a stretched rear-corner position and losing on it
    if t in POWER and h_dep < REAR_M and outcome == "lost" and next_type is None:
        return ("attack_off_balance", f"在后场被动位置强行{me}，直接丢分",
                None, ("你在后场还没站稳，进攻质量和回位都跟不上；先打一拍又高又远的过渡球，"
                      "把对手压到底线，自己回到中心再找进攻机会。"), 2.0 + bonus)

    # 2. long ball to an opponent already at the back
    if t in LONG and o_dep < REAR_M and (punished or easy):
        tgt = _target(h_dep, o_view_lat, o_dep, exclude_long=True, others=others)
        return ("clear_to_rear", f"对手已经在{opp_zone}，你还打了一拍{me}到后场{tail}",
                tgt, (f"对手本来就在后场，这拍不用移动就能舒服地出手；对手离网前最远，"
                     f"把球放到{tgt.zone}能逼对手上网，接得最勉强。"), 1.5 + bonus)

    # 3. short ball to an opponent already at the net
    if t in SHORT and o_dep >= FRONT_M and (punished or easy):
        tgt = _target(h_dep, o_view_lat, o_dep, exclude_short=True, others=others)
        return ("short_to_front", f"对手已经在{opp_zone}，你还打了一拍{me}到网前{tail}",
                tgt, (f"对手已经压在网前，短球正好送到对手拍子上，容易被扑或推；"
                     f"这时把球送到{tgt.zone}，逼对手转身后退。"), 1.5 + bonus)

    # 4. lifting from the front court and getting smashed
    if t == "lift" and h_dep >= FRONT_M and next_type in ATTACKS:
        tgt = _target(h_dep, o_view_lat, o_dep, exclude_long=True, others=others)
        return ("lift_from_front", f"在网前挑球，对手直接{ZH[next_type]}{'' if outcome != 'lost' else '得分'}",
                tgt, (f"你已经到了网前，挑球等于把进攻权交给对手（对手当时在{opp_zone}）；"
                     f"能放网就放到{tgt.zone}，让对手继续起球。"), 1.0 + bonus)

    # 5. any shot that landed where the opponent already was
    if reach is not None and reach < REACH_EASY_M and punished:
        moved = "几乎没有移动" if reach < 0.3 else f"只移动约 {reach:.1f} 米"
        tgt = _target(h_dep, o_view_lat, o_dep, others=others)
        return ("to_opponent", f"{me}打到了对手身边（对手在{opp_zone}，{moved}就接到）{tail}",
                tgt, (f"球路没有调动对手，对手有时间准备下一拍；离对手最远的是{tgt.zone}，"
                     f"从你的位置打{ZH[tgt.shot]}能到。"), 1.0 + bonus)
    return None


# --------------------------------------------------------------------------- footwork rules

FOOTWORK_ISSUES = {"slow_recovery", "late_start"}
FOOTWORK_SUGGESTION = {
    "slow_recovery": "击球后马上回中心，在对手出手前站好并做启动步",
    "late_start": "对手击球的一瞬间做启动步（双脚轻跳、落地即蹬出），第一步朝来球方向",
}


def _fw_conf(s: ShotEvent, metric_ok: bool, fw) -> str:
    c = _conf(s, metric_ok)
    if fw is not None and fw.confidence < 0.5:
        c = {"high": "medium", "medium": "low", "low": "low"}[c]
    return c


def _footwork_rule(s: ShotEvent, fw, h_dep: float, o_view_lat: float, o_dep: float, next_type: str | None,
                   outcome: str, punished: bool):
    """(issue, problem, target, why, severity) for a footwork problem on this shot, or None."""
    me = ZH.get(s.shot_type, s.shot_type)
    reply = _reply(next_type, outcome)
    tail = f"，{reply}" if reply else ""
    bonus = (1.0 if outcome == "lost" else 0.0) + (0.5 if next_type in ATTACKS else 0.0)
    if fw.recovery_gap_m is not None and fw.recovery_gap_m >= SLOW_RECOVERY_M and punished:
        return ("slow_recovery", f"打完这拍{me}没有及时回位，对手出手时你离中心还有 {fw.recovery_gap_m:.1f} 米{tail}",
                None, (f"回位慢，对手只要把球打到你离开的那一侧就能调动你；中心离四个角都差不多远，"
                      f"站在中心半径 {BASE_RADIUS_M:.0f} 米以内，下一拍到哪个角都来得及。"), 1.2 + bonus)
    late = fw.split_step in ("none", "late")
    slow = fw.reaction_ms is not None and fw.reaction_ms >= SLOW_REACTION_MS
    if late and slow and (punished or fw.finish == "lunge"):
        what = "没有做启动步" if fw.split_step == "none" else f"启动步落地晚了（对手击球后 {fw.split_offset_ms} ms）"
        return ("late_start", f"对手击球时{what}，{fw.reaction_ms} ms 后才起动去接这拍{me}{tail}",
                None, ("启动步让你在对手出手的瞬间处在可以向任何方向蹬地的状态；起动晚了，"
                      "到位时只能被动击球，球路选择也会变少。"), 1.0 + bonus)
    return None
