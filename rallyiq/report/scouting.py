"""How to beat each player: direct tactics against P1 and against P2 (singles).

For every singles player ("the target") this reads where and how the target got into trouble and
turns it into plain instructions for whoever plays them, e.g. "压 P1 的左后场". Rules only, no model.

Per target shot (from ``style._contexts``: hitter position in court metres, stroke, the reply, the
rally winner) a shot is **trouble** when the target lost the rally with it (net / out / no reply), or
the opponent's next shot was a smash / net kill (the target's ball got punished), or the opponent's
next shot won the rally. From that:

* **zones**: the target's hit position, in their own half, split into rear / mid / front x left / right
  (their left as they face the net). A zone whose trouble rate is well above the target's own average
  is where to send the ball; a zone where the target attacks most and rarely gets into trouble is
  where not to feed them.
* **stroke errors**: the stroke the target most often ended a lost rally with, mapped to how to make
  them play it (lose on net shots -> keep the ball short at the net).
* **movement**: trouble rate after a long move (>= 2 m to reach the ball) vs a short one (<= 1.2 m).
* **reply habits** (``PlayerStyle.habits``): after your X the target mostly plays Y -> be ready for Y.
* **attack / defense state** (``<Pn>.style.*``): holds badly under attack, errs when attacking, rarely
  attacks from a good position.
* **rally length / serve-receive** (``<Pn>.win.*``, needs entered winners) and the style counter
  (``style.COUNTER``) for a clear style.

Confidence: every tactic gets one from its sample size, capped by data quality: the far-side player
(away from the camera, hitters / positions / strokes are less reliable), uncertain hitters,
uncalibrated court (position tactics), few confirmed shots. Evidence numbers are added to the stats as
``<Pn>.scout.*`` metrics and distributions so the coach can cite them.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median

from ..contracts import (
    COURT_WIDTH_M,
    NET_Y_M,
    Distribution,
    EventsResult,
    Metric,
    PlayerStyle,
    PlayerTracks,
    Scouting,
    ScoutTactic,
)
from ..contracts.geometry import SINGLES_INSET
from .moments import ZH, own_half
from .style import CLEAR_STYLE, COUNTER, MIN_SHOTS, OPP_ATTACK, SERVES, _contexts

OTHER = {"P1": "P2", "P2": "P1"}
PASSIVE = {"lift", "block"}
_RANK = ("low", "medium", "high")

REAR_M, FRONT_M = 2.2, 4.47  # own-half depth bands, same as stats.DEPTH_ZONES
MIN_ZONE_N = 6  # target shots from a zone before it can be called weak
ZONE_GAP = 15.0  # percentage points above the target's own trouble rate
ZONE_MIN_RATE = 35.0
FAR_REACH_M, NEAR_REACH_M = 2.0, 1.2
MAX_TACTICS = 6

DEPTH_ZH = {"rear": "后场", "mid": "中场", "front": "网前"}
SIDE_ZH = {"left": "左", "center": "中路", "right": "右"}
SINGLES_W = COURT_WIDTH_M - 2 * SINGLES_INSET
LANE_L, LANE_R = SINGLES_INSET + SINGLES_W / 3, SINGLES_INSET + 2 * SINGLES_W / 3  # same as stats.LATERAL_ZONES
# what a zone is for a right-handed player (handedness is not measured)
ZONE_HAND = {("rear", "left"): "右手持拍时是头顶/反手区", ("rear", "right"): "右手持拍时是正手后场",
             ("front", "left"): "右手持拍时是反手网前", ("front", "right"): "右手持拍时是正手网前",
             ("mid", "left"): "右手持拍时是反手中场", ("mid", "right"): "右手持拍时是正手中场"}
# how to send the ball to a zone
ZONE_HOW = {
    "rear": "用高远球、推后场、挑底线把球压到他的{z}，逼他从那里出手",
    "mid": "用平抽、平推、半场球打他的{z}",
    "front": "用吊球、放网、挡网把球放到他的{z}",
}
# you just played a shot and the target mostly answers with this: how to be ready
READY = {
    "net_shot": "放完别退，马上准备扑或搓",
    "cross_net": "放完注意封对角网前",
    "lift": "回中后马上准备后场进攻（杀球或劈吊）",
    "clear": "提前后撤，争取在高点出手",
    "block": "杀完马上上网，准备扑或推",
    "drive": "拍子举在胸前，封住中路平抽",
    "push": "提前后撤，准备接后场球",
    "smash": "提前降低重心准备接杀，挡网或挑底线",
    "drop": "提前上网接吊",
    "net_kill": "网前球要放得更贴网，避免被扑",
}
# the target ends lost rallies with this stroke -> how to make them play it
FORCE = {
    "net_shot": "多放网、吊网前，逼他在网前处理小球",
    "cross_net": "多放网、吊网前，逼他在网前处理小球",
    "lift": "放网、吊球压在网前逼他挑球，他挑球常出问题；挑完要做好后场进攻准备",
    "clear": "压他的后场底线，逼他打高远球，他后场高远球容易不到位或出界",
    "drop": "推后场、打平高让他在后场处理，他吊球容易下网或吊高",
    "smash": "后场球可以适当让他杀：他杀球失误多；挑球要挑到底线，做好接杀准备",
    "drive": "多和他打平抽快挡，他平抽失误多",
    "push": "网前和他搓推，他推球失误多",
    "block": "连续进攻压他，他接杀挡网容易失误",
    "net_kill": "网前球放到贴网，他扑球容易下网或出界",
}


def _cap(conf: str, *caps: str | None) -> str:
    return min([conf, *[c for c in caps if c]], key=_RANK.index)


def _n_conf(n: int, ok: int = 20, review: int = 8) -> str:
    return "high" if n >= ok else "medium" if n >= review else "low"


def _zone(pos: tuple[float, float]) -> tuple[str, str]:
    """(depth, lane) in the player's own half; lanes are thirds of the singles width (their left as they face the net)."""
    lat, dep = own_half(*pos)
    depth = "rear" if dep < REAR_M else "front" if dep >= FRONT_M else "mid"
    return depth, "left" if lat < LANE_L else "right" if lat >= LANE_R else "center"


def _zname(z: tuple[str, str]) -> str:
    return f"{SIDE_ZH[z[1]]}{DEPTH_ZH[z[0]]}"


def _hand(z: tuple[str, str]) -> str:
    return f"（{ZONE_HAND[z]}）" if z in ZONE_HAND else ""


def analyze_scouting(events: EventsResult, players: PlayerTracks | None, fps: float, metric_ok: bool,
                     styles: list[PlayerStyle], shared: dict[str, Metric | Distribution],
                     confirmed_pct: float | None = None, hitters_unreliable: bool = False,
                     ) -> tuple[list[Scouting], dict[str, list[Metric]], dict[str, list[Distribution]]]:
    """``shared``: existing metrics / distributions by key (``Pn.win.*``, ``Pn.style.*``)."""
    from .stats import _conf, _dist, _metric  # stats imports this module

    if events.match_format == "doubles":
        return [], {}, {}
    rallies = _contexts(events, players, fps)
    last_id = {r[-1].shot.shot_id for r in rallies if r}
    style_of = {s.subject: s for s in styles}
    out: list[Scouting] = []
    metrics: dict[str, list[Metric]] = {}
    dists: dict[str, list[Distribution]] = {}

    def usable(key: str) -> Metric | None:
        e = shared.get(key)
        return e if isinstance(e, Metric) and e.value is not None and e.status != "undeterminable" else None

    for pid in ("P1", "P2"):
        opp = OTHER[pid]
        own = [c for r in rallies for c in r if c.side == pid and c.shot.shot_type and c.shot.shot_type not in SERVES]
        ms: list[Metric] = []
        ds: list[Distribution] = []

        # which end of the court the target played from (camera is behind the near baseline)
        ys = [c.pos[1] for c in own if c.pos is not None]
        side = None if len(ys) < 5 else ("far" if median(ys) > NET_Y_M else "near")
        caps: list[str] = []
        notes: list[str] = []
        if side == "far":
            caps.append("medium")
            notes.append(f"{pid} 在远端（离镜头远），击球者、站位和击球类型的识别误差更大，结论先当作参考")
        if hitters_unreliable:
            caps.append("low")
            notes.append("击球者识别不稳定（相邻两拍常被判给同一人），先在工作台修正击球者")
        if confirmed_pct is not None and confirmed_pct < 50:
            caps.append("medium")
            notes.append(f"只有 {confirmed_pct:.0f}% 的击球经过人工确认")
        pos_cap = None if metric_ok else "low"

        if len(own) < MIN_SHOTS:
            out.append(Scouting(target=pid, against=opp, side=side, summary="", tactics=[], n_shots=len(own),
                                status="undeterminable", confidence="low",
                                note=f"{pid} 只有 {len(own)} 拍已分类的非发球击球，至少需要 {MIN_SHOTS} 拍才给战术"))
            continue

        def is_trouble(c, opp: str = opp) -> bool:
            if c.is_last and c.rally_winner == opp:
                return True
            if c.shot.shot_type in PASSIVE or (c.shot.shot_type == "clear" and c.phase == "defense"):
                return True  # had to give the attack away: lift, block, clear under pressure
            nx = c.next
            if nx is None:
                return False
            return nx.shot_type in OPP_ATTACK or (nx.shot_id in last_id and c.rally_winner == opp)

        tactics: list[tuple[float, ScoutTactic]] = []

        # ---- 1. zones: where they get into trouble, where they attack from
        placed = [c for c in own if c.pos is not None]
        base_n = len(placed)
        base_bad = sum(is_trouble(c) for c in placed)
        base_rate = 100.0 * base_bad / base_n if base_n else 0.0
        zn: Counter = Counter()
        zbad: Counter = Counter()
        zatt: Counter = Counter()
        zex: dict = {}
        for c in placed:
            z = _zone(c.pos)
            zn[z] += 1
            zatt[z] += c.attack
            if is_trouble(c):
                zbad[z] += 1
                zex.setdefault(z, []).append(c.shot.t_ms)
        if base_n:
            k_n, k_bad = f"{pid}.scout.zone_shots", f"{pid}.scout.zone_trouble"
            ds.append(_dist(k_n, "被对手打到各区域后的击球数（本人视角）", {_zname(z): v for z, v in zn.most_common()},
                            30, 10, degrade=not metric_ok))
            ds.append(_dist(k_bad, "各区域陷入被动的击球数（只能挑球/挡网、被杀/扑、失误或被打死）",
                            {_zname(z): zbad.get(z, 0) for z, _ in zn.most_common()}, 30, 10, degrade=not metric_ok))
            k_base = f"{pid}.scout.trouble_pct"
            ms.append(_metric(k_base, "陷入被动的比例（全场平均）", base_rate, "%", base_n, 40, 12,
                              degrade=not metric_ok))
            weak = []
            for z, n in zn.items():
                rate = 100.0 * zbad[z] / n
                if n >= MIN_ZONE_N and rate >= ZONE_MIN_RATE and rate - base_rate >= ZONE_GAP:
                    weak.append(((rate - base_rate) * math.sqrt(n), z, n, rate))
            weak.sort(reverse=True)
            weak = weak[:2]
            if len(weak) == 2 and weak[0][1][0] == weak[1][1][0]:  # both sides of the same depth: one tactic
                (s1, z1, n1, _), (s2, z2, n2, _) = weak
                n = n1 + n2
                bad = zbad[z1] + zbad[z2]
                depth = z1[0]
                name = DEPTH_ZH[depth]
                k_rate = f"{pid}.scout.trouble_pct.{depth}"
                ms.append(_metric(k_rate, f"{name}击球后陷入被动的比例", 100 * bad / n, "%", n, 15, 6,
                                  degrade=not metric_ok))
                tactics.append((s1 + s2 + 30, ScoutTactic(
                    code="zone_weak", title=f"压 {pid} 的{name}",
                    how=ZONE_HOW[depth].format(z=name) + "，左右两边都可以。",
                    why=f"{pid} 在{name}击球 {n} 拍，{bad} 拍陷入被动（只能挑球/挡网、被下压或丢分，{100 * bad / n:.0f}%），全场平均 {base_rate:.0f}%。",
                    evidence=[k_rate, k_n, k_bad, k_base], n=n, confidence=_cap(_n_conf(n, 15, 6), pos_cap, *caps),
                    examples_ms=sorted(zex.get(z1, [])[:2] + zex.get(z2, [])[:1]))))
                weak = []
            for score, z, n, rate in weak:
                name = _zname(z)
                k_rate = f"{pid}.scout.trouble_pct.{z[0]}_{z[1]}"
                ms.append(_metric(k_rate, f"{name}击球后陷入被动的比例", rate, "%", n, 15, 6, degrade=not metric_ok))
                conf = _cap(_n_conf(n, 15, 6), pos_cap, *caps)
                tactics.append((score + 30, ScoutTactic(
                    code="zone_weak", title=f"压 {pid} 的{name}",
                    how=ZONE_HOW[z[0]].format(z=name) + f"{_hand(z)}。",
                    why=f"{pid} 在{name}击球 {n} 拍，{zbad[z]} 拍陷入被动（只能挑球/挡网、被下压或丢分，{rate:.0f}%），全场平均 {base_rate:.0f}%。",
                    evidence=[k_rate, k_n, k_bad, k_base], n=n, confidence=conf, examples_ms=zex.get(z, [])[:3])))
            strong = []
            for z, n in zn.items():
                att = 100.0 * zatt[z] / n
                rate = 100.0 * zbad[z] / n
                if n >= 5 and att >= 40 and rate <= base_rate:
                    strong.append((att * math.sqrt(n), z, n, att))
            strong.sort(reverse=True)
            for score, z, n, att in strong[:1]:
                if any(t.code == "zone_weak" and _zname(z) in t.title for _, t in tactics):
                    continue
                name = _zname(z)
                k_att = f"{pid}.scout.zone_attack"
                k_att_pct = f"{pid}.scout.attack_pct.{z[0]}_{z[1]}"
                ms.append(_metric(k_att_pct, f"{name}击球中进攻的比例", att, "%", n, 15, 6, degrade=not metric_ok))
                ds.append(_dist(k_att, "各区域的进攻性击球数（本人视角）",
                                {_zname(x): zatt.get(x, 0) for x, _ in zn.most_common()}, 30, 10,
                                degrade=not metric_ok))
                tactics.append((score / 10, ScoutTactic(
                    code="zone_strong", title=f"少给 {pid} {name}的球",
                    how=f"别把球送到他的{name}{_hand(z)}，他在那里最容易下压进攻。",
                    why=f"{pid} 在{name}击球 {n} 拍，其中 {zatt[z]} 拍是进攻（{att:.0f}%），陷入被动的比例不高于平均。",
                    evidence=[k_att_pct, k_n, k_att], n=n, confidence=_cap(_n_conf(n, 15, 6), pos_cap, *caps),
                    examples_ms=[])))

        # ---- 2. the stroke they lose points with
        decided_last = [c for c in own if c.is_last and c.rally_winner is not None]
        errs = Counter(c.shot.shot_type for c in decided_last if c.rally_winner == opp)
        typed = Counter(c.shot.shot_type for c in own)
        if errs:
            k_err = f"{pid}.scout.errors"
            ds.append(_dist(k_err, "失分时本人最后一拍的击球类型（近似看作失误）",
                            {ZH.get(t, t): v for t, v in errs.most_common()}, 12, 4))
            total_err = sum(errs.values())
            for t, cnt in errs.most_common(2):
                if t not in FORCE or cnt < 3 or cnt / total_err < 0.3:
                    continue
                rate = 100.0 * cnt / typed[t] if typed[t] else 0.0
                exs = [c.shot.t_ms for c in decided_last if c.rally_winner == opp and c.shot.shot_type == t][:3]
                tactics.append((cnt * 6 + rate / 3, ScoutTactic(
                    code="stroke_errors", title=f"逼 {pid} 打{ZH.get(t, t)}",
                    how=FORCE[t] + "。",
                    why=f"{pid} 丢掉的 {total_err} 分里，有 {cnt} 分是他自己的{ZH.get(t, t)}结束的"
                        f"（他一共打了 {typed[t]} 拍{ZH.get(t, t)}）。",
                    evidence=[k_err, f"{pid}.shot_types"], n=cnt, confidence=_cap(_n_conf(total_err, 12, 5), *caps), examples_ms=exs)))

        # ---- 3. moving them
        own_by_id = {c.shot.shot_id: c for c in own}
        reached = [c for r in rallies for c in r if c.side == opp and c.reach is not None and c.next is not None]
        far = near = far_bad = near_bad = 0
        far_ex: list[int] = []
        for c in reached:  # c.next is the target's reply; judge it as the target's shot
            reply = own_by_id.get(c.next.shot_id)
            if reply is None:
                continue
            bad = is_trouble(reply)
            if c.reach >= FAR_REACH_M:
                far += 1
                far_bad += bad
                if bad:
                    far_ex.append(reply.shot.t_ms)
            elif c.reach <= NEAR_REACH_M:
                near += 1
                near_bad += bad
        if far >= 5 and near >= 5:
            fr, nr = 100.0 * far_bad / far, 100.0 * near_bad / near
            k_far, k_near = f"{pid}.scout.trouble_far_pct", f"{pid}.scout.trouble_near_pct"
            ms.append(_metric(k_far, f"跑动 ≥{FAR_REACH_M:.0f} m 接球后陷入被动的比例", fr, "%", far, 20, 6,
                              degrade=not metric_ok))
            ms.append(_metric(k_near, f"跑动 ≤{NEAR_REACH_M} m 接球后陷入被动的比例", nr, "%", near, 20, 6,
                              degrade=not metric_ok))
            if fr - nr >= 15:
                tactics.append(((fr - nr) * math.sqrt(far) / 2, ScoutTactic(
                    code="move", title=f"多调动 {pid}",
                    how="打四角，一拍后场一拍网前、一拍左一拍右，不要把球打到他身边。",
                    why=f"{pid} 跑 {FAR_REACH_M:.0f} 米以上去接的 {far} 拍里有 {fr:.0f}% 陷入被动，"
                        f"在身边接的 {near} 拍只有 {nr:.0f}%。",
                    evidence=[k_far, k_near], n=far, confidence=_cap(_n_conf(far, 15, 6), pos_cap, *caps),
                    examples_ms=far_ex[:3])))

        # ---- 4. attack / defense state
        hold, aerr, conv = usable(f"{pid}.style.hold_pct"), usable(f"{pid}.style.attack_err_pct"), \
            usable(f"{pid}.style.convert_pct")
        if hold is not None and hold.n >= 8 and hold.value < 65:
            tactics.append(((65 - hold.value) * math.sqrt(hold.n) / 2, ScoutTactic(
                code="pressure", title=f"拿到机会就连续压 {pid}",
                how="对手起高球时果断杀、劈吊，杀完马上上网封挡网，不要给他喘息。",
                why=f"{pid} 被杀、被扑时只有 {hold.value:.0f}% 的球顶住了（{hold.n} 拍）。",
                evidence=[hold.key], n=hold.n, confidence=_cap(hold.confidence, *caps), examples_ms=[])))
        if aerr is not None and aerr.n >= 8 and aerr.value >= 30:
            tactics.append(((aerr.value - 30) * math.sqrt(aerr.n) / 2 + 5, ScoutTactic(
                code="let_attack", title=f"守住 {pid} 的进攻，等他失误",
                how="他进攻时先求稳：挡网、挑到底线两角，不急于反击，让他多打几拍。",
                why=f"{pid} 进攻后直接失误丢分占 {aerr.value:.0f}%（{aerr.n} 次进攻）。",
                evidence=[aerr.key], n=aerr.n, confidence=_cap(aerr.confidence, *caps), examples_ms=[])))
        if conv is not None and conv.n >= 8 and conv.value < 45:
            tactics.append(((45 - conv.value) * math.sqrt(conv.n) / 4, ScoutTactic(
                code="safe_lift", title=f"被动时可以放心给 {pid} 起高球过渡",
                how="自己被动时先挑高、挑远到底线，回到中心再找机会，不必冒险硬拼。",
                why=f"{pid} 拿到主动球时只有 {conv.value:.0f}% 选择进攻（{conv.n} 拍），对高球威胁不大。",
                evidence=[conv.key], n=conv.n, confidence=_cap(conv.confidence, *caps), examples_ms=[])))

        # ---- 5. rally length, serve / receive (needs entered winners)
        short, long_ = usable(f"{pid}.win.short_rate_pct"), usable(f"{pid}.win.long_rate_pct")
        if short is not None and long_ is not None and abs(short.value - long_.value) >= 20 \
                and min(short.n, long_.n) >= 3:
            conf = _cap(_n_conf(min(short.n, long_.n), 10, 4), *caps)
            if long_.value < short.value:
                tactics.append(((short.value - long_.value) / 2, ScoutTactic(
                    code="long_rallies", title=f"和 {pid} 拖长回合",
                    how="前几拍不急于下压，多拉开四角、控网，把回合打到 7 拍以上再找机会。",
                    why=f"{pid} 长回合胜率 {long_.value:.0f}%（{long_.n} 回合），短回合 {short.value:.0f}%（{short.n} 回合）。",
                    evidence=[long_.key, short.key], n=long_.n + short.n, confidence=conf, examples_ms=[])))
            else:
                tactics.append(((long_.value - short.value) / 2, ScoutTactic(
                    code="short_rallies", title=f"和 {pid} 打快、前三拍抢攻",
                    how="发接发后第一时间抢网、抢攻，不和他多拍相持。",
                    why=f"{pid} 短回合胜率 {short.value:.0f}%（{short.n} 回合），长回合 {long_.value:.0f}%（{long_.n} 回合）。",
                    evidence=[short.key, long_.key], n=long_.n + short.n, confidence=conf, examples_ms=[])))
        recv = usable(f"{pid}.win.receive_rate_pct")
        if recv is not None and recv.n >= 4 and recv.value < 40:
            tactics.append(((40 - recv.value) / 2, ScoutTactic(
                code="serve", title=f"发球回合打 {pid} 的接发",
                how="发球多变：短球贴网为主，偶尔突然发后场，第三拍准备抢攻。",
                why=f"{pid} 接发球回合只赢了 {recv.value:.0f}%（{recv.n} 回合）。",
                evidence=[recv.key], n=recv.n, confidence=_cap(_n_conf(recv.n, 10, 4), *caps), examples_ms=[])))

        # ---- 6. reply habits: be ready for their next ball
        st = style_of.get(pid)
        habit_items = []
        for h in (st.habits if st else [])[:3]:
            hint = READY.get(h.reply, "")
            if not hint or h.of < 5:
                continue
            k_h = f"{pid}.scout.reply.{h.after}"
            reply_counts = Counter(c.shot.shot_type for c in own if c.prev is not None and c.prev.shot_type == h.after)
            ds.append(_dist(k_h, f"你打{ZH.get(h.after, h.after)}后 {pid} 的回球",
                            {ZH.get(t, t): v for t, v in reply_counts.most_common()}, 15, 5))
            habit_items.append((h.pct / 40, ScoutTactic(
                code="habit", title=f"你{ZH.get(h.after, h.after)}后，{pid} 多半回{ZH.get(h.reply, h.reply)}",
                how=f"{hint}。",
                why=f"你打{ZH.get(h.after, h.after)}后，{pid} {h.pct:.0f}% 回{ZH.get(h.reply, h.reply)}（{h.n}/{h.of}）。",
                evidence=[k_h], n=h.of, confidence=_cap(_n_conf(h.of, 15, 6), *caps), examples_ms=[])))
        tactics += habit_items[:2]

        # ---- 7. against their style
        if st is not None and st.primary is not None and st.primary.score is not None \
                and st.primary.score >= CLEAR_STYLE and st.primary.code in COUNTER and st.status != "undeterminable":
            ev = [k for k in [f"{pid}.style.match_score", *st.evidence] if k in shared]
            ev = list(dict.fromkeys(ev))[:4]
            if ev:
                tactics.append((4.0, ScoutTactic(
                    code="style", title=f"针对 {pid} 的{st.primary.name}打法",
                    how=COUNTER[st.primary.code],
                    why=f"{pid} 的数据与{st.primary.name}原型吻合度 {st.primary.score:.0f}（满分 100）。",
                    evidence=ev, n=st.n_shots, confidence=_cap(st.confidence, *caps), examples_ms=[])))

        tactics.sort(key=lambda x: (x[1].confidence == "low", -x[0]))  # firm tactics first, then by strength
        chosen = [t for _, t in tactics][:MAX_TACTICS]
        firm = [t for t in chosen if t.confidence != "low"] or chosen
        conf = min((t.confidence for t in firm[:3]), key=_RANK.index) if chosen else "low"
        status, _ = _conf(len(own), 80, MIN_SHOTS)
        summary = (f"打 {pid}：" + "；".join(t.title for t in firm[:3]) + "。") if chosen else \
            f"{pid} 的数据里没有明显可利用的弱点。"
        if not metric_ok:
            notes.append("球场标定未确认，与站位和区域有关的战术只作参考")
        out.append(Scouting(target=pid, against=opp, side=side, summary=summary, tactics=chosen, n_shots=len(own),
                            status=status if conf != "low" else "needs_review", confidence=conf,
                            note="；".join(notes) or None))
        metrics[pid], dists[pid] = ms, ds
    return out, metrics, dists
