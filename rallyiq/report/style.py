"""Playing style: attack / neutral / defense states, tactical patterns and a style verdict per player.

Everything here is rules over the effective events (stroke types, hitters, times) and player tracks
(court positions). No model is trained: labelled style data is scarce, so each style is a *prototype*,
a weighted list of measurable features with the direction that is typical for it, and a player's
numbers are matched against every prototype (``score`` 0-100). See ``docs/playing-style.md`` for the
research behind the features and the reference ranges.

Per shot (both singles and doubles):

* **state** of the hitter, judged from the incoming ball (the user's 实时攻防状态):
  ``defense`` after an opponent smash / net kill; ``attack`` after an opponent lift, or a clear the
  opponent played while under pressure; ``neutral`` otherwise. A single 30 fps camera cannot see the
  contact height or balance directly, so the incoming stroke stands in for them.
* **attacking shot**: smash, net kill, push, or a drop / drive played from the attack state (劈吊、抢攻).
* **flight time**: time from this hit to the opponent's next hit (short = fast, flat or downward).
* **reach**: how far the opponent moved between this hit and their reply (调动).

Per player (singles) or team (doubles): shot mix, attack / defense shares, conversion of attack states,
consecutive attacks, early attacks (抢攻), build-then-attack (拉吊突击), counter after defending,
holding under pressure, attack success / errors, tempo (median flight, variation, sudden
accelerations), opponent movement, four-corner placement, straight vs cross, placement variety from
the same position (a deception proxy), forced replies; doubles add front-back vs side-by-side
formation and who plays the net.

Tactics (6, singles): 四角调动, 网前逼起球, 拉吊突击, 防守转攻, 重复落点陷阱, 节奏破坏, each counted
with how often the rally was won and how often the opponent was forced to defend.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import numpy as np

from ..contracts import (
    COURT_WIDTH_M,
    Distribution,
    EventsResult,
    Metric,
    PhaseChoice,
    PlayerStyle,
    PlayerTracks,
    ReplyHabit,
    ShotEvent,
    StyleScore,
    TacticUse,
    team_of,
)
from ..contracts.geometry import SINGLES_INSET
from .moments import FRONT_M, REAR_M, ZH, _Positions, own_half, timestamp

SERVES = {"short_serve", "long_serve"}
OPP_ATTACK = {"smash", "net_kill"}
ATTACK_SHOTS = {"smash", "net_kill", "push"}
NET = {"net_shot", "cross_net", "net_kill"}
FLAT = {"drive", "push"}
FORCED_REPLY = {"lift", "block"}

PHASE_ZH = {"attack": "主动", "neutral": "中立", "defense": "被动"}
# Recommended choices per state (the user's 实时战术决策 table).
RECOMMENDED = {
    "attack": {"smash", "drop", "net_kill", "push", "drive", "net_shot"},  # 杀球、点杀、劈吊、抢网、连续进攻
    "neutral": {"clear", "drop", "net_shot", "cross_net", "push", "drive"},  # 四角调动、直线控制、拉吊、节奏变化
    "defense": {"lift", "clear", "block", "drive", "net_shot", "cross_net"},  # 高质量解围、接杀挡网、防守反击
}
PHASE_ADVICE = {
    "attack": "杀球、点杀、劈吊、抢网、连续进攻",
    "neutral": "四角调动、直线控制、拉吊组织、节奏变化",
    "defense": "高质量解围（挑到底线两角）、接杀挡网、快速回位、防守反击",
}

MIN_SHOTS = 20  # classified non-serve shots below which no style is given
MIN_BASIS = 8  # a feature needs at least this many samples to count toward a style
REASON_MIN_N = 10  # ...and this many to be quoted as a reason for the verdict
MIN_CONFIRMED_TYPES = 0.5  # share of a player's shot types that must be confirmed for more than "low" confidence
SHRINK_K = 10  # small samples are pulled toward "typical": match = 0.5 + (m - 0.5) * n / (n + SHRINK_K)
MIN_ALTERNATION = 0.8  # singles: share of consecutive hits by different players below which hitters look wrong
UNUSABLE_ALTERNATION = 0.6  # below this the hitters are close to random: no style verdict at all
MIN_COVERAGE = 0.5  # share of a prototype's weight that must be measurable
CLOSE_POINTS = 5.0  # second style shown when within this many points
CLEAR_STYLE = 60.0  # below this the player is called balanced
SINGLES_W = COURT_WIDTH_M - 2 * SINGLES_INSET
LANE = SINGLES_W / 3


# --------------------------------------------------------------------------- features

# name -> (Chinese label, unit, lo, hi): lo maps to 0, hi to 1 when matching prototypes.
# Ranges are anchored on elite singles notational studies (docs/playing-style.md): e.g. smash ~10-14 %
# of strokes, clear 3-18 %, net 25-36 %, lift 19-27 %, drive 1-9 %, 5.4-10 shots per rally.
FEATURES: dict[str, tuple[str, str | None, float, float]] = {
    "smash_pct": ("杀球占比", "%", 5, 20),
    "net_pct": ("网前球占比（放网/勾对角/扑球）", "%", 15, 40),
    "clear_pct": ("高远球占比", "%", 5, 20),
    "drop_pct": ("吊球占比", "%", 5, 18),
    "clear_drop_pct": ("拉吊（高远球+吊球）占比", "%", 12, 35),
    "flat_pct": ("平抽推压（平抽+推球）占比", "%", 5, 30),
    "lift_block_pct": ("挑球+挡网占比", "%", 15, 35),
    "attack_pct": ("进攻性击球占比", "%", 10, 35),
    "defense_pct": ("被动状态下的击球占比", "%", 10, 35),
    "convert_pct": ("主动状态下选择进攻的比例", "%", 30, 80),
    "attack_run": ("连续进攻平均拍数", "拍", 1.0, 2.5),
    "early_attack_pct": ("前两拍就发起进攻的回合占比", "%", 15, 60),
    "build_attack_pct": ("先拉吊两拍再突击的回合占比", "%", 10, 50),
    "counter_pct": ("被动后两拍内转入进攻的比例", "%", 10, 50),
    "hold_pct": ("被动状态下顶住（没有直接丢分）的比例", "%", 60, 95),
    "attack_win_pct": ("进攻后两拍内得分的比例", "%", 15, 50),
    "attack_err_pct": ("进攻直接失误丢分的比例", "%", 5, 30),
    "flight_s": ("出球到对手击球的中位时间", "s", 0.7, 1.3),
    "tempo_cv": ("出球快慢变化（变异系数×100）", None, 25, 60),
    "accel_pct": ("突然加速的出球占比", "%", 5, 25),
    "opp_reach_m": ("对手接你的球平均移动距离", "m", 1.2, 2.8),
    "corner_pct": ("把对手调到四角的比例", "%", 30, 70),
    "straight_pct": ("边路出球中直线球的比例", "%", 40, 80),
    "variety": ("同一位置出球落点的多样性（0-100）", None, 40, 80),
    "forced_pct": ("逼对手挑球/挡网或直接得分的比例", "%", 10, 40),
    # doubles
    "fb_pct": ("前后站位（一前一后）的比例", "%", 30, 75),
    "fb_attack_pct": ("主动时前后站位的比例", "%", 40, 85),
    "sbs_defense_pct": ("被动时左右站位的比例", "%", 40, 85),
    "fixed_front_pct": ("前后站位时同一人在前的比例", "%", 50, 90),
    "smash_conc_pct": ("杀球集中在一人的比例", "%", 50, 90),
    "front_kill_pct": ("网前封扑/推压占比", "%", 3, 15),
}
# Existing match metrics a prototype may use: name -> (metric key template, lo, hi).
SHARED = {
    "rally_len": ("match.rally_len.mean", 5, 12),
    "long_win": ("{s}.win.long_rate_pct", 30, 70),
    "own_err": ("{s}.win.lost_own_last_shot_pct", 30, 70),
}
DOUBLES_RANGES = {"flat_pct": (15, 45), "flight_s": (0.5, 1.0), "smash_pct": (10, 30)}


@dataclass(frozen=True)
class Prototype:
    code: str
    name: str
    name_en: str
    representatives: tuple[str, ...]
    weights: dict[str, float]  # feature -> weight; negative = a low value is typical
    study: str = ""  # what to learn from the representative (学习重点)


SINGLES_STYLES: tuple[Prototype, ...] = (
    Prototype("power_attack", "强攻压制型", "Power Attacking", ("安赛龙 Viktor Axelsen", "李梓嘉 Lee Zii Jia"),
              {"smash_pct": 3, "attack_pct": 2, "attack_run": 2, "convert_pct": 2, "forced_pct": 1, "defense_pct": -1},
              "高点进攻、后场压制、连续杀球组织"),
    Prototype("speed_attack", "速度抢攻型", "Speed Attacking", ("李宗伟 Lee Chong Wei", "马林 Carolina Marín"),
              {"early_attack_pct": 3, "flight_s": -2, "flat_pct": 2, "attack_pct": 1, "net_pct": 1},
              "快速步法、提前启动、连续进攻、快速回位"),
    Prototype("rally_attack", "拉吊突击型", "Rally-to-Attack", ("林丹 Lin Dan", "石宇奇 Shi Yuqi"),
              {"clear_drop_pct": 3, "build_attack_pct": 3, "opp_reach_m": 1, "attack_win_pct": 1,
               "early_attack_pct": -1},
              "拉吊突击、进攻时机选择、节奏控制"),
    Prototype("control", "控制组织型", "Control-Based", ("桃田贤斗 Kento Momota", "陈雨菲 Chen Yufei"),
              {"straight_pct": 3, "net_pct": 2, "forced_pct": 2, "own_err": -2, "smash_pct": -1},
              "直线控制、网前技术、落点精准、限制对手回球选择"),
    Prototype("defensive_counter", "防守反击型", "Defensive Counterattack",
              ("谌龙 Chen Long", "昆拉武特 Kunlavut Vitidsarn", "安洗莹 An Se Young"),
              {"defense_pct": 2, "hold_pct": 3, "counter_pct": 3, "lift_block_pct": 1},
              "稳定防守、接杀变化、反击时机、长回合管理"),
    Prototype("endurance", "多拍消耗型", "Endurance / Rally-Based",
              ("奈良冈功大 Kodai Naraoka", "奥原希望 Nozomi Okuhara", "山口茜 Akane Yamaguchi"),
              {"rally_len": 3, "long_win": 2, "own_err": -2, "opp_reach_m": 1, "attack_pct": -1},
              "长回合相持、四角调动、减少主动失误"),
    Prototype("deceptive", "技术欺骗型", "Deceptive / Technical", ("戴资颖 Tai Tzu Ying", "因达农 Ratchanok Intanon"),
              {"variety": 3, "forced_pct": 2, "opp_reach_m": 2, "net_pct": 1},
              "假动作、隐蔽性击球、出球方向变化"),
    Prototype("tempo", "节奏变化型", "Tempo Variation", ("安东森 Anders Antonsen", "林丹 Lin Dan", "石宇奇 Shi Yuqi"),
              {"tempo_cv": 3, "accel_pct": 3, "clear_drop_pct": 1},
              "比赛快慢控制、相持中突然加速"),
)

DOUBLES_SYSTEMS: tuple[Prototype, ...] = (
    Prototype("rear_attack", "后场强攻型", "Rear-Court Attacking",
              ("兰基雷迪/谢提 Satwiksairaj Rankireddy / Chirag Shetty",),
              {"smash_pct": 3, "attack_run": 2, "fb_attack_pct": 2, "smash_conc_pct": 1}),
    Prototype("fast_drive", "平抽快挡型", "Fast Drive Exchange", ("吉迪恩/苏卡穆约 Marcus Gideon / Kevin Sukamuljo",),
              {"flat_pct": 3, "flight_s": -3, "early_attack_pct": 1}),
    Prototype("front_intercept", "网前预判控制型", "Front-Court Interception",
              ("阿山/塞蒂亚万 Mohammad Ahsan / Hendra Setiawan",),
              {"front_kill_pct": 3, "net_pct": 2, "fb_pct": 1}),
    Prototype("defensive_rotation", "防守转换型", "Defensive Rotation",
              ("松友美佐纪/高桥礼华 Misaki Matsutomo / Ayaka Takahashi",),
              {"hold_pct": 3, "counter_pct": 2, "defense_pct": 2, "sbs_defense_pct": 1}),
    Prototype("front_back_mixed", "混双前封后攻型", "Front-Back Mixed Doubles", ("郑思维/黄雅琼 Zheng Siwei / Huang Yaqiong",),
              {"fixed_front_pct": 3, "fb_pct": 2, "smash_pct": 1}),
)

TACTICS = {
    "space_control": "四角调动",
    "net_to_attack": "网前逼起球",
    "rally_to_attack": "拉吊突击",
    "defence_to_attack": "防守转攻",
    "pattern_breaking": "重复落点陷阱",
    "tempo_disruption": "节奏破坏",
}

# How to play against each style (the user's tactics, applied to the opponent's tendencies).
COUNTER = {
    "power_attack": "对手靠高点重杀压制：少给半高球，多打平高和网前控制，不让对手轻松到高点；接杀以挡网和挡对角为主，逼对手上网后再推后场空当。",
    "speed_attack": "对手速度快、喜欢抢攻：用又高又远的高远球放慢节奏，网前球放贴网，减少平抽对冲；把球拉到四角，让对手每拍多跑。",
    "rally_attack": "对手先拉吊再突击：别让对手轻松控制底线，多用平高、推后场和平抽压低弧线，不给对手组织空间；对手吊球时提前上网封网。",
    "control": "对手靠直线控制和网前限制回球路线：多变斜线和节奏，敢于抢网，打乱对手的组织；减少给对手直线控制的中路球。",
    "defensive_counter": "对手防守好、等你失位：不要盲目连续重杀，先用拉吊让对手失去平衡再进攻，多用点杀和劈吊打追身或空当；杀完迅速上网封挡网。",
    "endurance": "对手擅长多拍消耗：缩短回合，用网前逼起球和第三拍抢攻创造早期机会，同时控制自己的主动失误。",
    "deceptive": "对手出手隐蔽、方向多变：重心不要提前启动，回中后做好分腿垫步再判断；把对手压在被动位置，让他没时间做假动作。",
    "tempo": "对手善于变节奏：保持自己的回位和步法节奏，接突然加速的球以稳为主（挡、挑到底线），不要被带快而失误。",
    "rear_attack": "对方靠后场连续杀球：多平抽、推压中场，不给后场队员起跳机会；挑球一定挑到底线两角，接杀挡网逼对方前场队员接球。",
    "fast_drive": "对方擅长平抽快挡：适当挑高放慢节奏，打对方两人中间和追身，避免和对方硬拼平抽。",
    "front_intercept": "对方网前拦截强：减少平飞到网前的球，多用推后场和挑底线越过网前队员，让后场队员多跑。",
    "defensive_rotation": "对方防守轮转稳：杀球打中路和追身，杀吊结合，前场队员积极封网，逼对方挑短球。",
    "front_back_mixed": "对方固定前封后攻：把球推到后场，让前场队员被迫后退；挑球挑到对角，打乱对方前后分工。",
}

HABIT_HINT = {
    "net_shot": "可以提前上网封网", "cross_net": "可以提前上网准备封勾对角", "lift": "回中后提前准备后场进攻",
    "clear": "回中后提前后退，准备在后场高点出手", "block": "进攻后马上上网准备扑或推", "drive": "注意封住中路平抽",
    "push": "注意推后场后的快速后退", "smash": "提前降低重心准备接杀", "drop": "提前准备上网接吊",
    "net_kill": "网前球要放得更贴网，避免被扑", "short_serve": "", "long_serve": "",
}


# --------------------------------------------------------------------------- per-shot context


@dataclass
class _Ctx:
    shot: ShotEvent
    idx: int  # position in the rally
    rally: int  # rally number (0-based)
    side: str  # P1/P2 (singles) or A/B (doubles)
    phase: str  # hitter's state: attack / neutral / defense
    attack: bool
    prev: ShotEvent | None  # the opponent shot this one answers
    next: ShotEvent | None
    is_last: bool
    rally_winner: str | None
    flight: float | None  # s from this hit to the opponent's next hit
    pos: tuple[float, float] | None  # hitter, court metres
    reach: float | None  # opponent's move to reply
    next_pos: tuple[float, float] | None  # where the opponent hit the reply
    mate_pos: tuple[float, float] | None = None  # doubles: partner at this hit


def phase_of(incoming: str | None, incoming_phase: str | None, own_contact: str | None = None) -> str:
    """Hitter's state from the ball they receive. ``own_contact`` ("high" / "low", from the stroke
    detail) is where the hitter actually met it: a lift that is only reached late and below the net
    is not an attacking chance (amateurs often take a lift low at the back), so it stays neutral."""
    if incoming in OPP_ATTACK:
        return "defense"
    if incoming == "lift" or (incoming == "clear" and incoming_phase == "defense"):
        return "neutral" if own_contact == "low" else "attack"
    return "neutral"


def contact_of(s: ShotEvent) -> str | None:
    """"high" / "low" contact from the stroke detail (model or human), None when unknown."""
    d = s.stroke.detail if s.stroke is not None else None
    return d.contact.value if d is not None and d.contact is not None else None


def type_confirmed(s: ShotEvent) -> bool:
    """The shot type was set by a person, or the classifier was confident enough to mark it ok
    (a type with no stroke record at all came from outside the classifier and is taken as given)."""
    return s.stroke is None or s.stroke.source == "human" or s.stroke.status == "ok"


def is_attack(shot_type: str | None, phase: str) -> bool:
    return shot_type in ATTACK_SHOTS or (shot_type in ("drop", "drive") and phase == "attack")


def shot_phases(events: EventsResult) -> dict[str, str]:
    """shot_id -> hitter's state (attack / neutral / defense) for every shot with a known hitter."""
    shots_by_id = {s.shot_id: s for s in events.shots}
    out: dict[str, str] = {}
    for rally in events.rallies:
        seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        prev, prev_phase = None, None
        for s in seq:
            ph = phase_of(prev.shot_type if prev is not None else None, prev_phase, contact_of(s))
            if s.hitter is not None:
                out[s.shot_id] = ph
            prev, prev_phase = s, ph
    return out


def _contexts(events: EventsResult, players: PlayerTracks | None, fps: float) -> list[list[_Ctx]]:
    doubles = events.match_format == "doubles"
    side_of: Callable[[str | None], str | None] = (lambda p: team_of(p) if p else None) if doubles else (lambda p: p)
    pos = _Positions(players, fps) if players is not None else None
    partner = {"P1": "P3", "P3": "P1", "P2": "P4", "P4": "P2"}
    shots_by_id = {s.shot_id: s for s in events.shots}
    out: list[list[_Ctx]] = []
    for rno, rally in enumerate(sorted(events.rallies, key=lambda r: r.start_frame)):
        seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        winner = rally.winner
        ctxs: list[_Ctx] = []
        prev_phase = None
        for i, s in enumerate(seq):
            prev = seq[i - 1] if i > 0 else None
            phase = phase_of(prev.shot_type if prev is not None else None, prev_phase, contact_of(s))
            prev_phase = phase
            side = side_of(s.hitter)
            if side is None:
                continue
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if nxt is not None and side_of(nxt.hitter) == side:
                nxt = None  # same side twice: a missed or mis-assigned hit, not a reply
            flight = None
            if nxt is not None:
                dt = (nxt.t_ms - s.t_ms) / 1000
                flight = dt if 0.15 <= dt <= 3.0 else None
            hpos = _xy(s, pos)
            next_pos = _xy(nxt, pos) if nxt is not None else None
            reach = None
            if nxt is not None and pos is not None and next_pos is not None and nxt.hitter:
                before = pos.at(nxt.hitter, s.frame)
                if before is not None:
                    reach = math.dist(before, next_pos)
            mate = pos.at(partner[s.hitter], s.frame) if doubles and pos is not None and s.hitter in partner else None
            ctxs.append(_Ctx(shot=s, idx=i, rally=rno, side=side, phase=phase,
                             attack=is_attack(s.shot_type, phase), prev=prev, next=nxt, is_last=i == len(seq) - 1,
                             rally_winner=winner, flight=flight, pos=hpos, reach=reach, next_pos=next_pos,
                             mate_pos=mate))
        out.append(ctxs)
    return out


def _xy(s: ShotEvent | None, pos: _Positions | None) -> tuple[float, float] | None:
    if s is None:
        return None
    if s.hitter_court is not None:
        return float(s.hitter_court[0]), float(s.hitter_court[1])
    return pos.at(s.hitter, s.frame) if pos is not None and s.hitter else None


# --------------------------------------------------------------------------- features


def _pct(a: int, b: int) -> float | None:
    return 100.0 * a / b if b else None


def _lane(x_court: float) -> int:
    """0 / 1 / 2 = left / middle / right third of the singles court (absolute court x)."""
    return 0 if x_court < SINGLES_INSET + LANE else 1 if x_court < SINGLES_INSET + 2 * LANE else 2


def _band(depth: float) -> int:
    return 0 if depth < REAR_M else 1 if depth < FRONT_M else 2


def _entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values() if v) if n else 0.0


def features(rallies: list[list[_Ctx]], subject: str, doubles: bool = False) -> dict[str, tuple[float | None, int]]:
    """feature -> (value, sample size) for one player (singles) or team (doubles)."""
    mine = [c for r in rallies for c in r if c.side == subject and c.shot.shot_type]
    own = [c for c in mine if c.shot.shot_type not in SERVES]
    n = len(own)
    types = Counter(c.shot.shot_type for c in own)
    f: dict[str, tuple[float | None, int]] = {}

    def share(*names: str) -> tuple[float | None, int]:
        return _pct(sum(types[t] for t in names), n), n

    f["smash_pct"] = share("smash")
    f["net_pct"] = share(*NET)
    f["clear_pct"] = share("clear")
    f["drop_pct"] = share("drop")
    f["clear_drop_pct"] = share("clear", "drop")
    f["flat_pct"] = share(*FLAT)
    f["lift_block_pct"] = share("lift", "block")
    f["attack_pct"] = (_pct(sum(c.attack for c in own), n), n)
    f["defense_pct"] = (_pct(sum(c.phase == "defense" for c in own), n), n)
    att_state = [c for c in own if c.phase == "attack"]
    f["convert_pct"] = (_pct(sum(c.attack for c in att_state), len(att_state)), len(att_state))

    runs, early_hits, early_n, build_hits, build_n = [], 0, 0, 0, 0
    counter_hits, counter_n, hold_hits, hold_n = 0, 0, 0, 0
    att_win, att_err, att_n = 0, 0, 0
    for r in rallies:
        seq = [c for c in r if c.side == subject and c.shot.shot_type and c.shot.shot_type not in SERVES]
        if len(seq) >= 2:
            early_n += 1
            early_hits += any(c.attack for c in seq[:2])
        if len(seq) >= 3:  # rallies long enough to build: did the player attack right after two clears/drops?
            build_n += 1
            build_hits += any(c.attack and not seq[k - 1].attack and not seq[k - 2].attack
                              and all(p.shot.shot_type in ("clear", "drop") for p in seq[k - 2:k])
                              for k, c in enumerate(seq) if k >= 2)
        run = 0
        for k, c in enumerate(seq):
            if c.attack:
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 0
            if c.phase == "defense":
                if not c.is_last:
                    counter_n += 1
                    counter_hits += any(x.attack for x in seq[k + 1:k + 3])
                if not c.is_last or c.rally_winner is not None:
                    hold_n += 1
                    hold_hits += not c.is_last or c.rally_winner == subject
            if c.attack and c.rally_winner is not None:
                att_n += 1
                last_idx = r[-1].idx if r else c.idx
                att_win += c.rally_winner == subject and last_idx - c.idx <= 2
                att_err += c.is_last and c.rally_winner != subject
        if run:
            runs.append(run)
    f["attack_run"] = (float(np.mean(runs)) if runs else None, len(runs))
    f["early_attack_pct"] = (_pct(early_hits, early_n), early_n)
    f["build_attack_pct"] = (_pct(build_hits, build_n), build_n)
    f["counter_pct"] = (_pct(counter_hits, counter_n), counter_n)
    f["hold_pct"] = (_pct(hold_hits, hold_n), hold_n)
    f["attack_win_pct"] = (_pct(att_win, att_n), att_n)
    f["attack_err_pct"] = (_pct(att_err, att_n), att_n)

    # tempo
    flights = [c.flight for c in mine if c.flight is not None]
    f["flight_s"] = (float(median(flights)) if flights else None, len(flights))
    f["tempo_cv"] = (100 * float(np.std(flights) / np.mean(flights)) if len(flights) >= 2 else None, len(flights))
    acc_hits, acc_n = 0, 0
    for r in rallies:
        gaps: list[float] = []
        for c in r:
            if c.side == subject and c.flight is not None and len(gaps) >= 3:
                acc_n += 1
                base = median(gaps[-4:])
                acc_hits += base >= 0.8 and c.flight < 0.65 * base
            if c.flight is not None:
                gaps.append(c.flight)
    f["accel_pct"] = (_pct(acc_hits, acc_n), acc_n)

    # placement
    reach = [c.reach for c in own if c.reach is not None]
    f["opp_reach_m"] = (float(np.mean(reach)) if reach else None, len(reach))
    placed = [c for c in own if c.next_pos is not None]
    corners = 0
    straight_hits, straight_n = 0, 0
    by_band: dict[int, Counter] = defaultdict(Counter)
    for c in placed:
        lat, dep = own_half(*c.next_pos)
        lane = _lane(lat)
        corners += lane != 1 and _band(dep) != 1
        if c.pos is not None:
            by_band[_band(own_half(*c.pos)[1])][(_band(dep), lane)] += 1
            if _lane(c.pos[0]) != 1:
                straight_n += 1
                straight_hits += abs(c.next_pos[0] - c.pos[0]) < LANE
    f["corner_pct"] = (_pct(corners, len(placed)), len(placed))
    f["straight_pct"] = (_pct(straight_hits, straight_n), straight_n)
    nv = sum(sum(c.values()) for c in by_band.values())
    var = (sum(sum(c.values()) * _entropy(c) for c in by_band.values()) / nv / math.log(9) * 100) if nv >= 20 else None
    f["variety"] = (var, nv)
    forced_n = [c for c in own if not c.is_last or c.rally_winner is not None]
    forced = sum((c.next is not None and c.next.shot_type in FORCED_REPLY)
                 or (c.is_last and c.rally_winner == subject) for c in forced_n)
    f["forced_pct"] = (_pct(forced, len(forced_n)), len(forced_n))

    if doubles:
        f.update(_doubles_features(own))
    return f


# Shots behind a share feature, so a reason can point at moments to re-watch.
_EXAMPLE_TYPES = {"smash_pct": {"smash"}, "net_pct": NET, "clear_pct": {"clear"}, "drop_pct": {"drop"},
                  "clear_drop_pct": {"clear", "drop"}, "flat_pct": FLAT, "lift_block_pct": {"lift", "block"}}


def feature_examples(rallies: list[list[_Ctx]], subject: str, limit: int = 3) -> dict[str, list[int]]:
    """feature -> times (ms) of a few shots that make it up (share features and attack / defense shares)."""
    own = [c for r in rallies for c in r if c.side == subject and c.shot.shot_type
           and c.shot.shot_type not in SERVES]
    out: dict[str, list[int]] = {}
    for name, kinds in _EXAMPLE_TYPES.items():
        out[name] = [c.shot.t_ms for c in own if c.shot.shot_type in kinds][:limit]
    out["attack_pct"] = [c.shot.t_ms for c in own if c.attack][:limit]
    out["defense_pct"] = [c.shot.t_ms for c in own if c.phase == "defense"][:limit]
    out["convert_pct"] = [c.shot.t_ms for c in own if c.phase == "attack" and not c.attack][:limit]
    return {k: v for k, v in out.items() if v}


def _doubles_features(own: list[_Ctx]) -> dict[str, tuple[float | None, int]]:
    form = []  # (front_back, side_by_side, phase, front player)
    for c in own:
        if c.pos is None or c.mate_pos is None:
            continue
        _, d1 = own_half(*c.pos)
        _, d2 = own_half(*c.mate_pos)
        gap = abs(d1 - d2)
        front = c.shot.hitter if d1 > d2 else {"P1": "P3", "P3": "P1", "P2": "P4", "P4": "P2"}.get(c.shot.hitter or "")
        form.append((gap >= 1.5, gap < 1.0, c.phase, front))
    f: dict[str, tuple[float | None, int]] = {}
    f["fb_pct"] = (_pct(sum(x[0] for x in form), len(form)), len(form))
    att = [x for x in form if x[2] == "attack"]
    f["fb_attack_pct"] = (_pct(sum(x[0] for x in att), len(att)), len(att))
    dfn = [x for x in form if x[2] == "defense"]
    f["sbs_defense_pct"] = (_pct(sum(x[1] for x in dfn), len(dfn)), len(dfn))
    fronts = Counter(x[3] for x in form if x[0] and x[3])
    nf = sum(fronts.values())
    f["fixed_front_pct"] = (_pct(fronts.most_common(1)[0][1], nf) if nf else None, nf)
    smashers = Counter(c.shot.hitter for c in own if c.shot.shot_type == "smash")
    ns = sum(smashers.values())
    f["smash_conc_pct"] = (_pct(smashers.most_common(1)[0][1], ns) if ns else None, ns)
    fk = sum(c.shot.shot_type in ("net_kill", "push") and c.pos is not None and own_half(*c.pos)[1] >= FRONT_M
             for c in own)
    f["front_kill_pct"] = (_pct(fk, len(own)), len(own))
    return f


# --------------------------------------------------------------------------- tactics


def tactics(rallies: list[list[_Ctx]], subject: str) -> list[TacticUse]:
    uses: dict[str, list[_Ctx]] = {k: [] for k in TACTICS}
    for r in rallies:
        seq = [c for c in r if c.side == subject and c.shot.shot_type]
        for k, c in enumerate(seq):
            t = c.shot.shot_type
            before = seq[max(0, k - 3):k]
            # 四角调动: three own shots in a row that each made the opponent move far
            if k >= 2 and all(x.reach is not None and x.reach >= 1.8 for x in seq[k - 2:k + 1]) and not (
                    k >= 3 and seq[k - 3].reach is not None and seq[k - 3].reach >= 1.8):
                uses["space_control"].append(c)
            # 网前逼起球: own net shot -> opponent lift -> own attack
            if k >= 1 and c.attack and before and before[-1].shot.shot_type in ("net_shot", "cross_net") \
                    and before[-1].next is not None and before[-1].next.shot_type == "lift":
                uses["net_to_attack"].append(c)
            # 拉吊突击: two own clears/drops (at least one of each kind in the last three) then an attack
            if c.attack and t in ("smash", "drop", "net_kill") and len(before) >= 2:
                kinds = {x.shot.shot_type for x in before}
                if {"clear", "drop"} <= kinds and not any(x.attack for x in before[-2:]):
                    uses["rally_to_attack"].append(c)
            # 防守转攻: an own defense shot followed by an own attack within two own shots
            if c.attack and any(x.phase == "defense" for x in seq[max(0, k - 2):k]):
                uses["defence_to_attack"].append(c)
            # 重复落点陷阱: two straight shots from the side lanes, then a cross-court one
            if k >= 2 and all(_is_straight(x) is True for x in seq[k - 2:k]) and _is_straight(c) is False:
                uses["pattern_breaking"].append(c)
            # 节奏破坏: sudden acceleration after slow exchanges
            if c.flight is not None:
                gaps = [x.flight for x in r if x.idx < c.idx and x.flight is not None][-4:]
                if len(gaps) >= 3 and median(gaps) >= 0.8 and c.flight < 0.65 * median(gaps):
                    uses["tempo_disruption"].append(c)
    out = []
    for code, cs in uses.items():
        decided = [c for c in cs if c.rally_winner is not None]
        won = sum(c.rally_winner == subject for c in decided)
        fb = [c for c in cs if not c.is_last or c.rally_winner is not None]
        forced = sum((c.next is not None and c.next.shot_type in FORCED_REPLY)
                     or (c.is_last and c.rally_winner == subject) for c in fb)
        out.append(TacticUse(code=code, name=TACTICS[code], uses=len(cs), decided=len(decided), won=won,
                             success_pct=_round(_pct(won, len(decided))), forced_pct=_round(_pct(forced, len(fb))),
                             examples_ms=[c.shot.t_ms for c in cs[:3]]))
    return out


def _is_straight(c: _Ctx) -> bool | None:
    if c.pos is None or c.next_pos is None or _lane(c.pos[0]) == 1:
        return None
    return abs(c.next_pos[0] - c.pos[0]) < LANE


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


# --------------------------------------------------------------------------- habits & phases


def habits(rallies: list[list[_Ctx]], subject: str, min_of: int = 4, min_pct: float = 40.0) -> list[ReplyHabit]:
    """After the opponent plays X, the most common reply of ``subject`` (next-shot prediction)."""
    pairs: dict[str, Counter] = defaultdict(Counter)
    for r in rallies:
        for c in r:
            if c.side == subject and c.prev is not None and c.prev.shot_type and c.shot.shot_type:
                pairs[c.prev.shot_type][c.shot.shot_type] += 1
    out = []
    for after, cnt in pairs.items():
        of = sum(cnt.values())
        reply, n = cnt.most_common(1)[0]
        if of >= min_of and 100 * n / of >= min_pct:
            out.append(ReplyHabit(after=after, reply=reply, n=n, of=of, pct=round(100 * n / of, 1)))
    out.sort(key=lambda h: (-h.n, -h.pct))
    return out[:6]


def phase_choices(rallies: list[list[_Ctx]], subject: str) -> list[PhaseChoice]:
    out = []
    for ph in ("attack", "neutral", "defense"):
        cs = [c for r in rallies for c in r if c.side == subject and c.phase == ph and c.shot.shot_type
              and c.shot.shot_type not in SERVES]
        counts = Counter(c.shot.shot_type for c in cs)
        rec = sum(v for k, v in counts.items() if k in RECOMMENDED[ph])
        out.append(PhaseChoice(phase=ph, shots=len(cs), counts=dict(counts.most_common()),
                               recommended_pct=_round(_pct(rec, len(cs)))))
    return out


# --------------------------------------------------------------------------- matching


def _range(name: str, doubles: bool) -> tuple[float, float]:
    if doubles and name in DOUBLES_RANGES:
        return DOUBLES_RANGES[name]
    if name in SHARED:
        return SHARED[name][1], SHARED[name][2]
    return FEATURES[name][2], FEATURES[name][3]


def _fmt(v: float, unit: str | None) -> str:
    s = f"{v:.0f}" if abs(v) >= 10 or abs(v - round(v)) < 0.05 else f"{v:.1f}"
    return f"{s}%" if unit == "%" else f"{s} {unit}" if unit else s


def match_prototype(p: Prototype, values: dict[str, tuple[float | None, int, str, str | None]],
                    doubles: bool = False, examples: dict[str, list[int]] | None = None) -> StyleScore:
    """``values``: feature -> (value, n, label, unit). Weighted mean of how typical each feature is.
    Reasons quote only features measured on at least ``REASON_MIN_N`` samples, with the sample size
    and (``examples``: feature -> shot times) a few moments to re-watch."""
    total_w = sum(abs(w) for w in p.weights.values())
    got, wsum, parts = 0.0, 0.0, []
    # Unmeasurable features count as "typical" (0.5): a prototype is not matched on its easy features alone.
    for name, w in p.weights.items():
        v, n, label, unit = values.get(name, (None, 0, name, None))
        if v is None or n < MIN_BASIS:
            continue
        lo, hi = _range(name, doubles)
        x = min(1.0, max(0.0, (v - lo) / (hi - lo)))
        m = x if w > 0 else 1 - x
        m = 0.5 + (m - 0.5) * n / (n + SHRINK_K)  # a handful of samples cannot make a style
        got += abs(w) * m
        wsum += abs(w)
        if m >= 0.6 and n >= REASON_MIN_N:
            ex = (examples or {}).get(name) if w > 0 else None
            see = f"；回看 {'、'.join(timestamp(t) for t in ex)}" if ex else ""
            parts.append((abs(w) * m, f"{label} {_fmt(v, unit)}（{'偏高' if w > 0 else '偏低'}，{n} 个样本{see}）"))
    coverage = wsum / total_w if total_w else 0.0
    score = round(100 * (got + 0.5 * (total_w - wsum)) / total_w, 1) if wsum and coverage >= MIN_COVERAGE else None
    parts.sort(key=lambda t: -t[0])
    return StyleScore(code=p.code, name=p.name, name_en=p.name_en, score=score, coverage=round(coverage, 2),
                      reasons=[t for _, t in parts[:3]], representatives=list(p.representatives))


def _verdict(scores: list[StyleScore], n: int) -> tuple[StyleScore | None, StyleScore | None, str]:
    ranked = [s for s in scores if s.score is not None]
    if n < MIN_SHOTS or not ranked:
        return None, None, "证据不足"
    first = ranked[0]
    second = ranked[1] if len(ranked) > 1 and first.score - ranked[1].score <= CLOSE_POINTS else None
    if first.score < CLEAR_STYLE:
        return first, second, f"均衡型（没有明显倾向，最接近{first.name}）"
    return first, second, first.name + (f"（兼{second.name}）" if second else "")


# --------------------------------------------------------------------------- advice


def _advice(subject: str, kind: str, primary: StyleScore | None, feats: dict, phases: list[PhaseChoice],
            used: list[TacticUse], opp: PlayerStyle | None, opp_habits: list[ReplyHabit],
            protos: dict[str, Prototype]) -> list[str]:
    who = f"{subject} " if kind == "singles" else f"{subject} 队"
    out: list[str] = []

    def v(name: str) -> tuple[float | None, int]:
        return feats.get(name, (None, 0))

    # 1. execution under each state
    by = {p.phase: p for p in phases}
    att = by.get("attack")
    conv, nconv = v("convert_pct")
    if att and conv is not None and nconv >= 8 and conv < 45:
        wasted = [ZH.get(k, k) for k in att.counts if k not in RECOMMENDED["attack"]][:2]
        played = f"常打{'、'.join(wasted)}，" if wasted else ""
        if primary is not None and primary.code in ("rally_attack", "endurance", "control") \
                and (primary.score or 0) >= CLEAR_STYLE:
            out.append(f"{who}主动状态（对手挑球或被动起高球）下只有 {conv:.0f}% 选择进攻（{nconv} 拍），{played}"
                       f"符合{primary.name}先组织的思路，但对手回球一旦变短、自己站位到位，要果断下压，不要把机会拉没了。")
        else:
            out.append(f"{who}主动状态（对手挑球或被动起高球）下只有 {conv:.0f}% 选择进攻（{nconv} 拍），{played}"
                       f"错过了进攻窗口；主动时优先{PHASE_ADVICE['attack']}。")
    hold, nhold = v("hold_pct")
    if hold is not None and nhold >= 8 and hold < 65:
        out.append(f"{who}被动状态下只有 {hold:.0f}% 的球顶住了（{nhold} 拍）；被动时先求{PHASE_ADVICE['defense']}。")
    counter, ncounter = v("counter_pct")
    if counter is not None and ncounter >= 8 and counter < 15:
        out.append(f"{who}防守后两拍内转入进攻只有 {counter:.0f}%（{ncounter} 次）；接杀可挡对角网前或平抽到空当，"
                   "把对手调出位置后再反击。")
    err, nerr = v("attack_err_pct")
    if err is not None and nerr >= 8 and err >= 30:
        out.append(f"{who}进攻后直接失误丢分占 {err:.0f}%（{nerr} 次进攻）；进攻要在击球点高、站稳时出手，"
                   "后场被动时先过渡。")
    neu = by.get("neutral")
    if neu and neu.shots >= 10 and neu.counts.get("lift", 0) / neu.shots >= 0.25:
        out.append(f"{who}中立状态下挑球占 {100 * neu.counts['lift'] / neu.shots:.0f}%（{neu.shots} 拍），"
                   f"容易把主动权交给对手；中立时多用{PHASE_ADVICE['neutral']}。")

    # 2. tactics that worked / did not
    tried = [t for t in used if t.decided >= 3]
    if tried:
        best = max(tried, key=lambda t: (t.won / t.decided, t.uses))
        if best.won / best.decided >= 0.6:
            out.append(f"{who}最有效的套路是{best.name}：用了 {best.uses} 次，有胜负的 {best.decided} 个回合赢了 "
                       f"{best.won} 个，可以多用。")
        worst = min(tried, key=lambda t: (t.won / t.decided, -t.uses))
        if worst is not best and worst.won / worst.decided <= 0.35:
            out.append(f"{who}的{worst.name}用了 {worst.uses} 次，有胜负的 {worst.decided} 个回合只赢了 {worst.won} 个，"
                       "执行质量或出手时机需要调整。")

    # 3. style reference
    if primary is not None and primary.score is not None and primary.score >= CLEAR_STYLE:
        p = protos.get(primary.code)
        if p is not None and p.study:
            out.append(f"{who}的打法接近{p.name}，可以重点看{p.representatives[0].split(' ')[0]}的比赛，学习{p.study}。")

    # 4. against this opponent
    if opp is not None and opp.primary is not None and opp.primary.score is not None \
            and opp.primary.score >= CLEAR_STYLE and opp.primary.code in COUNTER:
        out.append(f"对手接近{opp.primary.name}。{COUNTER[opp.primary.code]}")
    for h in opp_habits[:2]:
        hint = HABIT_HINT.get(h.reply, "")
        if hint:
            out.append(f"对手接你的{ZH.get(h.after, h.after)}时 {h.pct:.0f}% 回{ZH.get(h.reply, h.reply)}"
                       f"（{h.n}/{h.of}），{hint}。")
    return out


# --------------------------------------------------------------------------- entry point


@dataclass
class StyleResult:
    styles: list[PlayerStyle]
    metrics: dict[str, list[Metric]]  # subject -> feature metrics
    distributions: dict[str, list[Distribution]]


def analyze_styles(events: EventsResult, players: PlayerTracks | None, fps: float, metric_ok: bool,
                   shared: dict[str, Metric]) -> StyleResult:
    """Style per player (singles) or team (doubles). ``shared`` holds existing metrics by key
    (``match.rally_len.mean``, ``Pn.win.*``) that some prototypes use."""
    from .stats import _conf, _dist, _metric  # stats imports this module

    doubles = events.match_format == "doubles"
    subjects = ("A", "B") if doubles else ("P1", "P2")
    protos = DOUBLES_SYSTEMS if doubles else SINGLES_STYLES
    by_code = {p.code: p for p in protos}
    rallies = _contexts(events, players, fps)
    alternation = hitter_alternation(events) if not doubles else None
    bad_hitters = alternation is not None and alternation[1] >= 20 and alternation[0] < MIN_ALTERNATION
    no_hitters = bad_hitters and alternation[0] < UNUSABLE_ALTERNATION
    geo = {"opp_reach_m", "corner_pct", "straight_pct", "variety", "fb_pct", "fb_attack_pct", "sbs_defense_pct",
           "fixed_front_pct", "front_kill_pct"}
    metrics: dict[str, list[Metric]] = {}
    dists: dict[str, list[Distribution]] = {}
    pre: dict[str, dict] = {}
    for s in subjects:
        feats = features(rallies, s, doubles)
        ms = []
        for name, (val, n) in feats.items():
            label, unit = FEATURES[name][0], FEATURES[name][1]
            ms.append(_metric(f"{s}.style.{name}", label, val, unit, n, 60, 15,
                              degrade=name in geo and not metric_ok,
                              digits=2 if name == "flight_s" else 1))
        own_n = feats["smash_pct"][1]
        phases = phase_choices(rallies, s)
        used = tactics(rallies, s) if not doubles else []
        values = {k: (val, n, FEATURES[k][0], FEATURES[k][1]) for k, (val, n) in feats.items()}
        for name, (tmpl, _, _) in SHARED.items():
            m = shared.get(tmpl.format(s=s))
            if m is not None and m.value is not None and m.status != "undeterminable":
                values[name] = (m.value, m.n, m.label, m.unit)
        examples = feature_examples(rallies, s)
        scores = sorted((match_prototype(p, values, doubles, examples) for p in protos),
                        key=lambda x: -(x.score if x.score is not None else -1))
        primary, second, label = _verdict(scores, own_n)
        status, conf = _conf(own_n, 80, MIN_SHOTS)
        if own_n < MIN_SHOTS:
            status = "undeterminable"
        elif no_hitters:  # hitters near random: a verdict would describe the tracker, not the player
            status, conf = "undeterminable", "low"
            primary, second, label = None, None, "证据不足（击球者识别不可靠）"
        elif bad_hitters:  # hits are misattributed: every per-player feature is suspect
            status, conf = "needs_review", "low"
        own_shots = [c.shot for r in rallies for c in r if c.side == s and c.shot.shot_type
                     and c.shot.shot_type not in SERVES]
        confirmed = sum(type_confirmed(x) for x in own_shots)
        unconfirmed_types = bool(own_shots) and confirmed / len(own_shots) < MIN_CONFIRMED_TYPES
        if unconfirmed_types and status != "undeterminable":
            # The style is built on shot types; on amateur footage the classifier gets about half of them
            # wrong (DJI_0009: 26 of 58 hand-labelled shots), so unconfirmed types cannot carry more.
            status, conf = "needs_review", "low"
        if primary is not None:
            ms.append(_metric(f"{s}.style.match_score", f"与{primary.name}原型的吻合度", primary.score, None,
                              own_n, 80, 30))
        ds = [_dist(f"{s}.style.phases", "攻防状态（拍数：主动/中立/被动）",
                    {PHASE_ZH[p.phase]: p.shots for p in phases}, 60, 15)]
        if used:
            ds.append(_dist(f"{s}.style.tactics", "战术套路使用次数", {t.name: t.uses for t in used}, 10, 3))
        metrics[s], dists[s] = ms, ds
        used_feats = set(by_code[primary.code].weights) if primary is not None else set()
        evidence = [m.key for m in ms if m.value is not None and m.status != "undeterminable"
                    and m.key.rsplit(".", 1)[-1] in used_feats | {"match_score"}]
        evidence += [SHARED[f][0].format(s=s) for f in sorted(used_feats & set(SHARED)) if f in values]
        pre[s] = dict(feats=feats, phases=phases, used=used, scores=scores, primary=primary, second=second,
                      label=label, status=status, conf=conf, n=own_n, evidence=evidence,
                      types=(confirmed, len(own_shots)) if unconfirmed_types else None,
                      habits=habits(rallies, s))

    styles = []
    for s in subjects:
        p = pre[s]
        opp = next(o for o in subjects if o != s)
        o = pre[opp]
        opp_style = PlayerStyle(subject=opp, kind="doubles" if doubles else "singles", primary=o["primary"],
                                label=o["label"], n_shots=o["n"], status=o["status"], confidence=o["conf"])
        # the opponent's replies to *my* shots: after = my shot, reply = theirs
        advice = _advice(s, "doubles" if doubles else "singles", p["primary"], p["feats"], p["phases"], p["used"],
                         opp_style if o["primary"] is not None else None, o["habits"], by_code) \
            if p["status"] != "undeterminable" else []
        notes = []
        if p["status"] == "undeterminable" and not no_hitters:
            notes.append(f"只有 {p['n']} 拍已分类的非发球击球，至少需要 {MIN_SHOTS} 拍才判断打法")
        elif bad_hitters:
            notes.append(f"相邻两拍由不同球员击出的只有 {100 * alternation[0]:.0f}%（单打应接近 100%），"
                         "击球者识别可能有误，打法判断只作参考；在工作台修正击球者后会更准")
        elif not metric_ok:
            notes.append("球场标定未确认，与站位相关的特征（调动距离、直线/斜线、四角）只作参考")
        if p["types"] is not None and p["status"] != "undeterminable":
            done, total = p["types"]
            notes.append(f"打法主要看击球类型，但 {total} 拍里只有 {done} 拍的类型经过确认；自动识别的类型在业余比赛视频上"
                         "大约一半会错（杀球、高远球常被认成吊球、挑球），所以只给“低”可信度，在工作台改正击球类型后会更准")
        note = "；".join(notes) or None
        styles.append(PlayerStyle(
            subject=s, players=(["P1", "P3"] if s == "A" else ["P2", "P4"]) if doubles else [s],
            kind="doubles" if doubles else "singles", primary=p["primary"], secondary=p["second"],
            scores=p["scores"], label=p["label"], phases=p["phases"], tactics=p["used"], habits=p["habits"],
            evidence=p["evidence"], advice=advice, n_shots=p["n"], status=p["status"], confidence=p["conf"],
            note=note,
        ))
    return StyleResult(styles=styles, metrics=metrics, distributions=dists)


def hitter_alternation(events: EventsResult) -> tuple[float, int] | None:
    """(share of consecutive in-rally hits by different players, number of pairs); singles hits must alternate."""
    shots_by_id = {s.shot_id: s for s in events.shots}
    pairs = alt = 0
    for rally in events.rallies:
        seq = [s for s in sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
               if s.hitter is not None]
        for a, b in zip(seq, seq[1:], strict=False):
            pairs += 1
            alt += a.hitter != b.hitter
    return (alt / pairs, pairs) if pairs else None


def style_by_subject(styles: list[PlayerStyle]) -> dict[str, PlayerStyle]:
    return {s.subject: s for s in styles}
