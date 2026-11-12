"""Doubles teamwork from player tracks + effective events (no model calls).

For each pair (team A = P1+P3, team B = P2+P4) this module looks at where the two partners stood
during every rally (sampled at ``SAMPLE_HZ``) and at who hit which shot:

* **Formation**: front-back (depth gap >= ``FB_GAP_M``), side-by-side (gap < ``SBS_GAP_M``) or in
  between, overall and split by the pair's attack / defense state (from the last shot: after the
  opponents lift, the pair attacks; after the pair lifts, it defends). Textbook doubles is
  front-back when attacking and side-by-side when defending.
* **Rotation**: how often the front player changes between front-back phases, and how often the
  pair switches between front-back and side-by-side, per rally. Short runs (< ``MIN_RUN_S``) are
  merged into the previous one so tracking jitter does not count as a rotation.
* **Spacing**: mean distance between the partners, share of time too close (< ``CLOSE_M``: they
  get in each other's way and leave space open) or too far apart (> ``WIDE_M``: a gap between them).
* **Coverage**: union of the cells both partners stood in, against the doubles half.
* **Roles**: per player, the share of front-back moments in which they were the front one
  (front / rear / rotating player), plus their share of the pair's smashes and net shots.
* **Who takes which shots**: per pair, counts per player for rear / mid / front-court hits, smashes,
  net shots, lifts and clears, drives, serves, returns of serve, balls down the middle while the
  pair stood side by side, and the last shot of rallies the pair lost.

Rule findings (``DoublesFinding``) turn these into strengths, weaknesses and patterns with advice.
Every number is a ``Metric`` / ``Distribution`` (team keys ``A.dbl.*``, player keys ``Pn.dbl.*``),
so the coach can cite them.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

from ..contracts import (
    COURT_WIDTH_M,
    NET_Y_M,
    Distribution,
    DoublesFinding,
    DoublesRole,
    DoublesTeam,
    EventsResult,
    Metric,
    PlayerTracks,
    ShotEvent,
    team_of,
)
from .moments import FRONT_M, REAR_M, _Positions, own_half
from .style import is_attack, phase_of, shot_phases

TEAMS = {"A": ("P1", "P3"), "B": ("P2", "P4")}
PARTNER = {"P1": "P3", "P3": "P1", "P2": "P4", "P4": "P2"}

SAMPLE_HZ = 10.0
FB_GAP_M = 1.5  # depth gap at or above which the pair stands front-back
SBS_GAP_M = 1.0  # depth gap below which the pair stands side by side
MIN_RUN_S = 0.4  # formation runs shorter than this are jitter
CLOSE_M = 1.2  # partners closer than this: crowding
WIDE_M = 5.0  # partners farther apart than this: a hole between them
MIDDLE_HALF_W = 0.75  # "down the middle": within this of the centre line
COVER_CELL_M = 0.5
COVER_MIN_SAMPLES = 2
DOUBLES_HALF_M2 = COURT_WIDTH_M * NET_Y_M

FRONT_ROLE_PCT = 65.0
REAR_ROLE_PCT = 35.0
MIN_FB_SAMPLES = 30  # 3 s of front-back before a role is named

NET_SHOTS = {"net_shot", "net_kill", "push", "cross_net", "block"}
LONG_SHOTS = {"lift", "clear", "long_serve"}
DRIVES = {"drive"}

WHO = {  # distribution suffix -> label
    "rear": "后场击球",
    "mid": "中场击球",
    "front": "网前击球",
    "smash": "杀球",
    "net": "网前小球（放网/扑/推/挡）",
    "lift_clear": "挑球和高远球",
    "drive": "平抽",
    "middle": "左右站位时的中路球",
    "serve": "发球",
    "receive": "接发球",
    "lost_last": "失分回合的最后一拍",
}


MIN_CONFIRMED_SHARE = 0.5  # below this share of confirmed hits, shot shares are only tendencies


def _conf(n: int, ok_n: int, review_n: int, degrade: bool = False) -> tuple[str, str]:
    if n >= ok_n:
        status, conf = "ok", "high"
    elif n >= review_n:
        status, conf = "needs_review", "medium"
    elif n > 0:
        status, conf = "needs_review", "low"
    else:
        return "undeterminable", "low"
    if degrade:
        status = "needs_review"
        conf = {"high": "medium", "medium": "low", "low": "low"}[conf]
    return status, conf


def _metric(key: str, label: str, value: float | None, unit: str | None, n: int, ok_n: int, review_n: int,
            degrade: bool = False, note: str | None = None, digits: int = 1) -> Metric:
    status, conf = _conf(n if value is not None else 0, ok_n, review_n, degrade)
    v = None if value is None else round(float(value), digits)
    return Metric(key=key, label=label, value=v, unit=unit, n=n, status=status, confidence=conf, note=note)


def _pct(a: int, b: int) -> float | None:
    return 100.0 * a / b if b else None


# --------------------------------------------------------------------------- samples


@dataclass
class _Sample:
    rally: int
    frame: int
    form: str  # fb / sbs / mid
    front: str  # player id of the deeper-in-court (closer to the net) partner
    dist: float  # partner distance, m
    state: str | None  # attack / neutral / defense of the pair, None before the first shot
    cells: tuple[tuple[int, int], tuple[int, int]]


@dataclass
class _TeamData:
    samples: list[_Sample] = field(default_factory=list)
    front_swaps: int = 0
    form_switches: int = 0
    rallies_seen: set[int] = field(default_factory=set)


def _state_after(shot: ShotEvent, phase: str | None, team: str) -> str | None:
    """The pair's attack / neutral / defense state after ``shot`` (until the next one)."""
    if shot.hitter is None or not shot.shot_type:
        return None
    if team_of(shot.hitter) != team:
        return phase_of(shot.shot_type, phase)  # the state they reply in
    if shot.shot_type in LONG_SHOTS:
        return "defense"  # they lifted: the opponents attack
    return "attack" if is_attack(shot.shot_type, phase or "neutral") else "neutral"


def _team_samples(events: EventsResult, pos: _Positions, fps: float, team: str) -> _TeamData:
    p, q = TEAMS[team]
    phases = shot_phases(events)
    shots_by_id = {s.shot_id: s for s in events.shots}
    step = max(1, round(fps / SAMPLE_HZ))
    min_run = max(1, round(MIN_RUN_S * SAMPLE_HZ))
    out = _TeamData()
    for rno, rally in enumerate(sorted(events.rallies, key=lambda r: r.start_frame)):
        seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        k = 0
        state: str | None = None
        rs: list[_Sample] = []
        for f in range(rally.start_frame, rally.end_frame + 1, step):
            while k < len(seq) and seq[k].frame <= f:
                state = _state_after(seq[k], phases.get(seq[k].shot_id), team)
                k += 1
            a, b = pos.at(p, f), pos.at(q, f)
            if a is None or b is None:
                continue
            la, da = own_half(*a)
            lb, db = own_half(*b)
            if (a[1] <= NET_Y_M) != (b[1] <= NET_Y_M):
                continue  # partners on different halves: a tracking or identity error
            gap = abs(da - db)
            form = "fb" if gap >= FB_GAP_M else "sbs" if gap < SBS_GAP_M else "mid"
            cells = ((int(la // COVER_CELL_M), int(da // COVER_CELL_M)), (int(lb // COVER_CELL_M), int(db // COVER_CELL_M)))
            rs.append(_Sample(rno, f, form, p if da > db else q, math.dist(a, b), state, cells))
        if not rs:
            continue
        out.rallies_seen.add(rno)
        # merge short runs (jitter) into the previous run
        runs: list[list] = []  # [form, count, Counter(front)]
        for s in rs:
            if runs and runs[-1][0] == s.form:
                runs[-1][1] += 1
                runs[-1][2][s.front] += 1
            else:
                runs.append([s.form, 1, Counter([s.front])])
        merged: list[list] = []
        for r in runs:
            if merged and (r[1] < min_run or merged[-1][0] == r[0]):
                if merged[-1][0] == r[0]:
                    merged[-1][2].update(r[2])
                merged[-1][1] += r[1]
            else:
                merged.append([r[0], r[1], Counter(r[2])])
        fb_fronts = [r[2].most_common(1)[0][0] for r in merged if r[0] == "fb"]
        out.front_swaps += sum(1 for x, y in pairwise(fb_fronts) if x != y)
        core = [r[0] for r in merged if r[0] != "mid"]
        out.form_switches += sum(1 for x, y in pairwise(core) if x != y)
        out.samples += rs
    return out


# --------------------------------------------------------------------------- track sanity

OUT_MARGIN_M = 0.5  # a foot this far outside the doubles court counts as off court
MAX_OFF_COURT_PCT = 50.0
STATIC_SPREAD_M = 1.0  # a "player" whose 5-95 % range is below this on both axes never moved
MIN_SAME_HALF_PCT = 60.0
MIN_CHECK_SAMPLES = 50


def track_issues(events: EventsResult, players: PlayerTracks | None) -> tuple[dict[str, str], dict[str, str]]:
    """Player tracks that cannot be a doubles player, and pairs whose partners are not on the same half.

    Returns (player id -> reason, team -> reason). An identity mix-up (a spectator or umpire tracked as
    P3, a near-court player put in the far team) would otherwise turn into confident but wrong findings.
    """
    if players is None:
        return {}, {}
    spans = sorted((r.start_frame, r.end_frame) for r in events.rallies)
    if not spans:
        return {}, {}
    starts = [a for a, _ in spans]
    from bisect import bisect_right

    def in_rally(f: int) -> bool:
        i = bisect_right(starts, f) - 1
        return i >= 0 and f <= spans[i][1]

    xy: dict[str, list[tuple[float, float]]] = {}
    half: dict[int, dict[str, bool]] = {}
    for pf in players.frames:
        if not in_rally(pf.frame):
            continue
        for d in pf.players:
            if d.player_id is None or d.foot_court is None or d.interpolated:
                continue
            x, y = float(d.foot_court[0]), float(d.foot_court[1])
            xy.setdefault(d.player_id, []).append((x, y))
            half.setdefault(pf.frame, {})[d.player_id] = y <= NET_Y_M
    bad: dict[str, str] = {}
    for pid, pts in xy.items():
        if len(pts) < MIN_CHECK_SAMPLES:
            continue
        xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        off = sum(1 for x, y in pts if x < -OUT_MARGIN_M or x > COURT_WIDTH_M + OUT_MARGIN_M
                  or y < -OUT_MARGIN_M - 0.5 or y > 2 * NET_Y_M + OUT_MARGIN_M + 0.5)
        lo, hi = int(0.05 * len(pts)), int(0.95 * len(pts)) - 1
        if 100 * off / len(pts) > MAX_OFF_COURT_PCT:
            bad[pid] = f"{pid} 在回合中 {100 * off / len(pts):.0f}% 的时间站在场外，可能是裁判、观众或邻场球员"
        elif xs[hi] - xs[lo] < STATIC_SPREAD_M and ys[hi] - ys[lo] < STATIC_SPREAD_M:
            bad[pid] = f"{pid} 整场几乎没有移动（活动范围不到 {STATIC_SPREAD_M:.0f} 米），可能是裁判或观众"
    team_bad: dict[str, str] = {}
    for team, (p, q) in TEAMS.items():
        both = [h for h in half.values() if p in h and q in h]
        if len(both) >= MIN_CHECK_SAMPLES:
            same = 100 * sum(h[p] == h[q] for h in both) / len(both)
            if same < MIN_SAME_HALF_PCT:
                team_bad[team] = f"{p} 和 {q} 只有 {same:.0f}% 的时间在同一个半场，队伍分配可能有误"
    return bad, team_bad


# --------------------------------------------------------------------------- shot division


def _hit_xy(s: ShotEvent, pos: _Positions | None) -> tuple[float, float] | None:
    if s.hitter_court is not None:
        return float(s.hitter_court[0]), float(s.hitter_court[1])
    return pos.at(s.hitter, s.frame) if pos is not None and s.hitter else None


def _division(events: EventsResult, pos: _Positions | None, team: str) -> dict[str, Counter]:
    p, q = TEAMS[team]
    who: dict[str, Counter] = {k: Counter({p: 0, q: 0}) for k in WHO}
    shots_by_id = {s.shot_id: s for s in events.shots}
    for rally in events.rallies:
        seq = sorted((shots_by_id[i] for i in rally.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        for i, s in enumerate(seq):
            if s.hitter not in (p, q):
                continue
            h = s.hitter
            xy = _hit_xy(s, pos)
            if xy is not None:
                lat, dep = own_half(*xy)
                who["rear" if dep < REAR_M else "mid" if dep < FRONT_M else "front"][h] += 1
                mate = pos.at(PARTNER[h], s.frame) if pos is not None else None
                if mate is not None and abs(lat - COURT_WIDTH_M / 2) <= MIDDLE_HALF_W \
                        and abs(dep - own_half(*mate)[1]) < SBS_GAP_M:
                    who["middle"][h] += 1
            t = s.shot_type
            if t == "smash":
                who["smash"][h] += 1
            elif t in NET_SHOTS:
                who["net"][h] += 1
            elif t in ("lift", "clear"):
                who["lift_clear"][h] += 1
            elif t in DRIVES:
                who["drive"][h] += 1
            if i == 0:
                who["serve"][h] += 1
            elif i == 1 and seq[0].hitter is not None and team_of(seq[0].hitter) != team:
                who["receive"][h] += 1
        if seq and rally.winner is not None and rally.winner != team and seq[-1].hitter in (p, q):
            who["lost_last"][seq[-1].hitter] += 1
    return who


# --------------------------------------------------------------------------- per team


def analyze_doubles(events: EventsResult, players: PlayerTracks | None, fps: float = 30.0, metric_ok: bool = True
                    ) -> tuple[list[DoublesTeam], list[Metric], list[Distribution], dict[str, list[Metric]]]:
    """(teams, team metrics, team distributions, player metrics by id). Empty for singles."""
    if events.match_format != "doubles":
        return [], [], [], {}
    pos = _Positions(players, fps) if players is not None else None
    degrade = not metric_ok
    bad_players, bad_teams = track_issues(events, players)
    teams: list[DoublesTeam] = []
    metrics: list[Metric] = []
    dists: list[Distribution] = []
    pmetrics: dict[str, list[Metric]] = {}
    for team, (p, q) in TEAMS.items():
        problems = [bad_players[x] for x in (p, q) if x in bad_players]
        if team in bad_teams:
            problems.append(bad_teams[team])
        trusted = pos is not None and not problems
        data = _team_samples(events, pos, fps, team) if trusted else _TeamData()
        S = data.samples
        n = len(S)
        nr = len(data.rallies_seen)
        pre = f"{team}.dbl"
        tm: dict[str, Metric] = {}
        # Mostly unconfirmed hits: false detections and wrong hitters are common, so shot shares, and the
        # attack / defense states read from the last shot, are only tendencies.
        team_shots = [s for s in events.shots if s.hitter in (p, q)]
        confirmed = sum(1 for s in team_shots if s.status == "ok")
        hit_deg = bool(team_shots) and confirmed < MIN_CONFIRMED_SHARE * len(team_shots)

        def add(key, label, value, unit, nn, ok_n, review_n, note=None, deg=False, digits=1, pre=pre, tm=tm):
            m = _metric(f"{pre}.{key}", label, value, unit, nn, ok_n, review_n, deg, note, digits)
            tm[key] = m
            metrics.append(m)

        fb = sum(s.form == "fb" for s in S)
        sbs = sum(s.form == "sbs" for s in S)
        no_tracks = "缺少两人同时被跟踪到的帧" if not n else None
        add("formation.fb_pct", "前后站位时间占比", _pct(fb, n), "%", n, 300, 60, no_tracks or
            f"两人离底线的距离相差 ≥ {FB_GAP_M} m 算前后站位（每秒取 {SAMPLE_HZ:.0f} 帧）", degrade)
        add("formation.sbs_pct", "左右站位时间占比", _pct(sbs, n), "%", n, 300, 60,
            no_tracks or f"相差 < {SBS_GAP_M} m 算左右站位，其余为过渡", degrade)
        att = [s for s in S if s.state == "attack"]
        dfn = [s for s in S if s.state == "defense"]
        add("formation.attack_fb_pct", "进攻时前后站位的比例", _pct(sum(s.form == "fb" for s in att), len(att)), "%",
            len(att), 150, 30, "进攻 = 对手刚挑球/起高球，或本方刚下压之后", degrade or hit_deg)
        add("formation.defense_sbs_pct", "防守时左右站位的比例", _pct(sum(s.form == "sbs" for s in dfn), len(dfn)), "%",
            len(dfn), 150, 30, "防守 = 本方刚挑球/起高球，或对手刚杀球/扑球之后", degrade or hit_deg)
        add("rotation.front_swaps_per_rally", "每回合前场换人次数", data.front_swaps / nr if nr else None, "次", nr,
            10, 3, "两段前后站位之间，在前面的人换了一次记 1 次")
        add("rotation.switches_per_rally", "每回合攻守站位转换次数", data.form_switches / nr if nr else None, "次", nr,
            10, 3, "前后站位 ↔ 左右站位切换的次数")
        add("gap.mean_m", "两人平均间距", sum(s.dist for s in S) / n if n else None, "m", n, 300, 60, None, degrade)
        add("gap.close_pct", f"两人过近（< {CLOSE_M} m）的时间占比", _pct(sum(s.dist < CLOSE_M for s in S), n), "%",
            n, 300, 60, "太近容易互相让球、撞拍，并空出大片场地", degrade)
        add("gap.wide_pct", f"两人过远（> {WIDE_M} m）的时间占比", _pct(sum(s.dist > WIDE_M for s in S), n), "%",
            n, 300, 60, "太远时两人中间留下空当", degrade)
        cells = Counter(c for s in S for c in s.cells)
        area = sum(1 for v in cells.values() if v >= COVER_MIN_SAMPLES) * COVER_CELL_M ** 2 if n else None
        add("cover.area_m2", "两人合计覆盖面积", area, "m²", n, 300, 60,
            f"按 {COVER_CELL_M} m 网格；双打半场 {DOUBLES_HALF_M2:.1f} m²", degrade)
        add("cover.ratio_pct", "覆盖率（占双打半场）", 100 * area / DOUBLES_HALF_M2 if area is not None else None, "%",
            n, 300, 60, None, degrade)

        # who takes which shots
        who = _division(events, pos, team)
        division = []
        for k, label in WHO.items():
            c = who[k]
            tot = sum(c.values())
            status, conf = _conf(tot, 20, 6, hit_deg)
            d = Distribution(key=f"{pre}.who.{k}", label=f"谁打{label}" if k not in ("lost_last",) else label,
                             counts={p: c[p], q: c[q]}, n=tot, status=status, confidence=conf)
            dists.append(d)
            division.append(d.key)
        whod = {k: who[k] for k in WHO}

        # roles + player metrics
        fbs = [s for s in S if s.form == "fb"]
        team_hits = sum(1 for s in events.shots if s.hitter in (p, q))
        team_smash = sum(whod["smash"].values())
        team_net = sum(whod["net"].values())
        roles = []
        for pid in (p, q):
            pm: list[Metric] = []
            front_pct = _pct(sum(s.front == pid for s in fbs), len(fbs))
            pm.append(_metric(f"{pid}.dbl.front_pct", "前后站位时在前场的比例", front_pct, "%", len(fbs), 150, 30,
                              degrade))
            mine = sum(1 for s in events.shots if s.hitter == pid)
            pm.append(_metric(f"{pid}.dbl.team_share_pct", "占本队击球的比例", _pct(mine, team_hits), "%", team_hits,
                              30, 8, hit_deg))
            pm.append(_metric(f"{pid}.dbl.smash_share_pct", "占本队杀球的比例", _pct(whod["smash"][pid], team_smash), "%",
                              team_smash, 15, 5, hit_deg))
            pm.append(_metric(f"{pid}.dbl.net_share_pct", "占本队网前小球的比例", _pct(whod["net"][pid], team_net), "%",
                              team_net, 15, 5, hit_deg))
            zone_n = whod["rear"][pid] + whod["mid"][pid] + whod["front"][pid]
            pm.append(_metric(f"{pid}.dbl.rear_hit_pct", "本人击球中在后场击球的比例", _pct(whod["rear"][pid], zone_n),
                              "%", zone_n, 20, 6, degrade or hit_deg))
            pmetrics[pid] = pm
            reasons = [bad_players[pid]] if pid in bad_players else []
            if front_pct is None or len(fbs) < MIN_FB_SAMPLES:
                role, label = "unknown", "不可判定"
            else:
                role, label = (("front", "前场型") if front_pct >= FRONT_ROLE_PCT else
                               ("rear", "后场型") if front_pct <= REAR_ROLE_PCT else ("rotating", "轮转型"))
                reasons.append(f"前后站位时 {front_pct:.0f}% 的时间在前场")
            sm = _pct(whod["smash"][pid], team_smash)
            if sm is not None and team_smash >= 5 and sm >= 50:
                reasons.append(f"打了本队 {sm:.0f}% 的杀球")
            nt = _pct(whod["net"][pid], team_net)
            if nt is not None and team_net >= 5 and nt >= 50:
                reasons.append(f"打了本队 {nt:.0f}% 的网前小球")
            roles.append(DoublesRole(player_id=pid, role=role, label=label,
                                     front_pct=None if front_pct is None else round(front_pct, 1), reasons=reasons))

        findings = _findings(team, (p, q), tm, whod, roles, pmetrics) if not problems else []
        status, conf = _conf(n, 300, 60, degrade)
        if not n and team_hits:
            status, conf = "needs_review", "low"
        note = None
        if pos is None:
            note = "缺少球员跟踪数据：只统计谁打了哪些球，站位和轮转不可判定。"
        elif problems:
            status, conf = "undeterminable", "low"
            note = ("；".join(problems) + "。球员身份需要先在工作台里修正，站位、分工和建议暂不给出；"
                    "“谁打什么球”只按当前识别列出，仅供参考。")
        elif not n:
            note = "两人很少同时被跟踪到，站位和轮转不可判定。"
        if hit_deg and not problems:
            extra = (f"本队 {len(team_shots)} 拍里只有 {confirmed} 拍经过人工确认，自动识别会把不是击球的动作算进来、"
                     "也会认错击球人，所以“谁打什么球”、杀球和网前的分工比例，以及按上一拍判断的攻防站位，都只作参考；整体站位、间距和轮转按球员跟踪计算，不受影响。")
            note = f"{note}{extra}" if note else extra
        teams.append(DoublesTeam(team=team, players=[p, q], roles=roles, division=division, findings=findings,
                                 status=status, confidence=conf, note=note))
    return teams, metrics, dists, pmetrics


# --------------------------------------------------------------------------- findings


def _weakest(*confs: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return min(confs, key=lambda c: order[c]) if confs else "low"


def _findings(team: str, pair: tuple[str, str], tm: dict[str, Metric], who: dict[str, Counter],
              roles: list[DoublesRole], pm: dict[str, list[Metric]]) -> list[DoublesFinding]:
    out: list[DoublesFinding] = []
    p, q = pair

    def ok(m: Metric, min_n: int) -> bool:
        return m.value is not None and m.n >= min_n

    def add(kind, title, detail, advice, keys, *extra_conf):
        conf = _weakest(*(tm[k].confidence for k in keys if k in tm), *extra_conf)
        out.append(DoublesFinding(kind=kind, title=title, detail=detail, advice=advice,
                                  evidence=[tm[k].key if k in tm else k for k in keys], confidence=conf))

    m = tm["formation.attack_fb_pct"]
    if ok(m, 30):
        if m.value >= 60:
            add("strength", "进攻时能站成前后", f"本方进攻时 {m.value:.0f}% 的时间是前后站位，后面的人连续下压、前面的人封网。",
                None, ["formation.attack_fb_pct"])
        elif m.value < 40:
            add("weakness", "进攻时没有形成前后站位",
                f"本方进攻时只有 {m.value:.0f}% 的时间是前后站位，两人经常还平行站着，对手的挡网和平抽没人封。",
                "本方杀球或下压时，杀球人的同伴马上上前到前发球线附近封网，杀球人留在后场继续进攻。",
                ["formation.attack_fb_pct"])
    m = tm["formation.defense_sbs_pct"]
    if ok(m, 30):
        if m.value >= 60:
            add("strength", "防守时能拉开成左右站位", f"本方防守时 {m.value:.0f}% 的时间是左右站位，两人各守半边。",
                None, ["formation.defense_sbs_pct"])
        elif m.value < 40:
            add("weakness", "防守时没有拉开成左右站位",
                f"本方挑球或起高球之后，只有 {m.value:.0f}% 的时间两人左右平行站开，对手杀球时有一侧没人。",
                "挑球后网前那位立刻退到中场自己那一侧，两人在中场左右平行，各守半边，再找机会挡网或平抽反击。",
                ["formation.defense_sbs_pct"])
    m = tm["gap.close_pct"]
    if ok(m, 60) and m.value >= 20:
        add("weakness", "两人经常挤在一起",
            f"{m.value:.0f}% 的时间两人相距不到 {CLOSE_M} m（平均间距 {_v(tm['gap.mean_m'])} m），容易互相让球或撞拍。",
            "保持约 2–3 米的间距；中路球提前约定由谁接（例如由正手在中路的一方，或刚才没击球的一方），接球前喊一声。",
            ["gap.close_pct", "gap.mean_m"])
    m = tm["gap.wide_pct"]
    if ok(m, 60) and m.value >= 15:
        add("weakness", "两人距离过大，中间有空当",
            f"{m.value:.0f}% 的时间两人相距超过 {WIDE_M:.0f} m，中路和两人之间容易被打穿。",
            "一人被调动到边线或角落后，同伴要向中间补位，让两人始终覆盖整个半场。",
            ["gap.wide_pct", "gap.mean_m"])

    known = [r for r in roles if r.role != "unknown" and r.front_pct is not None]
    swaps = tm["rotation.front_swaps_per_rally"]
    if len(known) == 2:
        fixed = next((r for r in known if r.front_pct >= 80), None)
        if fixed is not None and (swaps.value is None or swaps.value < 0.3):
            other = q if fixed.player_id == p else p
            add("pattern", "前后分工固定",
                f"{fixed.player_id} 在前后站位时 {fixed.front_pct:.0f}% 的时间在前场，{other} 基本在后场，很少轮转。",
                f"分工清楚是优点，但对手容易针对：把球打到 {other} 的反手后场，或用平高球逼 {fixed.player_id} 后退。"
                "练一练进攻转防守、防守转进攻时的轮转，让两人都能在前后场衔接。",
                ["rotation.front_swaps_per_rally", f"{fixed.player_id}.dbl.front_pct"],
                _pm(pm, fixed.player_id, "front_pct").confidence)
        elif ok(swaps, 3) and swaps.value >= 0.5:
            add("pattern", "前后场轮转积极",
                f"平均每回合前场换人 {swaps.value:.1f} 次，两人都会上网和后退，打法比较灵活。",
                None, ["rotation.front_swaps_per_rally"])

    def share(key: str, min_n: int, hi: float):
        c = who[key]
        tot = sum(c.values())
        if tot < min_n:
            return None
        top, k = c.most_common(1)[0]
        return (top, 100 * k / tot, tot) if 100 * k / tot >= hi else None

    if (s := share("smash", 5, 80)) is not None:
        other = q if s[0] == p else p
        add("pattern", f"进攻主要靠 {s[0]}",
            f"本队 {s[2]} 次杀球里 {s[1]:.0f}% 是 {s[0]} 打的，{other} 在后场的下压不多。",
            f"{other} 到后场时也要敢于下压（点杀、劈吊），否则对手只要把球挑给 {other} 就能缓解压力。",
            [f"{team}.dbl.who.smash", f"{s[0]}.dbl.smash_share_pct"], _pm(pm, s[0], "smash_share_pct").confidence)
    if (s := share("middle", 5, 75)) is not None:
        add("pattern", f"中路球多由 {s[0]} 接",
            f"左右站位时打到两人中间的 {s[2]} 个球里，{s[0]} 接了 {s[1]:.0f}%。",
            "这样分工清楚很好；如果是因为同伴让球，就提前约定中路球的归属，避免两人都不接。",
            [f"{team}.dbl.who.middle"], _dconf(s[2]))
    if (s := share("lost_last", 6, 70)) is not None:
        add("weakness", f"失分回合的最后一拍多由 {s[0]} 击出",
            f"本队 {s[2]} 个丢分回合里，{s[1]:.0f}% 的最后一拍是 {s[0]} 打的（下网、出界或被对手打死前的那一拍）。",
            f"回看这些回合，看是 {s[0]} 自己失误，还是对手专门把球打给 {s[0]}；如果是被针对，同伴要多补位、多分担。",
            [f"{team}.dbl.who.lost_last"], _dconf(s[2]))
    for pid in pair:
        m = _pm(pm, pid, "team_share_pct")
        if m.value is not None and m.n >= 20 and m.value >= 65:
            other = q if pid == p else p
            out.append(DoublesFinding(
                kind="pattern", title=f"{pid} 承担了大部分击球",
                detail=f"本队 {m.n} 拍里 {m.value:.0f}% 是 {pid} 打的，对手可能在专打 {pid}，或 {other} 接球不够主动。",
                advice=f"{other} 要更主动地抢中路和自己一侧的球；如果是对手针对 {pid}，{pid} 回球要更有质量，减少被连续压制。",
                evidence=[m.key], confidence=m.confidence))
    return out


def _pm(pm: dict[str, list[Metric]], pid: str, name: str) -> Metric:
    return next(m for m in pm[pid] if m.key == f"{pid}.dbl.{name}")


def _dconf(n: int) -> str:
    return _conf(n, 20, 6)[1]


def _v(m: Metric) -> str:
    return "—" if m.value is None else f"{m.value:.1f}"
