"""Match statistics from effective events + player tracks (no model calls).

Every number is a ``Metric`` carrying its sample size, ``status`` and a plain
``confidence`` level, so the coach (and the UI) can say how much to trust it.
Values that cannot be computed stay ``None``; nothing is filled with a default.

Court frame used for per-player positions ("own half"): each foot position is
mirrored into the player's own half so that ``depth`` is the distance from their
own baseline (0 = baseline, 6.70 = net) and ``lateral`` is measured from their
own left sideline as they face the net. That makes near/far and side switches
comparable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable
from statistics import median

import numpy as np

from ..contracts import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    CourtCalibration,
    Distribution,
    EventsResult,
    FootLandmarks,
    MatchStats,
    Metric,
    PlayerStats,
    PlayerTracks,
    Rally,
    ShotEvent,
    VideoMeta,
    team_of,
)
from ..contracts.geometry import SINGLES_INSET
from ..footwork import Footwork
from .doubles import analyze_doubles, track_issues
from .flow import flow_metrics, rally_log
from .moments import shot_moments
from .style import MIN_SHOTS, analyze_styles, shot_phases

log = logging.getLogger("rallyiq.report")

PLAYERS = ("P1", "P2")
LANDING_NOTE = "由脚跟和脚尖关键点判断，30fps 业余视频上人工核对约 7 成判断正确"
DOUBLES_PLAYERS = ("P1", "P3", "P2", "P4")

# Movement filtering.
MAX_SPEED_MPS = 8.0  # faster foot "moves" between samples are tracking glitches
MAX_GAP_S = 0.5  # do not bridge track gaps longer than this
SMOOTH_WINDOW = 5  # moving median over foot positions (frames)
COVER_CELL_M = 0.5
COVER_MIN_SAMPLES = 3

# Zones in the own-half frame.
DEPTH_ZONES = (("后场", 0.0, 2.2), ("中场", 2.2, 4.47), ("前场", 4.47, NET_Y_M + 1e-6))
SINGLES_W = COURT_WIDTH_M - 2 * SINGLES_INSET
LATERAL_ZONES = (
    ("左侧", -1.0, SINGLES_INSET + SINGLES_W / 3),
    ("中路", SINGLES_INSET + SINGLES_W / 3, SINGLES_INSET + 2 * SINGLES_W / 3),
    ("右侧", SINGLES_INSET + 2 * SINGLES_W / 3, COURT_WIDTH_M + 1.0),
)
RALLY_BUCKETS = (("1-3拍", 1, 3), ("4-6拍", 4, 6), ("7-10拍", 7, 10), ("11拍以上", 11, 10**9))
SHORT_RALLY_MAX = 4
LONG_RALLY_MIN = 7

# Known stroke types (filled later by the stroke classifier); unknown names are shown as-is.
SHOT_TYPE_LABELS = {
    "serve": "发球", "short_serve": "发短球", "long_serve": "发高远球", "clear": "高远球", "smash": "杀球",
    "drop": "吊球", "net": "网前球", "net_shot": "网前球", "lift": "挑球", "drive": "平抽", "push": "推球",
    "block": "挡网", "cross_net": "勾对角", "net_kill": "扑球", "kill": "扑球", "defense": "防守", "unknown": "未识别",
}
UNCLASSIFIED = "未分类"


# --------------------------------------------------------------------------- helpers


def _conf(n: int, ok_n: int, review_n: int, degrade: bool = False) -> tuple[str, str]:
    """(status, confidence) from sample size; ``degrade`` drops one level (e.g. no calibration)."""
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


def _dist(key: str, label: str, counts: dict[str, int], ok_n: int, review_n: int, degrade: bool = False,
          note: str | None = None) -> Distribution:
    n = sum(counts.values())
    status, conf = _conf(n, ok_n, review_n, degrade)
    return Distribution(key=key, label=label, counts=counts, n=n, status=status, confidence=conf, note=note)


def _pct(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def own_half(p: Iterable[float]) -> tuple[float, float]:
    """Court metres -> (lateral from own left sideline, depth from own baseline)."""
    x, y = p
    if y <= NET_Y_M:  # near half, player faces +y: their left is x = 0
        return x, y
    return COURT_WIDTH_M - x, COURT_LENGTH_M - y


def _zone(v: float, zones) -> str:
    for name, lo, hi in zones:
        if lo <= v < hi:
            return name
    return zones[-1][0] if v >= zones[-1][1] else zones[0][0]


def _moving_median(a: np.ndarray, w: int) -> np.ndarray:
    if len(a) < 3 or w < 3:
        return a
    h = w // 2
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = np.median(a[max(0, i - h): i + h + 1], axis=0)
    return out


def fingerprint(events: EventsResult, players: PlayerTracks | None, calibration: CourtCalibration | None,
                feet: FootLandmarks | None = None) -> str:
    h = hashlib.sha256()
    h.update(events.model_dump_json().encode())
    if players is not None:
        h.update(f"{players.model}|{len(players.frames)}|{players.side_switches}".encode())
        if players.frames:
            h.update(players.frames[0].model_dump_json().encode())
            h.update(players.frames[-1].model_dump_json().encode())
    if calibration is not None:
        h.update(json.dumps([(s.start_frame, s.end_frame, s.status, s.method) for s in calibration.segments]).encode())
    if feet is not None and feet.frames:  # absent / empty feet keep the old fingerprint
        h.update(f"feet|{feet.model}|{len(feet.frames)}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------- per player


def _rally_frames(rallies: list[Rally]) -> list[tuple[int, int]]:
    return sorted((r.start_frame, r.end_frame) for r in rallies)


def _span_of(frame: int, spans: list[tuple[int, int]], starts: list[int]) -> int | None:
    """Index of the rally span containing ``frame`` (spans sorted by start), else None."""
    i = bisect_right(starts, frame) - 1
    for j in range(i, max(i - 3, -1), -1):  # spans rarely overlap; look back a few for corrected rallies
        if spans[j][0] <= frame <= spans[j][1]:
            return j
    return None


def _player_samples(players: PlayerTracks, pid: str, spans: list[tuple[int, int]]) -> list[tuple]:
    """(rally_index, frame, t_ms, x, y) court-metre foot samples of ``pid`` inside rallies (detections only)."""
    starts = [a for a, _ in spans]
    out = []
    for pf in players.frames:
        k = _span_of(pf.frame, spans, starts)
        if k is None:
            continue
        for p in pf.players:
            if p.player_id != pid or p.foot_court is None or p.interpolated or p.on_court is False:
                continue
            out.append((k, pf.frame, pf.t_ms, float(p.foot_court[0]), float(p.foot_court[1])))
            break
    return out


def _movement(samples, spans, fps: float) -> dict:
    """Distance, coverage and position shares from rally-time foot samples."""
    total_rally_frames = sum(b - a + 1 for a, b in spans)
    res = {"distance": None, "rejected": 0, "steps": 0, "samples": len(samples),
           "tracked_ratio": _pct(len(samples), total_rally_frames), "cells": 0,
           "depth": Counter(), "lateral": Counter(), "depth_mean": None}
    if not samples:
        return res
    # split into continuous runs (same rally, no long gaps) and smooth each run
    max_gap_frames = max(1, int(round(MAX_GAP_S * fps)))
    runs, cur = [], [samples[0]]
    for prev, s in zip(samples, samples[1:]):
        if s[0] == prev[0] and s[1] - prev[1] <= max_gap_frames:
            cur.append(s)
        else:
            runs.append(cur)
            cur = [s]
    runs.append(cur)

    dist, rejected, steps = 0.0, 0, 0
    cells: Counter = Counter()
    depths = []
    for run in runs:
        xy = _moving_median(np.array([[s[3], s[4]] for s in run]), SMOOTH_WINDOW)
        t = np.array([s[2] for s in run], dtype=float) / 1000.0
        for i in range(1, len(run)):
            dt = t[i] - t[i - 1]
            d = float(np.hypot(*(xy[i] - xy[i - 1])))
            steps += 1
            if dt <= 0 or d / dt > MAX_SPEED_MPS:
                rejected += 1
                continue
            dist += d
        for x, y in xy:
            lat, dep = own_half((x, y))
            depths.append(dep)
            res["depth"][_zone(dep, DEPTH_ZONES)] += 1
            res["lateral"][_zone(lat, LATERAL_ZONES)] += 1
            cells[(int(lat // COVER_CELL_M), int(dep // COVER_CELL_M))] += 1
    res.update(distance=dist, rejected=rejected, steps=steps,
               cells=sum(1 for c in cells.values() if c >= COVER_MIN_SAMPLES),
               depth_mean=float(np.mean(depths)) if depths else None)
    return res


def _footwork_stats(pid: str, fw: Footwork, spans: list[tuple[int, int]], metric_ok: bool,
                    singles: bool) -> tuple[list[Metric], list[Distribution]]:
    """Footwork metrics of one player (脚步): split steps, reaction, recovery, steps, lunges, jumps, speed."""
    q = next((x for x in fw.quality(spans) if x.player_id == pid), None)
    if q is None:
        return [], []
    degrade = not metric_ok or q.status != "ok"
    noise = (f"脚部定位误差约 {q.jitter_m:.2f} 米，只作参考" if q.status != "ok" and q.jitter_m is not None else None)
    m: list[Metric] = []
    d: list[Distribution] = []
    starts = [a for a, _ in spans]
    in_rally = [e for e in fw.footwork_events if e.player_id == pid and _span_of(e.frame, spans, starts) is not None]
    shots = [x for x in fw.shots if x.player_id == pid]
    if singles:
        judged = [v for k, (p, v, _) in fw.split_judgements().items() if p == pid and v != "unknown"]
        split = [v for v in judged if v != "none"]
        on_time = [v for v in judged if v == "on_time"]
        offsets = [e.value for e in in_rally if e.kind == "split_step" and e.value is not None]
        m.append(_metric(f"{pid}.footwork.split_step_pct", "对手击球时做启动步的比例", _pct(len(split), len(judged)), "%",
                         len(judged), 30, 10, degrade, noise or "启动步 = 对手击球前后双脚同时轻跳再落地"))
        m.append(_metric(f"{pid}.footwork.split_on_time_pct", "启动步时机合适的比例", _pct(len(on_time), len(judged)), "%",
                         len(judged), 30, 10, degrade, "合适 = 在对手击球前 0.1 秒到击球后 0.3 秒之间落地"))
        m.append(_metric(f"{pid}.footwork.split_offset_ms", "启动步落地时刻（相对对手击球，中位数）",
                         float(median(offsets)) if offsets else None, "ms", len(offsets), 20, 6, degrade,
                         "负数 = 在对手击球前就落地了", digits=0))
        react = [x.reaction_ms for x in shots if x.reaction_ms is not None]
        m.append(_metric(f"{pid}.footwork.reaction_ms", "对手击球到起动（中位数）", float(median(react)) if react else None,
                         "ms", len(react), 20, 6, degrade, "朝自己下一拍击球位置方向的移动；0 = 对手击球时已在移动",
                         digits=0))
        gaps = [x.recovery_gap_m for x in shots if x.recovery_gap_m is not None]
        m.append(_metric(f"{pid}.footwork.recovery_gap_m", "对手击球时离中心的距离（中位数）",
                         float(median(gaps)) if gaps else None, "m", len(gaps), 20, 6, degrade, noise))
        m.append(_metric(f"{pid}.footwork.recovered_pct", "对手击球前回到中心的比例",
                         _pct(sum(g <= fw.base_radius for g in gaps), len(gaps)), "%", len(gaps), 20, 6, degrade,
                         f"中心 = 本方半场中线、离底线 {fw.base[1]:.1f} 米处，半径 {fw.base_radius:.1f} 米"))
        steps = [x.steps for x in shots if x.steps is not None]
        m.append(_metric(f"{pid}.footwork.steps_per_shot", "每拍移动步数（中位数）", float(median(steps)) if steps else None,
                         "步", len(steps), 20, 6, degrade, noise))
    kinds = Counter(e.kind for e in in_rally)
    m.append(_metric(f"{pid}.footwork.lunges", "弓步击球次数", kinds.get("lunge", 0), "次", len(shots), 20, 6, degrade,
                     "两脚相距 1 米以上且最后一步跨出 0.5 米以上", digits=0))
    m.append(_metric(f"{pid}.footwork.jumps", "起跳次数", kinds.get("jump", 0), "次", len(shots), 20, 6, degrade, digits=0))
    m.append(_metric(f"{pid}.footwork.steps", "回合内步数", sum(1 for st in fw.steps_of(pid)
                                                                  if _span_of(st.end_frame, spans, starts) is not None),
                     "步", len(shots), 20, 6, degrade, noise, digits=0))
    sp = fw.speeds(pid, spans)
    m.append(_metric(f"{pid}.footwork.top_speed_mps", "最快移动速度（前 5%）", float(np.percentile(sp, 95)) if len(sp) else None,
                     "m/s", len(sp), 300, 60, degrade, noise))
    # ---- heel / toe (needs the feet stage): how the feet land, and the lunge foot
    if fw.sole_ok.get(pid):
        judged = [st.landing for st in fw.steps_of(pid)
                  if st.landing is not None and _span_of(st.end_frame, spans, starts) is not None]
        lc = Counter(judged)
        # landings are always "needs review": checked by eye on 30 fps amateur video, about 7 in 10 calls are right
        m.append(_metric(f"{pid}.footwork.heel_first_pct", "移动中脚跟先着地的比例", _pct(lc.get("heel", 0), len(judged)),
                         "%", len(judged), 40, 15, True, "只作参考：脚跟先着地时重心偏后，下一步起动慢；弓步除外"))
        d.append(_dist(f"{pid}.footwork.landings", "移动中的着地方式（次数）",
                       {"前脚掌先着地": lc.get("forefoot", 0), "全脚掌": lc.get("flat", 0), "脚跟先着地": lc.get("heel", 0)},
                       40, 15, True, f"只作参考：{LANDING_NOTE}"))
        lunges = [x for x in shots if x.finish == "lunge"]
        landed = [x.lunge_landing for x in lunges if x.lunge_landing is not None]
        offs = [x.lunge_toe_off_deg for x in lunges if x.lunge_toe_off_deg is not None]
        m.append(_metric(f"{pid}.footwork.lunge_heel_first_pct", "弓步脚跟先着地的比例",
                         _pct(sum(v == "heel" for v in landed), len(landed)), "%", len(landed), 10, 3, True,
                         "只作参考：弓步应脚跟先着地、再过渡到全脚掌"))
        m.append(_metric(f"{pid}.footwork.lunge_toe_off_deg", "弓步脚尖偏离跨步方向（中位数）",
                         float(median(offs)) if offs else None, "°", len(offs), 10, 3, degrade,
                         "0° = 脚尖正对跨步方向；偏得多时膝盖容易扭转", digits=0))
    d.append(_dist(f"{pid}.footwork.patterns", "步法使用（次数）",
                   {"并步": kinds.get("chasse", 0), "交叉步": kinds.get("cross_step", 0), "跑动": kinds.get("run", 0),
                    "弓步": kinds.get("lunge", 0), "起跳": kinds.get("jump", 0)}, 30, 10, degrade,
                   "从对手击球到本人击球之间的步法"))
    return m, d


def _player_stats(pid: str, events: EventsResult, rallies_by_id: dict[str, Rally], players: PlayerTracks | None,
                  spans: list[tuple[int, int]], fps: float, metric_ok: bool, rally_minutes: float,
                  fw: Footwork | None = None) -> PlayerStats:
    shots = [s for s in events.shots if s.hitter == pid]
    n_shots_known = sum(1 for s in events.shots if s.hitter is not None)
    degrade = not metric_ok
    m: list[Metric] = []
    d: list[Distribution] = []

    m.append(_metric(f"{pid}.shots", "击球数", len(shots), "拍", n_shots_known, 30, 8, digits=0,
                     note=None if n_shots_known == len(events.shots) else "部分击球的击球者未确定"))
    m.append(_metric(f"{pid}.shot_share", "击球占比", _pct(len(shots), n_shots_known), "%", n_shots_known, 30, 8))

    # ---- movement & coverage
    mv = _movement(_player_samples(players, pid, spans), spans, fps) if players is not None else None
    n_rallies = len(spans)
    if mv is None:
        note = "缺少球员跟踪数据"
        for key, label, unit in [("move.distance_m", "回合内移动距离", "m"), ("move.per_rally_m", "每回合平均移动", "m"),
                                 ("cover.area_m2", "覆盖面积", "m²")]:
            m.append(_metric(f"{pid}.{key}", label, None, unit, 0, 1, 1, note=note))
    else:
        cal_note = None if metric_ok else "球场标定未确认，米制距离仅供参考"
        rej_note = f"已剔除 {mv['rejected']} 个疑似跟踪跳变" if mv["rejected"] else None
        note = "；".join(x for x in (cal_note, rej_note) if x) or None
        m.append(_metric(f"{pid}.move.distance_m", "回合内移动距离", mv["distance"], "m", mv["samples"], 300, 60,
                         degrade, note, digits=0))
        per_rally = mv["distance"] / n_rallies if mv["distance"] is not None and n_rallies else None
        m.append(_metric(f"{pid}.move.per_rally_m", "每回合平均移动", per_rally, "m", n_rallies, 10, 3, degrade, note))
        per_min = mv["distance"] / rally_minutes if mv["distance"] is not None and rally_minutes > 0 else None
        m.append(_metric(f"{pid}.move.per_min_m", "每分钟回合时间移动", per_min, "m/min", mv["samples"], 300, 60,
                         degrade, note, digits=0))
        m.append(_metric(f"{pid}.move.tracked_pct", "回合内有足点的帧占比", mv["tracked_ratio"], "%", mv["samples"],
                         300, 60, note="低于 60% 时移动和覆盖数据偏低"))
        area = mv["cells"] * COVER_CELL_M**2 if mv["samples"] else None
        doubles_fmt = events.match_format == "doubles"
        half_area = COURT_WIDTH_M * NET_Y_M / 2 if doubles_fmt else SINGLES_W * NET_Y_M
        m.append(_metric(f"{pid}.cover.area_m2", "覆盖面积（本方半场）", area, "m²", mv["samples"], 300, 60, degrade,
                         f"按 {COVER_CELL_M} m 网格，至少出现 {COVER_MIN_SAMPLES} 帧的格子计入", digits=1))
        m.append(_metric(f"{pid}.cover.ratio_pct", "覆盖率（占双打半场的一半）" if doubles_fmt else "覆盖率（占单打半场）",
                         100 * area / half_area if area is not None else None, "%", mv["samples"], 300, 60, degrade))
        m.append(_metric(f"{pid}.pos.depth_mean_m", "平均站位离底线距离", mv["depth_mean"], "m", mv["samples"], 300, 60,
                         degrade, "0 = 本方底线，6.7 = 球网"))
        d.append(_dist(f"{pid}.pos.depth_zones", "回合内站位：前中后场（帧数）",
                       {z[0]: mv["depth"].get(z[0], 0) for z in DEPTH_ZONES}, 300, 60, degrade))
        d.append(_dist(f"{pid}.pos.lateral_zones", "回合内站位：左中右（本人视角，帧数）",
                       {z[0]: mv["lateral"].get(z[0], 0) for z in LATERAL_ZONES}, 300, 60, degrade))

    # ---- footwork (脚步) from the ankle keypoints
    if fw is not None:
        fm, fd = _footwork_stats(pid, fw, spans, metric_ok, events.match_format == "singles")
        m += fm
        d += fd

    # ---- where they hit from
    hit_zones: Counter = Counter()
    for s in shots:
        if s.hitter_court is not None:
            lat, dep = own_half(s.hitter_court)
            hit_zones[f"{_zone(dep, DEPTH_ZONES)}{_zone(lat, LATERAL_ZONES)}"] += 1
    d.append(_dist(f"{pid}.hit.zones", "击球位置分布（本人视角）", dict(hit_zones.most_common()), 30, 8, degrade))

    # ---- shot types (filled by the stroke classifier when available)
    types = Counter(SHOT_TYPE_LABELS.get(s.shot_type, s.shot_type) if s.shot_type else UNCLASSIFIED for s in shots)
    classified = sum(v for k, v in types.items() if k != UNCLASSIFIED)
    st = _dist(f"{pid}.shot_types", "击球类型分布", dict(types.most_common()), 30, 8,
               note=None if classified else "击球类型分类尚未接入，暂不可判定")
    if not classified:
        st = st.model_copy(update={"status": "undeterminable", "confidence": "low"})
    d.append(st)

    # ---- win / loss patterns (only rallies with a known winner)
    # Doubles: rallies are won by a team, serve/receive is decided by the team.
    doubles = events.match_format == "doubles"

    def side(x: str | None) -> str | None:
        # Winners are already teams ("A"/"B"); hitters are players.
        return (team_of(x) or x) if doubles else x

    me = side(pid)
    decided = [r for r in rallies_by_id.values() if r.winner is not None]
    nd = len(decided)
    won = [r for r in decided if side(r.winner) == me]
    m.append(_metric(f"{pid}.win.won", "得分回合数", len(won) if nd else None, "分", nd, 15, 5, digits=0,
                     note=None if nd else "尚未标注回合胜负（在修正面板里设置胜者）"))
    m.append(_metric(f"{pid}.win.rate_pct", "回合胜率", _pct(len(won), nd), "%", nd, 15, 5))

    def rally_len(r: Rally) -> int:
        return len(r.shot_ids)

    for key, label, sel in [
        ("win.short_rate_pct", f"短回合（≤{SHORT_RALLY_MAX}拍）胜率", lambda r: rally_len(r) <= SHORT_RALLY_MAX),
        ("win.long_rate_pct", f"长回合（≥{LONG_RALLY_MIN}拍）胜率", lambda r: rally_len(r) >= LONG_RALLY_MIN),
    ]:
        sub = [r for r in decided if sel(r)]
        m.append(_metric(f"{pid}.{key}", label, _pct(sum(side(r.winner) == me for r in sub), len(sub)), "%", len(sub), 10, 4))

    shots_by_id = {s.shot_id: s for s in events.shots}

    def first_last(r: Rally) -> tuple[ShotEvent | None, ShotEvent | None]:
        ss = sorted((shots_by_id[i] for i in r.shot_ids if i in shots_by_id), key=lambda s: s.frame)
        return (ss[0], ss[-1]) if ss else (None, None)

    serve = [r for r in decided if (fl := first_last(r))[0] is not None and fl[0].hitter is not None
             and side(fl[0].hitter) == me]
    recv = [r for r in decided if (fl := first_last(r))[0] is not None and fl[0].hitter is not None
            and side(fl[0].hitter) != me]
    m.append(_metric(f"{pid}.win.serve_rate_pct", "发球回合胜率", _pct(sum(side(r.winner) == me for r in serve), len(serve)),
                     "%", len(serve), 10, 4, note="以回合第一拍的击球者视为发球方"))
    m.append(_metric(f"{pid}.win.receive_rate_pct", "接发球回合胜率",
                     _pct(sum(side(r.winner) == me for r in recv), len(recv)), "%", len(recv), 10, 4))
    won_last = [r for r in won if (fl := first_last(r))[1] is not None and fl[1].hitter is not None]
    m.append(_metric(f"{pid}.win.own_last_shot_pct", "得分回合中最后一拍由本人击出的比例",
                     _pct(sum(first_last(r)[1].hitter == pid for r in won_last), len(won_last)), "%", len(won_last),
                     10, 4, note="高说明多靠主动进攻得分，低说明多靠对手失误；落点未识别，仅为近似"))
    lost = [r for r in decided if side(r.winner) != me]
    lost_last = [r for r in lost if (fl := first_last(r))[1] is not None and fl[1].hitter is not None]
    m.append(_metric(f"{pid}.win.lost_own_last_shot_pct", "失分回合中最后一拍由本人击出的比例",
                     _pct(sum(first_last(r)[1].hitter == pid for r in lost_last), len(lost_last)), "%",
                     len(lost_last), 10, 4, note="高通常意味着本人失误（下网/出界）较多；近似判断"))
    # which of my shots ended the rallies I lost (stroke-specific error signal for the coach)
    lost_types = Counter(SHOT_TYPE_LABELS.get(fl[1].shot_type, fl[1].shot_type) if fl[1].shot_type else UNCLASSIFIED
                         for r in lost_last if (fl := first_last(r))[1].hitter == pid)
    lt = _dist(f"{pid}.lost.last_shot_types", "失分回合中本人最后一拍的击球类型", dict(lost_types.most_common()), 12, 4,
               note="近似看作失误或被压制的那一拍；落点未识别")
    if not any(k != UNCLASSIFIED for k in lost_types):
        lt = lt.model_copy(update={"status": "undeterminable", "confidence": "low"})
    d.append(lt)
    return PlayerStats(player_id=pid, metrics=m, distributions=d)


# --------------------------------------------------------------------------- match


def compute_match_stats(events: EventsResult, players: PlayerTracks | None = None,
                        calibration: CourtCalibration | None = None, meta: VideoMeta | None = None,
                        feet: FootLandmarks | None = None) -> MatchStats:
    """Statistics from the *effective* events (after human corrections). ``feet``: heel / toe keypoints
    (stage ``feet``), optional; footwork uses the ankles alone without them."""
    fps = meta.fps if meta else 30.0
    rallies = sorted(events.rallies, key=lambda r: r.start_frame)
    rallies_by_id = {r.rally_id: r for r in rallies}
    spans = _rally_frames(rallies)
    caveats: list[str] = []

    cal_statuses = [s.status for s in calibration.segments] if calibration and calibration.segments else []
    if not cal_statuses:
        cal_status = None
    elif all(s == "ok" for s in cal_statuses):
        cal_status = "ok"
    elif any(s == "ok" for s in cal_statuses):
        cal_status = "needs_review"
    else:
        cal_status = cal_statuses[0]
    metric_ok = cal_status == "ok"
    if not metric_ok:
        caveats.append("球场标定未确认：移动距离、覆盖面积等米制数据只作参考。")

    lens = [len(r.shot_ids) for r in rallies]
    durs = [(r.end_ms - r.start_ms) / 1000 for r in rallies]
    nr = len(rallies)
    confirmed_shots = sum(1 for s in events.shots if s.status == "ok" or s.source == "human")
    confirmed_rallies = sum(1 for r in rallies if r.status == "ok" or r.source == "human")
    decided = sum(1 for r in rallies if r.winner is not None)
    rally_minutes = sum(durs) / 60

    metrics = [
        _metric("match.rallies", "回合数", nr, "个", nr, 10, 3, digits=0),
        _metric("match.shots", "击球数", len(events.shots), "拍", len(events.shots), 50, 10, digits=0),
        _metric("match.rally_len.mean", "平均每回合拍数", float(np.mean(lens)) if lens else None, "拍", nr, 10, 3),
        _metric("match.rally_len.median", "回合拍数中位数", float(median(lens)) if lens else None, "拍", nr, 10, 3),
        _metric("match.rally_len.max", "最长回合拍数", max(lens) if lens else None, "拍", nr, 10, 3, digits=0),
        _metric("match.rally_dur.mean_s", "平均回合时长", float(np.mean(durs)) if durs else None, "s", nr, 10, 3),
        _metric("match.rally_dur.max_s", "最长回合时长", max(durs) if durs else None, "s", nr, 10, 3),
        _metric("match.rally_time_pct", "回合时间占视频比例",
                _pct(sum(durs), meta.duration_ms / 1000) if meta and meta.duration_ms else None, "%", nr, 10, 3),
        _metric("match.confirmed_shots_pct", "已人工确认的击球比例", _pct(confirmed_shots, len(events.shots)), "%",
                len(events.shots), 1, 1, note="其余为模型结果，未经复核"),
        _metric("match.confirmed_rallies_pct", "已人工确认的回合比例", _pct(confirmed_rallies, nr), "%", nr, 1, 1),
        _metric("match.winners_known", "已标注胜负的回合数", decided, "个", nr, 1, 1, digits=0),
    ]
    buckets = {name: sum(1 for x in lens if lo <= x <= hi) for name, lo, hi in RALLY_BUCKETS}
    distributions = [_dist("match.rally_len.buckets", "回合长度分布（回合数）", buckets, 10, 3)]

    if events.shots and confirmed_shots / len(events.shots) < 0.5:
        caveats.append(f"只有 {confirmed_shots}/{len(events.shots)} 拍经过人工确认，统计基于自动识别结果，可能有漏检或误检。")
    if nr and decided < nr:
        caveats.append(f"{nr} 个回合中有 {decided} 个标注了胜负；胜负相关的结论只基于这些回合。" if decided else
                       "还没有标注回合胜负，得失分模式暂不可判定。")
    if players is None:
        caveats.append("缺少球员跟踪数据，移动与覆盖统计不可用。")
    shot_types_available = any(s.shot_type for s in events.shots)
    if not shot_types_available:
        caveats.append("击球类型分类尚未接入，暂不评价具体技术动作或球路选择。")

    fw = None
    if players is not None and calibration is not None:
        try:
            fw = Footwork(events, players, calibration, fps, feet)
        except Exception:  # noqa: BLE001 - footwork is an extra; never fail the report on it
            log.warning("footwork analysis failed for %s", events.video_id, exc_info=True)
    if fw is not None and fw.runs:
        caveats.append("脚步数据来自脚踝" + ("和脚跟 / 脚尖" if feet is not None and feet.frames else "")
                       + "关键点：远端球员或低分辨率画面下步伐细节只作参考；启动步与回位以识别出的击球时刻为准。")
    moments = shot_moments(events, players, fps, metric_ok, footwork=fw) if shot_types_available else []
    if events.match_format == "doubles":
        # identity mix-ups (a spectator as P3, partners on opposite halves) make "the opponent" wrong
        bad_players, bad_teams = track_issues(events, players)
        if bad_players or bad_teams:
            def suspect(pid: str | None) -> bool:
                return pid in bad_players or team_of(pid) in bad_teams
            kept = [m for m in moments if not suspect(m.player_id) and not suspect(m.opponent_id)]
            dropped = f"，去掉了 {len(moments) - len(kept)} 条涉及这些球员的逐拍建议" if len(kept) < len(moments) else ""
            caveats.append(f"双打球员身份识别有问题（{'；'.join([*bad_players.values(), *bad_teams.values()])}）"
                           f"{dropped}；请先在工作台修正球员身份。")
            moments = kept
    pids = DOUBLES_PLAYERS if events.match_format == "doubles" else PLAYERS
    player_stats = [_player_stats(pid, events, rallies_by_id, players, spans, fps, metric_ok, rally_minutes, fw)
                    for pid in pids]
    # Doubles teamwork: formation, rotation, spacing, roles, who takes which shots (rallyiq/report/doubles.py).
    doubles, dbl_metrics, dbl_dists, dbl_players = analyze_doubles(events, players, fps, metric_ok)
    metrics += dbl_metrics
    distributions += dbl_dists
    player_stats = [p.model_copy(update={"metrics": [*p.metrics, *dbl_players.get(p.player_id, [])]})
                    for p in player_stats]
    if doubles and players is not None:
        caveats.append("双打站位按两人离底线的距离差判断前后 / 左右站位，攻防状态按上一拍推断，"
                       "球员身份互换（跟踪 ID 交换）会让分工统计偏差。")
    # Rally log, score flow and tempo (rallyiq/report/flow.py).
    lines, games = rally_log(events, lambda t: SHOT_TYPE_LABELS.get(t, t) if t else None)
    flow_match, flow_players = flow_metrics(events, lines, games, pids)
    metrics += flow_match
    player_stats = [p.model_copy(update={"metrics": [*p.metrics, *flow_players.get(p.player_id, [])]})
                    for p in player_stats]
    styles = []
    if shot_types_available:
        phases = shot_phases(events)
        moments = [m.model_copy(update={"phase": phases.get(m.shot_id)}) for m in moments]
        shared = {m.key: m for m in metrics}
        shared.update({m.key: m for p in player_stats for m in p.metrics})
        res = analyze_styles(events, players, fps, metric_ok, shared)
        styles = res.styles
        if events.match_format == "doubles":  # team features live at match level (keys A.style.*, B.style.*)
            for team in ("A", "B"):
                metrics += res.metrics.get(team, [])
                distributions += res.distributions.get(team, [])
        else:
            player_stats = [p.model_copy(update={"metrics": p.metrics + res.metrics.get(p.player_id, []),
                                                 "distributions": p.distributions + res.distributions.get(p.player_id, [])})
                            for p in player_stats]
        if any(s.note and s.note.startswith("相邻两拍") for s in styles):
            caveats.append(next(s.note for s in styles if s.note and s.note.startswith("相邻两拍")) + "。")
        if any(s.status == "undeterminable" and s.n_shots < MIN_SHOTS for s in styles):
            caveats.append(f"有球员已分类的击球不足 {MIN_SHOTS} 拍，打法风格暂不判断。")
        caveats.append("打法风格由规则原型匹配得出（没有用标注数据训练），参考区间来自职业单打统计，业余比赛只作倾向参考。")
    return MatchStats(
        video_id=events.video_id, fingerprint=fingerprint(events, players, calibration, feet),
        duration_ms=meta.duration_ms if meta else None, calibration_status=cal_status,
        shot_types_available=shot_types_available, metrics=metrics, distributions=distributions,
        players=player_stats, moments=moments, rallies=lines, games=games, styles=styles, doubles=doubles,
        caveats=caveats,
    )


def index(stats: MatchStats) -> dict[str, Metric | Distribution]:
    """All metrics and distributions by key."""
    out: dict[str, Metric | Distribution] = {}
    for m in stats.metrics:
        out[m.key] = m
    for d in stats.distributions:
        out[d.key] = d
    for p in stats.players:
        for m in p.metrics:
            out[m.key] = m
        for d in p.distributions:
            out[d.key] = d
    return out
