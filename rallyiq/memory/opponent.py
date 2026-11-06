"""Opponent report (README M6): conditional shot habits with sample size, evidence and counter-examples."""

from __future__ import annotations

from collections import Counter

from ..contracts import AnalysisEvent, Distribution, Habit, HabitEvidence, OpponentReport, Status
from .profile import _video, events_for

TYPE_ZH = {
    "short_serve": "发网前球", "long_serve": "发后场球", "clear": "高远球", "smash": "杀球", "drop": "吊球",
    "net_shot": "放网", "cross_net": "勾对角", "push": "推球", "net_kill": "扑球", "drive": "平抽",
    "lift": "挑球", "block": "挡网",
}
MIN_N = 5
MIN_P = 0.4
MAX_EVIDENCE = 5
MAX_COUNTER = 3


def _conf(n: int, p: float) -> str:
    if n >= 30 and p >= 0.6:
        return "high"
    if n >= 12:
        return "medium"
    return "low"


def _ev(e: AnalysisEvent, s) -> HabitEvidence:
    return HabitEvidence(video_id=e.video_id, shot_id=s.shot_id, t_ms=s.t_ms)


def _habits(rows: list[tuple[AnalysisEvent, object]], cond_of, cond_label) -> list[Habit]:
    groups: dict[str, list] = {}
    for e, s in rows:
        c = cond_of(s)
        if c is not None and s.shot_type:
            groups.setdefault(c, []).append((e, s))
    out = []
    for c, items in groups.items():
        n = len(items)
        if n < MIN_N:
            continue
        top, hits = Counter(s.shot_type for _, s in items).most_common(1)[0]
        p = hits / n
        if p < MIN_P:
            continue
        out.append(Habit(
            condition=cond_label(c), outcome=TYPE_ZH.get(top, top), p=round(p, 3), n=n, hits=hits,
            confidence=_conf(n, p),
            evidence=[_ev(e, s) for e, s in items if s.shot_type == top][:MAX_EVIDENCE],
            counter_examples=[_ev(e, s) for e, s in items if s.shot_type != top][:MAX_COUNTER],
        ))
    return sorted(out, key=lambda h: (h.p * min(h.n, 30)), reverse=True)


def build_opponent_report(all_events: list[AnalysisEvent], person: str) -> OpponentReport:
    evs = events_for(all_events, person)
    rows = [(e, s) for e in evs for s in e.shots]
    typed = [(e, s) for e, s in rows if s.shot_type]
    reviewed = sum(1 for _, s in typed if s.shot_type_status == Status.ok)
    dists = []
    if typed:
        dists.append(Distribution(
            key="shot_type", label="击球类型分布", counts=dict(Counter(TYPE_ZH.get(s.shot_type, s.shot_type) for _, s in typed)),
            n=len(typed), status=Status.ok if reviewed == len(typed) else Status.needs_review,
            confidence=_conf(len(typed), 1.0)))
    zoned = [s.zone for _, s in rows if s.zone]
    if zoned:
        dists.append(Distribution(key="zone", label="击球位置（本方半场）", counts=dict(Counter(zoned)), n=len(zoned),
                                  status=Status.needs_review, confidence=_conf(len(zoned), 1.0)))

    habits = _habits(rows, lambda s: "serve" if s.index_in_rally == 0 else None, lambda _c: "发球时")
    habits += _habits([(e, s) for e, s in rows if s.index_in_rally != 0], lambda s: s.zone,
                      lambda z: f"在{z}击球时")
    habits.sort(key=lambda h: (h.p * min(h.n, 30)), reverse=True)
    next_shot = _habits(rows, lambda s: s.prev_shot_type, lambda t: f"我方{TYPE_ZH.get(t, t)}之后")

    caveats = []
    if len(evs) < 3:
        caveats.append(f"只有 {len(evs)} 场比赛的数据，习惯判断样本有限")
    if typed and reviewed < len(typed):
        caveats.append(f"{len(typed) - reviewed}/{len(typed)} 拍的击球类型仍是待复核的模型判断（规则 v1）；"
                       "在工作台确认或修正类型后重新写入档案会更可靠")
    if not typed:
        caveats.append("还没有击球类型数据，无法给出条件化习惯")
    return OpponentReport(person=person, videos=[_video(e) for e in evs], shots=len(rows), typed_shots=len(typed),
                          distributions=dists, habits=habits[:3], next_shot=next_shot[:5], caveats=caveats)
