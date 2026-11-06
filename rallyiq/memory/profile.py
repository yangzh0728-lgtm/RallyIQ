"""Player profile recomputed from the append-only log (README M5 + M7 复测)."""

from __future__ import annotations

from ..contracts import (
    AnalysisEvent,
    PlayerProfile,
    ProfileMetric,
    ProfilePoint,
    ProfileVideo,
    RetestResult,
    Status,
)
from .log import current_events

# Relative change below this counts as "flat" (one match is noisy).
FLAT_REL = 0.10
# Two consecutive matches that differ by more than this are reported as a conflict to resolve with more data.
CONFLICT_REL = 0.35
# Metrics shown in the change list, in this order (others still appear in ``metrics``).
HEADLINE = ("win.rate_pct", "move.per_rally_m", "cover.ratio_pct", "pos.depth_mean_m", "shots", "shot_share")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def events_for(events: list[AnalysisEvent], person: str | None) -> list[AnalysisEvent]:
    cur = current_events(events)
    if person is None:
        return [e for e in cur if e.is_self]
    return [e for e in cur if e.person == person]


def people(events: list[AnalysisEvent]) -> list[dict]:
    cur = current_events(events)
    out: dict[str, dict] = {}
    for e in cur:
        if e.person is None:
            continue
        p = out.setdefault(e.person, {"person": e.person, "videos": 0, "is_self": False, "last": None})
        p["videos"] += 1
        p["is_self"] = p["is_self"] or e.is_self
        p["last"] = max(filter(None, [p["last"], e.recorded_at]))
    return sorted(out.values(), key=lambda p: (not p["is_self"], -p["videos"], p["person"]))


def _video(e: AnalysisEvent) -> ProfileVideo:
    return ProfileVideo(video_id=e.video_id, filename=e.filename, recorded_at=e.video_created_at or e.recorded_at,
                        player_id=e.player_id, opponents=e.opponents, conditions=e.conditions)


def build_profile(all_events: list[AnalysisEvent], person: str | None = None) -> PlayerProfile:
    evs = events_for(all_events, person)
    name = person or next((e.person for e in reversed(evs) if e.person), "我")
    series: dict[str, list[tuple[AnalysisEvent, object]]] = {}
    for e in evs:
        for m in e.metrics:
            series.setdefault(m.key, []).append((e, m))

    metrics: list[ProfileMetric] = []
    changes: list[str] = []
    for key, pts in series.items():
        last_m = pts[-1][1]
        usable = [(e, m) for e, m in pts if m.value is not None and m.status != Status.undeterminable]
        latest = usable[-1][1].value if usable else None
        prev = usable[-2][1].value if len(usable) >= 2 else None
        if not usable:
            trend = "insufficient"
        elif prev is None:
            trend = "new"
        elif _rel(latest, prev) < FLAT_REL:
            trend = "flat"
        else:
            trend = "up" if latest > prev else "down"
        note = None
        if prev is not None and _rel(latest, prev) > CONFLICT_REL and all(m.status == Status.ok for _, m in usable[-2:]):
            note = "与上一场差异较大，可能是对手或场地条件不同，需要更多比赛确认"
        conf = min((m.confidence for _, m in usable[-2:]), key=["low", "medium", "high"].index) if usable else "low"
        metrics.append(ProfileMetric(
            key=key, label=last_m.label, unit=last_m.unit,
            points=[ProfilePoint(video_id=e.video_id, filename=e.filename, recorded_at=e.video_created_at or e.recorded_at,
                                 value=m.value, n=m.n, status=m.status) for e, m in pts],
            latest=latest, previous=prev, delta=round(latest - prev, 3) if prev is not None else None,
            trend=trend, confidence=conf, note=note,
        ))
    order = {k: i for i, k in enumerate(HEADLINE)}
    metrics.sort(key=lambda m: (order.get(m.key, len(order)), m.key))

    for m in metrics:
        if m.key not in order:
            continue
        unit = m.unit or ""
        if m.trend == "new":
            changes.append(f"新增：{m.label} {m.latest:g}{unit}（第一场有效数据，n={m.points[-1].n}）")
        elif m.trend in ("up", "down"):
            arrow = "上升" if m.trend == "up" else "下降"
            line = f"趋势：{m.label} {arrow}，{m.previous:g} → {m.latest:g}{unit}"
            changes.append(line + (f"（{m.note}）" if m.note else ""))
    if len(evs) >= 2:
        a, b = evs[-2].conditions, evs[-1].conditions
        diff = [k for k in ("camera_view", "fps") if a.get(k) and b.get(k) and a[k] != b[k]]
        if diff:
            changes.append("冲突提醒：最近两场的拍摄条件不同（" + "、".join(diff) + "），对比结论只能作参考")

    return PlayerProfile(
        person=name, is_self=any(e.is_self for e in evs), videos=[_video(e) for e in evs], metrics=metrics,
        retests=retests(evs), changes=changes, log_events=len(all_events),
    )


def retests(evs: list[AnalysisEvent]) -> list[RetestResult]:
    """Each training item from one match, measured on the next committed match (README: 下次上传时自动对比)."""
    out = []
    for i, e in enumerate(evs):
        nxt = evs[i + 1] if i + 1 < len(evs) else None
        before_idx = {m.key: m for m in e.metrics}
        after_idx = {m.key: m for m in nxt.metrics} if nxt else {}
        for t in e.training:
            if not t.retest_metric:
                continue
            b = before_idx.get(t.retest_metric)
            a = after_idx.get(t.retest_metric)
            same = None
            if nxt is not None:
                same = all(e.conditions.get(k) == nxt.conditions.get(k) for k in ("camera_view", "fps"))
            if nxt is None:
                status = "pending"
            elif b is None or a is None or b.value is None or a.value is None:
                status = "not_measurable"
            elif _rel(a.value, b.value) < FLAT_REL:
                status = "unchanged"
            else:
                # "Better" is only clear for win and coverage rates; elsewhere report the change and its delta.
                if t.retest_metric.startswith(("win.", "cover.")):
                    status = "improved" if a.value > b.value else "worse"
                else:
                    status = "changed"
            out.append(RetestResult(
                drill=t.drill, why=t.why, metric=t.retest_metric, label=(b or a).label if (b or a) else None,
                unit=(b or a).unit if (b or a) else None, from_video_id=e.video_id,
                to_video_id=nxt.video_id if nxt else None,
                before=b.value if b else None, after=a.value if a else None,
                delta=round(a.value - b.value, 3) if (a and b and a.value is not None and b.value is not None) else None,
                same_conditions=same, status=status,
            ))
    return out
