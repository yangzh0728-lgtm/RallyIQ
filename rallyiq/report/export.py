"""Markdown rendering of a match report (download / print / paste into notes)."""

from __future__ import annotations

from ..contracts import CoachReport, Distribution, MatchStats, Metric
from .coach.advice import video_link

STATUS = {"ok": "可用", "needs_review": "待复核", "undeterminable": "不可判定"}
CONF = {"high": "高", "medium": "中", "low": "低"}
KIND = {"strength": "优势", "weakness": "问题", "pattern": "规律", "tactic": "战术", "data_quality": "数据质量"}


def _v(m: Metric) -> str:
    if m.value is None:
        return "—"
    v = f"{m.value:.0f}" if abs(m.value - round(m.value)) < 0.05 else f"{m.value:.1f}"
    return f"{v} {m.unit}" if m.unit and m.unit != "%" else f"{v}%" if m.unit == "%" else v


def _metric_rows(ms: list[Metric]) -> list[str]:
    rows = ["| 指标 | 数值 | 样本 | 状态 | 可信度 | 说明 |", "| --- | --- | --- | --- | --- | --- |"]
    for m in ms:
        rows.append(f"| {m.label} | {_v(m)} | {m.n} | {STATUS[m.status]} | {CONF[m.confidence]} | {m.note or ''} |")
    return rows


def _dist_line(d: Distribution) -> str:
    if not d.n:
        return f"- {d.label}：无数据" + (f"（{d.note}）" if d.note else "")
    parts = "，".join(f"{k} {c}（{100 * c / d.n:.0f}%）" for k, c in d.counts.items())
    tail = f"；{d.note}" if d.note else ""
    return f"- {d.label}：{parts}（n={d.n}，{STATUS[d.status]}{tail}）"


def render_markdown(stats: MatchStats, coach: CoachReport | None = None, title: str | None = None,
                    coach_stale: bool = False) -> str:
    out = [f"# {title or '赛后报告'}", ""]
    if coach is not None:
        out += ["## AI 教练点评", ""]
        if coach_stale:
            out += ["> 注意：这份点评生成后数据又有修正，内容可能已过时。", ""]
        src = f"{coach.provider}" + (f" · {coach.model}" if coach.model else "")
        out += [f"_生成：{src} · {coach.created_at:%Y-%m-%d %H:%M} UTC_", ""]
        if coach.fallback_reason:
            out += [f"> 改用 {coach.provider}：{coach.fallback_reason}", ""]
        out += [coach.summary, ""]
        for i in coach.insights:
            who = f"{i.player_id} · " if i.player_id and not i.title.startswith(i.player_id) else ""
            flag = "" if i.grounded else " ⚠ 含未核实数字"
            out.append(f"- **[{KIND[i.kind]}] {who}{i.title}**（可信度：{CONF[i.confidence]}{flag}）  ")
            out.append(f"  {i.detail}  ")
            if i.evidence:
                out.append(f"  依据：{', '.join(f'`{k}`' for k in i.evidence)}")
            for w in i.warnings:
                out.append(f"  ⚠ {w}")
        if coach.training:
            out += ["", "### 训练建议", ""]
            for t in coach.training:
                who = f"{t.player_id}：" if t.player_id else ""
                retest = f"（复测：`{t.retest_metric}`）" if t.retest_metric else ""
                focus = f"针对：{t.focus}。" if t.focus else ""
                out.append(f"1. {who}**{t.drill}** — {focus}{t.why}{retest}")
                if t.key_points:
                    out.append(f"   要点：{'；'.join(t.key_points)}")
                for r in t.sources:
                    drill = f" · 练习「{r.drill}」" if r.drill else ""
                    out.append(f"   参考：{r.coach}《{r.title}》{drill} {video_link(r)}")
                for w in t.warnings:
                    out.append(f"   ⚠ {w}")
        out.append("")

    caveats = coach.caveats if coach else stats.caveats
    if caveats:
        out += ["## 数据说明", ""] + [f"- {c}" for c in caveats] + [""]

    out += ["## 全场统计", ""] + _metric_rows(stats.metrics) + [""]
    out += [_dist_line(d) for d in stats.distributions] + [""]
    for p in stats.players:
        out += [f"## {p.player_id}", ""] + _metric_rows(p.metrics) + [""]
        out += [_dist_line(d) for d in p.distributions] + [""]
    out.append(f"_统计指纹 {stats.fingerprint}；状态：可用 / 待复核 / 不可判定，按样本量和标定情况给出。_")
    return "\n".join(out) + "\n"
