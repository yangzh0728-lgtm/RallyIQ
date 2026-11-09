"""Markdown rendering of a match report (download / print / paste into notes)."""

from __future__ import annotations

from ..contracts import CoachReport, Distribution, MatchStats, Metric
from .coach.advice import video_link
from .moments import ZH, timestamp

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


def _flow_lines(stats: MatchStats) -> list[str]:
    """Games from the entered winners, then the rally log (one row per rally)."""
    out: list[str] = []
    if stats.games:
        out += ["## 比分", ""]
        for g in stats.games:
            score = " : ".join(f"{k} {v}" for k, v in g.score.items())
            tail = f"，{g.winner} 胜" if g.winner else "（未完成或胜负未标全）"
            out.append(f"- 第 {g.game_no} 局：{score}{tail}")
        out.append("")
    if stats.rallies:
        out += ["## 逐回合", "", "| # | 开始 | 拍数 | 时长 | 发球方 | 得分方 | 比分 | 最后一拍 | 拍间间隔 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in stats.rallies:
            secs = r.start_ms // 1000
            score = "-".join(str(v) for v in r.score.values()) if r.score else "—"
            last = f"{r.last_hitter or '?'} {r.last_shot_type or ''}".strip() if r.last_hitter or r.last_shot_type else "—"
            iv = f"{r.mean_interval_s:.2f} s" if r.mean_interval_s is not None else "—"
            out.append(f"| {r.rally_no} | {secs // 60:02d}:{secs % 60:02d} | {r.shots} | {r.duration_s:.1f} s | "
                       f"{r.server or '—'} | {r.winner or '—'} | {score} | {last} | {iv} |")
        out.append("")
    return out


def render_markdown(stats: MatchStats, coach: CoachReport | None = None, title: str | None = None,
                    coach_stale: bool = False) -> str:
    out = [f"# {title or '赛后报告'}", ""]
    out += scouting_lines(stats)
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
        if coach.shot_advice:
            out += ["", "### 逐拍建议", "", "_结合击球类型和双方站位找出的、可以换一种打法的击球（时间为视频时间）。_", ""]
            for a in coach.shot_advice:
                out.append(f"- **{a.timestamp}** {a.situation}（可信度：{CONF[a.confidence]}）  ")
                out.append(f"  {a.advice}  ")
                out.append(f"  {a.why}")
                for w in a.warnings:
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

    out += style_lines(stats)
    out += doubles_lines(stats)

    caveats = coach.caveats if coach else stats.caveats
    if caveats:
        out += ["## 数据说明", ""] + [f"- {c}" for c in caveats] + [""]

    out += ["## 全场统计", ""] + _metric_rows(stats.metrics) + [""]
    out += [_dist_line(d) for d in stats.distributions] + [""]
    out += _flow_lines(stats)
    for p in stats.players:
        out += [f"## {p.player_id}", ""] + _metric_rows(p.metrics) + [""]
        out += [_dist_line(d) for d in p.distributions] + [""]
    out.append(f"_统计指纹 {stats.fingerprint}；状态：可用 / 待复核 / 不可判定，按样本量和标定情况给出。_")
    return "\n".join(out) + "\n"


PHASE = {"attack": "主动", "neutral": "中立", "defense": "被动"}
FINDING = {"strength": "优势", "weakness": "问题", "pattern": "规律"}

DOUBLES_TEAM_KEYS = [
    ("formation.fb_pct", "前后站位"), ("formation.sbs_pct", "左右站位"), ("formation.attack_fb_pct", "进攻时前后站位"),
    ("formation.defense_sbs_pct", "防守时左右站位"), ("rotation.front_swaps_per_rally", "每回合前场换人"),
    ("rotation.switches_per_rally", "每回合攻守站位转换"), ("gap.mean_m", "平均间距"), ("gap.close_pct", "过近"),
    ("gap.wide_pct", "过远"), ("cover.ratio_pct", "覆盖率"),
]


def _fmt_metric(m) -> str:
    if m is None or m.value is None:
        return "—"
    v = f"{m.value:.0f}" if m.unit in ("%", "m²") else f"{m.value:.1f}"
    return f"{v}{m.unit or ''}"


def doubles_lines(stats: MatchStats) -> list[str]:
    """The doubles teamwork section: formation, rotation, spacing, roles, who takes which shots, findings."""
    if not stats.doubles:
        return []
    ix = {m.key: m for m in stats.metrics}
    ix.update({m.key: m for p in stats.players for m in p.metrics})
    dists = {d.key: d for d in stats.distributions}
    out = ["## 双打配合", "",
           "_按两人离底线的距离差判断站位：相差 ≥ 1.5 m 为前后站位，< 1 m 为左右站位；攻防状态按上一拍推断。_", ""]
    for t in stats.doubles:
        out += [f"### {t.team} 队（{'+'.join(t.players)}）", ""]
        meta = f"{STATUS[t.status]} · 可信度 {CONF[t.confidence]}"
        out += [f"_{meta}{'；' + t.note if t.note else ''}_", ""]
        out.append("- " + "，".join(f"{label} {_fmt_metric(ix.get(f'{t.team}.dbl.{k}'))}" for k, label in DOUBLES_TEAM_KEYS))
        for r in t.roles:
            why = f"（{'，'.join(r.reasons)}）" if r.reasons else ""
            out.append(f"- {r.player_id}：**{r.label}**{why}")
        rows = [dists[k] for k in t.division if k in dists and dists[k].n]
        if rows:
            p, q = t.players
            out += ["", f"| 谁打什么球 | {p} | {q} |", "| --- | --- | --- |"]
            for d in rows:
                out.append(f"| {d.label.removeprefix('谁打')} | {d.counts.get(p, 0)} | {d.counts.get(q, 0)} |")
            out.append("")
        for f in t.findings:
            out.append(f"- **[{FINDING[f.kind]}] {f.title}**（可信度：{CONF[f.confidence]}）：{f.detail}"
                       + (f" 建议：{f.advice}" if f.advice else ""))
        out.append("")
    return out


def scouting_lines(stats: MatchStats) -> list[str]:
    """How to beat P1 / P2: a game plan per player, each tactic with its evidence, confidence and replay times."""
    if not stats.scouting:
        return []
    lead = ("_从每位球员陷入被动的区域、失分的拍法、被调动后的表现、回球习惯和攻防状态里找弱点。"
            "可信度低的先当作观察，回看例子确认后再用。_")
    out = ["## 怎么打 P1、怎么打 P2", "", lead, ""]
    for sc in stats.scouting:
        side = {"far": "，在远端", "near": "，在近端"}.get(sc.side or "", "")
        out += [f"### 怎么打 {sc.target}（可信度：{CONF[sc.confidence]}{side}）", ""]
        if sc.summary:
            out += [f"**{sc.summary}**", ""]
        if sc.note:
            out += [f"> {sc.note}", ""]
        for k, t in enumerate(sc.tactics, 1):
            ex = f" 回看：{'、'.join(timestamp(x) for x in t.examples_ms)}" if t.examples_ms else ""
            out.append(f"{k}. **{t.title}**（可信度：{CONF[t.confidence]}）  ")
            out.append(f"   {t.how}  ")
            out.append(f"   依据：{t.why}{ex}")
        out.append("")
    return out


def style_lines(stats: MatchStats) -> list[str]:
    """The playing-style section: verdict, style scores, attack/neutral/defense choices, tactics, habits, advice."""
    if not stats.styles:
        return []
    out = ["## 打法风格", "",
           "_规则原型匹配（未用标注数据训练）：吻合度 0–100，参考区间来自职业单打统计，业余比赛看倾向即可。_", ""]
    for st in stats.styles:
        who = st.subject if st.kind == "singles" else f"{st.subject} 队（{'+'.join(st.players)}）"
        out += [f"### {who}：{st.label}", ""]
        meta = f"依据 {st.n_shots} 拍已分类击球 · {STATUS[st.status]} · 可信度 {CONF[st.confidence]}"
        out.append(f"_{meta}{'；' + st.note if st.note else ''}_")
        out.append("")
        if st.status == "undeterminable":
            continue
        if st.primary is not None:
            reps = "、".join(r.split(" ")[0] for r in st.primary.representatives)
            out.append(f"- 最接近：**{st.primary.name}**（{st.primary.name_en}，代表：{reps}），"
                       f"吻合度 {st.primary.score:.0f}；依据：{'、'.join(st.primary.reasons) or '—'}")
        ranked = [f"{x.name} {x.score:.0f}" for x in st.scores if x.score is not None]
        if ranked:
            out.append(f"- 各打法吻合度：{'，'.join(ranked)}")
        for ph in st.phases:
            if not ph.shots:
                continue
            mix = "，".join(f"{ZH.get(k, k)} {v}" for k, v in ph.counts.items())
            rec = f"；符合推荐打法 {ph.recommended_pct:.0f}%" if ph.recommended_pct is not None else ""
            out.append(f"- {PHASE[ph.phase]}状态 {ph.shots} 拍：{mix}{rec}")
        used = [t for t in st.tactics if t.uses]
        if used:
            out += ["", "| 战术套路 | 次数 | 有胜负 | 赢 | 胜率 | 逼对手被动 | 回看 |", "| --- | --- | --- | --- | --- | --- | --- |"]
            for t in used:
                sp = f"{t.success_pct:.0f}%" if t.success_pct is not None else "—"
                fp = f"{t.forced_pct:.0f}%" if t.forced_pct is not None else "—"
                out.append(f"| {t.name} | {t.uses} | {t.decided} | {t.won} | {sp} | {fp} | "
                           f"{'、'.join(timestamp(x) for x in t.examples_ms)} |")
            out.append("")
        if st.habits:
            out.append("- 回球习惯（预测下一拍）：" + "；".join(
                f"接{ZH.get(h.after, h.after)}时 {h.pct:.0f}% 回{ZH.get(h.reply, h.reply)}（{h.n}/{h.of}）" for h in st.habits))
        if st.advice:
            out += ["", "**建议**", ""] + [f"- {a}" for a in st.advice]
        out.append("")
    return out
