"""Shareable match report as one self-contained HTML file (赛后报告 · 网页版).

``render_html(report, events=..., players=...)`` turns the match report and the effective events into a
single static page: inline CSS and inline SVG charts, no scripts, no fonts or images from the network.
The file opens offline in any browser, can be sent to a coach or a partner, and prints to PDF.
Nothing is uploaded anywhere and the video itself is not included.

Inputs are read as plain dicts (``model_dump(mode="json")`` or the JSON files), and every section reads its
fields with ``.get``: a section whose data is missing (no stroke types yet, no styles, no footwork, no
calibration) is left out instead of failing. That keeps the page working while newer analyses land:

- rallies / shots / stroke types / movement / coach: ``MatchReport`` + ``EventsResult``
- multi-layer shot description: ``ShotEvent.stroke.detail`` (hand, direction, contact, pace, role, intent)
- playing style: ``MatchStats.styles``
- footwork: ``Pn.footwork.*`` metrics and the ``Pn.footwork.patterns`` distribution
- court heatmaps: ``PlayerTracks`` foot positions (court metres) and ``ShotEvent.hitter_court``
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from html import escape
from typing import Any

from ..contracts.models import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M

STATUS = {"ok": "可用", "needs_review": "待复核", "undeterminable": "不可判定"}
CONF = {"high": "高", "medium": "中", "low": "低"}
KIND = {"strength": "优势", "weakness": "问题", "pattern": "规律", "tactic": "战术", "data_quality": "数据质量"}
END_REASON = {"out": "出界", "net": "下网", "shuttle_lost": "球丢失", "fault": "失误", "unknown": "—"}
SHOT_ZH = {
    "serve": "发球", "short_serve": "发短球", "long_serve": "发高远球", "clear": "高远球", "smash": "杀球",
    "drop": "吊球", "net": "网前球", "net_shot": "放网", "lift": "挑球", "drive": "平抽", "push": "推球",
    "block": "挡网", "cross_net": "勾对角", "net_kill": "扑球", "kill": "扑球", "defense": "防守", "unknown": "未识别",
}
# One row per layer of ShotEvent.stroke.detail; values in display order.
LAYERS: list[tuple[str, str, dict[str, str]]] = [
    ("hand", "正反手", {"forehand": "正手", "backhand": "反手", "around_head": "头顶"}),
    ("direction", "线路", {"straight": "直线", "cross": "斜线"}),
    ("contact", "击球点", {"high": "高点", "low": "低点"}),
    ("pace", "出球速度", {"fast": "快", "medium": "中", "slow": "慢"}),
    ("role", "攻防", {"attacking": "主动进攻", "neutral": "相持", "defensive": "被动防守"}),
    ("intent", "意图", {"attack": "施压", "move": "调动", "control": "控网", "defend": "防守", "transition": "过渡"}),
]
PLAYER_CLASS = {"P1": "p1", "P2": "p2", "P3": "p1", "P4": "p2", "A": "p1", "B": "p2"}
FOOTWORK_ORDER = ["split_step_pct", "split_on_time_pct", "split_offset_ms", "reaction_ms", "recovered_pct",
                  "recovery_gap_m", "steps_per_shot", "steps", "lunges", "jumps", "top_speed_mps"]


# ----------------------------------------------------------------------------- helpers


def _e(x: Any) -> str:
    return escape("" if x is None else str(x))


def _num(v: float | None, unit: str | None = None) -> str:
    if v is None:
        return "—"
    s = f"{v:.0f}" if abs(v - round(v)) < 0.05 or abs(v) >= 100 else f"{v:.1f}"
    if not unit:
        return s
    return f"{s}%" if unit == "%" else f"{s} {unit}"


def _ms(t_ms: float | None) -> str:
    if t_ms is None:
        return "—"
    s = int(t_ms // 1000)
    return f"{s // 60:02d}:{s % 60:02d}"


def _player_cls(pid: str | None) -> str:
    return PLAYER_CLASS.get(pid or "", "pn")


def _metrics(src: dict | None) -> dict[str, dict]:
    return {m["key"]: m for m in (src or {}).get("metrics", [])}


def _dists(src: dict | None) -> dict[str, dict]:
    return {d["key"]: d for d in (src or {}).get("distributions", [])}


def _badge(status: str | None, confidence: str | None = None) -> str:
    if not status and not confidence:
        return ""
    text = STATUS.get(status or "", "")
    if confidence:
        text += ("·" if text else "") + f"可信度{CONF.get(confidence, confidence)}"
    return f'<span class="badge {_e(status or "")}">{_e(text)}</span>'


def _section(sid: str, title: str, body: str, lead: str = "") -> str:
    lead_html = f'<p class="lead">{lead}</p>' if lead else ""
    return f'<section id="{sid}"><h2>{_e(title)}</h2>{lead_html}{body}</section>'


def _legend(players: Iterable[str]) -> str:
    items = "".join(f'<span class="key"><i class="sw {_player_cls(p)}"></i>{_e(p)}</span>' for p in players)
    return f'<div class="legend">{items}</div>'


def _hbars(counts: dict[str, int], total: int | None = None, cls: str = "pn", limit: int = 12,
           ordered: bool = False) -> str:
    """Horizontal bars, largest first (``ordered``: keep the given order, e.g. length buckets), with counts and
    shares as text (identity never by colour alone)."""
    items = [(k, v) for k, v in counts.items() if v]
    items = (items if ordered else sorted(items, key=lambda kv: -kv[1]))[:limit]
    if not items:
        return '<p class="muted">无数据</p>'
    total = total or sum(counts.values()) or 1
    top = max(v for _, v in items)
    rows = []
    for k, v in items:
        w = 100 * v / top
        rows.append(f'<div class="hb"><span class="hb-k">{_e(k)}</span><span class="hb-track">'
                    f'<span class="hb-bar {cls}" style="width:{w:.1f}%" title="{_e(k)}：{v}"></span></span>'
                    f'<span class="hb-v">{v}<small>{100 * v / total:.0f}%</small></span></div>')
    return '<div class="hbars">' + "".join(rows) + "</div>"


# ----------------------------------------------------------------------------- sections


def _header(stats: dict, title: str, video_name: str | None, generated: datetime) -> str:
    m = _metrics(stats)
    dur = stats.get("duration_ms")
    chips = []
    if dur:
        chips.append(f"视频时长 {_ms(dur)}")
    cal = stats.get("calibration_status")
    chips.append(f"球场标定：{STATUS.get(cal, '未标定') if cal else '未标定'}")
    conf = m.get("match.confirmed_shots_pct")
    if conf and conf.get("value") is not None:
        chips.append(f"人工确认击球 {_num(conf['value'], '%')}")
    chips.append(f"生成于 {generated:%Y-%m-%d %H:%M} UTC")
    sub = f'<p class="sub">{_e(video_name)}</p>' if video_name and video_name not in title else ""
    chip_html = "".join(f'<span class="chip">{_e(c)}</span>' for c in chips)
    return (f'<header><p class="eyebrow">RallyIQ · 赛后报告</p><h1>{_e(title)}</h1>{sub}'
            f'<div class="chips">{chip_html}</div></header>')


def _overview(stats: dict, rallies: list[dict]) -> str:
    m = _metrics(stats)
    tiles = []

    def tile(label: str, key: str, unit: str | None = None, value: str | None = None) -> None:
        mm = m.get(key)
        v = value if value is not None else (_num(mm.get("value"), unit or mm.get("unit")) if mm else "—")
        if v == "—" and value is None and mm is None:
            return
        tiles.append(f'<div class="tile"><div class="tile-v">{_e(v)}</div><div class="tile-k">{_e(label)}</div></div>')

    tile("回合", "match.rallies", "")
    tile("总拍数", "match.shots", "")
    tile("平均每回合拍数", "match.rally_len.mean", "")
    tile("最长回合（拍）", "match.rally_len.max", "")
    tile("平均回合时长", "match.rally_dur.mean_s", "s")
    tile("实际对打时间占比", "match.rally_time_pct", "%")
    won = {p["player_id"]: _metrics(p).get(f"{p['player_id']}.win.won") for p in stats.get("players", [])}
    if stats.get("doubles"):  # partners share the team's record: show A : B
        won = {team: won.get(pid) for team, pid in (("A 队", "P1"), ("B 队", "P2"))}
    if any(w and w.get("value") is not None for w in won.values()):
        score = " : ".join(_num(w.get("value")) if w else "—" for w in won.values())
        tiles.append(f'<div class="tile"><div class="tile-v">{_e(score)}</div>'
                     f'<div class="tile-k">得分（{_e(" : ".join(won))}，已标注胜负的回合）</div></div>')
    return _section("overview", "概览", '<div class="tiles">' + "".join(tiles) + "</div>")


def _rally_chart(rallies: list[dict], shots_by_id: dict[str, dict]) -> str:
    if not rallies:
        return ""
    lens = [len([s for s in r.get("shot_ids", []) if s in shots_by_id]) or len(r.get("shot_ids", [])) for r in rallies]
    n, top = len(rallies), max(max(lens), 1)
    w, h, pad_l, pad_b, pad_t = 720, 200, 30, 22, 10
    bw = (w - pad_l) / n
    gap = min(2.0, bw * 0.25)
    ticks = sorted({0, top, *([round(top / 2)] if top >= 4 else [])})
    grid = "".join(
        f'<line class="grid" x1="{pad_l}" x2="{w}" y1="{pad_t + (h - pad_t - pad_b) * (1 - t / top):.1f}" '
        f'y2="{pad_t + (h - pad_t - pad_b) * (1 - t / top):.1f}"/>'
        f'<text class="axis" x="{pad_l - 6}" y="{pad_t + (h - pad_t - pad_b) * (1 - t / top) + 4:.1f}" '
        f'text-anchor="end">{t}</text>' for t in ticks)
    bars = []
    for i, (r, k) in enumerate(zip(rallies, lens)):
        bh = (h - pad_t - pad_b) * k / top
        x = pad_l + i * bw + gap / 2
        y = h - pad_b - bh
        winner = r.get("winner")
        tip = f"第{i + 1}回合 · {_ms(r.get('start_ms'))} · {k} 拍" + (f" · {winner} 得分" if winner else "")
        bars.append(f'<rect class="bar {_player_cls(winner)}" x="{x:.1f}" y="{y:.1f}" width="{max(bw - gap, 1):.1f}" '
                    f'height="{max(bh, 1):.1f}" rx="{min(3, (bw - gap) / 2):.1f}"><title>{_e(tip)}</title></rect>')
    every = max(1, round(n / 12))
    labels = "".join(f'<text class="axis" x="{pad_l + (i + 0.5) * bw:.1f}" y="{h - 6}" text-anchor="middle">{i + 1}</text>'
                     for i in range(0, n, every))
    winners = [p for p in ("P1", "P2", "A", "B") if any(r.get("winner") == p for r in rallies)]
    legend = _legend(winners) + ('<span class="key"><i class="sw pn"></i>胜负未标注</span>'
                                 if any(not r.get("winner") for r in rallies) else "")
    return (f'<div class="chart"><div class="chart-head"><h3>每回合拍数</h3><div class="legend">{legend}</div></div>'
            f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="每回合拍数柱状图">{grid}{"".join(bars)}{labels}'
            f'<text class="axis" x="{w}" y="{h - 6}" text-anchor="end">回合</text></svg></div>')


def _rallies(stats: dict, rallies: list[dict], shots_by_id: dict[str, dict]) -> str:
    if not rallies:
        return ""
    rows = []
    for i, r in enumerate(rallies):
        k = len(r.get("shot_ids", []))
        dur = (r.get("end_ms", 0) - r.get("start_ms", 0)) / 1000
        winner = r.get("winner")
        w = f'<span class="who {_player_cls(winner)}">{_e(winner)}</span>' if winner else '<span class="muted">—</span>'
        rows.append(f"<tr><td>{i + 1}</td><td>{_ms(r.get('start_ms'))}</td><td class='num'>{k}</td>"
                    f"<td class='num'>{dur:.1f} s</td><td>{w}</td><td>{_e(END_REASON.get(r.get('end_reason'), '—'))}</td></tr>")
    table = ('<details><summary>回合明细（表格）</summary><div class="table-wrap"><table><thead><tr><th>#</th>'
             '<th>开始</th><th class="num">拍数</th><th class="num">时长</th><th>得分</th><th>结束原因</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div></details>')
    buckets = _dists(stats).get("match.rally_len.buckets")
    bucket_html = (f'<div class="chart"><h3>{_e(buckets["label"])}</h3>{_hbars(buckets["counts"], buckets["n"], ordered=True)}</div>'
                   if buckets and buckets.get("n") else "")
    return _section("rallies", "回合", f'<div class="grid2">{_rally_chart(rallies, shots_by_id)}{bucket_html}</div>{table}')


def _score_flow(stats: dict) -> str:
    """Games and a diverging bar per decided rally: how far ahead the leader was after it."""
    games = stats.get("games") or []
    lines = [r for r in stats.get("rallies") or [] if r.get("score")]
    if not games or not lines:
        return ""
    sides = list(lines[0]["score"])
    a, b = sides[0], sides[1]
    tiles = "".join(
        f'<div class="tile"><div class="tile-v">{g["score"].get(a, 0)} : {g["score"].get(b, 0)}</div>'
        f'<div class="tile-k">第 {g["game_no"]} 局 · '
        + (f'<span class="who {_player_cls(g.get("winner"))}">{_e(g.get("winner"))}</span> 胜' if g.get("winner")
           else "未完成") + "</div></div>" for g in games)
    diffs = [r["score"].get(a, 0) - r["score"].get(b, 0) for r in lines]
    top = max(3, *(abs(d) for d in diffs))
    w, h, pad_l, pad_t, pad_b = 720, 180, 30, 10, 18
    mid = pad_t + (h - pad_t - pad_b) / 2
    half_h = (h - pad_t - pad_b) / 2
    bw = (w - pad_l) / len(lines)
    gap = min(2.0, bw * 0.25)
    out = [f'<line class="grid" x1="{pad_l}" x2="{w}" y1="{mid:.1f}" y2="{mid:.1f}"/>',
           f'<text class="axis" x="{pad_l - 6}" y="{pad_t + 8}" text-anchor="end">+{top}</text>',
           f'<text class="axis" x="{pad_l - 6}" y="{mid + 4:.1f}" text-anchor="end">0</text>',
           f'<text class="axis" x="{pad_l - 6}" y="{h - pad_b}" text-anchor="end">-{top}</text>']
    prev_game = lines[0].get("game_no")
    for i, (r, d) in enumerate(zip(lines, diffs)):
        x = pad_l + i * bw
        if r.get("game_no") != prev_game:
            out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{h - pad_b}" stroke-dasharray="3 3"/>')
            prev_game = r.get("game_no")
        bh = max(half_h * abs(d) / top, 1.0)
        y = mid - bh if d > 0 else mid
        cls = _player_cls(a if d > 0 else b if d < 0 else None)
        sc = f'{r["score"].get(a, 0)}-{r["score"].get(b, 0)}'
        tip = f'第{r["rally_no"]}回合 · {_ms(r.get("start_ms"))} · {r.get("winner")} 得分 · 比分 {sc}'
        out.append(f'<rect class="sbar {cls}" x="{x + gap / 2:.1f}" y="{y:.1f}" width="{max(bw - gap, 1):.1f}" '
                   f'height="{bh:.1f}"><title>{_e(tip)}</title></rect>')
    chart = (f'<div class="chart"><div class="chart-head"><h3>领先分差（每个已标注胜负的回合之后）</h3>'
             f'<div class="legend">{_legend([a, b])}</div></div>'
             f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="比分走势">{"".join(out)}</svg>'
             '<p class="small muted">向上是 ' + _e(a) + ' 领先，向下是 ' + _e(b) + ' 领先；虚线是换局。</p></div>')
    m = _metrics(stats)
    facts = []
    lc = m.get("match.score.lead_changes")
    if lc and lc.get("value") is not None:
        facts.append(f"领先易手 {_num(lc['value'])} 次")
    for p in stats.get("players", []):
        pm = _metrics(p)
        run = pm.get(f"{p['player_id']}.win.longest_run")
        if run and run.get("value"):
            facts.append(f"{p['player_id']} 最长连续得分 {_num(run['value'])} 分")
    lead = "按已标注的回合胜负推算（21 分制，领先 2 分获胜，30 分封顶）。" + ("；".join(facts) + "。" if facts else "")
    return _section("score", "比分走势", f'<div class="tiles">{tiles}</div>{chart}', lead)


def _shot_types(stats: dict, shots: list[dict]) -> str:
    cards = []
    for p in stats.get("players", []) if stats.get("shot_types_available") else []:
        pid = p["player_id"]
        d = _dists(p).get(f"{pid}.shot_types")
        if not d or not d.get("n"):
            continue
        lost = _dists(p).get(f"{pid}.lost.last_shot_types")
        lost_html = ""
        if lost and lost.get("n"):
            lost_html = f'<h4>{_e(lost["label"])}</h4>{_hbars(lost["counts"], lost["n"], _player_cls(pid), 5)}'
        cards.append(f'<div class="card"><h3><span class="who {_player_cls(pid)}">{_e(pid)}</span> 击球类型 '
                     f'{_badge(d.get("status"), d.get("confidence"))}</h3>'
                     f'{_hbars(d["counts"], d["n"], _player_cls(pid))}{lost_html}</div>')
    layers = _layers(shots)
    if not cards and not layers:
        return ""
    return _section("shots", "击球类型", f'<div class="grid2">{"".join(cards)}</div>{layers}',
                    "按击球者统计（已修正的击球）。分类可信度低的击球也计入，请结合工作台复核。")


def _layers(shots: list[dict]) -> str:
    """Multi-layer description of the shots (hand, direction, contact, pace, role, intent), per player."""
    per: dict[str, dict[str, Counter]] = {}
    for s in shots:
        detail = ((s.get("stroke") or {}).get("detail")) or {}
        pid = s.get("hitter")
        if not detail or not pid:
            continue
        for key, _, _ in LAYERS:
            attr = detail.get(key)
            if attr and attr.get("value"):
                per.setdefault(pid, {}).setdefault(key, Counter())[attr["value"]] += 1
    if not per:
        return ""
    rows = []
    for key, label, values in LAYERS:
        cells = []
        for pid in sorted(per):
            c = per[pid].get(key, Counter())
            n = sum(c.values())
            if not n:
                cells.append('<td class="muted">—</td>')
                continue
            segs = []
            for i, (v, name) in enumerate(values.items()):
                if c.get(v):
                    pct = 100 * c[v] / n
                    segs.append(f'<span class="seg s{i}" style="width:{pct:.1f}%" title="{_e(name)}：{c[v]} 拍">'
                                f'</span>')
            text = " ".join(f'<span class="key"><i class="sw s{i}"></i>{_e(name)} {c[v]}</span>'
                            for i, (v, name) in enumerate(values.items()) if c.get(v))
            cells.append(f'<td><div class="stack">{"".join(segs)}</div>'
                         f'<div class="legend small">{text}<span class="muted">（{n} 拍）</span></div></td>')
        if any(per[p].get(key) for p in per):
            rows.append(f"<tr><th>{_e(label)}</th>{''.join(cells)}</tr>")
    head = "".join(f'<th><span class="who {_player_cls(p)}">{_e(p)}</span></th>' for p in sorted(per))
    return ('<div class="chart"><h3>击球的多层描述</h3><p class="small muted">每一拍除了击球类型，还按正反手、线路、击球点、'
            '出球速度、攻防和意图描述；色块的颜色只区分选项（与球员颜色无关），文字给出具体拍数。</p>'
            f'<div class="table-wrap"><table class="layers"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></div>')


def _styles(stats: dict) -> str:
    styles = stats.get("styles") or []
    if not styles:
        return ""
    cards = []
    for st in styles:
        subj = st.get("subject", "")
        scores = [s for s in st.get("scores", []) if s.get("score") is not None][:4]
        score_rows = "".join(
            f'<div class="hb"><span class="hb-k">{_e(s.get("name"))}</span><span class="hb-track">'
            f'<span class="hb-bar {_player_cls(subj)}" style="width:{s["score"]:.0f}%"></span></span>'
            f'<span class="hb-v">{s["score"]:.0f}</span></div>' for s in scores)
        reasons = ""
        primary = st.get("primary") or {}
        if primary.get("reasons"):
            reasons = '<ul class="small">' + "".join(f"<li>{_e(r)}</li>" for r in primary["reasons"][:4]) + "</ul>"
        reps = f'<p class="small muted">代表球员：{_e("、".join(primary["representatives"]))}</p>' \
            if primary.get("representatives") else ""
        tactics = [t for t in st.get("tactics", []) if t.get("uses")]
        tactic_html = ""
        if tactics:
            trs = "".join(
                f"<tr><td>{_e(t.get('name'))}</td><td class='num'>{t['uses']}</td>"
                f"<td class='num'>{_num(t.get('success_pct'), '%')}</td><td class='num'>{_num(t.get('forced_pct'), '%')}</td>"
                f"<td>{_e('、'.join(_ms(x) for x in t.get('examples_ms', [])))}</td></tr>" for t in tactics)
            tactic_html = ('<h4>战术套路</h4><div class="table-wrap"><table><thead><tr><th>套路</th><th class="num">次数</th>'
                           '<th class="num">得分率</th><th class="num">逼对手被动</th><th>回看</th></tr></thead>'
                           f"<tbody>{trs}</tbody></table></div>")
        phases = [ph for ph in st.get("phases", []) if ph.get("shots")]
        phase_html = ""
        if phases:
            names = {"attack": "进攻", "neutral": "相持", "defense": "防守"}
            items = "".join(
                f'<li><b>{names.get(ph["phase"], ph["phase"])}</b>（{ph["shots"]} 拍）：'
                + _e("、".join(f"{SHOT_ZH.get(k, k)} {v}" for k, v in sorted(ph.get("counts", {}).items(), key=lambda kv: -kv[1])[:4]))
                + (f'；合理选择 {_num(ph.get("recommended_pct"), "%")}' if ph.get("recommended_pct") is not None else "")
                + "</li>" for ph in phases)
            phase_html = f'<h4>攻防状态下的选择</h4><ul class="small">{items}</ul>'
        habits = st.get("habits", [])[:4]
        habit_html = ""
        if habits:
            items = "".join(f"<li>对手打{_e(SHOT_ZH.get(x['after'], x['after']))}后，多回{_e(SHOT_ZH.get(x['reply'], x['reply']))}"
                            f"（{x['n']}/{x['of']}，{x['pct']:.0f}%）</li>" for x in habits)
            habit_html = f'<h4>回球习惯</h4><ul class="small">{items}</ul>'
        advice = "".join(f"<li>{_e(a)}</li>" for a in st.get("advice", []))
        advice_html = f'<h4>建议</h4><ul>{advice}</ul>' if advice else ""
        note = f'<p class="note">{_e(st["note"])}</p>' if st.get("note") else ""
        cards.append(
            f'<div class="card"><h3><span class="who {_player_cls(subj)}">{_e(subj)}</span> {_e(st.get("label"))} '
            f'{_badge(st.get("status"), st.get("confidence"))}</h3>{note}'
            f'<p class="small muted">依据 {st.get("n_shots", 0)} 拍已分类的非发球击球，与 {len(st.get("scores", []))} 种打法原型比对。</p>'
            f'<div class="hbars">{score_rows}</div>{reasons}{reps}{phase_html}{tactic_html}{habit_html}{advice_html}</div>')
    return _section("style", "打法风格", f'<div class="grid2">{"".join(cards)}</div>',
                    "规则匹配，不是训练出来的模型：分数表示与每种打法原型的接近程度（0–100）。")


DOUBLES_TILES = [("formation.attack_fb_pct", "进攻时前后站位"), ("formation.defense_sbs_pct", "防守时左右站位"),
                 ("rotation.front_swaps_per_rally", "每回合前场换人"), ("rotation.switches_per_rally", "每回合攻守站位转换"),
                 ("gap.mean_m", "两人平均间距"), ("gap.close_pct", "两人过近"), ("gap.wide_pct", "两人过远"),
                 ("cover.ratio_pct", "覆盖双打半场")]
FINDING_KIND = {"strength": "优势", "weakness": "问题", "pattern": "规律"}


def _doubles(stats: dict) -> str:
    """Per pair: formation split, attack/defense formation, rotation, spacing, roles, who took which shots."""
    teams = stats.get("doubles") or []
    if not teams:
        return ""
    m = _metrics(stats)
    for p in stats.get("players", []):
        m.update(_metrics(p))
    dists = _dists(stats)
    cards = []
    for t in teams:
        team, (p, q) = t.get("team"), t.get("players", ["", ""])
        cls = _player_cls(team)
        note = f'<p class="note">{_e(t["note"])}</p>' if t.get("note") else ""
        fb, sbs = m.get(f"{team}.dbl.formation.fb_pct", {}), m.get(f"{team}.dbl.formation.sbs_pct", {})
        split = ""
        if fb.get("value") is not None and sbs.get("value") is not None:
            mid = max(0.0, 100 - fb["value"] - sbs["value"])
            segs = [("前后", fb["value"], "s0"), ("过渡", mid, "s2"), ("左右", sbs["value"], "s1")]
            bar = "".join(f'<span class="seg {c}" style="width:{v:.1f}%" title="{k} {v:.0f}%">{k if v >= 12 else ""}</span>'
                          for k, v, c in segs if v > 0)
            keys = "".join(f'<span class="key"><i class="sw {c}"></i>{k} {v:.0f}%</span>' for k, v, c in segs)
            split = (f'<h4>站位时间（{fb.get("n", 0)} 个采样）</h4><div class="stack">{bar}</div>'
                     f'<div class="legend">{keys}</div>')
        tiles = []
        for k, label in DOUBLES_TILES:
            mm = m.get(f"{team}.dbl.{k}")
            if not mm or mm.get("value") is None:
                continue
            tip = f' title="{_e(mm.get("note"))}"' if mm.get("note") else ""
            tiles.append(f'<div class="tile"{tip}><div class="tile-v">{_e(_num(mm["value"], mm.get("unit")))}</div>'
                         f'<div class="tile-k">{_e(label)}</div></div>')
        tile_html = f'<div class="tiles">{"".join(tiles)}</div>' if tiles else ""
        roles = "".join(
            f'<li><b>{_e(r.get("player_id"))}</b>：{_e(r.get("label"))}'
            + (f'<span class="muted">（{_e("，".join(r.get("reasons", [])))}）</span>' if r.get("reasons") else "")
            + "</li>" for r in t.get("roles", []))
        role_html = f'<h4>分工</h4><ul class="small">{roles}</ul>' if roles else ""
        rows = []
        for key in t.get("division", []):
            d = dists.get(key)
            if not d or not d.get("n"):
                continue
            a, b, n = d["counts"].get(p, 0), d["counts"].get(q, 0), d["n"]
            rows.append(f'<tr><th>{_e(d["label"].removeprefix("谁打"))}</th>'
                        f'<td class="num">{a}<div class="small muted">{100 * a / n:.0f}%</div></td>'
                        f'<td class="num">{b}<div class="small muted">{100 * b / n:.0f}%</div></td></tr>')
        div_html = ""
        if rows:
            div_html = (f'<h4>谁打什么球</h4><div class="table-wrap"><table><thead><tr><th>球</th>'
                        f'<th class="num">{_e(p)}</th><th class="num">{_e(q)}</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div>')
        fl = []
        for f in t.get("findings", []):
            kind = f.get("kind")
            adv = f'<p><b>建议：</b>{_e(f["advice"])}</p>' if f.get("advice") else ""
            fl.append(f'<li class="insight {_e(kind)}"><span class="kind">{_e(FINDING_KIND.get(kind, kind))}</span>'
                      f'<div><b>{_e(f.get("title"))}</b> {_badge(None, f.get("confidence"))}<p>{_e(f.get("detail"))}</p>'
                      f'{adv}</div></li>')
        find_html = f'<h4>观察与建议</h4><ul class="insights">{"".join(fl)}</ul>' if fl else ""
        cards.append(f'<div class="card"><h3><span class="who {cls}">{_e(team)} 队</span> {_e(p)} + {_e(q)} '
                     f'{_badge(t.get("status"), t.get("confidence"))}</h3>{note}{split}{tile_html}{role_html}'
                     f'{div_html}{find_html}</div>')
    return _section("doubles", "双打配合", f'<div class="grid2">{"".join(cards)}</div>',
                    "按两人离底线的距离差判断：相差 ≥ 1.5 米为前后站位（进攻阵型），< 1 米为左右站位（防守阵型）；"
                    "进攻 / 防守按上一拍推断（对手挑球后本方进攻，本方挑球后本方防守）。")


def _footwork(stats: dict) -> str:
    players = stats.get("players", [])
    per = {p["player_id"]: {k.split(".footwork.", 1)[1]: m for k, m in _metrics(p).items() if ".footwork." in k}
           for p in players}
    per = {pid: ms for pid, ms in per.items() if ms}
    if not per:
        return ""
    keys = [k for k in FOOTWORK_ORDER if any(k in ms for ms in per.values())]
    keys += sorted({k for ms in per.values() for k in ms} - set(keys))
    head = "".join(f'<th class="num"><span class="who {_player_cls(p)}">{_e(p)}</span></th>' for p in per)
    rows = []
    for k in keys:
        label = next(ms[k]["label"] for ms in per.values() if k in ms)
        cells = []
        for ms in per.values():
            m = ms.get(k)
            if not m:
                cells.append('<td class="num muted">—</td>')
                continue
            tip = f' title="{_e(m.get("note"))}"' if m.get("note") else ""
            cells.append(f'<td class="num"{tip}>{_e(_num(m.get("value"), m.get("unit")))}'
                         f'<div class="small muted">n={m.get("n", 0)} · {_e(STATUS.get(m.get("status"), ""))}</div></td>')
        rows.append(f"<tr><th>{_e(label)}</th>{''.join(cells)}</tr>")
    patterns = []
    for p in players:
        d = _dists(p).get(f"{p['player_id']}.footwork.patterns")
        if d and d.get("n"):
            patterns.append(f'<div class="card"><h3><span class="who {_player_cls(p["player_id"])}">{_e(p["player_id"])}</span> '
                            f'{_e(d["label"])}</h3>{_hbars(d["counts"], None, _player_cls(p["player_id"]))}</div>')
    notes = {m.get("note") for ms in per.values() for m in ms.values() if m.get("note") and "误差" in m.get("note", "")}
    note_html = "".join(f'<p class="note">{_e(n)}</p>' for n in sorted(notes))
    return _section("footwork", "脚步", f'<div class="table-wrap"><table><thead><tr><th>指标</th>{head}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>{note_html}<div class="grid2">{"".join(patterns)}</div>',
                    "从脚踝关键点投影到球场上得到；参考值：职业单打约 60% 的回合会在对手击球时做启动步。")


def _court_svg(cells: list[tuple[int, int, int]], cell_m: float, top: int, cls: str,
               dots: list[tuple[float, float, str]], label: str) -> str:
    sc = 26  # px per metre
    pad = 14
    W, L = COURT_WIDTH_M * sc, COURT_LENGTH_M * sc
    out = [f'<svg viewBox="{-pad} {-pad} {W + 2 * pad:.0f} {L + 2 * pad:.0f}" class="court" role="img" aria-label="{_e(label)}">',
           f'<rect class="floor" x="{-pad}" y="{-pad}" width="{W + 2 * pad:.0f}" height="{L + 2 * pad:.0f}" rx="6"/>']
    for ix, iy, c in cells:
        op = 0.12 + 0.88 * (c / top) ** 0.6
        out.append(f'<rect class="heat {cls}" x="{ix * cell_m * sc:.1f}" y="{iy * cell_m * sc:.1f}" '
                   f'width="{cell_m * sc:.1f}" height="{cell_m * sc:.1f}" fill-opacity="{op:.2f}"/>')
    # lines: outer (doubles), singles sidelines, short service lines, doubles long service lines, centre lines, net
    ln = []
    s_in = 0.46 * sc
    for y in (0, L):
        ln.append((0, y, W, y))
    ln += [(0, 0, 0, L), (W, 0, W, L), (s_in, 0, s_in, L), (W - s_in, 0, W - s_in, L)]
    net = NET_Y_M * sc
    for dy in (1.98 * sc, -1.98 * sc):
        ln.append((0, net + dy, W, net + dy))
    for y in (0.76 * sc, L - 0.76 * sc):
        ln.append((0, y, W, y))
    ln += [(W / 2, 0, W / 2, net - 1.98 * sc), (W / 2, net + 1.98 * sc, W / 2, L)]
    out += [f'<line class="cl" x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}"/>' for a, b, c, d in ln]
    out.append(f'<line class="net" x1="{-pad / 2}" y1="{net:.1f}" x2="{W + pad / 2}" y2="{net:.1f}"/>')
    for x, y, tip in dots:
        out.append(f'<circle class="dot {cls}" cx="{x * sc:.1f}" cy="{y * sc:.1f}" r="3.5"><title>{_e(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def _heatmaps(players_doc: dict | None, shots: list[dict], spans: list[tuple[int, int]], cell_m: float = 0.5) -> str:
    """Per player: where they stood during rallies (foot positions) and where they hit from."""
    nx, ny = int(-(-COURT_WIDTH_M // cell_m)), int(-(-COURT_LENGTH_M // cell_m))
    grids: dict[str, Counter] = {}
    if players_doc:
        starts = [a for a, _ in spans]
        for pf in players_doc.get("frames", []):
            t = pf.get("t_ms", 0)
            if spans:
                i = next((k for k in range(len(starts)) if starts[k] <= t <= spans[k][1]), None)
                if i is None:
                    continue
            for p in pf.get("players", []):
                fc, pid = p.get("foot_court"), p.get("player_id")
                if not fc or not pid or p.get("interpolated"):
                    continue
                ix, iy = int(fc[0] // cell_m), int(fc[1] // cell_m)
                if 0 <= ix < nx and 0 <= iy < ny:
                    grids.setdefault(pid, Counter())[(ix, iy)] += 1
    hits: dict[str, list[tuple[float, float, str]]] = {}
    for s in shots:
        pos, pid = s.get("hitter_court"), s.get("hitter")
        if pos and pid:
            tip = f"{_ms(s.get('t_ms'))} {SHOT_ZH.get(s.get('shot_type') or '', '击球')}"
            hits.setdefault(pid, []).append((pos[0], pos[1], tip))
    pids = sorted(set(grids) | set(hits))
    if not pids:
        return ""
    cards = []
    for pid in pids:
        g = grids.get(pid, Counter())
        top = max(g.values(), default=1)
        cells = [(ix, iy, c) for (ix, iy), c in g.items()]
        dots = hits.get(pid, [])
        cap = f"站位 {sum(g.values())} 帧" if g else "无站位数据"
        cap += f" · 击球点 {len(dots)} 拍" if dots else ""
        cards.append(f'<figure class="court-card"><figcaption><span class="who {_player_cls(pid)}">{_e(pid)}</span> {_e(cap)}'
                     f'</figcaption>{_court_svg(cells, cell_m, top, _player_cls(pid), dots, f"{pid} 站位热区与击球点")}</figure>')
    return _section("heatmap", "场地热区", f'<div class="courts">{"".join(cards)}</div>',
                    "色块越深 = 回合中站在那里的时间越长（0.5 米网格，按脚的位置）；圆点 = 击球时所在位置，悬停可看时间。"
                    "球场为视频中的固定方向，换边后两个半场都会出现。")


def _coach(coach: dict | None, stale: bool) -> str:
    if not coach:
        return ""
    parts = []
    if stale:
        parts.append('<p class="note">这份点评生成后数据又有修正，内容可能已过时。</p>')
    src = coach.get("provider", "") + (f" · {coach['model']}" if coach.get("model") else "")
    parts.append(f'<p class="small muted">生成：{_e(src)}</p><p class="summary">{_e(coach.get("summary"))}</p>')
    ins = []
    for i in coach.get("insights", []):
        who = f'<span class="who {_player_cls(i.get("player_id"))}">{_e(i["player_id"])}</span> ' if i.get("player_id") else ""
        warn = "" if i.get("grounded", True) else '<span class="badge needs_review">含未核实数字</span>'
        ins.append(f'<li class="insight {_e(i.get("kind"))}"><span class="kind">{_e(KIND.get(i.get("kind"), i.get("kind")))}</span>'
                   f'<div><b>{who}{_e(i.get("title"))}</b> {_badge(None, i.get("confidence"))}{warn}'
                   f'<p>{_e(i.get("detail"))}</p></div></li>')
    if ins:
        parts.append(f'<ul class="insights">{"".join(ins)}</ul>')
    train = []
    for t in coach.get("training", []):
        who = f'<span class="who {_player_cls(t.get("player_id"))}">{_e(t["player_id"])}</span> ' if t.get("player_id") else ""
        pts = "".join(f"<li>{_e(k)}</li>" for k in t.get("key_points", []))
        refs = "".join(f'<li>{_e(r.get("coach"))}《{_e(r.get("title"))}》</li>' for r in t.get("sources", []))
        focus = f"针对：{_e(t['focus'])}。" if t.get("focus") else ""
        train.append(f'<li><b>{who}{_e(t.get("drill"))}</b><p>{focus}{_e(t.get("why"))}</p>'
                     + (f'<ul class="small">{pts}</ul>' if pts else "")
                     + (f'<p class="small muted">参考：</p><ul class="small muted">{refs}</ul>' if refs else "") + "</li>")
    if train:
        parts.append(f'<h3>训练建议</h3><ol class="training">{"".join(train)}</ol>')
    return _section("coach", "AI 教练点评", "".join(parts))


def _shot_advice(coach: dict | None, stats: dict) -> str:
    items = (coach or {}).get("shot_advice") or []
    if items:
        rows = []
        for a in items:
            fw = f'<p class="small">脚步：{_e(a["footwork"])}</p>' if a.get("footwork") else ""
            rows.append(f'<li><span class="ts">{_e(a.get("timestamp"))}</span><div><b>{_e(a.get("situation"))}</b> '
                        f'{_badge(None, a.get("confidence"))}<p>{_e(a.get("advice"))}</p>'
                        f'<p class="small muted">{_e(a.get("why"))}</p>{fw}</div></li>')
        return _section("moments", "逐拍建议", _capped(rows),
                        "结合击球类型和双方站位找出的、可以换一种打法的击球；时间为视频时间，可在工作台跳转回看。")
    moments = stats.get("moments") or []
    if not moments:
        return ""
    rows = []
    for m in moments:
        label = m.get("shot_label_zh") or SHOT_ZH.get(m.get("shot_type"), m.get("shot_type"))
        rows.append(f'<li><span class="ts">{_ms(m.get("t_ms"))}</span><div><b>第{m.get("rally_no") or "?"}回合第{m.get("shot_no")}拍：'
                    f'{_e(m.get("player_id"))}在{_e(m.get("hitter_zone"))}打{_e(label)}</b> {_badge(None, m.get("confidence"))}'
                    f'<p>{_e(m.get("problem"))}。{_e(m.get("suggestion"))}。</p><p class="small muted">{_e(m.get("why"))}</p></div></li>')
    return _section("moments", "逐拍建议", _capped(rows))


def _capped(rows: list[str], first: int = 10) -> str:
    """The first ``first`` items in view, the rest behind a toggle (keeps the shared page short)."""
    head = f'<ol class="moments">{"".join(rows[:first])}</ol>'
    if len(rows) <= first:
        return head
    return (head + f'<details><summary>其余 {len(rows) - first} 拍</summary>'
            f'<ol class="moments" start="{first + 1}">{"".join(rows[first:])}</ol></details>')


def _details(stats: dict) -> str:
    blocks = []
    for title, src in [("全场", stats)] + [(p["player_id"], p) for p in stats.get("players", [])]:
        ms = [m for m in src.get("metrics", []) if ".footwork." not in m["key"] and ".style." not in m["key"]]
        if not ms:
            continue
        rows = "".join(f"<tr><td>{_e(m['label'])}</td><td class='num'>{_e(_num(m.get('value'), m.get('unit')))}</td>"
                       f"<td class='num'>{m.get('n', 0)}</td><td>{_badge(m.get('status'), m.get('confidence'))}</td>"
                       f"<td class='small muted'>{_e(m.get('note'))}</td></tr>" for m in ms)
        dl = "".join(f"<li>{_e(d['label'])}：" + _e("，".join(f"{k} {v}" for k, v in d['counts'].items())) + "</li>"
                     for d in src.get("distributions", []) if d.get("n") and ".footwork." not in d["key"])
        blocks.append(f'<details><summary>{_e(title)}：全部指标</summary><div class="table-wrap"><table><thead><tr>'
                      '<th>指标</th><th class="num">数值</th><th class="num">样本</th><th>状态</th><th>说明</th></tr></thead>'
                      f'<tbody>{rows}</tbody></table></div>' + (f'<ul class="small">{dl}</ul>' if dl else "") + "</details>")
    return _section("details", "详细数据", "".join(blocks)) if blocks else ""


def _caveats(stats: dict, coach: dict | None) -> str:
    cv = list(dict.fromkeys([*(stats.get("caveats") or []), *((coach or {}).get("caveats") or [])]))
    if not cv:
        return ""
    return _section("caveats", "数据说明", "<ul>" + "".join(f"<li>{_e(c)}</li>" for c in cv) + "</ul>")


# ----------------------------------------------------------------------------- page


def render_html(report: dict | Any, events: dict | Any | None = None, players: dict | Any | None = None,
                *, title: str | None = None, video_name: str | None = None,
                generated: datetime | None = None) -> str:
    """One self-contained HTML page for a match. ``report`` is a ``MatchReport`` (model or dict); ``events`` the
    effective ``EventsResult`` (rallies, shots with stroke details); ``players`` the ``PlayerTracks`` for the
    court heatmap. Pydantic models are dumped to dicts first; missing parts drop their section."""
    report, events, players = (x.model_dump(mode="json") if hasattr(x, "model_dump") else x
                               for x in (report, events, players))
    stats = report.get("stats") or {}
    coach = report.get("coach")
    rallies = sorted((events or {}).get("rallies", []), key=lambda r: r.get("start_ms", 0))
    shots = (events or {}).get("shots", [])
    shots_by_id = {s["shot_id"]: s for s in shots}
    spans = [(r.get("start_ms", 0), r.get("end_ms", 0)) for r in rallies]
    generated = generated or datetime.now(UTC)
    title = title or "赛后报告"
    body = [
        _header(stats, title, video_name, generated),
        _nav(),
        _overview(stats, rallies),
        _coach(coach, bool(report.get("coach_stale"))),
        _score_flow(stats),
        _rallies(stats, rallies, shots_by_id),
        _shot_types(stats, shots),
        _styles(stats),
        _doubles(stats),
        _footwork(stats),
        _heatmaps(players, shots, spans),
        _shot_advice(coach, stats),
        _caveats(stats, coach),
        _details(stats),
        f'<footer>由 RallyIQ 在本机生成 · 不含视频，不上传任何数据 · 统计指纹 {_e(stats.get("fingerprint"))}</footer>',
    ]
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_e(title)}</title><style>{CSS}</style></head>'
            f'<body><main>{"".join(b for b in body if b)}</main></body></html>\n')


def _nav() -> str:
    links = [("overview", "概览"), ("coach", "教练点评"), ("score", "比分"), ("rallies", "回合"), ("shots", "击球"), ("style", "风格"),
             ("doubles", "双打"), ("footwork", "脚步"), ("heatmap", "热区"), ("moments", "逐拍"), ("details", "数据")]
    return '<nav>' + "".join(f'<a href="#{a}">{b}</a>' for a, b in links) + "</nav>"


CSS = """
:root{color-scheme:light;--bg:#f6f5f2;--surface:#fcfcfb;--line:#e4e2dc;--grid:#ecebe6;--text:#0b0b0b;--text2:#52514e;
--muted:#7a7975;--p1:#2a78d6;--p2:#eb6834;--pn:#b9b7b0;--court:#eef3ee;--court-line:#9aa79a;--net:#52514e;
--warn-bg:#fff4dc;--warn:#8a5a00;--ok:#1d7a3a;--s0:#1baf7a;--s1:#6250d6;--s2:#eda100;--s3:#e87ba4;--s4:#008300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;--surface:#1a1a19;
--line:#2f2e2b;--grid:#262624;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e88;--p1:#3987e5;--p2:#d95926;--pn:#55544f;
--court:#18211a;--court-line:#4f5f50;--net:#c3c2b7;--warn-bg:#3a2d10;--warn:#f0c46a;--ok:#5fc27e;
--s0:#199e70;--s1:#9085e9;--s2:#c98500;--s3:#d55181;--s4:#008300}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--line:#2f2e2b;--grid:#262624;--text:#f0efec;
--text2:#c3c2b7;--muted:#8f8e88;--p1:#3987e5;--p2:#d95926;--pn:#55544f;--court:#18211a;--court-line:#4f5f50;--net:#c3c2b7;
--warn-bg:#3a2d10;--warn:#f0c46a;--ok:#5fc27e;--s0:#199e70;--s1:#9085e9;--s2:#c98500;--s3:#d55181;--s4:#008300}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",
"Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
header{padding:8px 0 4px}
.eyebrow{margin:0;color:var(--muted);font-size:13px;letter-spacing:.04em}
h1{margin:4px 0 2px;font-size:28px;line-height:1.25}
.sub{margin:0 0 8px;color:var(--text2)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{font-size:12px;color:var(--text2);border:1px solid var(--line);border-radius:999px;padding:2px 10px;background:var(--surface)}
nav{position:sticky;top:0;z-index:2;display:flex;gap:4px;overflow-x:auto;margin:16px -16px 8px;padding:8px 16px;
background:var(--bg);border-bottom:1px solid var(--line)}
nav a{flex:none;color:var(--text2);text-decoration:none;font-size:13px;padding:4px 10px;border-radius:6px}
nav a:hover{background:var(--surface);color:var(--text)}
section{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0;
scroll-margin-top:56px}
h2{margin:0 0 6px;font-size:19px}
h3{margin:0 0 10px;font-size:15px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
h4{margin:14px 0 6px;font-size:13px;color:var(--text2)}
p{margin:6px 0}
.lead{color:var(--text2);font-size:13px;margin:0 0 14px}
.small{font-size:13px}.muted{color:var(--muted)}
.note{background:var(--warn-bg);color:var(--warn);border-radius:8px;padding:6px 10px;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.tile{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile-v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.2}
.tile-k{font-size:12px;color:var(--text2)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:14px;align-items:start}
.card,.chart{border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.chart-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.chart svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.axis{fill:var(--muted);font-size:11px}
.bar.p1,.sbar.p1{fill:var(--p1)}.bar.p2,.sbar.p2{fill:var(--p2)}.bar.pn,.sbar.pn{fill:var(--pn)}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--text2)}
.key{display:inline-flex;align-items:center;gap:4px}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px}
.sw.p1,.hb-bar.p1{background:var(--p1)}.sw.p2,.hb-bar.p2{background:var(--p2)}.sw.pn,.hb-bar.pn{background:var(--pn)}
.who{display:inline-block;font-size:12px;font-weight:650;color:#fff;border-radius:5px;padding:0 6px;line-height:20px}
.who.p1{background:var(--p1)}.who.p2{background:var(--p2)}.who.pn{background:var(--pn)}
.hbars{display:grid;gap:5px}
.hb{display:grid;grid-template-columns:minmax(64px,auto) 1fr 64px;align-items:center;gap:8px;font-size:13px}
.hb-k{white-space:nowrap;color:var(--text2)}
.hb-track{height:12px;background:var(--grid);border-radius:4px;overflow:hidden}
.hb-bar{display:block;height:100%;border-radius:4px;min-width:2px}
.hb-v{text-align:right;font-variant-numeric:tabular-nums}.hb-v small{color:var(--muted);margin-left:4px}
.stack{display:flex;height:14px;margin-bottom:4px;border-radius:4px;overflow:hidden;gap:2px;min-width:160px}
.seg{display:block;height:100%;font-size:11px;color:#fff;text-align:center;line-height:18px;overflow:hidden;white-space:nowrap}
.seg.s0,.sw.s0{background:var(--s0)}.seg.s1,.sw.s1{background:var(--s1)}.seg.s2,.sw.s2{background:var(--s2)}.seg.s3,.sw.s3{background:var(--s3)}.seg.s4,.sw.s4{background:var(--s4)}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--text2)}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
table.layers th:first-child{white-space:nowrap}
details{margin-top:12px}summary{cursor:pointer;color:var(--text2);font-size:13px}
.badge{font-size:11px;font-weight:500;border-radius:5px;padding:1px 6px;border:1px solid var(--line);color:var(--text2)}
.badge.ok{color:var(--ok)}.badge.needs_review{background:var(--warn-bg);color:var(--warn);border-color:transparent}
.courts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,260px));gap:16px;justify-content:center}
.court-card{margin:0}.court-card figcaption{font-size:13px;color:var(--text2);margin-bottom:6px}
.court{width:100%;height:auto;display:block}
.floor{fill:var(--court)}.cl{stroke:var(--court-line);stroke-width:1.5}.net{stroke:var(--net);stroke-width:3;stroke-dasharray:4 3}
.heat.p1{fill:var(--p1)}.heat.p2{fill:var(--p2)}.heat.pn{fill:var(--pn)}
.dot{stroke:var(--surface);stroke-width:2;fill:var(--text)}.dot.p1{fill:var(--text)}.dot.p2{fill:var(--text)}
.summary{font-size:15px}
.insights{list-style:none;padding:0;margin:12px 0;display:grid;gap:8px}
.insight{display:grid;grid-template-columns:64px 1fr;gap:10px;border-top:1px solid var(--line);padding-top:8px}
.insight .kind{font-size:12px;color:var(--text2)}.insight.weakness .kind{color:var(--warn)}.insight.strength .kind{color:var(--ok)}
.insight p{margin:2px 0 0;color:var(--text2);font-size:14px}
.training li,.moments li{margin-bottom:10px}
.moments{list-style:none;padding:0}
.moments li{display:grid;grid-template-columns:56px 1fr;gap:10px;border-top:1px solid var(--line);padding-top:8px}
.ts{font-variant-numeric:tabular-nums;font-weight:650;color:var(--text2)}
.moments p{margin:2px 0}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:24px}
@media (max-width:560px){h1{font-size:23px}section{padding:14px}.insight,.moments li{grid-template-columns:1fr}}
@media print{nav{display:none}body{background:#fff}section{break-inside:avoid;border-color:#ddd}details{display:block}
details>summary{display:none}details:not([open])>*{display:block}}
"""
