"""Rule-based coach: deterministic Chinese comments from ``MatchStats``.

Used when no LLM is configured (or as an explicit choice). It only states what
the rules can read directly off the metrics, so it is always grounded, but it is
far less insightful than an LLM.
"""

from __future__ import annotations

from ...contracts import Distribution, MatchStats, Metric
from ..stats import index

_RANK = ["low", "medium", "high"]


def _weakest(*confs: str) -> str:
    return min(confs, key=_RANK.index)

OTHER = {"P1": "P2", "P2": "P1"}


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def _hedge(e: Metric | Distribution) -> str:
    return "" if e.confidence == "high" else f"（样本 {e.n}，初步判断）"


class OfflineProvider:
    name = "offline"
    model = "rules@1"

    def generate(self, stats: MatchStats, knowledge=None) -> dict:
        idx = index(stats)

        def m(key: str) -> Metric | None:
            e = idx.get(key)
            return e if isinstance(e, Metric) and e.value is not None and e.status != "undeterminable" else None

        def dist(key: str) -> Distribution | None:
            e = idx.get(key)
            return e if isinstance(e, Distribution) and e.n > 0 and e.status != "undeterminable" else None

        insights: list[dict] = []
        training: list[dict] = []

        # --- data quality first
        conf = m("match.confirmed_shots_pct")
        if conf is not None and conf.value < 50:
            insights.append(dict(kind="data_quality", player_id=None, title="多数击球尚未复核",
                                 detail=f"只有 {_fmt(conf.value)}% 的击球经过人工确认，以下结论基于自动识别，可能受漏检和误检影响。",
                                 evidence=[conf.key], confidence="medium"))

        # --- rally rhythm
        mean_len, rallies = m("match.rally_len.mean"), m("match.rallies")
        if mean_len is not None and rallies is not None:
            style = "以短回合为主，节奏偏快" if mean_len.value <= 5 else "多拍相持较多" if mean_len.value >= 9 else "回合长度适中"
            insights.append(dict(kind="pattern", player_id=None, title=f"本场{style}",
                                 detail=f"{_fmt(rallies.value)} 个回合平均每回合 {_fmt(mean_len.value)} 拍{_hedge(mean_len)}。",
                                 evidence=[mean_len.key, rallies.key], confidence=mean_len.confidence))

        # --- movement comparison
        mv = {p: m(f"{p}.move.per_rally_m") for p in ("P1", "P2")}
        if all(mv.values()):
            a, b = mv["P1"], mv["P2"]
            hi, lo = (a, b) if a.value >= b.value else (b, a)
            if lo.value > 0 and hi.value / lo.value >= 1.2:
                pid = hi.key.split(".")[0]
                insights.append(dict(kind="pattern", player_id=pid, title=f"{pid} 跑动明显更多",
                                     detail=f"{pid} 每回合平均移动 {_fmt(hi.value)} 米，对手为 {_fmt(lo.value)} 米{_hedge(hi)}。"
                                            "跑动多可能意味着被对手调动，结合胜负数据判断是否处于被动。",
                                     evidence=[hi.key, lo.key], confidence=_weakest(hi.confidence, lo.confidence)))

        for pid in ("P1", "P2"):
            depth = dist(f"{pid}.pos.depth_zones")
            if depth is not None:
                zone, cnt = max(depth.counts.items(), key=lambda kv: kv[1])
                share = 100 * cnt / depth.n
                if share >= 50:
                    insights.append(dict(kind="pattern", player_id=pid, title=f"{pid} 回合中多停留在{zone}",
                                         detail=f"回合时间里约 {_fmt(share)}% 的帧位于{zone}{_hedge(depth)}。",
                                         evidence=[depth.key], confidence=depth.confidence))
                    if zone == "后场":
                        training.append(dict(player_id=pid, drill="后场到网前的启动与回位步法（每组 6 个来回，4 组）",
                                             why=f"{pid} 回合中大部分时间位于后场，前场覆盖可能不足。",
                                             evidence=[depth.key], retest_metric=f"{pid}.pos.depth_mean_m"))

            cover = m(f"{pid}.cover.ratio_pct")
            if cover is not None and cover.value < 45:
                insights.append(dict(kind="weakness", player_id=pid, title=f"{pid} 场地覆盖偏窄",
                                     detail=f"{pid} 只到达了单打半场约 {_fmt(cover.value)}% 的区域{_hedge(cover)}。",
                                     evidence=[cover.key], confidence=cover.confidence))

            short, long_ = m(f"{pid}.win.short_rate_pct"), m(f"{pid}.win.long_rate_pct")
            if (short is not None and long_ is not None and abs(short.value - long_.value) >= 20
                    and _weakest(short.confidence, long_.confidence) != "low"):
                better, worse = (short, long_) if short.value > long_.value else (long_, short)
                what = "短回合" if better is short else "长回合"
                insights.append(dict(kind="strength", player_id=pid, title=f"{pid} 在{what}中更占优",
                                     detail=f"{pid} {better.label} {_fmt(better.value)}%（{better.n} 回合），"
                                            f"{worse.label} {_fmt(worse.value)}%（{worse.n} 回合）。",
                                     evidence=[better.key, worse.key],
                                     confidence=_weakest(better.confidence, worse.confidence)))
                if better is short:
                    training.append(dict(player_id=pid, drill="多拍相持练习：一攻一守连续 15 拍不失误",
                                         why=f"{pid} 长回合胜率低于短回合，需要提高相持稳定性。",
                                         evidence=[long_.key, short.key], retest_metric=long_.key))

            lost_own = m(f"{pid}.win.lost_own_last_shot_pct")
            if lost_own is not None and lost_own.value >= 60 and lost_own.confidence != "low":
                insights.append(dict(kind="weakness", player_id=pid, title=f"{pid} 失分多在自己最后一拍",
                                     detail=f"{pid} 失分回合中有 {_fmt(lost_own.value)}% 的最后一拍由自己击出"
                                            f"（{lost_own.n} 回合），可能是主动失误偏多{_hedge(lost_own)}。",
                                     evidence=[lost_own.key], confidence=lost_own.confidence))

        if not stats.shot_types_available:
            insights.append(dict(kind="data_quality", player_id=None, title="暂不评价技术动作",
                                 detail="击球类型分类尚未接入，本报告不涉及拍法和球路选择。", evidence=[], confidence="high"))

        winners = m("match.winners_known")
        if winners is not None and winners.value == 0:
            insights.append(dict(kind="data_quality", player_id=None, title="得失分模式证据不足",
                                 detail="还没有标注任何回合的胜者。在修正面板中为回合设置胜者后，可分析得失分规律。",
                                 evidence=[winners.key], confidence="high"))

        noted = _knowledge_training(knowledge)
        if noted:
            training = noted + [t for t in training if t["player_id"] not in {n["player_id"] for n in noted}]

        n_main = sum(1 for i in insights if i["kind"] != "data_quality")
        summary = (f"本场共 {_fmt(rallies.value)} 个回合。" if rallies is not None else "") + (
            f"根据规则模板得到 {n_main} 条数据观察。" if n_main else "现有数据不足以给出可靠的观察。"
        ) + "这是离线规则生成的点评，配置 Claude API 后可获得更完整的教练分析。"
        return {"summary": summary, "insights": insights, "training": training[:4], "caveats": []}


def _knowledge_training(knowledge) -> list[dict]:
    """One drill per weakness, taken from the best-matching coaching note."""
    if knowledge is None:
        return []
    out = []
    for f in knowledge.focus:
        ids = knowledge.by_focus.get(f.id) or []
        if not ids:
            continue
        note = knowledge.notes[ids[0]]
        drills = note.get("drills") or []
        d = drills[0] if drills else None
        drill = f"{d['name']}：{d['how']}" if d else str(note.get("topic", ""))
        hedge = "" if f.confidence == "high" else "（样本较少，先当作观察）"
        out.append(dict(
            player_id=f.player_id, focus_id=f.id, drill=drill,
            why=f"{f.title}{hedge}。参考{note.get('coach', '')}《{note.get('title', '')}》：{note.get('topic', '')}",
            evidence=f.evidence, retest_metric=f.retest_metric,
            key_points=list(note.get("coach_cues") or note.get("key_points") or [])[:4],
            sources=[{"note_id": i, "drill": d["name"] if d and i == ids[0] else None} for i in ids[:2]],
        ))
    return out
