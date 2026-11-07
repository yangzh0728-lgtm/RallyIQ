"""Prompt and output schema shared by every LLM provider."""

from __future__ import annotations

import json

from ...contracts import MatchStats
from ..moments import ZH, timestamp

SYSTEM_PROMPT = """你是 RallyIQ 的羽毛球单打教练助手。你会收到一场比赛（或训练）视频经自动识别和人工修正后计算出的统计数据（JSON），请据此给出中文赛后点评。

数据规则（必须遵守）：
1. 只依据给定数据说话。每条观点的 evidence 字段必须列出支撑它的指标 key（来自输入里的 key 字段），不得引用不存在的 key。
2. 文中出现的数字必须直接取自所引用指标的 value、n 或分布里的计数/占比，可以四舍五入，但不要自己推算新的数字。
3. 每个指标都带有 status（ok / needs_review / undeterminable）和 confidence（high / medium / low）。
   - 观点的 confidence 不能高于它所引用指标中最低的那个。
   - 引用 needs_review 或 low 的指标时，在 detail 里用“初步看”“样本较少”等措辞说明不确定性，并给出样本量。
   - status 为 undeterminable 或 value 为 null 的指标不能作为依据；相关方面可以说“证据不足”。
4. shot_types_available 为 false 时，击球类型分类尚未接入：不要评价具体技术动作、拍法或球路选择。
5. 落点、球速、关节角度等数据未提供，不要臆测。不对伤病、身体状况或情绪做判断。
6. 胜负相关指标只基于已标注胜者的回合，引用时注明回合数。

内容要求：
- summary：3–5 句话的总体评价，先说最重要的发现。
- insights：3–8 条，覆盖优势、问题、规律和战术建议；kind 取 strength / weakness / pattern / tactic / data_quality；如果数据质量明显限制了结论，加一条 data_quality。
- training：1–5 项可执行的训练建议，说明理由，并在 retest_metric 里给出下次上传后用来复测的指标 key（可为 null）。
- caveats：需要提醒使用者的数据局限。
- 球员用 P1、P2 称呼（双打还有 P3、P4：A 队 = P1+P3，B 队 = P2+P4，回合胜负记在队伍上）。语言简洁、具体，像教练当面讲解，不要空话。

教学笔记（knowledge 字段，可能为空）：
- 系统已根据数据找出每位球员的问题（focus），并为每个问题检索了职业球员（安赛龙、郑思维等）公开教学视频的中文学习笔记（notes）。
- training 优先针对这些 focus：每项填 focus_id，drill 写具体练法（组数/次数/要点，可改编笔记里的 drills），why 说明数据上的问题和这个练习为什么对症，key_points 写 2–4 条取自笔记的技术要点或口令（coach_cues），sources 列出所依据笔记的 note_id（以及用到的 drill 名称，没有则为 null）。
- 只能引用 knowledge 里给出的 note_id，不要编造视频、链接或时间点；笔记是改写的学习笔记，不要写成球员原话，可以说“安赛龙在视频里强调……”。
- 笔记只说明“怎么练”，不能当作数据证据：evidence 仍然只能是统计指标 key；retest_metric 优先用 focus 给出的复测指标。
- 如果某个 focus 的数据可信度低，可以写进 training，但在 why 里说明“样本较少，先当作观察”；没有合适笔记时 sources 留空。
- training 最多 5 项，按重要性排序；insights 中对应的问题可以简要提一句“见训练建议”。

打法风格（统计里的 styles 字段，可能为空）：
- 系统用规则把每位球员（双打为每队）的数据和打法原型做了匹配：单打 8 种（强攻压制、速度抢攻、拉吊突击、控制组织、防守反击、多拍消耗、技术欺骗、节奏变化），双打 5 种体系（后场强攻、平抽快挡、网前预判控制、防守转换、混双前封后攻）。每条有 label（结论）、primary/secondary（score 0–100 为吻合度，reasons 为主要依据）、phases（主动/中立/被动状态下各打了什么，recommended_pct 为符合该状态推荐打法的比例）、tactics（6 种战术套路的使用次数、所在回合胜率 success_pct、逼对手被动的比例 forced_pct）、habits（对手打某种球后此人最常见的回球，可用来预测下一拍）、advice（规则建议）。
- 特征数值在各球员的 <Pn>.style.* 指标里（双打为 A.style.* / B.style.*），引用打法结论时 evidence 写这些 key（如 P1.style.smash_pct、P1.style.match_score），数字只能取自这些指标。
- 写 1–2 条 pattern 观点说明每位球员的打法、为什么这么判断；再写 tactic 观点：这种打法执行得是否到位（主动时是否抓住机会进攻、被动时是否稳住、哪个战术套路有效或无效），以及针对对手打法和回球习惯，下一拍/下一局应该怎么选择。
- status 为 undeterminable 或 label 为“证据不足”时不要判断打法；score 低于 60 表示没有明显倾向（均衡型），不要硬套某种打法。
- 打法是倾向而不是定论：参考区间来自职业单打统计，业余比赛只作参考，措辞用“接近”“偏向”。

逐拍建议（moments 字段，可能为空）：
- 系统已结合击球类型和双方在球场上的站位，用规则找出了需要改变球路选择的击球（moments）。每条有 moment_id、视频时间 time、第几回合第几拍、击球者 player_id、击球类型、击球者位置 hitter_zone（本人视角）、对手位置 opponent_zone（从击球者这边看过去的左右）、对手接这一拍移动的距离 opponent_reach_m、对手的回球 next_shot、结果 outcome，以及规则给出的 problem / suggestion / why。
- shot_advice 为每个 moment 写一条：moment_id 原样照抄；advice 用一两句话说这一拍应该怎么打（打什么球、送到对方哪个区域），why 结合“我在哪、对手在哪、对手怎么回”解释为什么这样更好。可以参考规则的 suggestion，但要讲得像教练当面指出那一拍。
- 不要写或改动时间、回合号、拍数和站位，这些由系统显示；不要编造 moments 里没有的击球。confidence 为 low 的击球，在 why 里说“识别把握不大，可以回看这一拍确认”。
- 每条 moment 的 state 是击球者当时的攻防状态（主动 / 中立 / 被动，按来球判断）。主动时推荐杀球、点杀、劈吊、抢网、连续进攻；中立时推荐四角调动、直线控制、拉吊组织、节奏变化；被动时推荐高质量解围、接杀挡网、快速回位、防守反击。advice 要符合这一拍的状态。
- 输入里说明了谁是使用者本人时，用“你”称呼本人。"""

_PLAYER = {"anyOf": [{"type": "string", "enum": ["P1", "P2", "P3", "P4"]}, {"type": "null"}]}

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["strength", "weakness", "pattern", "tactic", "data_quality"]},
                    "player_id": _PLAYER,
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["kind", "player_id", "title", "detail", "evidence", "confidence"],
                "additionalProperties": False,
            },
        },
        "training": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "player_id": _PLAYER,
                    "drill": {"type": "string"},
                    "why": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "retest_metric": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "focus_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "note_id": {"type": "string"},
                                "drill": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            },
                            "required": ["note_id", "drill"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["player_id", "drill", "why", "evidence", "retest_metric", "focus_id", "key_points",
                             "sources"],
                "additionalProperties": False,
            },
        },
        "shot_advice": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "moment_id": {"type": "string"},
                    "advice": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["moment_id", "advice", "why"],
                "additionalProperties": False,
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "insights", "training", "shot_advice", "caveats"],
    "additionalProperties": False,
}


def stats_payload(stats: MatchStats) -> str:
    """Compact, deterministic JSON of the stats for the prompt (cache friendly)."""
    data = stats.model_dump(mode="json", exclude={"schema_version", "video_id", "fingerprint", "moments", "styles"})
    if stats.styles:
        data["styles"] = [style_card(st) for st in stats.styles]
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def style_card(st) -> dict:
    """Compact style summary for the prompt (a small local model has limited context)."""
    def score(x):
        return None if x is None else {"name": x.name, "score": x.score, "reasons": x.reasons}

    return {
        "subject": st.subject, "kind": st.kind, "label": st.label, "status": st.status, "confidence": st.confidence,
        "n_shots": st.n_shots, "primary": score(st.primary), "secondary": score(st.secondary),
        "others": {x.name: x.score for x in st.scores[1:4] if x.score is not None},
        "phases": [{"state": p.phase, "shots": p.shots, "counts": {ZH.get(k, k): v for k, v in p.counts.items()},
                    "recommended_pct": p.recommended_pct} for p in st.phases],
        "tactics": [t.model_dump(mode="json", exclude={"code", "examples_ms"}) for t in st.tactics if t.uses],
        "habits": [f"接{ZH.get(h.after, h.after)}时回{ZH.get(h.reply, h.reply)} {h.pct:.0f}%（{h.n}/{h.of}）"
                   for h in st.habits],
        "advice": st.advice, "evidence": st.evidence, "note": st.note,
    }


def knowledge_payload(knowledge) -> str:
    items = knowledge.prompt_payload() if knowledge is not None else []
    return json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def moment_card(m) -> dict:
    return {
        "moment_id": m.moment_id, "time": timestamp(m.t_ms), "rally_no": m.rally_no, "shot_no": m.shot_no,
        "player_id": m.player_id, "shot": ZH.get(m.shot_type, m.shot_type), "hitter_zone": m.hitter_zone,
        "opponent_zone": m.opponent_zone, "opponent_reach_m": m.opponent_reach_m,
        "next_shot": ZH.get(m.next_shot_type, m.next_shot_type) if m.next_shot_type else None,
        "state": {"attack": "主动", "neutral": "中立", "defense": "被动"}.get(m.phase or "", None),
        "outcome": m.outcome, "problem": m.problem, "suggestion": m.suggestion, "why": m.why,
        "confidence": m.confidence,
    }


def moments_payload(knowledge) -> str:
    items = [moment_card(m) for m in (knowledge.moments if knowledge is not None else [])]
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def user_message(stats: MatchStats, knowledge=None) -> str:
    msg = "以下是本场统计数据（JSON）。请按要求输出点评。\n\n" + stats_payload(stats)
    me = getattr(knowledge, "self_player", None)
    if me:
        msg += f"\n\n使用者本人是 {me}（点评里称“你”），对手是另一方。"
    if knowledge is not None and knowledge.focus:
        msg += "\n\n以下是系统找出的问题（focus）和检索到的教学笔记（knowledge，JSON）：\n\n" + knowledge_payload(knowledge)
    if knowledge is not None and knowledge.moments:
        msg += ("\n\n以下是需要改变球路选择的击球（moments，JSON，按时间顺序）。请为每条写一条 shot_advice：\n\n"
                + moments_payload(knowledge))
    return msg
