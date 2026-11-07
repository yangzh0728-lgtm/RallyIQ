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

逐拍建议（moments 字段，可能为空）：
- 系统已结合击球类型和双方在球场上的站位，用规则找出了需要改变球路选择的击球（moments）。每条有 moment_id、视频时间 time、第几回合第几拍、击球者 player_id、击球类型、击球者位置 hitter_zone（本人视角）、对手位置 opponent_zone（从击球者这边看过去的左右）、对手接这一拍移动的距离 opponent_reach_m、对手的回球 next_shot、结果 outcome，以及规则给出的 problem / suggestion / why。shot 可能带正反手和直线/斜线（如“反手斜线挑球”）；shot_detail 是这一拍的分层描述，其中“主动进攻/相持/被动防守”和“意图”是系统按规则推断的，只作参考，不要当成确定事实。
- shot_advice 为每个 moment 写一条：moment_id 原样照抄；advice 用一两句话说这一拍应该怎么打（打什么球、送到对方哪个区域），why 结合“我在哪、对手在哪、对手怎么回”解释为什么这样更好。可以参考规则的 suggestion，但要讲得像教练当面指出那一拍。
- 不要写或改动时间、回合号、拍数和站位，这些由系统显示；不要编造 moments 里没有的击球。confidence 为 low 的击球，在 why 里说“识别把握不大，可以回看这一拍确认”。
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
    data = stats.model_dump(mode="json", exclude={"schema_version", "video_id", "fingerprint", "moments"})
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def knowledge_payload(knowledge) -> str:
    items = knowledge.prompt_payload() if knowledge is not None else []
    return json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def moment_card(m) -> dict:
    return {
        "moment_id": m.moment_id, "time": timestamp(m.t_ms), "rally_no": m.rally_no, "shot_no": m.shot_no,
        "player_id": m.player_id, "shot": m.shot_label_zh or ZH.get(m.shot_type, m.shot_type),
        "shot_detail": m.shot_detail_zh, "hitter_zone": m.hitter_zone,
        "opponent_zone": m.opponent_zone, "opponent_reach_m": m.opponent_reach_m,
        "next_shot": ZH.get(m.next_shot_type, m.next_shot_type) if m.next_shot_type else None,
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
