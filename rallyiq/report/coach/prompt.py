"""Prompt and output schema shared by every LLM provider."""

from __future__ import annotations

import json

from ...contracts import MatchStats

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
- training：1–3 项可执行的训练建议，说明理由，并在 retest_metric 里给出下次上传后用来复测的指标 key（可为 null）。
- caveats：需要提醒使用者的数据局限。
- 球员用 P1、P2 称呼（双打还有 P3、P4：A 队 = P1+P3，B 队 = P2+P4，回合胜负记在队伍上）。语言简洁、具体，像教练当面讲解，不要空话。"""

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
                },
                "required": ["player_id", "drill", "why", "evidence", "retest_metric"],
                "additionalProperties": False,
            },
        },
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "insights", "training", "caveats"],
    "additionalProperties": False,
}


def stats_payload(stats: MatchStats) -> str:
    """Compact, deterministic JSON of the stats for the prompt (cache friendly)."""
    data = stats.model_dump(mode="json", exclude={"schema_version", "video_id", "fingerprint"})
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def user_message(stats: MatchStats) -> str:
    return "以下是本场统计数据（JSON）。请按要求输出点评。\n\n" + stats_payload(stats)
