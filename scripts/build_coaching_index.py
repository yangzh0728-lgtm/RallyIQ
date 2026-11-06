"""Regenerate the human-readable views of the coaching notes.

Writes ``knowledge/coaching/INDEX.md`` (one row per episode, with the specific shots and
footwork it teaches) and ``knowledge/coaching/readable/<coach>/<id>.md`` (each JSON note
rendered as a page). The JSON notes are the source of truth.

    python scripts/build_coaching_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallyiq.report.coach.knowledge import KNOWLEDGE_DIR, taxonomy  # noqa: E402

HEADER = "由 `scripts/build_coaching_index.py` 从 JSON 笔记生成，不要手改。"


def bullets(items: list[str]) -> list[str]:
    return [f"- {x}" for x in items] or ["- （无）"]


def render_note(n: dict, tax: dict) -> str:
    out = [f"# {n['title']}", "", HEADER, "",
           f"- 球员：{n['coach']}　平台：{n['platform']}　发布：{n.get('published', '')}　时长：{n.get('duration_s', '?')} 秒",
           f"- 原视频：{n['url']}",
           f"- 难度：{n['level']}　字幕来源：{n['transcript_source']}",
           f"- 主题：{n['topic']}", "", "## 概述", "", n["summary"], "",
           "## 详细描述", "", n["detailed_description"], ""]
    if n["shots"]:
        out += ["## 具体击球", "", "| 击球 | 类型 | 正/反手 | 场区 | 线路 |", "|---|---|---|---|---|"]
        for s in n["shots"]:
            out.append(f"| {s['name_zh']} | {tax['strokes'][s['stroke']]['zh']} | {tax['shot_hand'][s['hand']]} "
                       f"| {tax['shot_zone'][s['zone']]} | {tax['shot_direction'][s['direction']]} |")
        out.append("")
    if n["footwork"]:
        out += ["## 具体步法", "", *bullets(n["footwork"]), ""]
    if n["technique_breakdown"]:
        out += ["## 动作分解", "", *[f"- **{k}**：{v}" for k, v in n["technique_breakdown"].items()], ""]
    out += ["## 技术要点", "", *bullets(n["key_points"]), ""]
    if n["when_to_use"]:
        out += ["## 使用时机", "", *bullets(n["when_to_use"]), ""]
    out += ["## 常见错误", "", *bullets(n["common_mistakes"]), ""]
    if n["corrections"]:
        out += ["## 纠错实例", "", "| 问题 | 球员的提示 | 效果 |", "|---|---|---|",
                *[f"| {c['problem']} | {c['cue']} | {c['result']} |" for c in n["corrections"]], ""]
    if n["drills"]:
        out += ["## 练习方法", "", *[f"- **{d['name']}**：{d['how']}" for d in n["drills"]], ""]
    if n["progression"]:
        out += ["## 进阶顺序", "", *[f"{i}. {p}" for i, p in enumerate(n["progression"], 1)], ""]
    if n["self_check"]:
        out += ["## 自查清单", "", *[f"- [ ] {q}" for q in n["self_check"]], ""]
    out += ["## 口令", "", "、".join(n["coach_cues"]), ""]
    if n["related_metrics"]:
        out += ["## 相关 RallyIQ 指标", "", *[f"- `{m}`" for m in n["related_metrics"]], ""]
    if n.get("content_caveats"):
        out += ["## 说明", "", n["content_caveats"], ""]
    return "\n".join(out)


def main() -> None:
    tax = taxonomy()
    rows = []
    for f in sorted((KNOWLEDGE_DIR / "notes").glob("*/*.json")):
        n = json.loads(f.read_text(encoding="utf-8"))
        page = KNOWLEDGE_DIR / "readable" / f.parent.name / f"{f.stem}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_note(n, tax), encoding="utf-8")
        shots = "<br>".join(s["name_zh"] for s in n["shots"]) or "—"
        footwork = "<br>".join(n["footwork"]) or "—"
        title = n["title"].replace("|", "/")
        rows.append((n["coach"], n.get("published", ""),
                     f"| {n.get('published', '')} | [{title}]({n['url']}) "
                     f"([详细笔记]({page.relative_to(KNOWLEDGE_DIR).as_posix()})) "
                     f"| {n['topic']} | {shots} | {footwork} | {n['level']} |"))
    lines = ["# 教学笔记索引", "", HEADER, "",
             "| 发布 | 视频 | 主题 | 具体击球 | 具体步法 | 难度 |", "|---|---|---|---|---|---|"]
    lines += [r[2] for r in sorted(rows)]
    (KNOWLEDGE_DIR / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} notes")


if __name__ == "__main__":
    main()
