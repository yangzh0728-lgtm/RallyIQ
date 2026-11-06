"""Regenerate knowledge/coaching/INDEX.md from the per-episode notes.

    python scripts/build_coaching_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallyiq.report.coach.knowledge import KNOWLEDGE_DIR, taxonomy  # noqa: E402


def main() -> None:
    tax = taxonomy()
    rows = []
    for f in sorted((KNOWLEDGE_DIR / "notes").glob("*/*.json")):
        n = json.loads(f.read_text(encoding="utf-8"))
        strokes = "、".join(tax["strokes"][s]["zh"] for s in n["strokes"]) or "—"
        aspects = "、".join(tax["aspects"][a] for a in n["aspects"])
        title = n["title"].replace("|", "/")
        note = f.relative_to(KNOWLEDGE_DIR).as_posix()
        rows.append((n["coach"], n.get("published", ""),
                     f"| {n.get('published', '')} | {n['coach']} | [{title}]({n['url']}) ([笔记]({note})) "
                     f"| {n['topic']} | {strokes} | {aspects} | {n['level']} |"))
    lines = ["# 教学笔记索引", "", "由 `scripts/build_coaching_index.py` 生成，不要手改。", "",
             "| 发布 | 球员 | 标题 | 主题 | 击球 | 方面 | 难度 |", "|---|---|---|---|---|---|---|"]
    lines += [r[2] for r in sorted(rows)]
    (KNOWLEDGE_DIR / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} notes")


if __name__ == "__main__":
    main()
