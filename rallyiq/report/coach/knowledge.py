"""Coaching knowledge base: per-episode notes from pro players' public tutorials.

Notes live in ``knowledge/coaching/notes/<coach>/<video_id>.json`` at the repo root
(see ``knowledge/coaching/README.md``). They are paraphrased study notes with a link to
the source video; no transcripts or video files are stored. The coach prompt does not
use them yet: this module is the lookup a later retrieval step will call.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge" / "coaching"

REQUIRED_FIELDS = (
    "id", "coach", "platform", "url", "title", "language", "transcript_source", "level", "topic",
    "strokes", "aspects", "summary", "key_points", "common_mistakes", "drills", "coach_cues",
    "related_metrics", "shots", "footwork", "technique_breakdown", "when_to_use", "corrections", "progression",
    "self_check", "detailed_description",
)


@lru_cache(maxsize=1)
def taxonomy() -> dict:
    return json.loads((KNOWLEDGE_DIR / "taxonomy.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_notes() -> tuple[dict, ...]:
    files = sorted((KNOWLEDGE_DIR / "notes").glob("*/*.json"))
    return tuple(json.loads(f.read_text(encoding="utf-8")) for f in files)


def find_notes(*, strokes: list[str] | None = None, aspects: list[str] | None = None,
               level: str | None = None, coach: str | None = None) -> list[dict]:
    """Notes matching any of ``strokes`` / ``aspects`` (each filter is optional), narrowed by
    ``level`` (notes tagged ``all`` always match) and ``coach``."""
    out = []
    for n in load_notes():
        if strokes and not set(strokes) & set(n["strokes"]):
            continue
        if aspects and not set(aspects) & set(n["aspects"]):
            continue
        if level and n["level"] not in (level, "all"):
            continue
        if coach and n["coach"] != coach:
            continue
        out.append(n)
    return out
