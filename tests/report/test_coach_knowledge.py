import json
import re

from rallyiq.report.coach.knowledge import KNOWLEDGE_DIR, REQUIRED_FIELDS, find_notes, load_notes, taxonomy

# Metric keys the stats module can emit; "{pid}." is the per-player prefix.
METRIC_KEY = re.compile(r"^(\{pid\}\.)?[a-z_]+(\.[a-z_]+)*$")


def test_notes_exist_and_follow_schema():
    notes = load_notes()
    assert len(notes) >= 20
    tax = taxonomy()
    ids = set()
    for n in notes:
        missing = [k for k in REQUIRED_FIELDS if k not in n]
        assert not missing, (n.get("id"), missing)
        assert n["id"] not in ids
        ids.add(n["id"])
        assert n["url"].startswith("https://")
        assert set(n["strokes"]) <= set(tax["strokes"]), n["id"]
        assert set(n["aspects"]) <= set(tax["aspects"]), n["id"]
        assert n["level"] in tax["levels"], n["id"]
        assert n["transcript_source"] in tax["transcript_sources"], n["id"]
        assert n["key_points"] and n["summary"].strip(), n["id"]
        for d in n["drills"]:
            assert set(d) == {"name", "how"}, n["id"]
        for m in n["related_metrics"]:
            assert METRIC_KEY.match(m), (n["id"], m)


def test_note_file_name_matches_source_id():
    for f in (KNOWLEDGE_DIR / "notes").glob("*/*.json"):
        n = json.loads(f.read_text(encoding="utf-8"))
        assert n["id"].endswith(f.stem)
        assert f.stem in n["url"]


def test_catalog_points_at_existing_notes():
    catalog = json.loads((KNOWLEDGE_DIR / "catalog.json").read_text(encoding="utf-8"))
    noted = {n["url"] for n in load_notes()}
    for coach in catalog["coaches"]:
        for v in coach["videos"]:
            if v["status"] == "noted":
                assert (KNOWLEDGE_DIR / v["note"]).exists()
                assert v["url"] in noted
            elif v["status"] == "skipped":
                assert v["reason"].strip()


def test_find_notes_filters():
    nets = find_notes(strokes=["net_shot"])
    assert nets and all("net_shot" in n["strokes"] for n in nets)
    beginners = find_notes(level="beginner")
    assert all(n["level"] in ("beginner", "all") for n in beginners)
    assert find_notes(strokes=["net_shot"], aspects=["equipment"]) == []
