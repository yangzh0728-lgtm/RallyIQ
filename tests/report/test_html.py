import re
from datetime import UTC, datetime

import pytest

from rallyiq import contracts as c
from rallyiq.pipeline import demo
from rallyiq.report import compute_match_stats, render_html
from rallyiq.report.coach import OfflineProvider, build_report


@pytest.fixture(scope="module")
def match():
    meta = c.VideoMeta(video_id="v", filename="m.mp4", fps=30, frame_count=1800, width=1920, height=1080,
                       duration_ms=60000)
    cal = demo.demo_calibration(meta)
    players = demo.demo_players(meta, cal)
    events = demo.demo_events(meta, players)
    events = events.model_copy(update={"rallies": [r.model_copy(update={"winner": "P1" if i % 3 else "P2"})
                                                   for i, r in enumerate(events.rallies)]})
    stats = compute_match_stats(events, players, cal, meta)
    coach = build_report(OfflineProvider().generate(stats), stats, OfflineProvider())
    return c.MatchReport(stats=stats, coach=coach), events, players


def sections(page: str) -> list[str]:
    return re.findall(r'<section id="([a-z]+)"', page)


def test_self_contained_page(match):
    report, events, players = match
    page = render_html(report, events, players, title="赛后报告 · m.mp4", video_name="m.mp4",
                       generated=datetime(2026, 11, 7, tzinfo=UTC))
    assert page.startswith("<!doctype html>")
    # offline: no scripts, no external stylesheets / images / fonts
    assert "<script" not in page and not re.search(r'(src|href)="(https?:)?//', page)
    assert {"overview", "coach", "rallies", "heatmap", "details"} <= set(sections(page))
    # one bar per rally, with a tooltip naming the winner
    assert page.count('<rect class="bar') == len(events.rallies)
    assert "P1 得分" in page and "P2 得分" in page
    # both players get a court heatmap with their hit points
    assert page.count('<figure class="court-card">') == 2
    assert page.count('<circle class="dot') == sum(1 for s in events.shots if s.hitter_court and s.hitter)
    assert "2026-11-07 00:00 UTC" in page


def test_text_is_escaped(match):
    report, events, players = match
    page = render_html(report, events, players, title="<b>x</b> & y")
    assert "<title>&lt;b&gt;x&lt;/b&gt; &amp; y</title>" in page


def test_missing_parts_drop_sections(match):
    report, _, _ = match
    page = render_html(report.model_copy(update={"coach": None}))
    got = set(sections(page))
    assert "overview" in got and not {"coach", "rallies", "heatmap"} & got


def test_optional_analyses_from_dicts(match):
    """Styles, footwork metrics and stroke layers are read from plain dicts, so the page shows them as soon as
    the analyses fill those fields."""
    report, events, players = match
    rep = report.model_dump(mode="json")
    rep["stats"]["styles"] = [{
        "subject": "P1", "kind": "singles", "label": "强攻压制型", "status": "ok", "confidence": "medium", "n_shots": 40,
        "primary": {"name": "强攻压制型", "score": 75, "reasons": ["杀球占比 21%"], "representatives": ["安赛龙"]},
        "scores": [{"name": "强攻压制型", "score": 75}, {"name": "拉吊突击型", "score": 46}],
        "tactics": [{"name": "拉吊突击", "uses": 3, "success_pct": 66.7, "forced_pct": 100, "examples_ms": [61000]}],
        "phases": [{"phase": "attack", "shots": 10, "counts": {"smash": 8, "drop": 2}, "recommended_pct": 80}],
        "habits": [{"after": "lift", "reply": "smash", "n": 5, "of": 8, "pct": 62.5}], "advice": ["多用劈吊"],
    }]
    rep["stats"]["players"][0]["metrics"].append({
        "key": "P1.footwork.split_step_pct", "label": "对手击球时做启动步的比例", "value": 33.0, "unit": "%", "n": 33,
        "status": "ok", "confidence": "high", "note": None})
    ev = events.model_dump(mode="json")
    ev["shots"][1]["stroke"] = {"detail": {"hand": {"value": "backhand", "source": "model"},
                                           "direction": {"value": "cross", "source": "measured"}}}
    page = render_html(rep, ev, players)
    assert {"style", "footwork"} <= set(sections(page))
    assert "强攻压制型" in page and "01:01" in page and "杀球 8" in page and "多回杀球" in page
    assert "对手击球时做启动步的比例" in page and "33%" in page
    assert "击球的多层描述" in page and "反手 1" in page and "斜线 1" in page
    assert "<th>击球点</th>" not in page  # empty layers are left out
