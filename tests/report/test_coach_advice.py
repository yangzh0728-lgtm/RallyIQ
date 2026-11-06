"""Knowledge-grounded advice: weaknesses -> coaching notes -> cited drills -> retests."""

import json

import pytest

from rallyiq import contracts as c
from rallyiq.memory.profile import retests
from rallyiq.pipeline import demo
from rallyiq.report import compute_match_stats, generate_coach_report, render_markdown
from rallyiq.report.coach import AnthropicProvider, build_report, knowledge_context
from rallyiq.report.coach.advice import _load, detect_focus, retrieve, video_link
from rallyiq.report.coach.prompt import user_message
from rallyiq.report.stats import index


def _note(nid, coach, strokes, aspects, drills=("练习A",), related=(), level="intermediate"):
    return {
        "id": nid, "coach": coach, "url": f"https://example.com/{nid}", "title": f"title {nid}", "level": level,
        "topic": f"topic {nid}", "strokes": list(strokes), "aspects": list(aspects),
        "key_points": ["要点1", "要点2"], "common_mistakes": ["错误1"], "coach_cues": ["口令1", "口令2"],
        "drills": [{"name": d, "how": f"{d}的做法"} for d in drills], "related_metrics": list(related),
        "corrections": [], "self_check": [], "when_to_use": [],
    }


NOTES = (
    _note("a-lift", "Axelsen", ["lift"], ["technique"], related=["{pid}.win.lost_own_last_shot_pct"]),
    _note("z-lift", "郑思维", ["lift", "net_shot"], ["technique", "grip"]),
    _note("z-split", "郑思维", [], ["footwork", "split_step"], related=["{pid}.cover.ratio_pct"]),
    _note("a-smash", "Axelsen", ["smash"], ["technique"]),
    _note("a-racket", "Axelsen", [], ["equipment"]),
)


@pytest.fixture(scope="module")
def stats():
    meta = c.VideoMeta(video_id="v", filename="m.mp4", fps=30, frame_count=1800, width=1920, height=1080,
                       duration_ms=60000)
    cal = demo.demo_calibration(meta)
    players = demo.demo_players(meta, cal)
    events = demo.demo_events(meta, players)
    types = ["net_shot", "smash", "net_shot", "clear", "net_shot", "drop", "lift"]
    events = events.model_copy(update={
        "shots": [s.model_copy(update={"shot_type": types[i % 7]}) for i, s in enumerate(events.shots)],
        "rallies": [r.model_copy(update={"winner": "P2" if i % 4 else "P1"}) for i, r in enumerate(events.rallies)],
    })
    return compute_match_stats(events, players, cal, meta)


def test_lost_last_shot_distribution(stats):
    d = index(stats)["P1.lost.last_shot_types"]
    assert d.n > 0 and d.status != "undeterminable"
    assert sum(d.counts.values()) == d.n


def test_focus_uses_only_existing_metrics(stats):
    idx = index(stats)
    focus = detect_focus(stats)
    assert focus, "demo match with a losing P1 should show weaknesses"
    ids = {f.id for f in focus}
    assert "P1.lost_stroke" in ids and "P1.coverage" in ids
    for f in focus:
        assert all(k in idx for k in f.evidence)
        assert f.retest_metric in idx


def test_retrieve_matches_tags_and_skips_off_topic(stats):
    focus = {f.id: f for f in detect_focus(stats)}
    lift = retrieve(focus["P1.lost_stroke"], NOTES)
    assert [n["id"] for n in lift][:2] == ["a-lift", "z-lift"]  # one per coach first, metric match ranks first
    assert all("lift" in n["strokes"] for n in lift)
    cover = retrieve(focus["P1.coverage"], NOTES)
    assert [n["id"] for n in cover] == ["z-split"]


def test_offline_report_cites_notes_and_sets_retest(stats):
    k = knowledge_context(stats, NOTES)
    rep = generate_coach_report(stats, "offline", knowledge=k)
    cited = [t for t in rep.training if t.sources]
    assert cited
    t = next(t for t in cited if t.focus and "挑球" in t.focus)
    assert t.sources[0].note_id == "a-lift" and t.sources[0].url == "https://example.com/a-lift"
    assert t.sources[0].drill == "练习A"
    assert t.retest_metric == "P1.win.lost_own_last_shot_pct"
    assert t.key_points
    md = render_markdown(stats, rep)
    assert "https://example.com/a-lift" in md and "针对：" in md


def test_model_citations_are_grounded(stats):
    k = knowledge_context(stats, NOTES)
    raw = {"summary": "s", "insights": [], "caveats": [], "training": [{
        "player_id": "P1", "drill": "挑球三线", "why": "w", "evidence": [], "retest_metric": "not.a.key",
        "focus_id": "P1.lost_stroke", "key_points": ["拇指定方向"],
        "sources": [{"note_id": "a-lift", "drill": "练习A"}, {"note_id": "made-up", "drill": None},
                    {"note_id": "a-smash", "drill": None}],
    }]}
    rep = build_report(raw, stats, AnthropicProvider(client=object()), knowledge=k)
    t = rep.training[0]
    # a-smash exists on disk but was not retrieved for this match, so it cannot be cited either
    assert [r.note_id for r in t.sources] == ["a-lift"]
    assert any("made-up" in w and "a-smash" in w for w in t.warnings)
    assert t.retest_metric == "P1.win.lost_own_last_shot_pct"  # falls back to the focus retest metric
    assert "P1.lost.last_shot_types" in t.evidence
    assert t.focus and "挑球" in t.focus


def test_prompt_carries_focus_and_note_cards(stats):
    k = knowledge_context(stats, NOTES)
    msg = user_message(stats, k)
    payload = json.loads(msg.split("（knowledge，JSON）：\n\n", 1)[1])
    first = payload[0]
    assert first["focus_id"] and first["notes"][0]["note_id"]
    assert "url" not in first["notes"][0]  # links come from the files, not from the model
    assert user_message(stats) == user_message(stats, None)


def test_anthropic_provider_sends_knowledge(stats):
    from tests.report.test_report import FakeClient, _Msg

    k = knowledge_context(stats, NOTES)
    client = FakeClient(_Msg(json.dumps({"summary": "", "insights": [], "training": [], "caveats": []})))
    AnthropicProvider(client=client).generate(stats, knowledge=k)
    assert "a-lift" in client.calls[0]["messages"][0]["content"]


def test_custom_provider_without_knowledge_arg_still_works(stats):
    from rallyiq.report import register_provider

    class Old:
        name, model = "old", None

        def generate(self, st):
            return {"summary": "x", "insights": [], "training": [], "caveats": []}

    register_provider("old", Old)
    assert generate_coach_report(stats, "old", knowledge=knowledge_context(stats, NOTES)).summary == "x"


def test_retests_keep_sources():
    ref = c.NoteRef(note_id="a-lift", coach="Axelsen", title="t", url="https://example.com/a", t_s=42)
    assert video_link(ref) == "https://example.com/a?t=42"
    m = lambda v: c.Metric(key="win.rate_pct", label="胜率", value=v, unit="%", n=20, status="ok", confidence="high")  # noqa: E731
    base = dict(kind="match_committed", recorded_at="2026-11-01T00:00:00Z", person="me", is_self=True)
    e1 = c.AnalysisEvent(event_id="1", seq=1, video_id="a", metrics=[m(40)], **base, training=[
        c.TrainingItem(drill="d", why="w", retest_metric="win.rate_pct", focus="f", sources=[ref])])
    e2 = c.AnalysisEvent(event_id="2", seq=2, video_id="b", metrics=[m(60)], **base)
    r = retests([e1, e2])[0]
    assert r.status == "improved" and r.focus == "f" and r.sources[0].t_s == 42


def test_loader_reads_note_dir(tmp_path, monkeypatch):
    d = tmp_path / "notes" / "x"
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps(NOTES[0]), encoding="utf-8")
    (d / "bad.json").write_text("{", encoding="utf-8")
    monkeypatch.setenv("RALLYIQ_KNOWLEDGE_DIR", str(tmp_path))
    _load.cache_clear()
    from rallyiq.report.coach.advice import load_notes

    assert [n["id"] for n in load_notes()] == ["a-lift"]
    _load.cache_clear()


# --------------------------------------------------------------------------- local model (Ollama)


class _Resp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ollama_out(content, done_reason="stop"):
    return {"message": {"role": "assistant", "content": content}, "done_reason": done_reason}


def test_ollama_provider_request_and_citations(stats):
    from rallyiq.report.coach import OllamaProvider
    from rallyiq.report.coach.prompt import OUTPUT_SCHEMA

    calls = []
    out = {"summary": "本地点评", "insights": [], "caveats": [], "training": [{
        "player_id": "P1", "drill": "挑球三线", "why": "w", "evidence": [], "retest_metric": None,
        "focus_id": "P1.lost_stroke", "key_points": ["拇指"], "sources": [{"note_id": "a-lift", "drill": None}]}]}

    def opener(req, timeout):
        if req.full_url.endswith("/api/tags"):
            assert timeout <= 5
            return _Resp({"models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]})
        calls.append((req.full_url, json.loads(req.data), timeout))
        return _Resp(_ollama_out(json.dumps(out, ensure_ascii=False)))

    k = knowledge_context(stats, NOTES)
    p = OllamaProvider(url="http://mac:11434", opener=opener)
    rep = build_report(p.generate(stats, knowledge=k), stats, p, knowledge=k)
    url, body, _ = calls[0]
    assert url == "http://mac:11434/api/chat"
    assert body["model"] == "qwen2.5:7b" and body["format"] == OUTPUT_SCHEMA and body["stream"] is False
    assert body["options"]["num_ctx"] >= 16384
    assert "a-lift" in body["messages"][1]["content"]
    assert rep.provider == "ollama" and rep.model == "qwen2.5:7b"
    assert rep.training[0].sources[0].url == "https://example.com/a-lift"


def test_ollama_errors_are_provider_errors(stats):
    import urllib.error

    from rallyiq.report.coach import OllamaProvider
    from rallyiq.report.coach.base import CoachProviderError

    def refused(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    with pytest.raises(CoachProviderError, match="Ollama"):
        OllamaProvider(opener=refused).generate(stats)
    def serve(chat, models=("qwen2.5:7b",)):
        def opener(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return _Resp({"models": [{"name": m, "model": m} for m in models]})
            return _Resp(chat)
        return opener

    with pytest.raises(CoachProviderError, match="ollama pull"):
        OllamaProvider(opener=serve(_ollama_out("{}"), models=("llama3:8b",))).generate(stats)
    with pytest.raises(CoachProviderError, match="截断"):
        OllamaProvider(opener=serve(_ollama_out("{", "length"))).generate(stats)
    with pytest.raises(CoachProviderError):
        OllamaProvider(opener=serve(_ollama_out("not json"))).generate(stats)


def test_auto_prefers_local_then_claude_then_offline(stats, monkeypatch):
    from rallyiq.report import register_provider
    from rallyiq.report.coach import auto_chain
    from rallyiq.report.coach.base import CoachProviderError

    class Fake:
        model = None

        def __init__(self, name, fail=None):
            self.name, self.fail = name, fail

        def generate(self, st, knowledge=None):
            if self.fail:
                raise CoachProviderError(self.fail)
            return {"summary": self.name, "insights": [], "training": [], "caveats": []}

    monkeypatch.delenv("RALLYIQ_COACH_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert auto_chain() == ["ollama"]
    k = knowledge_context(stats, NOTES)
    try:
        register_provider("ollama", lambda: Fake("ollama"))
        assert generate_coach_report(stats, knowledge=k).provider == "ollama"

        register_provider("ollama", lambda: Fake("ollama", fail="连不上"))
        rep = generate_coach_report(stats, knowledge=k)
        assert rep.provider == "offline" and "本地模型不可用：连不上" in rep.fallback_reason
        assert any(t.sources for t in rep.training)  # offline advice still cites notes

        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        register_provider("anthropic", lambda: Fake("anthropic"))
        rep = generate_coach_report(stats, knowledge=k)
        assert rep.provider == "anthropic" and "本地模型" in rep.fallback_reason
        rep = generate_coach_report(stats, "auto", knowledge=k)
        assert rep.provider == "anthropic"
    finally:
        from rallyiq.report.coach import AnthropicProvider, OllamaProvider

        register_provider("ollama", OllamaProvider)
        register_provider("anthropic", AnthropicProvider)
