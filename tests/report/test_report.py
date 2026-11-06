import pytest

from rallyiq import contracts as c
from rallyiq.pipeline import demo
from rallyiq.report import compute_match_stats, generate_coach_report, register_provider, render_markdown
from rallyiq.report.coach import AnthropicProvider, CoachUnavailable, build_report
from rallyiq.report.coach.base import CoachProviderError
from rallyiq.report.coach.grounding import ground_insights, unverified_numbers
from rallyiq.report.coach.prompt import OUTPUT_SCHEMA
from rallyiq.report.stats import index, own_half


@pytest.fixture(scope="module")
def match():
    meta = c.VideoMeta(video_id="v", filename="m.mp4", fps=30, frame_count=1800, width=1920, height=1080,
                       duration_ms=60000)
    cal = demo.demo_calibration(meta)
    players = demo.demo_players(meta, cal)
    events = demo.demo_events(meta, players)
    return meta, cal, players, events


def with_winners(events, pattern):
    return events.model_copy(update={"rallies": [r.model_copy(update={"winner": pattern(i)})
                                                 for i, r in enumerate(events.rallies)]})


def test_own_half_mirrors_far_side():
    assert own_half((1.0, 2.0)) == (1.0, 2.0)
    lat, dep = own_half((1.0, 12.0))
    assert lat == pytest.approx(5.1) and dep == pytest.approx(1.4)


def test_rally_and_player_stats(match):
    meta, cal, players, events = match
    st = compute_match_stats(events, players, cal, meta)
    idx = index(st)
    assert idx["match.rallies"].value == len(events.rallies) == 6
    assert idx["match.rally_len.mean"].value == 7
    assert idx["P1.shots"].value + idx["P2.shots"].value == len(events.shots)
    # demo P1 moves on a smooth curve; distance must be positive and plausible (< 8 m/s * rally time)
    d = idx["P1.move.distance_m"].value
    rally_s = sum((r.end_ms - r.start_ms) / 1000 for r in events.rallies)
    assert 0 < d < 8 * rally_s
    assert idx["P1.move.tracked_pct"].value == pytest.approx(100)
    assert 0 < idx["P1.cover.ratio_pct"].value <= 100
    zones = idx["P1.pos.depth_zones"]
    assert sum(zones.counts.values()) == zones.n > 0
    # no winners and no shot types -> undeterminable, never a fake number
    assert idx["P1.win.rate_pct"].value is None and idx["P1.win.rate_pct"].status == "undeterminable"
    assert idx["P1.shot_types"].status == "undeterminable"
    assert not st.shot_types_available
    assert any("胜负" in x for x in st.caveats)


def test_movement_rejects_tracking_jumps(match):
    meta, cal, players, events = match
    base = index(compute_match_stats(events, players, cal, meta))["P1.move.distance_m"].value
    # teleport P1 across the court for one frame in the middle of a rally (identity swap glitch)
    r = events.rallies[0]
    f = (r.start_frame + r.end_frame) // 2
    frames = [pf.model_copy(deep=True) for pf in players.frames]
    for p in frames[f].players:
        if p.player_id == "P1":
            p.foot_court = [0.2, 6.5]
    glitched = players.model_copy(update={"frames": frames})
    st = compute_match_stats(events, glitched, cal, meta)
    m = index(st)["P1.move.distance_m"]
    assert m.value == pytest.approx(base, rel=0.05)


def test_uncalibrated_degrades_metric_distances(match):
    meta, cal, players, events = match
    bad = cal.model_copy(update={"segments": [cal.segments[0].model_copy(update={"status": "needs_review"})]})
    st = compute_match_stats(events, players, bad, meta)
    m = index(st)["P1.move.distance_m"]
    assert m.status == "needs_review" and m.confidence != "high"
    assert st.calibration_status == "needs_review"


def test_win_patterns(match):
    meta, cal, players, events = match
    ev = with_winners(events, lambda i: "P1" if i % 3 else "P2")
    idx = index(compute_match_stats(ev, players, cal, meta))
    assert idx["P1.win.won"].value == 4 and idx["P2.win.won"].value == 2
    assert idx["P1.win.rate_pct"].value == pytest.approx(66.7)
    assert idx["P1.win.rate_pct"].n == 6
    # demo rallies all start with P1 hitting
    assert idx["P1.win.serve_rate_pct"].n == 6 and idx["P2.win.receive_rate_pct"].n == 6


def test_shot_types_plug_in(match):
    meta, cal, players, events = match
    shots = [s.model_copy(update={"shot_type": "smash" if i % 2 else "clear"}) for i, s in enumerate(events.shots)]
    st = compute_match_stats(events.model_copy(update={"shots": shots}), players, cal, meta)
    d = index(st)["P1.shot_types"]
    assert st.shot_types_available and set(d.counts) == {"杀球", "高远球"} and d.status != "undeterminable"


def test_fingerprint_changes_with_corrections(match):
    meta, cal, players, events = match
    a = compute_match_stats(events, players, cal, meta).fingerprint
    b = compute_match_stats(with_winners(events, lambda i: "P1"), players, cal, meta).fingerprint
    assert a != b


def test_grounding_numbers():
    m = c.Metric(key="P1.win.rate_pct", label="回合胜率", value=66.7, unit="%", n=6, status="needs_review",
                 confidence="medium")
    assert unverified_numbers("P1 胜率约 67%，6 个回合", [m]) == []
    assert unverified_numbers("P1 胜率 90%", [m]) == ["90"]


def test_grounding_drops_and_caps(match):
    meta, cal, players, events = match
    st = compute_match_stats(with_winners(events, lambda i: "P1" if i % 3 else "P2"), players, cal, meta)
    raw = [
        {"kind": "strength", "player_id": "P1", "title": "P1 胜率高", "detail": "胜率 66.7%（6 回合）",
         "evidence": ["P1.win.rate_pct"], "confidence": "high"},
        {"kind": "weakness", "player_id": "P2", "title": "瞎编", "detail": "杀球成功率 12%",
         "evidence": ["P2.smash_rate"], "confidence": "high"},
        {"kind": "pattern", "player_id": "P2", "title": "P2 跑动", "detail": "每回合跑 99 米",
         "evidence": ["P2.move.per_rally_m"], "confidence": "medium"},
        {"kind": "weakness", "player_id": "P1", "title": "短回合", "detail": "短回合胜率低",
         "evidence": ["P1.win.short_rate_pct"], "confidence": "high"},  # undeterminable metric
    ]
    ins, dropped = ground_insights(raw, st)
    assert dropped == 2
    assert ins[0].confidence == "medium" and ins[0].grounded  # capped by needs_review evidence
    assert not ins[1].grounded and "99" in ins[1].warnings[0]


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, out=None, fail=None):
        self.out, self.fail = out, fail

    def generate(self, stats):
        if self.fail:
            raise CoachProviderError(self.fail)
        return self.out


def test_offline_report_is_grounded(match):
    meta, cal, players, events = match
    st = compute_match_stats(with_winners(events, lambda i: "P1" if i % 3 else "P2"), players, cal, meta)
    rep = generate_coach_report(st, "offline")
    assert rep.provider == "offline" and rep.summary and rep.insights
    assert all(i.grounded for i in rep.insights), [i.warnings for i in rep.insights]
    assert rep.stats_fingerprint == st.fingerprint
    md = render_markdown(st, rep)
    assert "AI 教练点评" in md and "P1" in md and "回合胜率" in md


def test_default_provider_falls_back_to_offline(match, monkeypatch):
    meta, cal, players, events = match
    st = compute_match_stats(events, players, cal, meta)
    register_provider("broken", lambda: FakeProvider(fail="no key"))
    monkeypatch.setenv("RALLYIQ_COACH_PROVIDER", "broken")
    rep = generate_coach_report(st)
    assert rep.provider == "offline" and rep.fallback_reason == "no key"
    with pytest.raises(CoachUnavailable):
        generate_coach_report(st, "broken")
    with pytest.raises(CoachUnavailable):
        generate_coach_report(st, "nope")


def test_build_report_keeps_data_caveats(match):
    meta, cal, players, events = match
    st = compute_match_stats(events, players, cal, meta)
    raw = {"summary": "总结", "insights": [], "training": [
        {"player_id": "P1", "drill": "步法", "why": "覆盖", "evidence": ["P1.cover.ratio_pct", "bogus"],
         "retest_metric": "bogus"}], "caveats": ["模型提醒"]}
    rep = build_report(raw, st, FakeProvider())
    assert rep.caveats[0] == "模型提醒" and set(st.caveats) <= set(rep.caveats)
    assert rep.training[0].evidence == ["P1.cover.ratio_pct"] and rep.training[0].retest_metric is None


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Msg:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason, self.model = [_Block(text)], stop, "claude-opus-5-5"


class _Stream:
    def __init__(self, msg, calls, kwargs):
        self.msg = msg
        calls.append(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, msg):
        self.calls = []
        outer = self

        class Messages:
            def stream(self, **kw):
                return _Stream(msg, outer.calls, kw)

        class Beta:
            messages = Messages()

        self.beta = Beta()


def test_anthropic_provider_request_shape(match):
    import json

    meta, cal, players, events = match
    st = compute_match_stats(events, players, cal, meta)
    out = {"summary": "好", "insights": [], "training": [], "caveats": []}
    client = FakeClient(_Msg(json.dumps(out, ensure_ascii=False)))
    p = AnthropicProvider(client=client)
    assert p.generate(st) == out
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["format"]["schema"] == OUTPUT_SCHEMA
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    with pytest.raises(CoachProviderError):
        AnthropicProvider(client=FakeClient(_Msg("{}", stop="refusal"))).generate(st)
    with pytest.raises(CoachProviderError):
        AnthropicProvider(client=FakeClient(_Msg("not json"))).generate(st)


def test_output_schema_is_strict():
    def walk(s):
        if isinstance(s, dict):
            if s.get("type") == "object":
                assert s.get("additionalProperties") is False
                assert set(s["required"]) == set(s["properties"])
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    walk(OUTPUT_SCHEMA)
