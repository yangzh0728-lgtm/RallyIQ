"""How to beat P1 / P2: weaknesses -> direct tactics, confidence caps, report and coach wiring."""

from rallyiq.report import compute_match_stats, render_markdown
from rallyiq.report.coach import build_report
from rallyiq.report.coach.offline import OfflineProvider
from rallyiq.report.html import render_html
from rallyiq.report.stats import index
from tests.report.test_style import POWER, _cal, _rallies

# P1 (near) moves P2 (far) to P2's rear left (lane "R" seen from the camera) and P2 gets smashed there;
# P2 is fine at the front. Lanes for P2 are mirrored: x = 4.9 is P2's own left.
PRESSED = [("P1", [
    ("P1", "short_serve", "mid", "C", 0.9),
    ("P2", "push", "front", "C", 0.6),
    ("P1", "clear", "rear", "L", 1.4),
    ("P2", "clear", "rear", "R", 1.4),
    ("P1", "smash", "rear", "C", 0.5),
    ("P2", "block", "mid", "C", 0.8),
    ("P1", "net_kill", "front", "C", 0.6),
]) for _ in range(10)]
# P2 nets a net shot right after the serve.
NET_ERRORS = [("P1", [
    ("P1", "short_serve", "mid", "C", 0.9),
    ("P2", "net_shot", "front", "C", 0.6),
]) for _ in range(6)]


def _stats(spec):
    events, players = _rallies(spec)
    return compute_match_stats(events, players, _cal()), events, players


def test_tactics_against_the_far_player():
    stats, _, _ = _stats(PRESSED + NET_ERRORS)
    by = {s.target: s for s in stats.scouting}
    p2 = by["P2"]
    assert p2.side == "far" and p2.against == "P1"
    assert p2.confidence in ("low", "medium") and "远端" in p2.note  # far side caps confidence
    codes = {t.code: t for t in p2.tactics}
    zone = next(t for t in p2.tactics if t.code == "zone_weak" and "后场" in t.title)
    assert zone.title == "压 P2 的左后场" and "头顶/反手区" in zone.how
    assert zone.n == 10 and len(zone.examples_ms) == 3 and zone.confidence != "high"
    assert codes["stroke_errors"].title == "逼 P2 打放网"
    assert "6 分是他自己的放网" in codes["stroke_errors"].why
    assert p2.summary.startswith("打 P2：")
    # every cited key is a real stats key (the coach can ground on it)
    idx = index(stats)
    assert all(k in idx for t in p2.tactics for k in t.evidence)
    assert idx["P2.scout.trouble_pct.rear_left"].value == 100.0
    assert by["P1"].side == "near"


def test_offline_coach_markdown_and_html_show_the_plan():
    stats, events, players = _stats(PRESSED + NET_ERRORS)
    rep = build_report(OfflineProvider().generate(stats), stats, OfflineProvider())
    plan = [i for i in rep.insights if i.title.startswith("怎么打 P2")]
    assert plan and all(i.kind == "tactic" and i.player_id == "P2" for i in plan)
    assert all(i.grounded for i in plan), [(i.title, i.warnings) for i in plan]
    md = render_markdown(stats, rep)
    assert md.index("## 怎么打 P1、怎么打 P2") < md.index("## AI 教练点评")
    assert "压 P2 的左后场" in md and "回看：" in md
    from rallyiq.contracts import MatchReport
    page = render_html(MatchReport(stats=stats, coach=rep), events, players)
    assert 'id="scout"' in page and "压 P2 的左后场" in page


def test_habits_and_style_counter():
    stats, _, _ = _stats(POWER)
    p2 = {s.target: s for s in stats.scouting}["P1"]
    titles = [t.title for t in p2.tactics]
    assert "你挑球后，P1 多半回杀球" in titles
    assert any(t.code == "style" and "强攻压制型" in t.title for t in p2.tactics)


def test_too_few_shots_and_doubles():
    stats, _, _ = _stats(NET_ERRORS)
    assert all(s.status == "undeterminable" and not s.tactics for s in stats.scouting)
    events, players = _rallies(POWER, doubles=False)
    events = events.model_copy(update={"match_format": "doubles"})
    assert compute_match_stats(events, players, _cal()).scouting == []


def test_winners_inferred_from_next_server():
    from rallyiq.report.scouting import infer_winners
    spec = [(None, seq) for _, seq in PRESSED]
    events, _ = _rallies(spec)
    inferred = infer_winners(events)
    # every rally but the last is followed by a P1 serve; the last has no next rally
    assert inferred == {f"r{k}": "P1" for k in range(9)}
    assert infer_winners(events, max_gap_s=1.0) == {}  # gap between rallies is longer than that
    stats = compute_match_stats(events, *_stats_args(spec))
    p2 = {s.target: s for s in stats.scouting}["P2"]
    assert "按“下一分谁发球”推断" in p2.note
    assert all(t.confidence != "high" for t in p2.tactics)


def test_one_shot_rallies_are_left_out():
    pickups = [(None, [("P1", "clear", "rear", "C", 0.9)]) for _ in range(30)]
    stats, _, _ = _stats(PRESSED + pickups)
    p1 = {s.target: s for s in stats.scouting}["P1"]
    assert p1.n_shots == 10 * 3  # clear, smash, kill per rally (serves aside); the 30 lone clears are not counted


def _stats_args(spec):
    _, players = _rallies(spec)
    return players, _cal()


def test_warm_up_rally_is_left_out_of_style_and_tactics():
    drill = [("P1", "clear", "rear", "C", 1.2) if k % 2 == 0 else ("P2", "clear", "rear", "C", 1.2)
             for k in range(40)]
    stats, _, _ = _stats([(None, drill), *PRESSED])
    assert any("热身或练习" in c for c in stats.caveats)
    p1 = {s.target: s for s in stats.scouting}["P1"]
    assert p1.n_shots == 10 * 3  # the 20 drill clears of P1 are not counted
    assert stats.rallies and len(stats.rallies) == 11  # the rally log still shows it


def test_rally_is_cut_at_a_long_pause():
    from rallyiq.report.scouting import split_at_gaps
    # a point, then 6 s of nobody hitting while P1 gets ready, then the next serve, all in one detected rally
    seq = [("P1", "short_serve", "mid", "C", 0.9), ("P2", "push", "front", "C", 0.6),
           ("P1", "clear", "rear", "L", 6.0), ("P1", "short_serve", "mid", "C", 0.9),
           ("P2", "net_shot", "front", "C", 0.6)]
    events, _ = _rallies([("P1", seq)])
    parts = split_at_gaps(events).rallies
    assert [len(r.shot_ids) for r in parts] == [3, 2]
    assert [r.winner for r in parts] == [None, "P1"] and parts[1].rally_id == "r0~1"


def test_first_shot_counts_as_the_serve():
    # every rally opens with a shot labelled "block" from P2's mid court: that is the serve, not a weakness
    spec = [("P2", [("P2", "block", "mid", "C", 0.9), ("P1", "lift", "front", "C", 1.2),
                    ("P2", "smash", "rear", "C", 0.5), ("P1", "block", "mid", "C", 0.8)]) for _ in range(12)]
    stats, _, _ = _stats(spec)
    p2 = {s.target: s for s in stats.scouting}["P2"]
    assert not any(t.code == "zone_weak" and "中场" in t.title for t in p2.tactics)
