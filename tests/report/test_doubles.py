"""Doubles teamwork: formation, rotation, spacing, roles, who takes which shots, doubles shot advice."""

import pytest

from rallyiq import contracts as c
from rallyiq.report import compute_match_stats, render_markdown
from rallyiq.report.doubles import analyze_doubles
from rallyiq.report.html import render_html
from rallyiq.report.moments import shot_moments
from rallyiq.report.stats import index

FPS = 30.0
SEC = 30  # frames between shots

# Positions in the near half (y < 6.7); the far half is mirrored (y -> 13.4 - y, x -> 6.1 - x).
FORMS = {
    "fb_first_front": {"first": (3.05, 5.2), "second": (3.05, 1.5)},
    "sbs": {"first": (1.5, 3.3), "second": (4.6, 3.3)},
    "crowded": {"first": (2.8, 3.3), "second": (3.4, 3.3)},
}
TEAM = {"P1": ("A", "first"), "P3": ("A", "second"), "P2": ("B", "first"), "P4": ("B", "second")}


def _xy(pid, forms):
    team, slot = TEAM[pid]
    x, y = FORMS[forms[team]][slot]
    return [x, y] if team == "A" else [6.1 - x, 13.4 - y]


def _rally(rid, start, hits, forms, winner):
    shots, frames = [], []
    for k, (hitter, typ) in enumerate(hits):
        f = start + SEC * k
        shots.append(c.ShotEvent(shot_id=f"{rid}s{k}", rally_id=rid, frame=f, t_ms=int(f / FPS * 1000), hitter=hitter,
                                 hitter_court=_xy(hitter, forms), shot_type=typ, confidence=0.9, status="ok"))
    end = start + SEC * len(hits)
    for f in range(start, end + 1, 3):
        frames.append(c.PlayerFrame(frame=f, t_ms=int(f / FPS * 1000), players=[
            c.PlayerDetection(player_id=pid, track_id=n, bbox=[0, 0, 1, 1], foot_court=_xy(pid, forms), confidence=0.9)
            for n, pid in enumerate(("P1", "P2", "P3", "P4"), 1)]))
    rally = c.Rally(rally_id=rid, start_frame=start, end_frame=end, start_ms=int(start / FPS * 1000),
                    end_ms=int(end / FPS * 1000), shot_ids=[s.shot_id for s in shots], winner=winner, confidence=0.9)
    return rally, shots, frames


# A attacks front-back (P1 at the net, P3 smashing); B defends side by side but crowded together.
A_ATTACKS = [("P2", "short_serve"), ("P1", "net_shot"), ("P4", "lift"), ("P3", "smash"), ("P2", "lift"),
             ("P3", "smash"), ("P4", "lift"), ("P3", "smash")]
# B attacks front-back (P2 at the net, P4 smashing); A defends side by side, well spread.
B_ATTACKS = [("P1", "short_serve"), ("P2", "net_shot"), ("P1", "lift"), ("P4", "smash"), ("P3", "lift"),
             ("P4", "smash"), ("P1", "lift"), ("P4", "smash")]


@pytest.fixture(scope="module")
def match():
    rallies, shots, frames = [], [], []
    start = 0
    for k in range(12):
        if k % 2 == 0:
            r = _rally(f"r{k}", start, A_ATTACKS, {"A": "fb_first_front", "B": "crowded"}, "A")
        else:
            r = _rally(f"r{k}", start, B_ATTACKS, {"A": "sbs", "B": "fb_first_front"}, "B")
        rallies.append(r[0])
        shots += r[1]
        frames += r[2]
        start += SEC * 8 + 150
    events = c.EventsResult(video_id="v", rallies=rallies, shots=shots, match_format="doubles")
    players = c.PlayerTracks(video_id="v", frames=frames, match_format="doubles")
    return events, players


def test_formation_follows_attack_and_defense(match):
    events, players = match
    st = compute_match_stats(events, players)
    ix = index(st)
    assert ix["A.dbl.formation.attack_fb_pct"].value == pytest.approx(100, abs=1)
    assert ix["A.dbl.formation.defense_sbs_pct"].value == pytest.approx(100, abs=1)
    assert ix["B.dbl.formation.defense_sbs_pct"].value == pytest.approx(100, abs=1)
    assert ix["A.dbl.formation.fb_pct"].value == pytest.approx(50, abs=5)
    assert ix["A.dbl.rotation.front_swaps_per_rally"].value == 0
    # B stood 0.6 m apart in half the rallies
    assert ix["B.dbl.gap.close_pct"].value == pytest.approx(50, abs=5)
    assert ix["A.dbl.gap.close_pct"].value == 0
    assert 0 < ix["A.dbl.cover.ratio_pct"].value < 100


def test_roles_and_shot_division(match):
    events, players = match
    st = compute_match_stats(events, players)
    ix = index(st)
    a = next(t for t in st.doubles if t.team == "A")
    roles = {r.player_id: r for r in a.roles}
    assert (roles["P1"].role, roles["P3"].role) == ("front", "rear")
    assert roles["P1"].front_pct == 100
    assert ix["A.dbl.who.smash"].counts == {"P1": 0, "P3": 18}
    assert ix["P3.dbl.smash_share_pct"].value == 100
    assert ix["A.dbl.who.serve"].counts == {"P1": 6, "P3": 0}
    assert ix["B.dbl.who.receive"].counts == {"P2": 6, "P4": 0}
    assert ix["A.dbl.who.rear"].counts["P3"] == 18
    assert set(a.division) <= set(ix)


def test_findings_cite_existing_keys(match):
    events, players = match
    st = compute_match_stats(events, players)
    ix = index(st)
    by = {t.team: {f.title: f for f in t.findings} for t in st.doubles}
    assert "进攻时能站成前后" in by["A"] and "防守时能拉开成左右站位" in by["A"]
    assert "前后分工固定" in by["A"] and "进攻主要靠 P3" in by["A"]
    assert "两人经常挤在一起" in by["B"] and by["B"]["两人经常挤在一起"].kind == "weakness"
    for t in st.doubles:
        for f in t.findings:
            assert all(k in ix for k in f.evidence), f.evidence


def test_doubles_shot_advice_uses_the_replying_opponent(match):
    events, players = match
    ms = shot_moments(events, players, FPS)
    assert ms
    lifts = [m for m in ms if m.issue == "clear_to_rear"]
    # both pairs lift to the opponents' rear smasher: P3 smashes for A, P4 for B
    assert lifts and all(m.opponent_id == ("P4" if m.player_id in ("P1", "P3") else "P3") for m in lifts)
    assert {m.player_id for m in lifts} == {"P1", "P2", "P3", "P4"}
    # P2 waits at the net and P4 at the back: the open spot is a front corner on the doubles court
    assert all(m.target_zone.startswith("网前") for m in lifts)
    assert all(m.outcome in ("lost", "continued") for m in lifts)


def test_report_shows_doubles_section(match):
    events, players = match
    st = compute_match_stats(events, players)
    report = c.MatchReport(stats=st)
    html = render_html(report, events, players)
    assert 'id="doubles"' in html and "前后分工固定" in html and "谁打杀球" in html
    md = render_markdown(st)
    assert "## 双打配合" in md and "前场型" in md


def test_singles_and_untracked():
    r, s, _ = _rally("r", 0, A_ATTACKS, {"A": "fb_first_front", "B": "sbs"}, "A")
    singles = c.EventsResult(video_id="v", rallies=[r], shots=s)
    assert analyze_doubles(singles, None) == ([], [], [], {})
    doubles = singles.model_copy(update={"match_format": "doubles"})
    teams, metrics, dists, _ = analyze_doubles(doubles, None)
    assert [t.team for t in teams] == ["A", "B"] and teams[0].note.startswith("缺少球员跟踪")
    assert next(m for m in metrics if m.key == "A.dbl.formation.fb_pct").value is None
    assert next(d for d in dists if d.key == "A.dbl.who.smash").counts == {"P1": 0, "P3": 3}


def test_offline_coach_uses_doubles_findings(match):
    from rallyiq.report.coach import build_report
    from rallyiq.report.coach.offline import OfflineProvider

    events, players = match
    st = compute_match_stats(events, players)
    provider = OfflineProvider()
    rep = build_report(provider.generate(st), st, provider)
    titles = {i.title: i for i in rep.insights}
    assert "B队 两人经常挤在一起" in titles and titles["B队 两人经常挤在一起"].grounded
    assert not any("场地覆盖偏窄" in t for t in titles)  # per-player coverage is not judged in doubles
    assert any(t.drill.startswith("B队配合") for t in rep.training)


def test_identity_mixups_are_flagged_not_analysed(match):
    """A spectator tracked as P3 at the net post, and B's players on opposite halves: no findings, a clear note."""
    events, players = match
    frames = []
    for pf in players.frames:
        dets = []
        for d in pf.players:
            if d.player_id == "P3":
                d = d.model_copy(update={"foot_court": [-0.8, 6.6]})  # stands by the post the whole match
            if d.player_id == "P4":
                d = d.model_copy(update={"foot_court": [3.0, 3.0]})  # tracked on the near half: wrong team
            dets.append(d)
        frames.append(pf.model_copy(update={"players": dets}))
    st = compute_match_stats(events, players.model_copy(update={"frames": frames}))
    by = {t.team: t for t in st.doubles}
    assert by["A"].status == by["B"].status == "undeterminable"
    assert not by["A"].findings and not by["B"].findings
    assert "P3" in by["A"].note and "场外" in by["A"].note
    assert "同一个半场" in by["B"].note
    assert all(r.role == "unknown" for t in st.doubles for r in t.roles)
    assert not st.moments and any("身份识别" in x for x in st.caveats)
