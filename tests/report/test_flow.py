import pytest

from rallyiq import contracts as c
from rallyiq.report import compute_match_stats
from rallyiq.report.flow import flow_metrics, rally_log
from rallyiq.report.stats import index


def _events(winners, hitters_per_rally=None, gap_ms=800, fmt="singles"):
    """Rallies 10 s apart; each rally's shots alternate P1/P2 (or the given hitters), ``gap_ms`` apart."""
    rallies, shots = [], []
    for i, w in enumerate(winners):
        start = i * 10_000
        hitters = (hitters_per_rally or {}).get(i, ["P1", "P2", "P1"])
        ids = []
        for k, h in enumerate(hitters):
            t = start + 500 + k * gap_ms
            sid = f"s{i}_{k}"
            ids.append(sid)
            shots.append(c.ShotEvent(shot_id=sid, rally_id=f"r{i}", frame=t * 30 // 1000, t_ms=t, hitter=h,
                                     confidence=0.9, shot_type="smash" if k == len(hitters) - 1 else None))
        end = start + 500 + len(hitters) * gap_ms
        rallies.append(c.Rally(rally_id=f"r{i}", start_frame=start * 30 // 1000, end_frame=end * 30 // 1000,
                               start_ms=start, end_ms=end, shot_ids=ids, winner=w, confidence=0.9))
    return c.EventsResult(video_id="v", match_format=fmt, rallies=rallies, shots=shots)


def label(t):
    return {"smash": "杀球"}.get(t, t) if t else None


def test_rally_log_running_score_and_game_split():
    # P1 wins 21-0, then the second game starts and P2 leads 1-0
    ev = _events(["P1"] * 21 + ["P2", None])
    lines, games = rally_log(ev, label)
    assert [ln.rally_no for ln in lines[:3]] == [1, 2, 3]
    assert lines[0].score == {"P1": 1, "P2": 0} and lines[0].game_no == 1
    assert lines[20].score == {"P1": 21, "P2": 0}
    assert lines[21].score == {"P1": 0, "P2": 1} and lines[21].game_no == 2
    assert lines[22].score is None and lines[22].winner is None  # no winner -> no score move
    assert games[0].winner == "P1" and games[0].score == {"P1": 21, "P2": 0}
    assert games[1].winner is None and games[1].rallies == 1
    ln = lines[0]
    assert ln.shots == 3 and ln.server == "P1" and ln.last_hitter == "P1" and ln.last_shot_type == "杀球"
    assert ln.mean_interval_s == pytest.approx(0.8)


def test_deuce_needs_two_point_lead_and_caps_at_30():
    # 20-20, then alternate up to 29-29, then P2 takes the 30th point
    seq = ["P1", "P2"] * 29 + ["P2"]
    lines, games = rally_log(_events(seq), label)
    assert len(games) == 1 and games[0].score == {"P1": 29, "P2": 30} and games[0].winner == "P2"
    seq = ["P1", "P2"] * 20 + ["P1", "P1"]  # 22-20 ends the game
    _, games = rally_log(_events(seq), label)
    assert games[0].score == {"P1": 22, "P2": 20} and games[0].winner == "P1"


def test_runs_lead_changes_and_tempo():
    # P1 3-0 run, P2 4 in a row (lead changes once), P1 1
    ev = _events(["P1"] * 3 + ["P2"] * 4 + ["P1"], gap_ms=1000)
    lines, games = rally_log(ev, label)
    match, per = flow_metrics(ev, lines, games, ("P1", "P2"))
    m = {x.key: x for x in match + per["P1"] + per["P2"]}
    assert m["P1.win.longest_run"].value == 3 and m["P2.win.longest_run"].value == 4
    assert m["match.score.lead_changes"].value == 1
    assert m["P1.win.biggest_lead"].value == 3 and m["P2.win.biggest_lead"].value == 1
    assert m["match.tempo.interval_mean_s"].value == pytest.approx(1.0)
    # each rally P1-P2-P1: P2 prepares once, P1 once; P1 gives time once, P2 once
    assert m["P1.tempo.prep_s"].n == 8 and m["P2.tempo.given_s"].n == 8


def test_tempo_ignores_same_side_and_implausible_gaps():
    ev = _events([None], hitters_per_rally={0: ["P1", "P1", "P2"]}, gap_ms=5000)
    lines, games = rally_log(ev, label)
    match, per = flow_metrics(ev, lines, games, ("P1", "P2"))
    m = {x.key: x for x in match}
    assert m["match.tempo.interval_mean_s"].value is None
    assert m["match.tempo.interval_mean_s"].status == "undeterminable"
    assert m["match.score.lead_changes"].value is None  # no winners entered
    assert lines[0].mean_interval_s is None


def test_doubles_score_by_team():
    ev = _events(["A", "B", "A"], hitters_per_rally={i: ["P1", "P2", "P3"] for i in range(3)}, fmt="doubles")
    lines, games = rally_log(ev, label)
    assert lines[-1].score == {"A": 2, "B": 1}
    assert lines[0].server == "A"
    _, per = flow_metrics(ev, lines, games, ("P1", "P3", "P2", "P4"))
    p3 = {x.key: x for x in per["P3"]}
    assert p3["P3.win.longest_run"].value == 1


def test_match_stats_carry_rally_log_and_flow_metrics():
    ev = _events(["P1", "P2", "P1"])
    st = compute_match_stats(ev)
    assert [ln.rally_id for ln in st.rallies] == ["r0", "r1", "r2"]
    assert st.games and st.games[0].score == {"P1": 2, "P2": 1}
    idx = index(st)
    for key in ("match.tempo.interval_mean_s", "match.score.lead_changes", "P1.tempo.prep_s", "P2.win.longest_run"):
        assert key in idx
