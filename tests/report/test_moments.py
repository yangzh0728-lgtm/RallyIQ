"""Shot-by-shot advice: stroke type + both players' court positions -> timed shots to change."""

import json

import pytest

from rallyiq import contracts as c
from rallyiq.report import compute_match_stats, generate_coach_report, render_markdown
from rallyiq.report.coach import build_report, knowledge_context
from rallyiq.report.coach.grounding import ground_shot_advice
from rallyiq.report.coach.offline import OfflineProvider
from rallyiq.report.coach.prompt import OUTPUT_SCHEMA, user_message
from rallyiq.report.moments import shot_moments, timestamp

FPS = 30.0


def _rally(rid, start, shots, winner):
    """shots: (hitter, type, (x, y) of hitter, (x, y) of the other player) per shot, one second apart."""
    evs, frames = [], []
    for k, (hitter, typ, hpos, opos) in enumerate(shots):
        f = start + 30 * k
        evs.append(c.ShotEvent(shot_id=f"{rid}s{k}", rally_id=rid, frame=f, t_ms=int(f / FPS * 1000), hitter=hitter,
                               hitter_court=list(hpos), shot_type=typ, confidence=0.9, status="ok"))
        other = "P2" if hitter == "P1" else "P1"
        frames.append(c.PlayerFrame(frame=f, t_ms=int(f / FPS * 1000), players=[
            c.PlayerDetection(player_id=hitter, track_id=1 if hitter == "P1" else 2, bbox=[0, 0, 1, 1],
                              foot_court=list(hpos), confidence=0.9),
            c.PlayerDetection(player_id=other, track_id=1 if other == "P1" else 2, bbox=[0, 0, 1, 1],
                              foot_court=list(opos), confidence=0.9),
        ]))
    end = start + 30 * len(shots)
    rally = c.Rally(rally_id=rid, start_frame=start, end_frame=end, start_ms=int(start / FPS * 1000),
                    end_ms=int(end / FPS * 1000), shot_ids=[e.shot_id for e in evs], winner=winner, confidence=0.9)
    return rally, evs, frames


def _match(*rallies):
    rs, shots, frames = [], [], []
    for r, e, f in rallies:
        rs.append(r)
        shots += e
        frames += f
    events = c.EventsResult(video_id="v", rallies=rs, shots=shots)
    players = c.PlayerTracks(video_id="v", frames=sorted(frames, key=lambda f: f.frame))
    return events, players


# P1 plays the near half (y < 6.7), P2 the far half.
LIFT_TO_REAR = _rally("r1", 0, [
    ("P1", "short_serve", (3.0, 5.0), (3.0, 9.0)),
    ("P2", "net_shot", (3.0, 8.0), (3.0, 4.0)),
    ("P1", "lift", (3.0, 5.5), (2.0, 12.5)),  # P2 already deep (depth 0.9), on P1's left
    ("P2", "smash", (2.2, 12.3), (3.0, 5.5)),  # P2 barely moved, smashed and won
], winner="P2")
DROP_TO_NET = _rally("r2", 300, [
    ("P2", "short_serve", (3.0, 8.5), (3.0, 4.0)),
    ("P1", "drop", (1.0, 1.0), (4.5, 7.6)),  # P2 already at the net
    ("P2", "net_kill", (4.4, 7.5), (1.0, 1.0)),
], winner="P2")
OFF_BALANCE = _rally("r3", 600, [
    ("P2", "short_serve", (3.0, 8.5), (3.0, 4.0)),
    ("P1", "clear", (3.0, 4.0), (3.0, 9.0)),
    ("P2", "clear", (3.0, 12.0), (3.0, 4.0)),
    ("P1", "smash", (0.8, 0.8), (3.0, 10.0)),  # from the rear corner, last shot, lost
], winner="P2")
TO_BODY = _rally("r4", 900, [
    ("P2", "short_serve", (3.0, 8.5), (3.0, 4.0)),
    ("P1", "drive", (3.0, 4.0), (3.0, 9.4)),
    ("P2", "drive", (3.2, 9.5), (3.0, 4.0)),  # 0.2 m from where P2 stood
], winner="P2")


@pytest.fixture(scope="module")
def match():
    return _match(LIFT_TO_REAR, DROP_TO_NET, OFF_BALANCE, TO_BODY)


def test_rules_find_each_shot_with_time_and_positions(match):
    events, players = match
    ms = {m.shot_id: m for m in shot_moments(events, players, FPS)}
    assert set(ms) == {"r1s2", "r2s1", "r3s3", "r4s1"}

    lift = ms["r1s2"]
    assert (lift.issue, lift.rally_no, lift.shot_no, lift.player_id) == ("clear_to_rear", 1, 3, "P1")
    assert lift.t_ms == 2000 and timestamp(lift.t_ms) == "00:02"
    assert lift.hitter_zone.startswith("网前") and lift.opponent_zone == "后场左侧"
    assert lift.opponent_reach_m == pytest.approx(0.3, abs=0.05)
    assert (lift.next_shot_type, lift.outcome) == ("smash", "lost")
    # farthest open corner from an opponent deep on the left, reachable from the net: net shot to the front right
    assert (lift.suggested_shot, lift.target_zone) == ("net_shot", "网前右侧")

    drop = ms["r2s1"]
    assert drop.issue == "short_to_front" and drop.suggested_shot == "clear" and drop.target_zone.startswith("后场")
    assert ms["r3s3"].issue == "attack_off_balance" and ms["r3s3"].suggested_shot == "clear"
    body = ms["r4s1"]
    assert body.issue == "to_opponent" and body.opponent_reach_m < 1.0 and body.opponent_zone.startswith("中场")
    assert [m.moment_id for m in shot_moments(events, players, FPS)] == ["m1", "m2", "m3", "m4"]  # video order


def test_no_moments_without_strokes_tracks_or_in_doubles(match):
    events, players = match
    untyped = events.model_copy(update={"shots": [s.model_copy(update={"shot_type": None}) for s in events.shots]})
    assert shot_moments(untyped, players, FPS) == []
    assert shot_moments(events, None, FPS) == []
    assert shot_moments(events.model_copy(update={"match_format": "doubles"}), players, FPS) == []


def test_won_rally_is_not_flagged():
    r, e, f = _rally("r", 0, [
        ("P2", "short_serve", (3.0, 8.5), (3.0, 4.0)),
        ("P1", "drive", (3.0, 4.0), (3.0, 9.4)),
        ("P2", "drive", (3.2, 9.5), (3.0, 4.0)),
    ], winner="P1")
    assert shot_moments(*_match((r, e, f)), FPS) == []


def test_stats_carry_moments_but_prompt_stats_do_not(match):
    events, players = match
    stats = compute_match_stats(events, players)
    assert len(stats.moments) == 4
    k = knowledge_context(stats, notes=(), self_player="P1")
    msg = user_message(stats, k)
    head, moments = msg.split("请为每条写一条 shot_advice：\n\n", 1)
    assert '"moments"' not in head and "使用者本人是 P1" in head
    cards = json.loads(moments)
    assert [m["moment_id"] for m in cards] == ["m1", "m2", "m3", "m4"]
    assert cards[0]["time"] == "00:02" and cards[0]["opponent_zone"] == "后场左侧"
    assert "shot_advice" in OUTPUT_SCHEMA["required"]


def test_self_player_selects_their_shots(match):
    events, players = match
    stats = compute_match_stats(events, players)
    assert len(knowledge_context(stats, notes=(), self_player="P1").moments) == 4
    assert knowledge_context(stats, notes=(), self_player="P2").moments == []  # never advise on the opponent
    assert len(knowledge_context(stats, notes=()).moments) == 4


def test_grounding_keeps_time_from_the_moment(match):
    events, players = match
    ms = compute_match_stats(events, players).moments
    raw = [
        {"moment_id": "m1", "advice": "这拍放网到右边网前。", "why": "对手在后场左侧，离右前角最远。"},
        {"moment_id": "m2", "advice": "05:31 这拍应该挑后场", "why": "x"},  # wrong time -> rule text
        {"moment_id": "m99", "advice": "编的", "why": "编的"},  # unknown moment -> ignored
    ]
    out = ground_shot_advice(raw, ms, "P1")
    assert [a.moment_id for a in out] == ["m1", "m2", "m3", "m4"]
    assert out[0].source == "model" and out[0].advice == "这拍放网到右边网前。" and out[0].timestamp == "00:02"
    assert out[0].situation.startswith("第1回合第3拍：你在网前") and "对手在后场左侧" in out[0].situation
    assert out[1].source == "rules" and out[1].warnings and "05:31" in out[1].warnings[0]
    assert out[2].source == "rules" and out[2].advice.endswith("。")


def test_offline_report_lists_shot_advice_and_markdown(match):
    events, players = match
    stats = compute_match_stats(events, players)
    rep = generate_coach_report(stats, "offline", self_player="P1")
    assert rep.self_player == "P1" and [a.timestamp for a in rep.shot_advice] == ["00:02", "00:11", "00:23", "00:31"]
    assert all(a.source == "rules" for a in rep.shot_advice)
    md = render_markdown(stats, rep)
    assert "### 逐拍建议" in md and "**00:02** 第1回合第3拍：你在网前" in md
    # older reports (no shot_advice) still build
    old = build_report({"summary": "", "insights": [], "training": [], "caveats": []}, stats, OfflineProvider(),
                       knowledge=knowledge_context(stats, notes=()))
    assert len(old.shot_advice) == 4
