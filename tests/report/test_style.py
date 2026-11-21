"""Playing style: attack/neutral/defense states, tactics, prototype matching, report and coach wiring."""

import pytest

from rallyiq import contracts as c
from rallyiq.report import compute_match_stats, render_markdown
from rallyiq.report.coach import build_report, knowledge_context
from rallyiq.report.coach.advice import detect_focus
from rallyiq.report.coach.offline import OfflineProvider
from rallyiq.report.coach.prompt import user_message
from rallyiq.report.style import (
    COUNTER,
    DOUBLES_SYSTEMS,
    SINGLES_STYLES,
    TACTICS,
    is_attack,
    match_prototype,
    phase_of,
)

FPS = 30.0
NEAR = {"rear": 1.0, "mid": 3.3, "front": 5.6}  # y of the near half (P1 / team A)
FAR = {k: 13.4 - v for k, v in NEAR.items()}  # y of the far half
X = {"L": 1.2, "C": 3.05, "R": 4.9}


def _pos(pid, depth, lane):
    near = pid in ("P1", "P3")
    return [X[lane], (NEAR if near else FAR)[depth]]


def _rallies(specs, doubles=False):
    """specs: list of (winner, [(hitter, type, depth, lane, seconds to next shot), ...])."""
    rallies, shots, frames = [], [], []
    f = 0
    for k, (winner, seq) in enumerate(specs):
        rid = f"r{k}"
        start = f
        ids = []
        for j, (hitter, typ, depth, lane, dt) in enumerate(seq):
            sid = f"{rid}s{j}"
            hpos = _pos(hitter, depth, lane)
            shots.append(c.ShotEvent(shot_id=sid, rally_id=rid, frame=f, t_ms=int(f / FPS * 1000), hitter=hitter,
                                     hitter_court=hpos, shot_type=typ, confidence=0.9, status="ok"))
            ids.append(sid)
            # everyone else stands at their base (mid court centre) at this frame
            others = ["P1", "P2", "P3", "P4"] if doubles else ["P1", "P2"]
            dets = []
            for n, pid in enumerate(others):
                fc = hpos if pid == hitter else _pos(pid, "mid", "C")
                if doubles and pid != hitter and pid in ("P3", "P4"):
                    fc = _pos(pid, "front", "C")
                dets.append(c.PlayerDetection(player_id=pid, track_id=n + 1, bbox=[0, 0, 1, 1], foot_court=fc,
                                              confidence=0.9))
            frames.append(c.PlayerFrame(frame=f, t_ms=int(f / FPS * 1000), players=dets))
            f += max(1, round(dt * FPS))
        rallies.append(c.Rally(rally_id=rid, start_frame=start, end_frame=f, start_ms=int(start / FPS * 1000),
                               end_ms=int(f / FPS * 1000), shot_ids=ids, winner=winner, confidence=0.9))
        f += 150
    events = c.EventsResult(video_id="v", rallies=rallies, shots=shots,
                            match_format="doubles" if doubles else "singles")
    players = c.PlayerTracks(video_id="v", frames=frames)
    return events, players


def _cal():
    return c.CourtCalibration(video_id="v", segments=[c.CalibrationSegment(
        start_frame=0, end_frame=10**7, status="ok", method="manual", homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        confidence=0.9)])


# P1 attacks: push / smash, and kills the block; P2 lifts and blocks.
POWER = [("P1" if k % 4 else "P2", [
    ("P2", "short_serve", "mid", "C", 0.9),
    ("P1", "push", "front", "L", 0.6),
    ("P2", "lift", "rear", "R", 1.3),
    ("P1", "smash", "rear", "L", 0.5),
    ("P2", "block", "mid", "L", 0.8),
    ("P1", "push", "front", "R", 0.6),
    ("P2", "lift", "rear", "L", 1.3),
    ("P1", "smash", "rear", "R", 0.5),
]) for k in range(12)]

# P1 builds with clears and drops (four corners), then attacks; P2 clears and lifts back.
RALLY_ATTACK = [("P1" if k % 3 else "P2", [
    ("P1", "short_serve", "mid", "C", 0.9),
    ("P2", "lift", "front", "C", 1.4),
    ("P1", "clear", "rear", "L", 1.4),
    ("P2", "clear", "rear", "R", 1.4),
    ("P1", "drop", "rear", "R", 1.1),
    ("P2", "lift", "front", "L", 1.4),
    ("P1", "clear", "rear", "R", 1.4),
    ("P2", "clear", "rear", "L", 1.4),
    ("P1", "drop", "rear", "L", 1.1),
    ("P2", "lift", "front", "R", 1.4),
    ("P1", "smash", "rear", "L", 0.5),
    ("P2", "block", "mid", "C", 0.8),
]) for k in range(9)]


def test_phase_rules():
    assert phase_of("smash", None) == "defense"
    assert phase_of("net_kill", "attack") == "defense"
    assert phase_of("lift", None) == "attack"
    assert phase_of("clear", "defense") == "attack"
    assert phase_of("clear", "neutral") == "neutral"
    assert phase_of(None, None) == "neutral"
    # a lift met late and below the net is no attacking chance
    assert phase_of("lift", None, "low") == "neutral" and phase_of("lift", None, "high") == "attack"
    assert phase_of("smash", None, "high") == "defense"
    assert is_attack("smash", "neutral") and is_attack("drop", "attack") and not is_attack("drop", "neutral")


def test_power_attacker_vs_defender():
    events, players = _rallies(POWER)
    stats = compute_match_stats(events, players, _cal())
    by = {s.subject: s for s in stats.styles}
    p1, p2 = by["P1"], by["P2"]
    assert p1.primary.code == "power_attack", [(s.code, s.score) for s in p1.scores]
    assert p1.primary.score >= 70
    assert p1.n_shots == 48 and p1.status == "needs_review" and p1.confidence == "medium"  # ok from 80 shots
    # every P1 shot attacks; P2 defends after each smash and holds (the rally goes on or P1 errs)
    ph1 = {p.phase: p for p in p1.phases}
    assert ph1["attack"].counts == {"smash": 24}
    assert ph1["attack"].recommended_pct == 100.0
    ph2 = {p.phase: p for p in p2.phases}
    assert ph2["defense"].counts == {"block": 12}
    assert p2.primary.code in {"defensive_counter", "control", "endurance"}
    # features are metrics the coach can cite
    keys = {m.key: m for p in stats.players for m in p.metrics}
    assert keys["P1.style.smash_pct"].value == 50.0
    assert keys["P1.style.attack_pct"].value == 100.0
    assert keys["P1.style.convert_pct"].value == 100.0
    assert keys["P1.style.match_score"].value == p1.primary.score
    assert set(p1.evidence) >= {"P1.style.smash_pct", "P1.style.match_score"}
    # P2 always lifts after P1's push: next-shot prediction
    assert any(h.after == "push" and h.reply == "lift" and h.pct == 100.0 for h in p2.habits)
    # advice to P1 uses the opponent's habit; P2 gets the counter-plan when P1's style is clear
    assert any("挑球" in a for a in p1.advice)
    assert any(COUNTER["power_attack"] in a for a in p2.advice)


def test_rally_to_attack_and_tactics():
    events, players = _rallies(RALLY_ATTACK)
    stats = compute_match_stats(events, players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert p1.primary.code == "rally_attack", [(s.code, s.score) for s in p1.scores]
    tac = {t.code: t for t in p1.tactics}
    assert set(tac) == set(TACTICS)
    assert tac["rally_to_attack"].uses == 9
    assert tac["rally_to_attack"].decided == 9 and tac["rally_to_attack"].won == 6
    assert tac["rally_to_attack"].success_pct == pytest.approx(66.7)
    assert tac["rally_to_attack"].examples_ms  # times to replay
    # the clears go corner to corner: P2 moves far on each
    assert tac["space_control"].uses >= 9
    # sudden acceleration: the smash after slow clears and drops
    assert tac["tempo_disruption"].uses >= 9
    md = render_markdown(stats)
    assert "## 打法风格" in md and "拉吊突击型" in md and "| 拉吊突击 | 9 | 9 | 6 | 67% |" in md


def test_net_to_attack_and_defence_to_attack():
    specs = [("P1", [
        ("P2", "short_serve", "mid", "C", 0.9),
        ("P1", "net_shot", "front", "L", 0.9),
        ("P2", "lift", "front", "R", 1.3),
        ("P1", "smash", "rear", "L", 0.5),
        ("P2", "smash", "rear", "R", 0.5),  # P2 smashes, P1 defends then attacks again
        ("P1", "block", "mid", "C", 0.8),
        ("P2", "lift", "front", "L", 1.3),
        ("P1", "smash", "rear", "R", 0.5),
    ])] * 6
    events, players = _rallies(specs)
    stats = compute_match_stats(events, players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    tac = {t.code: t for t in p1.tactics}
    # the net shot forced the lift every time, but P2 answered the smash with a smash: not forced
    assert tac["net_to_attack"].uses == 6 and tac["net_to_attack"].forced_pct == 0.0
    assert tac["defence_to_attack"].uses == 6
    keys = {m.key: m for p in stats.players for m in p.metrics}
    assert keys["P1.style.counter_pct"].value == 100.0
    assert keys["P1.style.hold_pct"].value == 100.0


def test_too_few_shots_is_undeterminable():
    events, players = _rallies(POWER[:2])
    stats = compute_match_stats(events, players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert p1.status == "undeterminable" and p1.label == "证据不足" and p1.primary is None
    assert p1.advice == []
    assert any("打法风格暂不判断" in x for x in stats.caveats)


def test_no_styles_without_stroke_types():
    events, players = _rallies(POWER)
    events = events.model_copy(update={"shots": [s.model_copy(update={"shot_type": None}) for s in events.shots]})
    stats = compute_match_stats(events, players, _cal())
    assert stats.styles == []
    assert "打法风格" not in render_markdown(stats)


def test_moments_carry_the_hitters_state():
    events, players = _rallies(POWER)
    stats = compute_match_stats(events, players, _cal())
    for m in stats.moments:
        assert m.phase in ("attack", "neutral", "defense")


def test_prototype_scoring():
    p = next(x for x in SINGLES_STYLES if x.code == "power_attack")
    big = 10**6  # large samples: no shrinkage toward "typical"
    hi = {"smash_pct": (25.0, big, "杀球占比", "%"), "attack_pct": (40.0, big, "", "%"),
          "attack_run": (3.0, big, "", "拍"), "convert_pct": (90.0, big, "", "%"), "forced_pct": (45.0, big, "", "%"),
          "defense_pct": (5.0, big, "", "%")}
    s = match_prototype(p, hi)
    assert s.score == 100.0 and s.coverage == 1.0 and s.reasons[0].startswith("杀球占比 25%")
    lo = {k: (0.0, big, "", None) for k in hi}
    assert match_prototype(p, lo).score == pytest.approx(100 * 1 / 11, abs=0.1)  # only defense_pct (low) matches
    # ten samples per feature: halfway between the data and "typical" (50)
    ten = {k: (v[0], 10, "", None) for k, v in hi.items()}
    assert match_prototype(p, ten).score == pytest.approx(75.0, abs=0.1)
    few = {"smash_pct": (25.0, 50, "", "%")}
    assert match_prototype(p, few).score is None  # under half of the weight measurable
    # a feature from a handful of samples may count, but is never quoted as a reason
    eight = {**{k: (v[0], 100, "", None) for k, v in hi.items()}, "smash_pct": (25.0, 8, "杀球占比", "%")}
    assert not any(r.startswith("杀球占比") for r in match_prototype(p, eight).reasons)
    with_ex = match_prototype(p, hi, examples={"smash_pct": [61_000, 125_500]})
    assert with_ex.reasons[0].endswith("回看 01:01、02:05）")
    tiny = {k: (v[0], 2, "", None) for k, v in hi.items()}
    assert match_prototype(p, tiny).score is None  # samples too small
    assert len(SINGLES_STYLES) == 8 and len(DOUBLES_SYSTEMS) == 5


def test_offline_coach_and_prompt_use_styles():
    events, players = _rallies(POWER)
    stats = compute_match_stats(events, players, _cal())
    rep = build_report(OfflineProvider().generate(stats), stats, OfflineProvider())
    style = [i for i in rep.insights if "打法" in i.title]
    assert any(i.title.startswith("P1 打法：强攻压制型") for i in style)
    assert all(i.grounded for i in style), [(i.title, i.warnings) for i in style]
    assert any(i.kind == "tactic" and "对阵强攻压制型" in i.title for i in style)
    msg = user_message(stats, knowledge_context(stats, notes=[]))
    assert '"styles"' in msg and "强攻压制型" in msg and '"state"' in msg


def test_focus_from_wasted_attack_chances():
    # P1 gets lifts (attack state) but keeps clearing them back
    specs = [("P2", [
        ("P2", "short_serve", "mid", "C", 0.9),
        ("P1", "lift", "front", "C", 1.3),
        ("P2", "clear", "rear", "L", 1.3),
        ("P1", "clear", "rear", "R", 1.3),
        ("P2", "lift", "front", "R", 1.3),
        ("P1", "clear", "rear", "L", 1.3),
        ("P2", "lift", "front", "L", 1.3),
        ("P1", "clear", "rear", "C", 1.3),
        ("P2", "smash", "rear", "C", 0.5),
    ])] * 6
    events, players = _rallies(specs)
    stats = compute_match_stats(events, players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert any("主动状态" in a and "高远球" in a for a in p1.advice)
    assert any(f.id == "P1.convert" for f in detect_focus(stats))


def test_doubles_systems_and_formation():
    # team A: P1 smashes from the back while P3 stays at the net (front-back)
    specs = [("A", [
        ("P2", "short_serve", "mid", "C", 0.9),
        ("P3", "push", "front", "C", 0.6),
        ("P2", "lift", "rear", "C", 1.2),
        ("P1", "smash", "rear", "L", 0.4),
        ("P4", "block", "mid", "C", 0.7),
        ("P3", "net_kill", "front", "R", 0.5),
    ])] * 10
    events, players = _rallies(specs, doubles=True)
    stats = compute_match_stats(events, players, _cal())
    by = {s.subject: s for s in stats.styles}
    assert set(by) == {"A", "B"} and by["A"].kind == "doubles" and by["A"].players == ["P1", "P3"]
    codes = {s.code for s in by["A"].scores}
    assert codes == {p.code for p in DOUBLES_SYSTEMS}
    assert by["A"].primary.code in {"rear_attack", "front_intercept", "front_back_mixed"}
    keys = {m.key: m for m in stats.metrics}
    assert keys["A.style.fb_pct"].value == 100.0  # P1 back / P3 front at every A shot
    assert keys["A.style.smash_conc_pct"].value == 100.0
    assert by["A"].tactics == []  # singles tactics are not counted for doubles


def test_misattributed_hitters_lower_confidence():
    # the same player credited with consecutive hits: singles hits must alternate
    events, players = _rallies(POWER)
    shots = [sh.model_copy(update={"hitter": "P1"}) if k % 3 == 0 else sh for k, sh in enumerate(events.shots)]
    events = events.model_copy(update={"shots": shots})
    stats = compute_match_stats(events, players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert p1.confidence == "low" and p1.status == "needs_review"
    assert p1.note.startswith("相邻两拍由不同球员击出的只有")
    assert any(c.startswith("相邻两拍") for c in stats.caveats)


def test_random_hitters_give_no_verdict():
    events, players = _rallies(POWER)
    shots = [sh.model_copy(update={"hitter": "P1" if k % 4 < 2 else "P2"}) for k, sh in enumerate(events.shots)]
    stats = compute_match_stats(events.model_copy(update={"shots": shots}), players, _cal())
    for st in stats.styles:
        assert st.status == "undeterminable" and st.primary is None and st.advice == []
        assert st.label == "证据不足（击球者识别不可靠）"
    assert not any(m.key.endswith("style.match_score") for p in stats.players for m in p.metrics)


def test_unconfirmed_stroke_types_lower_confidence():
    events, players = _rallies(POWER)
    auto = c.StrokeInfo(shot_type="smash", confidence=0.4, status="needs_review", source="model")
    shots = [s.model_copy(update={"stroke": auto.model_copy(update={"shot_type": s.shot_type})}) for s in events.shots]
    stats = compute_match_stats(events.model_copy(update={"shots": shots}), players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert p1.primary.code == "power_attack" and p1.confidence == "low"
    assert "只有 0 拍的类型经过确认" in p1.note
    # once a person has checked the types the confidence comes back
    human = [s.model_copy(update={"stroke": s.stroke.model_copy(update={"source": "human", "status": "ok"})})
             for s in shots]
    stats = compute_match_stats(events.model_copy(update={"shots": human}), players, _cal())
    p1 = next(s for s in stats.styles if s.subject == "P1")
    assert p1.confidence == "medium" and (p1.note is None or "经过确认" not in p1.note)
