"""The same footwork filmed at 30 and 60 fps reads the same: every window and tolerance is in seconds."""

from __future__ import annotations

import pytest

import test_footwork as tf
from rallyiq.footwork import Footwork

from test_footwork import Script, calibration, events, opponent, shot, tracks


@pytest.fixture(params=[30.0, 60.0], ids=["30fps", "60fps"])
def fps(request, monkeypatch):
    monkeypatch.setattr(tf, "FPS", request.param)
    return request.param


def at(fps: float):
    return lambda t: int(round(t * fps))


def run(me: Script, sh, n: int, fps: float) -> Footwork:
    return Footwork(events(sh, n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), fps)


def test_split_step_reaction_and_steps(fps):
    F = at(fps)
    n = F(5)
    me = Script(n, [2.7, 3.3], [3.4, 3.3])
    me.hop(F(1.93), F(2.2), 0.07)  # lands 0.2 s after the opponent's hit at 2.0 s
    me.move(1, F(2.33), F(2.73), [4.6, 3.6], 0.1)
    me.move(0, F(2.8), F(3.13), [3.9, 3.6], 0.08)
    fw = run(me, [shot("s1", F(2.0), "P2"), shot("s2", F(3.2), "P1")], n, fps)
    sf = fw.shot("s2")
    assert sf.split_step == "on_time"
    assert sf.split_offset_ms == pytest.approx(200, abs=70)
    assert sf.reaction_ms is not None and 100 <= sf.reaction_ms <= 500
    assert sf.steps == 2


def test_jump_belongs_to_the_hit(fps):
    F = at(fps)
    n = F(4)
    me = Script(n, [2.7, 2.0], [3.4, 2.0])
    me.hop(F(1.67), F(2.13), 0.35)
    # the hit is logged at the very start of the take-off: still the jump's own hit
    fw = run(me, [shot("s1", F(1.62), "P1")], n, fps)
    jumps = [e for e in fw.footwork_events if e.kind == "jump"]
    assert len(jumps) == 1 and jumps[0].shot_id == "s1"
    assert fw.shot("s1").finish == "jump"


def test_lunge(fps):
    F = at(fps)
    n = F(5)
    me = Script(n, [2.7, 3.6], [3.4, 3.6])
    me.move(1, F(1.67), F(2.2), [4.3, 5.0], 0.12)
    fw = run(me, [shot("s1", F(1.0), "P2"), shot("s2", F(2.27), "P1")], n, fps)
    sf = fw.shot("s2")
    assert sf.finish == "lunge" and sf.stance_m == pytest.approx(2.13, abs=0.2)


def test_recovery(fps):
    F = at(fps)
    n = F(6.7)
    me = Script(n, [0.9, 1.0], [1.6, 1.0])  # hits from the rear-left corner, walks back to the base
    me.move(1, F(1.33), F(1.73), [3.5, 3.2], 0.1)
    me.move(0, F(1.87), F(2.27), [2.8, 3.2], 0.1)
    fw = run(me, [shot("s1", F(1.0), "P1"), shot("s2", F(3.0), "P2"), shot("s3", F(5.0), "P1")], n, fps)
    sf = fw.shot("s1")
    assert sf.recovery_gap_m == pytest.approx(0.42, abs=0.1)
    assert sf.recovery_ms == pytest.approx(1050, abs=120)


def test_quiet_feet_have_the_same_jitter(fps):
    F = at(fps)
    n = F(4)
    me = Script(n, [2.7, 3.0], [3.4, 3.0])
    me.move(1, F(1.0), F(1.33), [4.0, 3.4])
    fw = run(me, [], n, fps)
    assert fw.jitter["P1"] < 0.03
