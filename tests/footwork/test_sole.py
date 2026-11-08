"""Heel / toe (足底) analysis on synthetic foot landmarks: foot direction, heel-strike vs forefoot landings,
the lunge front foot, the feet stage's frame selection and occlusion skip."""

from __future__ import annotations

import numpy as np
import pytest

from rallyiq import contracts as c
from rallyiq.footwork import Footwork
from rallyiq.footwork.tracks import apply_h
from rallyiq.perception.feet.stage import detect_feet, wanted_frames
from rallyiq.report.stats import compute_match_stats

from test_footwork import FPS, HINV, Script, ankle_px, calibration, events, opponent, scale_at, shot, tracks

HEEL_BACK_M, TOE_FWD_M = 0.07, 0.19  # floor point under the ankle -> heel / toe along the foot


class Feet:
    """Foot direction (unit court vector) and pitch (m, + = toe up, - = heel up) per frame and foot."""

    def __init__(self, sc: Script, direction=(0.0, 1.0)):
        n = len(sc.pos)
        self.sc = sc
        self.dir = np.zeros((n, 2, 2))
        self.dir[:] = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.pitch = np.zeros((n, 2))
        self.length = HEEL_BACK_M + TOE_FWD_M

    def turn(self, foot: int, f0: int, direction) -> None:
        self.dir[f0:, foot] = np.asarray(direction, dtype=float) / np.linalg.norm(direction)

    def landmarks(self, f: int, foot: int) -> tuple[list[float], list[float], list[float]]:
        pos, u, lift = self.sc.pos[f, foot], self.dir[f, foot], self.sc.lift[f, foot]
        s = scale_at(pos)
        heel = apply_h(HINV, np.asarray([pos - HEEL_BACK_M * u]))[0]
        toe = apply_h(HINV, np.asarray([pos + TOE_FWD_M * u]))[0]
        p = self.pitch[f, foot]
        heel[1] -= (lift + max(-p, 0.0)) * s
        toe[1] -= (lift + max(p, 0.0)) * s
        a = ankle_px(pos, lift)
        return [a[0], a[1], 0.95], [float(heel[0]), float(heel[1]), 0.95], [float(toe[0]), float(toe[1]), 0.95]


def landmarks(feet: dict[str, Feet], n: int, swap_frames=(), noise_px: float = 0.4) -> c.FootLandmarks:
    rng = np.random.default_rng(1)
    frames = []
    for f in range(n):
        ps = []
        for k, (pid, ft) in enumerate(feet.items()):
            got = [ft.landmarks(f, j) for j in range(2)]
            got = [[[v[0] + rng.normal(0, noise_px), v[1] + rng.normal(0, noise_px), v[2]] for v in g] for g in got]
            if f in swap_frames:
                got = got[::-1]
            ps.append(c.PlayerFeet(player_id=pid, track_id=k, ankle=[got[0][0], got[1][0]],
                                   heel=[got[0][1], got[1][1]], toe=[got[0][2], got[1][2]]))
        frames.append(c.FeetFrame(frame=f, t_ms=round(f * 1000 / FPS), players=ps))
    return c.FootLandmarks(video_id="v", model="test", frames=frames)


def run(me: Script, feet: dict[str, Feet], n: int, shots=(), **kw) -> Footwork:
    return Footwork(events(list(shots), n), tracks({"P1": me, "P2": feet["P2"].sc}, n), calibration(n), FPS,
                    landmarks(feet, n, **kw))


def standing(n: int) -> tuple[Script, dict[str, Feet]]:
    me = Script(n, [2.7, 3.0], [3.4, 3.0])
    return me, {"P1": Feet(me), "P2": Feet(opponent(n), (0.0, -1.0))}


def test_foot_direction_in_the_players_own_half():
    n = 90
    me, feet = standing(n)
    feet["P1"].turn(1, 0, (1.0, 1.0))  # right foot turned out 45 degrees
    fw = run(me, feet, n)
    assert fw.sole_ok == {"P1": True, "P2": True}
    mine = {x.foot: x for x in fw.result().contacts if x.player_id == "P1"}
    assert mine["left"].toe_dir_deg == pytest.approx(0, abs=8)
    assert mine["right"].toe_dir_deg == pytest.approx(45, abs=8)
    assert mine["right"].heel_image is not None and mine["right"].toe_image is not None
    # the far player faces the other way: pointing at the net is 0 in their own half too
    far = [x for x in fw.result().contacts if x.player_id == "P2"]
    assert all(x.toe_dir_deg == pytest.approx(0, abs=8) for x in far)


def step_with(me: Script, ft: Feet, f0: int, f1: int, to, pitch: float) -> None:
    """Right foot steps to ``to`` and touches down with the toe (pitch > 0) or heel (< 0) still up."""
    me.move(1, f0, f1, to, 0.1)
    ft.pitch[f1 - 3:f1 + 1, 1] = pitch
    ft.pitch[f1 + 1, 1] = pitch / 2


def test_heel_strike_and_forefoot_landings():
    n = 160
    me, feet = standing(n)
    step_with(me, feet["P1"], 30, 42, [3.4, 3.8], 0.09)  # heel first
    step_with(me, feet["P1"], 90, 102, [3.4, 4.6], -0.09)  # on the ball of the foot
    fw = run(me, feet, n)
    steps = [st for st in fw.result().steps if st.player_id == "P1"]
    assert [st.landing for st in steps] == ["heel", "forefoot"]


def test_flat_landing_is_not_called_heel_or_forefoot():
    n = 100
    me, feet = standing(n)
    me.move(1, 30, 42, [3.4, 3.8], 0.1)
    fw = run(me, feet, n)
    assert [st.landing for st in fw.result().steps if st.player_id == "P1"] == ["flat"]


def test_foot_model_left_right_swaps_are_paired_by_ankle():
    n = 90
    me, feet = standing(n)
    feet["P1"].turn(1, 0, (1.0, 1.0))
    fw = run(me, feet, n, swap_frames=set(range(10, 80, 3)))
    mine = {x.foot: x for x in fw.result().contacts if x.player_id == "P1"}
    assert mine["right"].toe_dir_deg == pytest.approx(45, abs=8)
    assert mine["left"].toe_dir_deg == pytest.approx(0, abs=8)


def test_no_soles_where_the_projection_breaks_the_shoe():
    n = 90
    me, feet = standing(n)
    feet["P1"].length = 0.9
    bad = Feet(me)
    # a projection that stretches the shoe to ~0.8 m: no directions, no landings, a caveat instead
    bad.landmarks = lambda f, foot: (lambda a, h, t: (a, h, [t[0], t[1] - 0.6 * scale_at(me.pos[f, foot]), t[2]]))(
        *Feet.landmarks(bad, f, foot))
    feet["P1"] = bad
    fw = run(me, feet, n)
    assert fw.sole_ok["P1"] is False
    assert all(x.toe_dir_deg is None and x.landing is None for x in fw.result().contacts if x.player_id == "P1")
    assert any("P1 的脚跟 / 脚尖" in x for x in fw.result().caveats)


def test_lunge_front_foot_landing_and_direction():
    n = 150
    me = Script(n, [2.7, 3.6], [3.4, 3.6])
    me.move(1, 50, 66, [4.3, 5.0], 0.12)  # forward-right lunge
    feet = {"P1": Feet(me), "P2": Feet(opponent(n), (0.0, -1.0))}
    feet["P1"].turn(1, 50, (1.0, 0.0))  # front toe points sideways: ~57 degrees off the lunge step
    feet["P1"].pitch[63:67, 1] = 0.1  # heel strike
    fw = run(me, feet, n, shots=[shot("s1", 30, "P2"), shot("s2", 68, "P1")])
    sf = fw.shot("s2")
    assert sf.finish == "lunge"
    assert sf.lunge_landing == "heel"
    assert sf.lunge_toe_off_deg == pytest.approx(57, abs=10)
    assert "脚跟先着地" in sf.summary and "脚尖偏离跨步方向" in sf.summary


def test_stats_report_landings_only_with_feet():
    n = 160
    me, feet = standing(n)
    step_with(me, feet["P1"], 30, 42, [3.4, 3.8], 0.09)
    step_with(me, feet["P1"], 90, 102, [3.4, 4.6], -0.09)
    ev = events([shot("s1", 20, "P1"), shot("s2", 140, "P2")], n)
    pt = tracks({"P1": me, "P2": feet["P2"].sc}, n)
    with_feet = compute_match_stats(ev, pt, calibration(n), None, landmarks(feet, n))
    without = compute_match_stats(ev, pt, calibration(n), None)
    keys = {m.key: m for p in with_feet.players for m in p.metrics}
    assert keys["P1.footwork.heel_first_pct"].value == pytest.approx(50)
    assert not any(m.key.endswith("heel_first_pct") for p in without.players for m in p.metrics)
    assert with_feet.fingerprint != without.fingerprint


# ----------------------------------------------------------------------------- feet stage


def test_wanted_frames_cover_rallies_with_a_margin():
    w = wanted_frames([(100, 200)], 30.0, 1000)
    assert w(70) and w(150) and w(230)
    assert not w(60) and not w(240)
    assert wanted_frames([], 30.0, 10)(5)


def test_detect_feet_skips_a_player_hidden_behind_the_near_one():
    near = c.PlayerDetection(player_id="P1", track_id=0, bbox=[100, 200, 300, 700], confidence=0.9)
    far = c.PlayerDetection(player_id="P2", track_id=1, bbox=[150, 100, 250, 400], confidence=0.9)
    clear = c.PlayerDetection(player_id="P2", track_id=1, bbox=[600, 100, 700, 400], confidence=0.9)
    pt = c.PlayerTracks(video_id="v", frames=[c.PlayerFrame(frame=0, t_ms=0, players=[near, far]),
                                              c.PlayerFrame(frame=1, t_ms=33, players=[near, clear])])
    calls = []

    def fake(img, bbox):
        calls.append(tuple(bbox))
        return np.ones((3, 2, 3))

    img = np.zeros((10, 10, 3), np.uint8)
    out = detect_feet([(0, 0.0, img), (1, 33.0, img)], pt, fake, lambda f: True)
    assert [[p.player_id for p in fr.players] for fr in out] == [["P1"], ["P1", "P2"]]
    assert len(calls) == 3
