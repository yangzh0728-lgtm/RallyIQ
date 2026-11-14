"""Footwork analysis on synthetic ankle tracks with known contacts, hops, lunges and jumps."""

from __future__ import annotations

import numpy as np
import pytest

from rallyiq import contracts as c
from rallyiq.contracts.geometry import homography_from_corners, invert_homography
from rallyiq.footwork import Footwork, analyze_footwork
from rallyiq.footwork.tracks import ANKLE_HEIGHT_M, _fix_swaps, apply_h

FPS = 30.0
# end-on camera, like an amateur phone/DJI video from behind the near baseline
CORNERS = [[60, 1020], [1860, 1020], [1150, 560], [770, 560]]
H = np.asarray(homography_from_corners(CORNERS))
HINV = np.asarray(invert_homography(H.tolist()))


def scale_at(court) -> float:
    a = apply_h(HINV, np.asarray([court]))[0]
    b = apply_h(HINV, np.asarray([[court[0] + 0.5, court[1]]]))[0]
    return float(np.linalg.norm(b - a) * 2)


def ankle_px(court, lift_m: float = 0.0) -> list[float]:
    """Image ankle keypoint for a foot standing at ``court`` lifted ``lift_m`` off the floor."""
    p = apply_h(HINV, np.asarray([court]))[0]
    s = scale_at(court)
    return [float(p[0]), float(p[1] - (ANKLE_HEIGHT_M + lift_m) * s)]


class Script:
    """Per-frame foot positions (court m) and lifts (m) for one player."""

    def __init__(self, n: int, left, right):
        self.pos = np.zeros((n, 2, 2))
        self.pos[:, 0] = left
        self.pos[:, 1] = right
        self.lift = np.zeros((n, 2))

    def move(self, foot: int, f0: int, f1: int, to, height: float = 0.08) -> None:
        """Swing ``foot`` from its position at f0 to ``to`` over frames f0..f1 (lifted on the way)."""
        start = self.pos[f0, foot].copy()
        to = np.asarray(to, dtype=float)
        for f in range(f0, f1 + 1):
            u = (f - f0) / (f1 - f0)
            self.pos[f, foot] = start + (to - start) * (0.5 - 0.5 * np.cos(np.pi * u))
            self.lift[f, foot] = max(self.lift[f, foot], height * np.sin(np.pi * u))
        self.pos[f1 + 1:, foot] = to

    def hop(self, f0: int, f1: int, height: float) -> None:
        for f in range(f0, f1 + 1):
            u = (f - f0) / (f1 - f0)
            self.lift[f] = np.maximum(self.lift[f], height * np.sin(np.pi * u))


def tracks(scripts: dict[str, Script], n: int, noise_px: float = 0.6, swap_frames=()) -> c.PlayerTracks:
    rng = np.random.default_rng(0)
    frames = []
    for f in range(n):
        dets = []
        for k, (pid, sc) in enumerate(scripts.items()):
            la = ankle_px(sc.pos[f, 0], sc.lift[f, 0])
            ra = ankle_px(sc.pos[f, 1], sc.lift[f, 1])
            la = [la[0] + rng.normal(0, noise_px), la[1] + rng.normal(0, noise_px)]
            ra = [ra[0] + rng.normal(0, noise_px), ra[1] + rng.normal(0, noise_px)]
            if f in swap_frames:
                la, ra = ra, la
            mid = np.mean([la, ra], axis=0)
            h = 3.2 * scale_at(sc.pos[f].mean(axis=0)) * 0.55
            kps = [[float(mid[0]), float(mid[1] - h * (1 - i / 17)), 0.9] for i in range(17)]
            kps[15] = [la[0], la[1], 0.95]
            kps[16] = [ra[0], ra[1], 0.95]
            court = sc.pos[f].mean(axis=0).tolist()
            dets.append(c.PlayerDetection(
                player_id=pid, track_id=k, side="near" if court[1] < c.NET_Y_M else "far",
                bbox=[float(mid[0] - 60), float(mid[1] - h), float(mid[0] + 60), float(mid[1])], keypoints=kps,
                foot_px=[float(mid[0]), float(mid[1])], foot_court=court, on_court=True, confidence=0.9))
        frames.append(c.PlayerFrame(frame=f, t_ms=round(f * 1000 / FPS), players=dets))
    return c.PlayerTracks(video_id="v", frames=frames)


def calibration(n: int) -> c.CourtCalibration:
    return c.CourtCalibration(video_id="v", segments=[c.CalibrationSegment(
        start_frame=0, end_frame=n - 1, homography=H.tolist(), status="ok", confidence=1.0)])


def shot(sid: str, frame: int, hitter: str, rally: str = "r001") -> c.ShotEvent:
    return c.ShotEvent(shot_id=sid, rally_id=rally, frame=frame, t_ms=round(frame * 1000 / FPS), hitter=hitter,
                       shot_type="clear", confidence=0.9)


def events(shots: list[c.ShotEvent], n: int) -> c.EventsResult:
    return c.EventsResult(video_id="v", rallies=[c.Rally(
        rally_id="r001", start_frame=0, end_frame=n - 1, start_ms=0, end_ms=round((n - 1) * 1000 / FPS),
        shot_ids=[s.shot_id for s in shots], confidence=0.9)], shots=shots)


def opponent(n: int) -> Script:
    return Script(n, [2.6, 10.0], [3.5, 10.0])


# --------------------------------------------------------------------------- tests


def test_contacts_and_steps_measured_on_court():
    n = 120
    me = Script(n, [2.7, 3.0], [3.4, 3.0])
    me.move(1, 30, 40, [4.0, 3.4])  # right foot steps 0.72 m
    me.move(0, 60, 70, [3.3, 3.4])  # left foot follows 0.72 m
    res = analyze_footwork(events([], n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    steps = [s for s in res.steps if s.player_id == "P1"]
    assert [s.foot for s in steps] == ["right", "left"]
    assert [s.length_m for s in steps] == pytest.approx([0.72, 0.72], abs=0.08)
    assert steps[0].to_court == pytest.approx([4.0, 3.4], abs=0.08)
    contacts = [x for x in res.contacts if x.player_id == "P1"]
    assert len(contacts) == 4  # each foot planted before and after its step
    assert not [s for s in res.steps if s.player_id == "P2"]  # standing still: no steps from keypoint noise
    q = {x.player_id: x for x in res.quality}
    assert q["P1"].status == "ok" and q["P1"].jitter_m < 0.03


def test_split_step_timed_against_the_opponents_hit():
    n = 150
    me = Script(n, [2.7, 3.3], [3.4, 3.3])
    me.hop(58, 66, 0.07)  # small hop, landing at frame 66 = 0.2 s after the opponent hits at 60
    me.move(1, 70, 82, [4.6, 3.6], 0.1)  # then off to the right
    me.move(0, 84, 94, [3.9, 3.6], 0.08)
    sh = [shot("s1", 60, "P2"), shot("s2", 96, "P1")]
    fw = Footwork(events(sh, n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    sf = fw.shot("s2")
    assert sf.split_step == "on_time"
    assert sf.split_offset_ms == pytest.approx(200, abs=70)
    assert sf.reaction_ms is not None and 100 <= sf.reaction_ms <= 500
    assert sf.steps == 2
    assert "启动步" in sf.summary
    ev = [e for e in fw.footwork_events if e.kind == "split_step"]
    assert len(ev) == 1 and ev[0].player_id == "P1" and ev[0].shot_id == "s1"


def test_no_split_step_without_a_hop():
    n = 150
    me = Script(n, [2.7, 3.3], [3.4, 3.3])
    me.move(1, 70, 82, [4.6, 3.6], 0.1)
    me.move(0, 84, 94, [3.9, 3.6], 0.08)
    sh = [shot("s1", 60, "P2"), shot("s2", 96, "P1")]
    fw = Footwork(events(sh, n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    assert fw.shot("s2").split_step == "none"
    assert fw.split_judgements()["s1"] == ("P1", "none", None)


def test_early_split_step():
    n = 150
    me = Script(n, [2.7, 3.3], [3.4, 3.3])
    me.hop(40, 48, 0.07)  # lands 0.4 s before the hit at 60
    sh = [shot("s1", 60, "P2"), shot("s2", 96, "P1")]
    fw = Footwork(events(sh, n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    assert fw.shot("s2").split_step == "early"


def test_jump_detected():
    n = 120
    me = Script(n, [2.7, 2.0], [3.4, 2.0])
    me.hop(50, 64, 0.35)  # 0.47 s in the air, 35 cm high
    fw = Footwork(events([shot("s1", 57, "P1")], n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    jumps = [e for e in fw.footwork_events if e.kind == "jump"]
    assert len(jumps) == 1 and jumps[0].value == pytest.approx(0.35, abs=0.1)
    assert fw.shot("s1").finish == "jump"


def test_lunge_finish():
    n = 150
    me = Script(n, [2.7, 3.6], [3.4, 3.6])
    me.move(1, 50, 66, [4.3, 5.0], 0.12)  # long last step forward-right: stance ~2.1 m
    fw = Footwork(events([shot("s1", 30, "P2"), shot("s2", 68, "P1")], n),
                  tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    sf = fw.shot("s2")
    assert sf.finish == "lunge" and sf.stance_m == pytest.approx(2.13, abs=0.2)
    assert any(e.kind == "lunge" for e in fw.footwork_events)


def test_recovery_gap_when_the_opponent_hits_next():
    n = 200
    me = Script(n, [0.9, 1.0], [1.6, 1.0])  # hits from the rear-left corner ...
    me.move(1, 40, 52, [3.2, 2.6], 0.1)  # ... then walks back to the base
    me.move(0, 56, 68, [2.6, 2.6], 0.1)
    sh = [shot("s1", 30, "P1"), shot("s2", 90, "P2"), shot("s3", 150, "P1")]
    fw = Footwork(events(sh, n), tracks({"P1": me, "P2": opponent(n)}, n), calibration(n), FPS)
    sf = fw.shot("s1")
    assert sf.recovery_gap_m is not None and sf.recovery_gap_m < 1.0
    assert sf.recovery_ms is not None and sf.recovery_ms < 2000
    assert "回到中心" in sf.summary


def test_left_right_swaps_are_undone():
    feet = np.array([[[0.0, 0.0], [10.0, 0.0]]] * 6)
    feet[2] = feet[2][::-1]
    feet[3] = feet[3][::-1]
    fixed = _fix_swaps(feet)
    assert np.allclose(fixed[:, 0], [0.0, 0.0]) and np.allclose(fixed[:, 1], [10.0, 0.0])
    n = 120
    me = Script(n, [2.7, 3.0], [3.4, 3.0])
    me.move(1, 30, 40, [4.0, 3.4])
    pl = tracks({"P1": me, "P2": opponent(n)}, n, swap_frames=set(range(50, 54)))
    res = analyze_footwork(events([], n), pl, calibration(n), FPS)
    assert [s.foot for s in res.steps if s.player_id == "P1"] == ["right"]


def test_nothing_without_calibration():
    n = 60
    me = Script(n, [2.7, 3.0], [3.4, 3.0])
    res = analyze_footwork(events([], n), tracks({"P1": me}, n), None, FPS)
    assert res.contacts == [] and res.steps == []
    assert any("标定" in x for x in res.caveats)


def test_stats_and_moments_use_footwork():
    """A shot whose hitter did not recover before the opponent's attack is flagged as a footwork moment."""
    from rallyiq.report import compute_match_stats

    n = 240
    me = Script(n, [0.9, 0.8], [1.6, 0.8])  # P1 stays stuck in the rear-left corner
    sh = [shot("s1", 20, "P2"), shot("s2", 60, "P1"), shot("s3", 100, "P2"), shot("s4", 140, "P1")]
    sh[1] = sh[1].model_copy(update={"shot_type": "drive"})
    sh[2] = sh[2].model_copy(update={"shot_type": "smash"})
    ev = events(sh, n)
    opp = opponent(n)
    opp.move(1, 66, 78, [5.0, 8.4])  # the opponent had to run to reach P1's drive
    opp.move(0, 80, 92, [4.4, 8.4])
    meta = c.VideoMeta(video_id="v", filename="x.mp4", fps=FPS, frame_count=n, width=1920, height=1080,
                       duration_ms=8000)
    stats = compute_match_stats(ev, tracks({"P1": me, "P2": opp}, n), calibration(n), meta)
    keys = {m.key: m for p in stats.players for m in p.metrics}
    assert keys["P1.footwork.recovered_pct"].value == 0
    assert keys["P1.footwork.recovery_gap_m"].value > 2
    m = next(m for m in stats.moments if m.shot_id == "s2")
    assert m.issue == "slow_recovery" and m.footwork is not None and m.suggested_shot is None
    assert "回中心" in m.suggestion
