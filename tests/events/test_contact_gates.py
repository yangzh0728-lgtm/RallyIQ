"""Contact gates: a shuttle that arrives slowly at the hitter's feet is a pickup or a bounce, not a stroke."""

from dataclasses import replace

from rallyiq.events.config import EventsConfig
from rallyiq.events.hits import R_FLOOR_CONTACT, R_OUT_OF_REACH, Candidate, _contact_gate, contact_height, reach_torso
from rallyiq.events.players import PoseIndex, Proximity
from rallyiq.events.trajectory import Point, Track
from rallyiq.events.types import Keypoint, PlayerPose, PoseFrame

CFG = EventsConfig()


def _player() -> PlayerPose:
    """Standing player, head at y=100, ankles at y=300 (200 px tall), wrists at hip height (y=200)."""
    kp = [Keypoint(500, 100, 0.9)] * 5  # nose, eyes, ears
    kp += [Keypoint(480, 140, 0.9), Keypoint(520, 140, 0.9)]  # shoulders
    kp += [Keypoint(470, 170, 0.9), Keypoint(530, 170, 0.9)]  # elbows
    kp += [Keypoint(465, 200, 0.9), Keypoint(535, 200, 0.9)]  # wrists
    kp += [Keypoint(485, 200, 0.9), Keypoint(515, 200, 0.9)]  # hips
    kp += [Keypoint(485, 250, 0.9), Keypoint(515, 250, 0.9)]  # knees
    kp += [Keypoint(485, 300, 0.9), Keypoint(515, 300, 0.9)]  # ankles
    return PlayerPose("P1", kp, side="near")


def _setup(y_at_contact: float, speed_px_per_frame: float):
    """Shuttle moving straight up at constant speed, contact at frame 10."""
    pts = [Point(f, f * 1000 / 30, 540.0, y_at_contact - (f - 10) * speed_px_per_frame, 0.9, False, None)
           for f in range(21)]
    track = Track(pts)
    p = _player()
    poses = PoseIndex([PoseFrame(f, f * 1000 / 30, [p]) for f in range(21)], CFG)
    c = Candidate(10, 0.0, 1.0, 1.0, 1.0, prox=[Proximity("P1", 0.8, 0.1, True, p)], t_ms=pts[10].t_ms)
    return c, track, poses


def test_contact_height_and_reach_are_in_body_units():
    p = _player()
    assert abs(contact_height(p, (500, 300), 0.3)) < 1e-9
    assert abs(contact_height(p, (500, 100), 0.3) - 1.0) < 1e-9
    assert abs(reach_torso(p, (535, 260), 0.3) - 1.0) < 1e-9  # 60 px from the wrist, torso 60 px


def test_slow_shuttle_at_the_feet_is_rejected_as_floor_contact():
    c, track, poses = _setup(y_at_contact=295.0, speed_px_per_frame=1.0)  # 30 px/s, ~0.15 H/s
    assert _contact_gate(c, track, poses, CFG)
    assert R_FLOOR_CONTACT in c.reasons


def test_fast_low_shuttle_is_a_real_low_return():
    c, track, poses = _setup(y_at_contact=295.0, speed_px_per_frame=20.0)  # 600 px/s, 3 H/s: a lift
    assert not _contact_gate(c, track, poses, CFG)
    assert R_FLOOR_CONTACT not in c.reasons


def test_slow_shuttle_at_waist_height_is_not_floor_contact():
    c, track, poses = _setup(y_at_contact=200.0, speed_px_per_frame=1.0)
    assert not _contact_gate(c, track, poses, CFG)


def test_gates_can_be_switched():
    c, track, poses = _setup(y_at_contact=295.0, speed_px_per_frame=1.0)
    assert not _contact_gate(c, track, poses, replace(CFG, floor_contact_h=0.0))
    # Reach gate is off by default; on, a shuttle 4 torso lengths from the wrist is out of reach.
    c, track, poses = _setup(y_at_contact=200.0 - 240.0, speed_px_per_frame=20.0)
    assert not _contact_gate(c, track, poses, CFG)
    assert _contact_gate(c, track, poses, replace(CFG, max_reach_torso=2.5))
    assert R_OUT_OF_REACH in c.reasons
