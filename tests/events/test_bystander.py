"""A person who stands still outside the lines all video (umpire, spectator at the net post) is never a hitter."""

import math

from rallyiq.contracts import PlayerDetection, PlayerFrame, PlayerTracks
from rallyiq.events.adapters import bystander_ids, poses_from_contract

KP = [[0.0, 0.0, 0.9]] * 17


def _det(pid, x, y, side):
    return PlayerDetection(player_id=pid, track_id=int(pid[1]), side=side, bbox=[0, 0, 10, 20], keypoints=KP,
                           foot_court=[x, y], on_court=True, confidence=0.9,
                           team="A" if pid in ("P1", "P3") else "B")


def _tracks(n=600, spectator=True, fmt="doubles"):
    frames = []
    for f in range(n):
        a = f / 30.0
        ps = [_det("P1", 3 + 2 * math.sin(a), 3 + 1.5 * math.cos(a), "near"),
              _det("P2", 3 + 2 * math.cos(a), 11 + 1.5 * math.sin(a), "far")]
        if fmt == "doubles":
            ps.append(_det("P4", 2 + math.sin(a), 10 + math.cos(a), "far"))
            # P3: standing at the left net post, half a metre outside the sideline, barely moving
            x, y = (-0.8 + 0.05 * math.sin(a), 6.5 + 0.1 * math.cos(a)) if spectator else (3 + math.cos(a), 4 + math.sin(a))
            ps.append(_det("P3", x, y, "near"))
        frames.append(PlayerFrame(frame=f, t_ms=round(f * 1000 / 30), players=ps))
    return PlayerTracks(video_id="v", match_format=fmt, frames=frames)


def test_static_track_outside_the_lines_is_a_bystander():
    assert bystander_ids(_tracks()) == {"P3"}
    assert bystander_ids(_tracks(spectator=False)) == set()
    assert bystander_ids(_tracks(fmt="singles")) == set()
    assert bystander_ids(_tracks(n=100)) == set()  # too short to tell
    assert bystander_ids(None) == set()


def test_bystander_kept_for_proximity_but_never_reported():
    poses = poses_from_contract(_tracks())
    by = [p for p in poses[0].players if p.player_id.startswith("bystander")]
    assert len(by) == 1 and by[0].side is None and by[0].team is None
    assert {p.player_id for p in poses[0].players} == {"P1", "P2", "P4", "bystanderP3"}


def test_player_who_steps_outside_now_and_then_is_not_a_bystander():
    t = _tracks(spectator=False)
    for k, fr in enumerate(t.frames):  # P3 waits outside the line between rallies, a third of the time
        if k % 3 == 0:
            p3 = next(p for p in fr.players if p.player_id == "P3")
            p3.foot_court = [-0.8, 6.5]
    assert bystander_ids(t) == set()

