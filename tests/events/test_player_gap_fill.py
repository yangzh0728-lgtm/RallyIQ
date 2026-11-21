"""A player missing from the poses around a hit (hidden behind the other player) is bridged from the
frames before and after (``EventsConfig.player_gap_fill_ms``)."""

from __future__ import annotations

from dataclasses import replace

from rallyiq.events import Keypoint, PlayerPose, PoseFrame, segment_events
from rallyiq.events.config import EventsConfig
from rallyiq.events.metrics import match_hits
from rallyiq.events.players import PoseIndex
from rallyiq.events.synthetic import FAR, make_video


def _pose(pid: str, x: float, y: float, side: str) -> PlayerPose:
    # standing figure 100 px tall, feet at (x, y)
    kps = [Keypoint(x, y - 100 + 6 * k, 0.9) for k in range(17)]
    return PlayerPose(pid, kps, side)


def _frames(gap: range) -> list[PoseFrame]:
    out = []
    for f in range(0, 31):
        ps = [_pose("P1", 900, 800, "near")]
        if f not in gap:
            ps.append(_pose("P2", 900 + 2 * f, 400, "far"))
        out.append(PoseFrame(f, f * 33.3, ps))
    return out


def test_missing_player_is_bridged_between_sightings():
    idx = PoseIndex(_frames(range(10, 21)), EventsConfig(player_gap_fill_ms=1500.0, player_gap_fill_floor=0.2))
    prox = {p.player_id: p for p in idx.proximities(15, (930, 330))}
    assert set(prox) == {"P1", "P2"}
    p2 = prox["P2"]
    assert p2.filled and not prox["P1"].filled
    assert abs(p2.pose.keypoints[0].x - 930) < 1e-6  # halfway between the sightings at frame 9 (x=918) and 21 (942)
    assert p2.pose.side == "far"
    assert p2.score >= 0.2


def test_no_bridge_when_disabled_or_gap_too_long():
    off = PoseIndex(_frames(range(10, 21)), EventsConfig(player_gap_fill_ms=0.0))
    assert {p.player_id for p in off.proximities(15, (930, 330))} == {"P1"}
    short = PoseIndex(_frames(range(10, 21)), EventsConfig(player_gap_fill_ms=100.0))
    assert {p.player_id for p in short.proximities(15, (930, 330))} == {"P1"}
    # never seen after the gap: not invented
    tail = PoseIndex(_frames(range(10, 40)), EventsConfig(player_gap_fill_ms=1500.0))
    assert {p.player_id for p in tail.proximities(15, (930, 330))} == {"P1"}


def _hide_far_at_hits(v, half: int = 12):
    """Remove the far player's pose for ``half`` frames around each of their hits (hidden behind the near one)."""
    hidden = {f for frame, pid in v.gt_hits if pid == FAR.player_id for f in range(frame - half, frame + half + 1)}
    return [
        PoseFrame(pf.frame, pf.t_ms, [p for p in pf.players if not (p.player_id == FAR.player_id and pf.frame in hidden)])
        for pf in v.poses
    ]


def test_bridging_recovers_hits_of_a_hidden_far_player():
    # 4 seeds, 81 true hits: rules 39 -> 77 found with the right hitter, learned v2 41 -> 56.
    base = EventsConfig(hit_scorer="off")
    right = {}
    for name, cfg in (("off", replace(base, player_gap_fill_ms=0.0)), ("on", base)):
        tp = 0
        for seed in range(4):
            v = make_video(n_rallies=3, seed=seed)
            rallies = segment_events(v.shuttle, _hide_far_at_hits(v), cfg)
            pred = [(h.frame, h.hitter_id) for r in rallies for h in r.hits]
            tp += match_hits(pred, v.gt_hits).tp
        right[name] = tp
    assert right["on"] >= right["off"] + 25
