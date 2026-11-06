"""Deterministic synthetic perception results.

Used by the stage stubs when ``RALLYIQ_DEMO=1`` (or ``config["demo"]``) so the API and
web UI can be built before the real models land, and to generate the JSON examples
in ``contracts/examples``. Not a model: everything here is fake.
"""

from __future__ import annotations

import math

from ..contracts import (
    CalibrationSegment,
    CourtCalibration,
    CourtKeypoint,
    EventsResult,
    PlayerDetection,
    PlayerFrame,
    PlayerTracks,
    Rally,
    ShotEvent,
    ShuttlePoint,
    ShuttleTrack,
    VideoMeta,
)
from ..contracts.geometry import (
    COURT_CORNERS_M,
    apply_homography,
    homography_from_corners,
    invert_homography,
)

RALLY_PERIOD_S = 10.0
RALLY_LEN_S = 7.0
SHOT_INTERVAL_S = 1.0


def demo_corners(meta: VideoMeta) -> list[list[float]]:
    w, h = meta.width, meta.height
    return [[0.20 * w, 0.93 * h], [0.80 * w, 0.93 * h], [0.645 * w, 0.31 * h], [0.355 * w, 0.31 * h]]


def demo_calibration(meta: VideoMeta) -> CourtCalibration:
    corners = demo_corners(meta)
    h = homography_from_corners(corners)
    names = ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"]
    seg = CalibrationSegment(
        start_frame=0, end_frame=max(meta.frame_count - 1, 0), homography=h, image_corners=corners,
        keypoints=[CourtKeypoint(name=n, image=i, court=c) for n, i, c in zip(names, corners, COURT_CORNERS_M)],
        reprojection_error_px=0.0, method="auto", confidence=0.9, status="ok", notes=["demo data"],
    )
    return CourtCalibration(video_id=meta.video_id, segments=[seg])


def _shots(meta: VideoMeta) -> list[tuple[int, int, int]]:
    """(rally_index, shot_index, frame) for every synthetic hit."""
    out = []
    dur = meta.frame_count / meta.fps
    r = 0
    while r * RALLY_PERIOD_S + 1.0 < dur:
        start = r * RALLY_PERIOD_S + 1.0
        k = 0
        while k * SHOT_INTERVAL_S < RALLY_LEN_S and start + k * SHOT_INTERVAL_S < dur:
            out.append((r, k, int(round((start + k * SHOT_INTERVAL_S) * meta.fps))))
            k += 1
        r += 1
    return out


def _player_court(pid: str, t: float) -> list[float]:
    if pid == "P1":
        return [3.05 + 1.6 * math.sin(t * 1.3), 2.6 + 1.2 * math.sin(t * 0.9)]
    return [3.05 + 1.6 * math.sin(t * 1.1 + 1), 10.8 + 1.2 * math.sin(t * 0.7 + 2)]


def _skeleton(foot: list[float], height_px: float) -> list[list[float]]:
    x, y = foot
    s = height_px
    rel = [  # (dx, dy) as fraction of height, COCO-17 order
        (0, -0.95), (-0.02, -0.97), (0.02, -0.97), (-0.04, -0.95), (0.04, -0.95),
        (-0.12, -0.80), (0.12, -0.80), (-0.18, -0.62), (0.18, -0.62), (-0.2, -0.46), (0.24, -0.9),
        (-0.08, -0.48), (0.08, -0.48), (-0.09, -0.25), (0.09, -0.25), (-0.1, -0.02), (0.1, -0.02),
    ]
    return [[x + dx * s, y + dy * s, 0.9] for dx, dy in rel]


def demo_players(meta: VideoMeta, cal: CourtCalibration) -> PlayerTracks:
    h = cal.segments[0].homography
    inv = invert_homography(h)
    frames = []
    for f in range(meta.frame_count):
        t = f / meta.fps
        dets = []
        for track_id, (pid, side) in enumerate([("P1", "near"), ("P2", "far")], start=1):
            court = _player_court(pid, t)
            foot = apply_homography(inv, [court])[0]
            height = meta.height * (0.32 if side == "near" else 0.17)
            kps = _skeleton(foot, height)
            xs, ys = [k[0] for k in kps], [k[1] for k in kps]
            dets.append(PlayerDetection(
                player_id=pid, track_id=track_id, side=side,
                bbox=[min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5], keypoints=kps,
                foot_px=foot, foot_court=court, on_court=True, confidence=0.9,
            ))
        frames.append(PlayerFrame(frame=f, t_ms=meta.t_ms(f), players=dets))
    return PlayerTracks(video_id=meta.video_id, model="demo@0", frames=frames)


def demo_shuttle(meta: VideoMeta, players: PlayerTracks) -> ShuttleTrack:
    by_frame = {pf.frame: pf for pf in players.frames}
    shots = _shots(meta)
    points = []
    for (r, k, f0), nxt in zip(shots, shots[1:] + [None]):
        if nxt is None or nxt[0] != r:
            continue
        f1 = nxt[2]
        a = by_frame[f0].players[k % 2].keypoints[10]
        b = by_frame[f1].players[(k + 1) % 2].keypoints[10]
        for f in range(f0, f1):
            u = (f - f0) / (f1 - f0)
            x = a[0] + (b[0] - a[0]) * u
            y = a[1] + (b[1] - a[1]) * u - meta.height * 0.35 * 4 * u * (1 - u)
            interp = (f % 23) in (5, 6)  # pretend the detector missed a few frames
            points.append(ShuttlePoint(frame=f, t_ms=meta.t_ms(f), x=x, y=y,
                                       confidence=0.5 if interp else 0.85,
                                       source="interpolated" if interp else "detected"))
    return ShuttleTrack(video_id=meta.video_id, detector="demo@0", points=points)


def demo_events(meta: VideoMeta, players: PlayerTracks) -> EventsResult:
    by_frame = {pf.frame: pf for pf in players.frames}
    shots: list[ShotEvent] = []
    rallies: dict[int, list[ShotEvent]] = {}
    for r, k, f in _shots(meta):
        hitter = by_frame[f].players[k % 2]
        s = ShotEvent(
            shot_id=f"s{len(shots) + 1:04d}", rally_id=f"r{r + 1:03d}", frame=f, t_ms=meta.t_ms(f),
            hitter=hitter.player_id, hitter_confidence=0.8, shuttle_px=hitter.keypoints[10][:2],
            hitter_court=hitter.foot_court, confidence=0.7, status="needs_review",
            reasons=["demo: trajectory turn", "demo: wrist within 40 px"],
        )
        shots.append(s)
        rallies.setdefault(r, []).append(s)
    out = []
    for r, ss in rallies.items():
        end = min(ss[-1].frame + int(meta.fps), meta.frame_count - 1)
        out.append(Rally(
            rally_id=f"r{r + 1:03d}", start_frame=max(ss[0].frame - int(meta.fps // 2), 0), end_frame=end,
            start_ms=meta.t_ms(max(ss[0].frame - int(meta.fps // 2), 0)), end_ms=meta.t_ms(end),
            shot_ids=[s.shot_id for s in ss], end_reason="shuttle_lost", confidence=0.7, status="needs_review",
        ))
    return EventsResult(video_id=meta.video_id, rallies=out, shots=shots)
