"""Stage ``overlay``: merge calibration, players and shuttle into per-frame draw data."""

from __future__ import annotations

from ..contracts import (
    CourtLines,
    OverlayData,
    OverlayFrame,
    OverlayPlayer,
    OverlayShuttle,
)
from ..contracts.geometry import project_court_lines
from .context import StageContext

COCO17_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
]


def run(ctx: StageContext) -> OverlayData:
    meta = ctx.require_meta()
    court: list[CourtLines] = []
    if (cal := ctx.calibration) is not None:
        for seg in cal.segments:
            if seg.homography is None:
                continue
            court.append(CourtLines(start_frame=seg.start_frame, end_frame=seg.end_frame,
                                    lines=project_court_lines(seg.homography), status=seg.status,
                                    image_corners=seg.image_corners, other_courts=seg.other_courts))

    frames: dict[int, OverlayFrame] = {}

    def at(frame: int, t_ms: int) -> OverlayFrame:
        if frame not in frames:
            frames[frame] = OverlayFrame(frame=frame, t_ms=t_ms)
        return frames[frame]

    if (players := ctx.players) is not None:
        for pf in players.frames:
            if not pf.players:
                continue
            at(pf.frame, pf.t_ms).players = [
                OverlayPlayer(player_id=p.player_id, side=p.side, bbox=p.bbox, keypoints=p.keypoints,
                              foot_court=p.foot_court, interpolated=p.interpolated)
                for p in pf.players
            ]
    if (shuttle := ctx.shuttle) is not None:
        for sp in shuttle.points:
            at(sp.frame, sp.t_ms).shuttle = OverlayShuttle(x=sp.x, y=sp.y, source=sp.source,
                                                           confidence=sp.confidence)

    return OverlayData(video=meta, court=court, skeleton=COCO17_SKELETON,
                       frames=[frames[k] for k in sorted(frames)])


