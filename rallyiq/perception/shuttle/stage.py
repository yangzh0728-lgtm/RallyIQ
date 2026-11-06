"""Stage ``shuttle``. STUB owned by the shuttle tracking thread: replace ``run``.

Contract: ``run(ctx) -> ShuttleTrack``. Keep raw detections; interpolated points only fill gaps.
"""

from ...contracts import ShuttleTrack
from ...pipeline.context import StageContext, demo_mode


def run(ctx: StageContext) -> ShuttleTrack:
    meta = ctx.require_meta()
    if demo_mode(ctx) and (players := ctx.players) is not None and players.frames:
        from ...pipeline.demo import demo_shuttle

        return demo_shuttle(meta, players)
    return ShuttleTrack(video_id=meta.video_id, points=[])
