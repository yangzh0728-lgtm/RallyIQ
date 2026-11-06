"""Stage ``players``. STUB owned by the player pose tracking thread: replace ``run``.

Contract: ``run(ctx) -> PlayerTracks``. ``ctx.calibration`` holds the court homography.
"""

from ...contracts import PlayerTracks
from ...pipeline.context import StageContext, demo_mode


def run(ctx: StageContext) -> PlayerTracks:
    meta = ctx.require_meta()
    if demo_mode(ctx) and (cal := ctx.calibration) is not None and cal.segments[0].homography:
        from ...pipeline.demo import demo_players

        return demo_players(meta, cal)
    return PlayerTracks(video_id=meta.video_id, frames=[])
