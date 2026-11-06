"""Stage ``events``. STUB owned by the rally/hit segmentation thread: replace ``run``.

Contract: ``run(ctx) -> EventsResult`` using ``ctx.players`` and ``ctx.shuttle``.
"""

from ..contracts import EventsResult
from ..pipeline.context import StageContext, demo_mode


def run(ctx: StageContext) -> EventsResult:
    meta = ctx.require_meta()
    if demo_mode(ctx) and (players := ctx.players) is not None and players.frames:
        from ..pipeline.demo import demo_events

        return demo_events(meta, players)
    return EventsResult(video_id=meta.video_id, rallies=[], shots=[])
