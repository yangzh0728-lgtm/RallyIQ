"""Stage ``events``: rule-based rally segmentation and hit candidates.

Contract: ``run(ctx) -> EventsResult`` using ``ctx.players`` and ``ctx.shuttle``.
Rule thresholds can be overridden with ``ctx.config["events"] = {name: value}``
(field names of :class:`~rallyiq.events.config.EventsConfig`).
"""

from __future__ import annotations

from dataclasses import fields, replace

from ..contracts import EventsResult
from ..pipeline.context import StageContext, demo_mode
from .adapters import events_to_contract, poses_from_contract, shuttle_from_contract
from .config import EventsConfig
from .pipeline import segment_events


def _config(ctx: StageContext) -> EventsConfig:
    overrides = ctx.config.get("events") or {}
    known = {f.name for f in fields(EventsConfig)}
    return replace(EventsConfig(), **{k: v for k, v in overrides.items() if k in known})


def run(ctx: StageContext) -> EventsResult:
    meta = ctx.require_meta()
    shuttle = ctx.shuttle
    players = ctx.players
    if shuttle is None or not shuttle.points:
        if demo_mode(ctx) and players is not None and players.frames:
            from ..pipeline.demo import demo_events

            return demo_events(meta, players)
        return EventsResult(video_id=meta.video_id, rallies=[], shots=[])

    ctx.report(0.1)
    rallies = segment_events(
        shuttle_from_contract(shuttle),
        poses_from_contract(players),
        _config(ctx),
        video_end_ms=float(meta.duration_ms),
    )
    ctx.report(0.9)
    return events_to_contract(meta.video_id, rallies)
