"""What every pipeline stage receives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import CourtCalibration, EventsResult, PlayerTracks, ShuttleTrack, VideoMeta
from ..store import Store


@dataclass
class StageContext:
    """Passed to ``run(ctx)`` of every stage.

    Upstream artifacts are loaded lazily: ``ctx.calibration`` / ``ctx.players`` /
    ``ctx.shuttle`` / ``ctx.events`` return the saved model or ``None``.
    Call ``ctx.report(0.0..1.0)`` to publish progress for the web UI.
    """

    video_id: str
    video_path: Path
    work_dir: Path
    store: Store
    meta: VideoMeta | None = None
    config: dict[str, Any] = field(default_factory=dict)
    report: Callable[[float], None] = lambda p: None

    def _load(self, name, model):
        return self.store.load_artifact(self.video_id, name, model)

    @property
    def calibration(self) -> CourtCalibration | None:
        return self._load("calibration", CourtCalibration)

    @property
    def players(self) -> PlayerTracks | None:
        return self._load("players", PlayerTracks)

    @property
    def shuttle(self) -> ShuttleTrack | None:
        return self._load("shuttle", ShuttleTrack)

    @property
    def events(self) -> EventsResult | None:
        return self._load("events", EventsResult)

    def require_meta(self) -> VideoMeta:
        if self.meta is None:
            raise RuntimeError("video metadata missing: the probe stage must run first")
        return self.meta


def demo_mode(ctx: StageContext) -> bool:
    """True when stages should emit synthetic data (``RALLYIQ_DEMO=1`` or ``config['demo']``)."""
    import os

    return bool(ctx.config.get("demo")) or os.environ.get("RALLYIQ_DEMO") == "1"
