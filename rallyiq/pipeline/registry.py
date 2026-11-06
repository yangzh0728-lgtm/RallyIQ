"""Stage registry. Each stage is ``run(ctx: StageContext) -> <contract model>``.

Entries are import strings so heavy model dependencies are only imported when the
stage actually runs (the API process never imports torch).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from pydantic import BaseModel

from .context import StageContext

STAGE_ENTRYPOINTS: dict[str, str] = {
    "probe": "rallyiq.pipeline.probe:run",
    "calibration": "rallyiq.perception.calibration.stage:run",
    "players": "rallyiq.perception.players.stage:run",
    "shuttle": "rallyiq.perception.shuttle.stage:run",
    "events": "rallyiq.events.stage:run",
    "strokes": "rallyiq.strokes.stage:run",
    "overlay": "rallyiq.pipeline.overlay:run",
}


def load_stage(name: str) -> Callable[[StageContext], BaseModel]:
    module, func = STAGE_ENTRYPOINTS[name].split(":")
    return getattr(importlib.import_module(module), func)
