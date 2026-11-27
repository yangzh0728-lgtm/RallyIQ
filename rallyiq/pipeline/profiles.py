"""Pipeline config profiles.

``own``: sam's own (amateur, fixed phone/DJI camera) footage. Turns on the own-footage stroke residual
(``strokes.residual = "auto"``, train_v4): leave-one-video-out on 138 trusted own stroke labels 65.9% -> 76.1%
(train_v8, CI over videos [+5, +16] points) and 44% -> 56% on 78 held-out fresh hits, but it costs 5-8 points on broadcast
ShuttleSet clips, so it is never on by default.

``broadcast``: TV broadcast footage. Uses the rally sequence hit model (``events.hit_model = "seq"``, train_v10): grouped
5-fold by match over 43 ShuttleSet matches, production code path, hit F1 0.787 -> 0.811 (per-match CI [+0.012, +0.037]).
On own footage that model is 0.07 worse, so ``own`` and unset keep the rules hit detector. Unset: the defaults.

Selection: ``RALLYIQ_PIPELINE_PROFILE=own`` (demo API worker, ``python -m rallyiq.pipeline worker``), or
``--profile own`` on the pipeline CLI. ``RALLYIQ_PIPELINE_CONFIG=/path.json`` merges a JSON config on top.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "broadcast": {"events": {"hit_model": "seq"}},
    "own": {"strokes": {"residual": "auto"}},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def profile_config(name: str | None) -> dict[str, Any]:
    if not name:
        return {}
    if name not in PROFILES:
        raise ValueError(f"unknown pipeline profile {name!r} (known: {', '.join(sorted(PROFILES))})")
    return copy.deepcopy(PROFILES[name])


def config_from_env(base: dict[str, Any] | None = None, profile: str | None = None) -> dict[str, Any]:
    """``base`` + the named (or ``RALLYIQ_PIPELINE_PROFILE``) profile + ``RALLYIQ_PIPELINE_CONFIG`` JSON."""
    cfg = _merge(dict(base or {}), profile_config(profile or os.environ.get("RALLYIQ_PIPELINE_PROFILE") or None))
    path = os.environ.get("RALLYIQ_PIPELINE_CONFIG")
    if path:
        cfg = _merge(cfg, json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
    return cfg
