"""Serialise tracking results in the shared ``PlayerTracks`` JSON shape."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import PlayerFrameResult, PlayerState


def _r(a: np.ndarray | None, nd: int = 2) -> list | None:
    return None if a is None else [round(float(v), nd) for v in np.asarray(a).ravel()]


def player_to_dict(p: PlayerState) -> dict[str, Any]:
    return {
        "player_id": p.player_id,
        "track_id": p.track_id,
        "side": p.side,
        "bbox": _r(p.bbox, 1),
        "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in p.keypoints],
        "foot_px": _r(p.foot_px, 1),
        "foot_court": _r(p.foot_court, 3),
        "on_court": p.on_court,
        "confidence": round(float(p.confidence), 3),
        "interpolated": p.interpolated,
    }


def tracks_to_dict(results: list[PlayerFrameResult], video_id: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "keypoint_format": "coco17",
        "frames": [
            {"frame": r.frame, "t_ms": round(r.t_ms, 2), "players": [player_to_dict(p) for p in r.players]}
            for r in results
        ],
    }
