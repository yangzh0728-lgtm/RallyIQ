"""Write a ShuttleTrack in the evaluation benchmark's prediction format.

Only observed detections are reported as visible. Interpolated frames are left
out entirely (neither visible nor invisible), so a fill can never be scored as a
detection. Frames in gaps are reported as ``visible: false``.
"""

from __future__ import annotations

from typing import Any

from ...contracts import ShuttleTrack, VideoMeta


def to_benchmark_prediction(track: ShuttleTrack, meta: VideoMeta, clip_id: str, start_ms: int = 0) -> dict[str, Any]:
    def t(frame: int) -> int:
        return int(round(start_ms + frame * 1000.0 / meta.fps))

    shuttle: list[dict[str, Any]] = [
        {"t_ms": t(p.frame), "visible": True, "xy": [p.x, p.y], "score": p.confidence}
        for p in track.points if p.source == "detected"
    ]
    for g in track.gaps:
        shuttle.extend({"t_ms": t(f), "visible": False, "xy": None} for f in range(g.start_frame, g.end_frame + 1))
    shuttle.sort(key=lambda s: s["t_ms"])
    return {
        "schema_version": 1,
        "clip_id": clip_id,
        "kind": "prediction",
        "status": "draft",
        "video": {"width": meta.width, "height": meta.height, "fps": meta.fps},
        "range_ms": [int(start_ms), int(start_ms + max(meta.duration_ms, 1))],
        "court": [],
        "shuttle": shuttle,
        "hits": [],
        "rallies": [],
        "detector": track.detector,
    }
