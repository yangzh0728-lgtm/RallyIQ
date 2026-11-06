"""Pipeline stage ``players``: ``run(ctx) -> PlayerTracks``.

Runs pose estimation + singles tracking on the video, using the court
calibration from the previous stage when it is available.

Configuration (``ctx.config`` or environment):

* ``pose_model`` / ``RALLYIQ_POSE_MODEL`` - path to the RTMO ONNX file.
  Without it the stage returns empty tracks so the rest of the pipeline still
  runs, and the ``model`` field stays ``None`` to show nothing was detected.
* ``players_stride`` - run pose on every n-th frame (default 1).
* ``players_far_boost`` - second pass for small far-side players.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ...contracts import PlayerTracks
from ...contracts.models import PlayerDetection, PlayerFrame
from ...pipeline.context import StageContext, demo_mode
from .models import PlayerFrameResult, PlayerState

log = logging.getLogger(__name__)

MODEL_NAME = "rtmo-onnx"


def _r(a: np.ndarray | None, nd: int) -> list[float] | None:
    return None if a is None else [round(float(v), nd) for v in np.asarray(a).ravel()]


def to_contract(p: PlayerState) -> PlayerDetection:
    return PlayerDetection(
        player_id=p.player_id,
        track_id=p.track_id,
        side=p.side,
        bbox=_r(p.bbox, 1),
        keypoints=[[round(float(x), 1), round(float(y), 1), round(float(min(max(c, 0.0), 1.0)), 3)]
                   for x, y, c in p.keypoints],
        foot_px=_r(p.foot_px, 1),
        foot_court=_r(p.foot_court, 3),
        on_court=p.on_court,
        confidence=round(float(min(max(p.confidence, 0.0), 1.0)), 3),
        interpolated=p.interpolated,
    )


def side_switches(results: list[PlayerFrameResult]) -> list[int]:
    """Frames where P1 is first observed on the other half than before."""
    out: list[int] = []
    last = None
    for r in results:
        for p in r.players:
            if p.player_id == "P1" and p.side is not None and not p.interpolated:
                if last is not None and p.side != last:
                    out.append(r.frame)
                last = p.side
    return out


def build_tracks(video_id: str, results: list[PlayerFrameResult], model: str | None) -> PlayerTracks:
    frames = [
        PlayerFrame(frame=r.frame, t_ms=max(int(round(r.t_ms)), 0), players=[to_contract(p) for p in r.players])
        for r in results
    ]
    return PlayerTracks(video_id=video_id, model=model, frames=frames, side_switches=side_switches(results))


def run(ctx: StageContext) -> PlayerTracks:
    meta = ctx.require_meta()
    cal = ctx.calibration
    if demo_mode(ctx) and cal is not None and cal.segments and cal.segments[0].homography:
        from ...pipeline.demo import demo_players

        return demo_players(meta, cal)

    model_path = ctx.config.get("pose_model") or os.environ.get("RALLYIQ_POSE_MODEL")
    if not model_path or not os.path.exists(model_path):
        log.warning("players: no pose model configured (RALLYIQ_POSE_MODEL); returning empty tracks")
        return PlayerTracks(video_id=meta.video_id, model=None, frames=[])

    from .pose import RTMOOnnx
    from .runner import PlayerTrackingConfig, iter_video, track_players

    stride = int(ctx.config.get("players_stride", 1))
    cfg = PlayerTrackingConfig(stride=stride)
    cfg.tracker.fps = meta.fps
    est = RTMOOnnx(model_path, tile_far_half=bool(ctx.config.get("players_far_boost", False)))

    def homography_for(frame: int):
        h = cal.homography_for(frame) if cal is not None else None
        return np.asarray(h, dtype=np.float64) if h is not None else None

    total = max(meta.frame_count, 1)

    def frames():
        for i, t_ms, img in iter_video(str(ctx.video_path), stride):
            if i % 30 == 0:
                ctx.report(min(i / total, 0.99))
            # Prefer the probe's timeline so all stages agree on t_ms.
            yield i, float(meta.t_ms(i)) if not meta.variable_frame_rate else t_ms, img

    results = track_players(frames(), est, homography_for, cfg)
    ctx.report(1.0)
    return build_tracks(meta.video_id, results, f"{MODEL_NAME}:{os.path.basename(model_path)}")
