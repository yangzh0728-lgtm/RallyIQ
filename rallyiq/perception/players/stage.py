"""Pipeline stage ``players``: ``run(ctx) -> PlayerTracks``.

Runs pose estimation + player tracking (singles or doubles) on the video, using the court
calibration from the previous stage when it is available.

Configuration (``ctx.config`` or environment):

* ``pose_model`` / ``RALLYIQ_POSE_MODEL`` - path to the RTMO ONNX file (default: ``weights/pose.onnx``,
  see ``pose.find_pose_model``).
  Without it the stage returns empty tracks so the rest of the pipeline still
  runs, and the ``model`` field stays ``None`` to show nothing was detected.
* ``players_stride`` - run pose on every n-th frame (default 1).
* ``players_far_boost`` - second pose pass on an upscaled crop of the far half
  of the court: ``auto`` (default, only when the far player is small),
  ``on``/``true`` or ``off``/``false``.
* ``match_format`` / ``RALLYIQ_MATCH_FORMAT`` - ``auto`` (default), ``singles``
  or ``doubles``. ``auto`` decides from the first seconds of the video.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ...contracts import PlayerTracks
from ...contracts.models import PlayerDetection, PlayerFrame, team_of
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
        team=p.team or team_of(p.player_id),
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
    """Frames where the teams really changed ends (sustained, spaced like games).

    ``track_players`` already worked them out while fixing the labels; for other
    result lists they are recomputed without touching the labels.
    """
    known = getattr(results, "side_switches", None)
    if known is not None:
        return list(known)
    from .sides import stabilise_sides

    return stabilise_sides(list(results), apply=False)


def build_tracks(video_id: str, results: list[PlayerFrameResult], model: str | None) -> PlayerTracks:
    frames = [
        PlayerFrame(frame=r.frame, t_ms=max(int(round(r.t_ms)), 0), players=[to_contract(p) for p in r.players])
        for r in results
    ]
    fmt = getattr(results, "match_format", "singles")
    conf = getattr(results, "format_confidence", None)
    return PlayerTracks(
        video_id=video_id, model=model, match_format=fmt,
        format_confidence=None if conf is None else round(float(conf), 3),
        frames=frames, side_switches=side_switches(results),
    )


def match_format_setting(ctx: StageContext) -> str:
    """Per-video setting from the UI, else config / environment, else ``auto``."""
    from ...contracts import MatchFormatSetting
    from ...pipeline.runner import MATCH_FORMAT

    chosen = ctx.store.load_artifact(ctx.video_id, MATCH_FORMAT, MatchFormatSetting)
    if chosen is not None and chosen.match_format != "auto":
        return chosen.match_format
    fmt = str(ctx.config.get("match_format") or os.environ.get("RALLYIQ_MATCH_FORMAT") or "auto").lower()
    return fmt if fmt in ("auto", "singles", "doubles") else "auto"


def far_boost_setting(ctx: StageContext) -> str:
    v = ctx.config.get("players_far_boost", "auto")
    if v is None or v == "auto":
        return "auto"
    if isinstance(v, str):
        return "off" if v.lower() in ("off", "false", "0", "no") else "on"
    return "on" if v else "off"


def _rechoose_court(ctx: StageContext, cal, est):
    """Switch the target court to the one the players stand on when calibration
    picked a neighbour (see ``court_choice``). Saves the corrected calibration so
    the shuttle court lock and the overlay use it too."""
    from ..court_choice import rechoose

    def detect_frames(indices):
        import cv2

        cap = cv2.VideoCapture(str(ctx.video_path))
        out = []
        try:
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, img = cap.read()
                if ok:
                    out.append(est(img))
        finally:
            cap.release()
        return out

    new = rechoose(cal, detect_frames)
    if new is None:
        return cal
    for seg in new.segments:
        if seg.notes and seg.notes[-1].startswith(("target court re-chosen", "calibration contradicted")):
            log.warning("players: %s (frames %d-%d)", seg.notes[-1], seg.start_frame, seg.end_frame)
    ctx.store.save_artifact(ctx.video_id, "calibration", new)
    return new


def run(ctx: StageContext) -> PlayerTracks:
    meta = ctx.require_meta()
    cal = ctx.calibration
    if demo_mode(ctx) and cal is not None and cal.segments and cal.segments[0].homography:
        from ...pipeline.demo import demo_players

        fmt = match_format_setting(ctx)
        return demo_players(meta, cal, "doubles" if fmt == "doubles" else "singles")

    from .pose import find_pose_model

    model_path = find_pose_model(ctx.config.get("pose_model"))
    if not model_path:
        log.warning("players: no pose model found (scripts/fetch_weights.sh or RALLYIQ_POSE_MODEL); "
                    "returning empty tracks")
        return PlayerTracks(video_id=meta.video_id, model=None, frames=[])

    from .pose import RTMOOnnx
    from .runner import PlayerTrackingConfig, iter_video, track_players

    stride = int(ctx.config.get("players_stride", 1))
    cfg = PlayerTrackingConfig(stride=stride, match_format=match_format_setting(ctx), far_boost=far_boost_setting(ctx))
    cfg.tracker.fps = meta.fps
    est = RTMOOnnx(model_path)

    if cal is not None and str(ctx.config.get("court_choice", "auto")).lower() != "off":
        cal = _rechoose_court(ctx, cal, est)

    def homography_for(frame: int):
        seg = cal.segment_for(frame) if cal is not None else None
        # An undeterminable court is not used to filter people (it may be the wrong lines).
        h = seg.homography if seg is not None and seg.status != "undeterminable" else None
        return np.asarray(h, dtype=np.float64) if h is not None else None

    total = max(meta.frame_count, 1)

    def frames():
        for i, t_ms, img in iter_video(str(ctx.video_path), stride):
            if i % 30 == 0:
                ctx.report(min(i / total, 0.99))
            # Prefer the probe's timeline so all stages agree on t_ms.
            yield i, float(meta.t_ms(i)) if not meta.variable_frame_rate else t_ms, img

    from ..target_court import SegmentRegions

    regions = SegmentRegions(cal)
    results = track_players(frames(), est, homography_for, cfg, regions.for_frame if regions else None)
    ctx.report(1.0)
    return build_tracks(meta.video_id, results, f"{MODEL_NAME}:{os.path.basename(model_path)}")
