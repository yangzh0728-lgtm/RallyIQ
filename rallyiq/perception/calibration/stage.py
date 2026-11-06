"""Stage ``calibration``: automatic court calibration for a whole video.

Contract: ``run(ctx) -> CourtCalibration``. A human 4-corner override, when present,
is applied by the pipeline runner and this stage is not called.

Config (``ctx.config["calibration"]``, all optional): ``check_every_ms``,
``move_frac``, ``drop_ratio``, ``cut_correlation`` (see ``TrackerConfig``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts import CalibrationSegment, CourtCalibration, CourtKeypoint
from ...pipeline.context import StageContext, demo_mode
from . import court_model as cm

if TYPE_CHECKING:
    from .calibrate import FrameCalibration


def _keypoints(cal: "FrameCalibration") -> list[CourtKeypoint]:
    xy = {k.name: k.xy for k in cm.keypoints()}
    return [
        CourtKeypoint(name=name, image=img, court=list(xy[name]), used_for_fit=True)
        for name, img in cal.image_keypoints().items()
    ]


def to_contract_segment(start: int, end: int, cal: "FrameCalibration", notes: list[str] | None = None) -> CalibrationSegment:
    notes = list(notes or [])
    m = cal.metrics
    if cal.reason:
        notes.append(cal.reason)
    notes.append(
        f"{cal.version}: inliers={m.inlier_ratio:.2f} verified_keypoints={m.verified_keypoints}/{m.visible_keypoints} "
        f"ambiguity={m.ambiguity:.2f} court_in_frame={m.visible_fraction:.2f}"
        + (f" other_courts={m.other_courts}" if m.other_courts else "")
    )
    if cal.method == "tracked":
        notes.append("refined from the previous segment")
    corners = None
    if cal.image_from_court is not None:
        corners = [[float(x), float(y)] for x, y in cal.to_image(list(cm.CORNERS.values()))]
        notes.append("mirroring cannot be told from court lines (the court is symmetric); mirrored is not checked")
    extra = {}
    if "other_courts" in CalibrationSegment.model_fields:  # neighbouring courts (hall footage)
        extra["other_courts"] = cal.other_court_corners()
    residual = m.residual_px if m.residual_px == m.residual_px else None  # NaN -> None
    return CalibrationSegment(
        start_frame=start,
        end_frame=max(start, end),
        homography=None if cal.court_from_image is None else cal.court_from_image.tolist(),
        image_corners=corners,
        keypoints=_keypoints(cal),
        reprojection_error_px=residual,
        method="manual" if cal.method == "manual" else "auto",
        mirrored=False,
        confidence=float(min(1.0, max(0.0, cal.confidence))),
        status=cal.status,
        notes=notes,
        **extra,
    )


def run(ctx: StageContext) -> CourtCalibration:
    meta = ctx.require_meta()
    if demo_mode(ctx):
        from ...pipeline.demo import demo_calibration

        return demo_calibration(meta)

    from .tracker import TrackerConfig, calibrate_video

    cfg = TrackerConfig(**(ctx.config.get("calibration") or {}))
    segs = calibrate_video(str(ctx.video_path), cfg, frame_count=meta.frame_count, report=ctx.report)
    if not segs:
        seg = CalibrationSegment(
            start_frame=0, end_frame=max(meta.frame_count - 1, 0), homography=None,
            status="undeterminable", notes=["no frames could be read"],
        )
        return CourtCalibration(video_id=meta.video_id, segments=[seg])
    out = []
    for s in segs:
        notes = [f"segment start: {s.reason}"] + s.notes
        out.append(to_contract_segment(s.start_frame, s.end_frame, s.calibration, notes))
    return CourtCalibration(video_id=meta.video_id, segments=out)
