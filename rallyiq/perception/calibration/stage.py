"""Stage ``calibration``. STUB owned by the court calibration thread: replace ``run``.

Contract: ``run(ctx) -> CourtCalibration``. A human 4-corner override, when present,
is applied by the pipeline runner and this stage is not called.
"""

from ...contracts import CalibrationSegment, CourtCalibration
from ...pipeline.context import StageContext, demo_mode


def run(ctx: StageContext) -> CourtCalibration:
    meta = ctx.require_meta()
    if demo_mode(ctx):
        from ...pipeline.demo import demo_calibration

        return demo_calibration(meta)
    seg = CalibrationSegment(start_frame=0, end_frame=max(meta.frame_count - 1, 0), homography=None,
                             status="undeterminable", notes=["auto calibration not implemented yet"])
    return CourtCalibration(video_id=meta.video_id, segments=[seg])
