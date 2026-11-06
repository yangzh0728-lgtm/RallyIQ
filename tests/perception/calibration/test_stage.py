from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rallyiq.contracts import CourtCalibration, VideoMeta  # noqa: E402
from rallyiq.contracts.geometry import apply_homography  # noqa: E402
from rallyiq.perception.calibration import court_model as cm  # noqa: E402
from rallyiq.perception.calibration.stage import run  # noqa: E402
from rallyiq.pipeline.context import StageContext  # noqa: E402

from .synthetic import Camera, Scene, render  # noqa: E402


def test_stage_produces_contract_segments(tmp_path: Path):
    cam = Camera((cm.COURT_WIDTH / 2, -6, 6), size=(640, 360), focal=500)
    img, H_true = render(Scene(cam))
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, cam.size)
    for _ in range(45):
        vw.write(img)
    vw.release()
    meta = VideoMeta(video_id="v1", filename="clip.avi", fps=30, frame_count=45, width=640, height=360, duration_ms=1500)
    ctx = StageContext(video_id="v1", video_path=path, work_dir=tmp_path, store=None, meta=meta)
    out = run(ctx)
    assert isinstance(out, CourtCalibration)
    CourtCalibration.model_validate_json(out.model_dump_json())
    assert len(out.segments) == 1
    seg = out.segments[0]
    assert seg.start_frame == 0 and seg.end_frame == 44
    assert str(getattr(seg.status, "value", seg.status)) != "undeterminable" and seg.homography is not None
    # homography is image px -> court metres; corners come back to the court corners (up to symmetry).
    back = np.array(apply_homography(seg.homography, seg.image_corners))
    assert np.allclose(back, np.array(list(cm.CORNERS.values())), atol=0.05)
    assert len(seg.keypoints) >= 10
