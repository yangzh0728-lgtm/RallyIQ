import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("manual_court", ROOT / "scripts" / "manual_court.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _court_to_px(shift_x: float) -> np.ndarray:
    # A plausible court seen from behind the near baseline in a 3840x2160 frame.
    src = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], float)
    dst = np.array([[900, 2000], [2900, 2000], [2300, 1100], [1500, 1100]], float) + [shift_x, 0]
    import cv2
    H, _ = cv2.findHomography(src, dst)
    return H


def test_labels_become_rescaled_segments(tmp_path):
    m = _mod()
    labels = tmp_path / "court.json"
    labels.write_text(json.dumps({"frames": {
        "300": {"image_from_court": _court_to_px(0).tolist()},
        "900": {"image_from_court": _court_to_px(40).tolist()},
    }}))
    size, poses = m.load_labels(labels)
    cal = m.build("vid_x", size, poses, (1920, 1080), 1200)

    segs = cal.segments
    assert [(s.start_frame, s.end_frame) for s in segs] == [(0, 600), (601, 1199)]
    assert all(s.method == "manual" and s.status == "ok" for s in segs)
    # corners are in the video's own pixels (half of the labelled 4K frame)
    np.testing.assert_allclose(segs[0].image_corners[0], [450, 1000], atol=1e-6)
    np.testing.assert_allclose(segs[1].image_corners[0], [470, 1000], atol=1e-6)
    # the stored homography goes image px -> court metres
    Hi = np.array(cal.homography_for(100))
    q = Hi @ [1150, 550, 1.0]
    np.testing.assert_allclose(q[:2] / q[2], [6.1, 13.4], atol=1e-6)


def test_single_pose_file_covers_whole_clip(tmp_path):
    m = _mod()
    labels = tmp_path / "court.json"
    labels.write_text(json.dumps({"image_size": [1920, 1080],
                                  "image_from_court": _court_to_px(0).tolist()}))
    size, poses = m.load_labels(labels)
    cal = m.build("vid_x", size, poses, (1920, 1080), 500)
    assert [(s.start_frame, s.end_frame) for s in cal.segments] == [(0, 499)]
    np.testing.assert_allclose(cal.segments[0].image_corners[2], [2300, 1100], atol=1e-6)
