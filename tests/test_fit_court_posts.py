"""fit_court_posts recovers the court whose net posts are given, not a neighbouring court."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_court_posts as fcp  # noqa: E402


def _true_h() -> np.ndarray:
    # court metres -> 1280x720 px, a low camera behind the near baseline, slightly to the right
    src = np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])
    dst = np.float32([[180, 700], [1150, 690], [820, 300], [470, 305]])
    return cv2.getPerspectiveTransform(src, dst)


def _draw_court(img, H, dx=0.0):
    for (x0, y0), (x1, y1) in fcp.court_segments():
        a, _ = fcp.project(H, np.array([[x0 + dx, y0]]))
        b, _ = fcp.project(H, np.array([[x1 + dx, y1]]))
        cv2.line(img, tuple(a[0].astype(int)), tuple(b[0].astype(int)), (255, 255, 255), 3)


def test_recovers_the_posted_court_not_the_neighbour():
    H = _true_h()
    img = np.full((720, 1280, 3), (60, 140, 40), np.uint8)  # green floor
    _draw_court(img, H)
    _draw_court(img, H, dx=7.28)  # neighbouring court to the right, same paint
    posts, _ = fcp.project(H, np.array([[0.0, 6.7], [6.1, 6.7]]))
    res = fcp.fit(img, tuple(posts[0]), tuple(posts[1]))
    Hf = np.array(res["image_from_court"])
    probe = np.array([[0, 0], [6.1, 0], [3.05, 4.72], [0, 13.4], [6.1, 13.4]], float)
    got, _ = fcp.project(Hf, probe)
    want, _ = fcp.project(H, probe)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 6.0
    assert max(res["post_error_px"]) < 3.0
    assert res["residual"]["median_px"] <= 2


def test_solve_is_exact_from_posts_and_three_lines():
    H = _true_h()
    posts, _ = fcp.project(H, np.array([[0.0, 6.7], [6.1, 6.7]]))

    def img_line(kind, c):
        pts = np.array([[c, 0.0], [c, 13.4]]) if kind == "x" else np.array([[0.0, c], [6.1, c]])
        p, _ = fcp.project(H, pts)
        l = np.cross(np.r_[p[0], 1], np.r_[p[1], 1])
        return l / np.linalg.norm(l[:2])

    ids = [("x", 0.0), ("x", 6.1), ("y", 4.72)]
    Hs = fcp.solve(posts, [img_line(*i) for i in ids], ids, 1280.0)
    assert fcp.solve(posts, [img_line(*i) for i in ids[1:]], ids[1:], 1280.0) is None  # 2 lines: a homology family
    a, _ = fcp.project(Hs, np.array([[0.0, 0.0], [6.1, 13.4]]))
    b, _ = fcp.project(H, np.array([[0.0, 0.0], [6.1, 13.4]]))
    assert np.allclose(a, b, atol=0.5)


def _key(img, H, color):
    # a basketball-style key: a wide rectangle from the near baseline, in white
    a, _ = fcp.project(H, np.array([[1.0, 0.0], [5.1, 0.0], [5.1, 5.8], [1.0, 5.8]]))
    cv2.polylines(img, [a.astype(np.int32)], True, color, 3)


def _colour_case(line_bgr, floor_bgr, lines):
    H = _true_h()
    img = np.full((720, 1280, 3), floor_bgr, np.uint8)
    for (x0, y0), (x1, y1) in fcp.court_segments():  # distractor: a white court offset by half a court width
        a, _ = fcp.project(H, np.array([[x0 + 3.0, y0]]))
        b, _ = fcp.project(H, np.array([[x1 + 3.0, y1]]))
        cv2.line(img, tuple(a[0].astype(int)), tuple(b[0].astype(int)), (255, 255, 255), 3)
    _key(img, H, (255, 255, 255))
    for (x0, y0), (x1, y1) in fcp.court_segments():  # the court in use, in its own colour
        a, _ = fcp.project(H, np.array([[x0, y0]]))
        b, _ = fcp.project(H, np.array([[x1, y1]]))
        cv2.line(img, tuple(a[0].astype(int)), tuple(b[0].astype(int)), line_bgr, 3)
    posts, _ = fcp.project(H, np.array([[0.0, 6.7], [6.1, 6.7]]))
    res = fcp.fit(img, tuple(posts[0]), tuple(posts[1]), lines_color=lines)
    Hf = np.array(res["image_from_court"])
    probe = np.array([[0, 0], [6.1, 0], [3.05, 4.72], [0, 13.4], [6.1, 13.4]], float)
    got, _ = fcp.project(Hf, probe)
    want, _ = fcp.project(H, probe)
    assert np.max(np.linalg.norm(got - want, axis=1)) < 6.0


def test_blue_tape_court_beats_white_distractors():
    _colour_case((200, 90, 30), (110, 160, 200), "blue")  # blue tape on wood


def test_dark_line_court_beats_white_distractors():
    _colour_case((40, 40, 40), (120, 170, 210), "dark")  # dark lines on light wood
