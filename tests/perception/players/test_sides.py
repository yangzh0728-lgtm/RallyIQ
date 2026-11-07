"""Side consistency: real changes of ends vs. labelling errors."""

import numpy as np

from rallyiq.perception.players.geometry import far_court_roi
from rallyiq.perception.players.models import PlayerFrameResult, PlayerState
from rallyiq.perception.players.runner import track_players
from rallyiq.perception.players.sides import stabilise_sides
from rallyiq.perception.players.tracker import SinglesTracker, TrackerConfig
from rallyiq.perception.players.filtering import build_candidates

from .synth import H_COURT_TO_IMG, H_IMG_TO_COURT, IMAGE_SIZE, person_at_court

FPS = 5  # processed frames per second in these synthetic sequences
RED = np.eye(36, dtype=np.float32)[0]
BLUE = np.eye(36, dtype=np.float32)[20]


def _p(pid, track, y, colour=None):
    return PlayerState(pid, track, "near" if y < 6.7 else "far", np.zeros(4), np.zeros((17, 3)),
                       np.zeros(2), np.array([3.0, y]), True, 0.9, appearance=colour,
                       team="A" if pid in ("P1", "P3") else "B")


def _seq(spec):
    """spec: list of (seconds, [(pid, track, y, colour), ...]) stretches, or (seconds, None) for no players."""
    frames, k = [], 0
    for secs, people in spec:
        for _ in range(int(secs * FPS)):
            players = [_p(*x) for x in people] if people else []
            frames.append(PlayerFrameResult(k * 6, k * 200.0, players))
            k += 1
    return frames


def _p1_side(f):
    return next((p.side for p in f.players if p.player_id == "P1"), None)


def test_mislabelled_stretch_after_cut_is_swapped_back():
    frames = _seq([
        (30, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)]),
        (3, None),
        (8, [("P1", 2, 10.5, None), ("P2", 3, 2.5, None)]),  # re-acquired the wrong way round
        (3, None),
        (30, [("P1", 4, 2.5, None), ("P2", 5, 2.5 + 8, None)]),
    ])
    assert stabilise_sides(frames) == []
    assert all(_p1_side(f) == "near" for f in frames if f.players)
    assert all(p.team == ("A" if p.player_id == "P1" else "B") for f in frames for p in f.players)


def test_long_mislabel_without_colour_evidence_is_not_a_switch():
    # Sustained, but after a re-identification and nothing says it is the same people.
    frames = _seq([
        (30, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)]),
        (3, None),
        (40, [("P1", 2, 10.5, None), ("P2", 3, 2.5, None)]),
    ])
    assert stabilise_sides(frames) == []
    assert all(_p1_side(f) == "near" for f in frames if f.players)


def test_change_of_ends_after_break_backed_by_colour():
    frames = _seq([
        (30, [("P1", 0, 2.5, RED), ("P2", 1, 10.5, BLUE)]),
        (60, None),
        (30, [("P1", 2, 10.5, RED), ("P2", 3, 2.5, BLUE)]),
    ])
    assert stabilise_sides(frames) == [90 * FPS * 6]
    assert _p1_side(frames[-1]) == "far"


def test_colour_says_labels_are_swapped_so_no_switch():
    frames = _seq([
        (30, [("P1", 0, 2.5, RED), ("P2", 1, 10.5, BLUE)]),
        (5, None),
        (30, [("P1", 2, 10.5, BLUE), ("P2", 3, 2.5, RED)]),
    ])
    assert stabilise_sides(frames) == []
    late = frames[-1]
    assert _p1_side(late) == "near"
    assert next(p for p in late.players if p.player_id == "P1").appearance is RED


def test_net_flicker_is_not_a_switch():
    frames = _seq([(5, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)])])
    for k in range(400):  # P1 hovers on the net line for 80 s
        y = 6.7 + 0.4 * np.sin(k)
        frames.append(PlayerFrameResult(len(frames) * 6, len(frames) * 200.0,
                                        [_p("P1", 0, y), _p("P2", 1, 10.5)]))
    assert stabilise_sides(frames) == []


def test_changes_of_ends_too_close_together_count_once():
    frames = _seq([
        (30, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)]),
        (40, [("P1", 0, 10.5, None), ("P2", 1, 2.5, None)]),  # walks over: real change
        (40, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)]),  # back again 40 s later: impossible
    ])
    assert stabilise_sides(frames) == [30 * FPS * 6]
    assert _p1_side(frames[-1]) == "far"


def test_apply_false_leaves_labels_alone():
    frames = _seq([
        (30, [("P1", 0, 2.5, None), ("P2", 1, 10.5, None)]),
        (3, None),
        (30, [("P1", 2, 10.5, None), ("P2", 3, 2.5, None)]),
    ])
    assert stabilise_sides(frames, apply=False) == []
    assert _p1_side(frames[-1]) == "far"


def test_tracker_reacquires_lookalikes_on_their_old_halves():
    tr = SinglesTracker(TrackerConfig())
    for i in range(10):
        c = build_candidates([person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)], H_IMG_TO_COURT, IMAGE_SIZE)
        tr.update(i, c, calibrated=True)
    # Long cut, then the far player is listed first by the detector.
    c = build_candidates([person_at_court(2.0, 11.0), person_at_court(4.0, 2.0)], H_IMG_TO_COURT, IMAGE_SIZE)
    out = {p.player_id: p.side for p in tr.update(200, c, calibrated=True)}
    assert out == {"P1": "near", "P2": "far"}


# -- far-half second pose pass ----------------------------------------------------

def test_far_court_roi_covers_far_half_with_head_room():
    from rallyiq.perception.players.geometry import apply_homography

    roi = far_court_roi(H_IMG_TO_COURT, IMAGE_SIZE)
    assert roi is not None
    x1, y1, x2, y2, player_px = roi
    far_base = apply_homography(H_COURT_TO_IMG, np.array([[0, 13.4], [6.1, 13.4]]))
    assert x1 <= far_base[:, 0].min() and x2 >= far_base[:, 0].max()
    assert y1 <= far_base[:, 1].min() - player_px  # a far player's head is inside
    net = apply_homography(H_COURT_TO_IMG, np.array([[3.05, 6.7]]))[0]
    assert y2 >= net[1] - 5
    assert y2 < IMAGE_SIZE[1]  # near half excluded
    assert 20 < player_px < 200


class RoiEstimator:
    supports_far_roi = True

    def __init__(self, per_frame, px):
        self.per_frame = iter(per_frame)
        self.px = px
        self.rois = []

    def far_player_px(self, shape, roi):
        return self.px

    def __call__(self, img, far_roi=None):
        self.rois.append(far_roi)
        return next(self.per_frame)


def _run(px, boost):
    from rallyiq.perception.players.runner import PlayerTrackingConfig

    per_frame = [[person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)]] * 3
    est = RoiEstimator(per_frame, px)
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    track_players(((i, i * 33.3, img) for i in range(3)), est, lambda f: H_IMG_TO_COURT,
                  PlayerTrackingConfig(match_format="singles", far_boost=boost))
    return est.rois


def test_far_boost_auto_only_when_far_player_is_small():
    assert all(r is not None for r in _run(40.0, "auto"))
    assert all(r is None for r in _run(200.0, "auto"))
    assert all(r is not None for r in _run(200.0, "on"))
    assert all(r is None for r in _run(40.0, "off"))


# -- duplicate detections of one person (far-boost regression) --------------------

def _widened(det, left=0.0, right=0.0):
    from rallyiq.perception.players.models import PersonDetection

    b = det.bbox.copy()
    b[0] -= left
    b[2] += right
    return PersonDetection(b, det.keypoints.copy(), det.score * 0.95)


def test_wide_racket_box_and_narrow_box_are_one_person():
    from rallyiq.perception.players.pose import same_person

    narrow = person_at_court(3.3, 10.5)
    w = narrow.bbox[2] - narrow.bbox[0]
    wide = _widened(narrow, left=0.6 * w, right=0.6 * w)  # racket arm out, ~2x wider
    assert same_person(wide, narrow) and same_person(narrow, wide)
    other = person_at_court(4.6, 11.8)
    assert not same_person(narrow, other)


def test_duplicate_candidates_are_dropped_but_close_partners_kept():
    near = person_at_court(3.0, 2.5)
    far = person_at_court(3.3, 10.5)
    w = far.bbox[2] - far.bbox[0]
    dup = _widened(far, left=0.6 * w, right=0.6 * w)
    cands = build_candidates([near, far, dup], H_IMG_TO_COURT, IMAGE_SIZE)
    assert len(cands) == 2
    # Doubles partners one behind the other overlap on screen but stand apart.
    front, back = person_at_court(3.0, 8.6), person_at_court(3.2, 11.0)
    assert len(build_candidates([front, back], H_IMG_TO_COURT, IMAGE_SIZE)) == 2


def test_singles_with_doubled_far_player_stays_singles():
    from rallyiq.perception.players.runner import PlayerTrackingConfig

    per_frame = []
    for i in range(120):
        t = i / 30
        near = person_at_court(3.0 + np.sin(t), 2.5)
        far = person_at_court(3.3 - np.sin(t), 10.5)
        w = far.bbox[2] - far.bbox[0]
        per_frame.append([near, far, _widened(far, left=0.6 * w, right=0.6 * w)])
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)

    class Est:
        def __init__(self):
            self.it = iter(per_frame)

        def __call__(self, im):
            return next(self.it)

    res = track_players(((i, i * 33.3, img) for i in range(120)), Est(), lambda f: H_IMG_TO_COURT,
                        PlayerTrackingConfig())
    assert res.match_format == "singles"
    assert all(len(r.players) <= 2 for r in res)


def test_singles_side_memory_ignored_when_both_slots_remember_the_same_half():
    # sx36_set2: before calibration settled, two people beyond the net seeded both singles slots on the
    # far half. P2 keeps tracking the far player; P1's far "memory" must not keep it off the near player,
    # who also looks nothing like the person P1 was seeded with.
    def cands(*people):
        out = build_candidates([person_at_court(*xy) for xy, _ in people], H_IMG_TO_COURT, IMAGE_SIZE)
        for c, (_, colour) in zip(out, people):
            c.appearance = colour
        return out

    tr = SinglesTracker(TrackerConfig())
    for i in range(10):
        tr.update(i, cands(((1.5, 10.0), RED), ((4.5, 12.0), RED)), calibrated=True)
    for i in range(10, 80):  # one of them leaves; the other is the real far player
        tr.update(i, cands(((4.5, 12.0), RED)), calibrated=True)
    out = {p.side for p in tr.update(80, cands(((4.5, 12.0), RED), ((3.0, 2.5), BLUE)), calibrated=True)}
    assert out == {"near", "far"}


def test_side_memory_still_steers_reacquisition_after_a_cut():
    tr = SinglesTracker(TrackerConfig())
    for i in range(10):
        tr.update(i, build_candidates([person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)],
                                      H_IMG_TO_COURT, IMAGE_SIZE), calibrated=True)
    # Long cut; only one person comes back, on the far half: it goes to the slot last seen there.
    out = tr.update(200, build_candidates([person_at_court(3.0, 11.0)], H_IMG_TO_COURT, IMAGE_SIZE), calibrated=True)
    assert [(p.player_id, p.side) for p in out] == [("P2", "far")]


# -- far player without calibration ------------------------------------------------

class UncalibratedEstimator(RoiEstimator):
    """Full frame finds only the near player; the far pass finds the far one."""

    def __init__(self, per_frame, px, far):
        super().__init__(per_frame, px)
        self.far = far
        self.detect_rois = []

    def detect_roi(self, img, roi, dets):
        self.detect_rois.append(roi)
        x1, y1, x2, y2 = roi[:4]
        fx, fy = (self.far.bbox[0] + self.far.bbox[2]) / 2, self.far.bbox[3]
        return dets + ([self.far] if x1 <= fx <= x2 and y1 <= fy <= y2 else [])


def _run_uncalibrated(per_frame, boost="auto", px=40.0):
    from rallyiq.perception.players.runner import PlayerTrackingConfig

    far = person_at_court(3.0, 10.5)
    est = UncalibratedEstimator(per_frame, px, far)
    img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), np.uint8)
    out = track_players(((i, i * 33.3, img) for i in range(len(per_frame))), est, lambda f: None,
                        PlayerTrackingConfig(match_format="singles", far_boost=boost))
    return est, out


def test_far_player_found_without_calibration():
    near = person_at_court(3.0, 2.5)
    est, out = _run_uncalibrated([[near]] * 5)
    assert len(est.detect_rois) == 5
    *_, y2, far_px = est.detect_rois[0]
    assert y2 < near.bbox[3] and far_px < near.height
    assert all(len(f.players) == 2 for f in out)


def test_far_pass_without_calibration_skipped_when_not_needed():
    near, far = person_at_court(3.0, 2.5), person_at_court(3.0, 10.5)
    assert _run_uncalibrated([[near, far]] * 3)[0].detect_rois == []  # both already found
    assert _run_uncalibrated([[near]] * 3, boost="off")[0].detect_rois == []
    assert _run_uncalibrated([[near]] * 3, px=200.0)[0].detect_rois == []  # far player big enough
    assert _run_uncalibrated([[]] * 3)[0].detect_rois == []  # nobody to anchor on
