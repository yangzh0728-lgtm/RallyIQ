"""Rally sequence hit model (train_v10): bundled weights load, decode, frame->time and the config switch."""
import math

import numpy as np

from rallyiq.events import seqhits as SH
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from rallyiq.events.types import Keypoint, PlayerPose, PoseFrame, RallyEndReason, RallySegment, ShuttleObs


def test_default_is_rules_and_bundled_model_loads():
    assert EventsConfig().hit_model == "rules"
    assert SH.get_model("rules") is None
    m = SH.get_model("seq")
    assert m is not None and len(m.members) >= 1 and 0 < m.thr < 1 and m.gap_s > 0
    h, f = m.predict(np.zeros((50, SH.NF), np.float32))
    assert h.shape == f.shape == (50,) and np.all((h >= 0) & (h <= 1))


def test_peaks_threshold_and_gap():
    grid = np.arange(0, 30)
    p = np.zeros(30)
    p[5], p[7], p[20] = 0.9, 0.8, 0.3
    assert SH.peaks(grid, p, 30.0, 0.5, 0.2) == [5]  # 7 is within 0.2 s of a stronger peak; 20 below threshold
    assert SH.peaks(grid, p, 30.0, 0.2, 0.05) == [5, 7, 20]


def _obs(n, fps):
    sh = [ShuttleObs(frame=i, t_ms=round(i * 1000 / fps), x=100.0 + i, y=200.0, visible=True, confidence=0.9)
          for i in range(n)]
    kp = [Keypoint(x=100.0, y=300.0 + 10 * j, confidence=0.9) for j in range(17)]
    poses = [PoseFrame(frame=i, t_ms=round(i * 1000 / fps), players=[
        PlayerPose(player_id="A", side="near", keypoints=kp)])
        for i in range(n)]
    return sh, poses


class _Fixed:
    """Stand-in model: a hit peak at frame 1800 (60 s at 29.97 fps)."""
    thr, gap_s = 0.5, 0.2

    def __init__(self, n):
        self.n = n

    def predict(self, X):
        h = np.zeros(len(X))
        h[1800] = 0.95
        return h, np.zeros(len(X))


def test_apply_uses_observed_timestamps_not_rounded_fps():
    fps = 30000 / 1001  # 29.97: the integer-ms step median is 33 ms (30.3 fps), which drifts ~600 ms by a minute
    sh, poses = _obs(2000, fps)
    r = RallySegment(rally_id="R1", start_frame=1768, end_frame=1828, start_ms=59000.0, end_ms=61000.0,
                     end_reason=list(RallyEndReason)[0])
    SH.apply([r], sh, poses, _Fixed(2000), 1000.0 / 33.0, (1920, 1080))
    assert len(r.hits) == 1
    h = r.hits[0]
    assert h.frame == 1800 and abs(h.t_ms - 1800 * 1000 / fps) < 2
    assert h.hitter_id == "A" and h.hit_id == "R1_S01" and h.source == SH.SOURCE


def test_segment_events_hit_model_needs_frame_size():
    sh, poses = _obs(300, 30.0)
    cfg = EventsConfig(hit_model="seq")
    a = segment_events(sh, poses, EventsConfig())
    b = segment_events(sh, poses, cfg)  # no frame size: falls back to the rules hits
    assert [len(r.hits) for r in a] == [len(r.hits) for r in b]
    c = segment_events(sh, poses, cfg, frame_size=(1920, 1080))
    assert [(r.start_ms, r.end_ms) for r in c] == [(r.start_ms, r.end_ms) for r in a]
    assert all(h.source == SH.SOURCE for r in c for h in r.hits)
