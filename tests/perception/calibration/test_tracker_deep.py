import numpy as np
import pytest

pytest.importorskip("cv2")

from rallyiq.perception.calibration import tracker as tr  # noqa: E402
from rallyiq.perception.calibration.calibrate import _failed  # noqa: E402


def test_deep_search_is_rate_limited_while_no_court_is_found(monkeypatch):
    calls = []

    def fake_calibrate(frame, previous=None, extra_hypotheses=(), method="auto", deep="auto"):
        calls.append(deep)
        return _failed((64, 48), "auto", "court lines not found")

    monkeypatch.setattr(tr, "calibrate_frame", fake_calibrate)
    t = tr.CourtTracker(tr.TrackerConfig(deep_every_checks=5))
    frame = np.full((48, 64, 3), 90, np.uint8)
    for i in range(12):
        t.update(i * 15, frame)
    assert calls[0] == "auto"  # first frame
    assert sum(1 for d in calls if d) == 3  # then once every 5 checks
    assert len(calls) == 12
