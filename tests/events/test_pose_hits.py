"""Pose-sequence hit recovery: model format, exporter and the recovery pass (synthetic rallies)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rallyiq.events import pose_hits as PH
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from rallyiq.events.synthetic import make_video


def _const(raw: float, **thr) -> dict:
    leaf = {"f": [0], "t": [0.0], "m": [0], "l": [0], "r": [0], "leaf": [1], "v": [0.0]}
    return {"kind": "gbt", "features": list(PH.FEATURES), "baseline": raw, "trees": [leaf], "thresholds": thr}


def _rallies(seed=0):
    v = make_video(n_rallies=2, seed=seed)
    return v, segment_events(v.shuttle, v.poses, replace(EventsConfig(), hit_scorer="off"))


def test_window_features_shape_and_side_flag():
    v, _ = _rallies()
    ser = PH.side_series(v.poses)
    assert ser["near"].bodies and ser["far"].bodies
    f = PH.window_features(ser["far"], ser["far"].t[len(ser["far"].t) // 2], 1000.0 / v.fps)
    assert len(f) == len(PH.FEATURES)
    assert f[PH.FEATURES.index("far")] == 1.0
    empty = PH.SideSeries("near")
    g = PH.window_features(empty, 1000.0, 33.3)
    assert np.isnan(g[0]) and g[PH.FEATURES.index("n_seen")] == 0.0


def test_recover_fills_serve_and_same_side_gap():
    v, rallies = _rallies()
    r = rallies[0]
    assert len(r.hits) >= 4
    first_t = r.hits[0].t_ms
    del r.hits[2]  # hits 1 and 3 now come from the same side
    del r.hits[0]  # and the serve is gone
    n = len(r.hits)
    added = PH.recover(rallies, v.poses, PH.PoseHitModel(_const(5.0)), PH.RecoverConfig(), 1000.0 / v.fps)
    assert added >= 2 and len(r.hits) >= n + 2
    rec = [h for h in r.hits if PH.R_POSE_RECOVERED in h.reasons]
    assert rec and all(h.source == PH.SOURCE and h.status.value == "needs_review" for h in rec)
    assert r.hits[0].t_ms < first_t + 1 and r.hits[0].reasons == [PH.R_POSE_RECOVERED]
    assert [h.index_in_rally for h in r.hits] == list(range(1, len(r.hits) + 1))
    assert len({h.hit_id for h in r.hits}) == len(r.hits)
    assert all(a.t_ms < b.t_ms for a, b in zip(r.hits, r.hits[1:]))


def test_recover_adds_nothing_when_model_says_no():
    v, rallies = _rallies(1)
    del rallies[0].hits[0]
    before = [len(r.hits) for r in rallies]
    assert PH.recover(rallies, v.poses, PH.PoseHitModel(_const(-5.0)), None, 1000.0 / v.fps) == 0
    assert [len(r.hits) for r in rallies] == before


def test_pipeline_option(tmp_path):
    p = tmp_path / "pose.json"
    p.write_text(json.dumps(_const(-5.0)))
    v = make_video(n_rallies=2, seed=2)
    cfg = replace(EventsConfig(), hit_scorer="off")
    a = segment_events(v.shuttle, v.poses, cfg)
    b = segment_events(v.shuttle, v.poses, replace(cfg, pose_hits=str(p)))
    assert [len(r.hits) for r in a] == [len(r.hits) for r in b]
    assert PH.get_model("off") is None and PH.get_model(str(tmp_path / "missing.json")) is None


def test_export_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(PH.FEATURES)))
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 3]) + 0.5 * np.nan_to_num(X[:, 10]) > 0.3).astype(float)
    h = HistGradientBoostingClassifier(max_iter=30).fit(X, y)
    m = PH.PoseHitModel(json.loads(json.dumps(PH.export_hgb(h, {"gap_threshold": 0.4}))))
    assert m.thresholds == {"gap_threshold": 0.4}
    assert np.abs(m.predict(X) - h.predict_proba(X)[:, 1]).max() < 1e-4
