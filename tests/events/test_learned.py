"""Learned hit scorer: training script plumbing on synthetic rallies."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rallyiq.events.config import EventsConfig
from rallyiq.events.learned import FEATURES, HitScorer, get_scorer
from rallyiq.events.pipeline import segment_events
from rallyiq.events.synthetic import make_video

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import learn_hits as LH  # noqa: E402


def _clip(seed: int, **kw) -> "LH.Clip":
    v = make_video(n_rallies=3, seed=seed, **kw)
    c = LH.Clip.__new__(LH.Clip)
    c.cid, c.group, c.offset, c.fps = f"syn{seed}", f"g{seed % 3}", 0, v.fps
    c.shuttle, c.poses, c.match_format = v.shuttle, v.poses, "singles"
    c.duration_ms = max(o.t_ms for o in v.shuttle) + 100
    side = {"P1": "near", "P2": "far"}
    hits = [{"t_ms": f * 1000.0 / v.fps, "side": side.get(p), "player": None} for f, p in v.gt_hits]
    c.label = {"range_ms": [0, int(c.duration_ms)], "video": {"fps": v.fps, "width": 1920, "height": 1080},
               "hits": hits, "rallies": []}
    return c


def test_simulation_matches_pipeline_for_rules():
    for cfg in (EventsConfig(hit_scorer="off"), EventsConfig(hit_scorer="off", hit_frame_offset=2)):
        _check_sim(cfg)


def test_estimate_offset_recovers_shift():
    clips = [_clip(s) for s in range(3)]
    for c in clips:  # labels two frames later than the synthetic truth
        for h in c.label["hits"]:
            h["t_ms"] += 2 * 1000.0 / c.fps
    off, gaps = LH.estimate_offset(clips, EventsConfig())
    assert off == 2 and gaps


def _check_sim(cfg):
    for seed in range(4):
        c = _clip(seed, dropout=0.1, outlier_rate=0.02)
        c.collect(cfg, 3)
        sim = sorted(round(h["t_ms"], 3) for h in LH.simulate(c, None, cfg))
        real = sorted(round(h.t_ms, 3) for r in c.run(cfg) for h in r.hits)
        assert sim == real


def test_train_and_use_scorer(tmp_path):
    clips = [_clip(s, dropout=0.1, outlier_rate=0.03) for s in range(6)]
    for c in clips:
        c.collect(EventsConfig(), 3)
    X = np.vstack([c.X for c in clips])
    y = np.concatenate([c.y for c in clips])
    assert X.shape[1] == len(FEATURES) and 0 < y.sum() < len(y)
    spec = LH.fit_mlp(X, y, hidden=8, l2=1e-3, epochs=300)
    p = HitScorer(spec).predict(X)
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.8
    spec["thresholds"] = {"keep_threshold": 0.4, "ok_threshold": 0.7, "min_candidate_conf": 0.1}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(spec))
    assert get_scorer(str(path)) is not None
    v = make_video(n_rallies=2, seed=11)
    rallies = segment_events(v.shuttle, v.poses, EventsConfig(hit_scorer=str(path)))
    assert sum(len(r.hits) for r in rallies) >= len(v.gt_hits) * 0.8
    # with a learned scorer, confidences are probabilities and the model's thresholds apply
    assert all(0.0 <= h.confidence <= 1.0 for r in rallies for h in r.hits)


def test_off_and_missing_model_fall_back_to_rules(tmp_path):
    assert get_scorer("off") is None
    assert get_scorer(str(tmp_path / "missing.json")) is None
    v = make_video(n_rallies=2, seed=3)
    a = segment_events(v.shuttle, v.poses, EventsConfig(hit_scorer="off"))
    b = segment_events(v.shuttle, v.poses, replace(EventsConfig(), hit_scorer=str(tmp_path / "missing.json")))
    assert [h.t_ms for r in a for h in r.hits] == [h.t_ms for r in b for h in r.hits]


def test_side_model_plumbing():
    clips = [_clip(s, dropout=0.1) for s in range(6)]
    for c in clips:
        c.collect(EventsConfig(), 3)
    assert sum((~np.isnan(c.ys)).sum() for c in clips) > 0
    folds = LH.folds_by_group(clips, 3)
    probs = LH.oof_probs(clips, folds, 0, 1e-2)
    p_near = LH.oof_side(clips, folds)
    thr, s = LH.tune_side(clips, probs, p_near, EventsConfig())
    assert s["f1"] is not None and 0.5 <= thr <= 1.01


def test_evaluate_config_runs():
    clips = [_clip(s, dropout=0.1) for s in range(6)]
    folds = LH.folds_by_group(clips, 3)
    r = LH.evaluate_config(clips, replace(EventsConfig(), min_hit_gap_ms=240.0), 3, folds, 0, 1e-2)
    assert r["hits"]["f1"] is not None and 0 < r["ceiling"] <= 1


@pytest.mark.bundled_hit_scorer
def test_bundled_model_loads_and_runs_on_a_synthetic_rally():
    from rallyiq.events import learned
    from rallyiq.events.pipeline import with_scorer_thresholds

    learned._load.cache_clear()
    scorer = learned.get_scorer("auto")
    assert scorer is not None, "rallyiq/events/models/hit_scorer.json missing"
    cfg = with_scorer_thresholds(EventsConfig())
    # Event parameters and thresholds tuned with the model travel with it.
    assert cfg.keep_threshold == scorer.thresholds["keep_threshold"]
    assert cfg.hit_frame_offset == scorer.thresholds.get("hit_frame_offset", 0)
    v = make_video(n_rallies=2, seed=3)
    hits = [h for r in segment_events(v.shuttle, v.poses) for h in r.hits]
    assert hits and all(0.0 <= h.confidence <= 1.0 for h in hits)
    learned._load.cache_clear()
