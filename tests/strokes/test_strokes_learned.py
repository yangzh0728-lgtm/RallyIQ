"""scikit-learn stroke-type backend: training, grouped CV, save/load and the pipeline stage.

Trained and scored on the synthetic generator only: the floors are regression checks, not
claims about real video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from rallyiq.contracts import SHOT_TYPES, StrokesResult, VideoMeta
from rallyiq.pipeline.context import StageContext
from rallyiq.store import Store
from rallyiq.strokes import classify, shot_features
from rallyiq.strokes import learned as L
from rallyiq.strokes.synthetic import make_rallies


def _data(seeds):
    X, y, g = [], [], []
    for seed in seeds:
        v = make_rallies(n_rallies=8, seed=seed)
        feats = shot_features(v.events, v.shuttle, v.players, v.fps)
        X.append(L.feature_matrix(feats))
        y += [v.truth[f.shot_id] for f in feats]
        g += [f"s{seed}"] * len(feats)
    return np.vstack(X), y, g


@pytest.fixture(scope="module")
def trained():
    X, y, g = _data(range(8))
    return L.train(X, y, g, L.TrainConfig(folds=4, max_iter=150))


def test_feature_matrix_keeps_missing_as_nan():
    v = make_rallies(n_rallies=2, seed=1, court_known=False)
    feats = shot_features(v.events, v.shuttle, v.players, v.fps)
    X = L.feature_matrix(feats)
    assert X.shape == (len(feats), len(L.FEATURES))
    assert np.isnan(X[:, L.FEATURES.index("hitter_depth_m")]).all()  # no calibration -> not measured


def test_grouped_cv_report(trained):
    _, rep = trained
    assert rep.folds == 4 and rep.n_groups == 8
    assert rep.accuracy is not None and rep.accuracy >= 0.8
    assert len(rep.confusion) == len(rep.classes) and sum(map(sum, rep.confusion)) == rep.n
    assert "Confusion matrix" in rep.markdown()


def test_save_load_and_classify(trained, tmp_path: Path):
    model, _ = trained
    path = tmp_path / "strokes.joblib"
    L.save(model, path)
    fn = L.make_model_fn(path)
    v = make_rallies(n_rallies=4, seed=99)
    res = classify(v.events, v.shuttle, v.players, v.fps, model=fn, model_name=L.SKLEARN_NAME)
    assert res.classifier == L.SKLEARN_NAME
    right = sum(s.stroke.shot_type == v.truth[s.shot_id] for s in res.shots)
    assert right / len(res.shots) >= 0.75
    for s in res.shots:
        assert s.stroke.candidates and s.stroke.candidates[0].shot_type in SHOT_TYPES


def test_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(L.SklearnUnavailable):
        L.load(tmp_path / "nope.joblib")


def test_stage_uses_model_and_falls_back(trained, tmp_path: Path):
    model, _ = trained
    path = tmp_path / "m.joblib"
    L.save(model, path)
    v = make_rallies(n_rallies=1, seed=5)
    store = Store(tmp_path / "store")
    vid, dest = store.create_video("x.mp4", ".mp4")
    dest.write_bytes(b"")
    meta = VideoMeta(video_id=vid, filename="x.mp4", fps=v.fps, frame_count=len(v.players.frames), width=1920,
                     height=1080, duration_ms=int(len(v.players.frames) * 1000 / v.fps))
    store.save_artifact(vid, "events", v.events.model_copy(update={"video_id": vid}))
    store.save_artifact(vid, "shuttle", v.shuttle)
    store.save_artifact(vid, "players", v.players)
    from rallyiq.strokes.stage import run

    ctx = StageContext(video_id=vid, video_path=dest, work_dir=tmp_path, store=store, meta=meta,
                       config={"strokes": {"classifier": "sklearn", "sklearn": {"model": str(path)}}})
    res = run(ctx)
    assert isinstance(res, StrokesResult) and res.classifier == L.SKLEARN_NAME
    ctx.config["strokes"]["sklearn"] = {"model": str(tmp_path / "missing.joblib")}
    assert run(ctx).classifier == "rules_v1"
    ctx.config["strokes"]["require_model"] = True
    with pytest.raises(L.SklearnUnavailable):
        run(ctx)
