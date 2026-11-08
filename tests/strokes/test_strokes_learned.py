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


def test_context_features_shape_and_missing():
    from rallyiq.strokes.context import CONTEXT_FEATURES, context_features

    v = make_rallies(n_rallies=2, seed=3)
    feats = shot_features(v.events, v.shuttle, v.players, v.fps)
    C = context_features(v.events, v.shuttle, v.players, feats)
    assert C.shape == (len(feats), len(CONTEXT_FEATURES))
    side = C[:, CONTEXT_FEATURES.index("side")]
    assert set(np.unique(side[~np.isnan(side)])) <= {-1.0, 1.0}
    prev = C[:, CONTEXT_FEATURES.index("prev_flight_ms")]
    firsts = [i for i, f in enumerate(feats) if f.first_in_rally]
    assert firsts and np.isnan(prev[firsts]).all()  # the serve has no incoming shot
    v2 = make_rallies(n_rallies=2, seed=3, court_known=False)
    f2 = shot_features(v2.events, v2.shuttle, v2.players, v2.fps)
    assert np.isnan(context_features(v2.events, v2.shuttle, v2.players, f2)[:, 0]).all()


def test_v2_model_round_trip(tmp_path: Path):
    X, y, g = [], [], []
    for seed in range(6):
        v = make_rallies(n_rallies=8, seed=seed)
        feats = shot_features(v.events, v.shuttle, v.players, v.fps)
        X.append(L.full_matrix(feats, v.events, v.shuttle, v.players, "v2"))
        y += [v.truth[f.shot_id] for f in feats]
        g += [f"s{seed}"] * len(feats)
    X = np.vstack(X)
    assert X.shape[1] == len(L.FEATURES_V2)
    model, _ = L.train(X, y, g, L.TrainConfig(folds=3, max_iter=100))
    path = tmp_path / "v2.joblib"
    L.save(model, path, feature_set="v2")
    assert L.load(path)["feature_set"] == "v2"
    v = make_rallies(n_rallies=4, seed=99)
    res = classify(v.events, v.shuttle, v.players, v.fps, model=L.make_model_fn(path), model_name=L.SKLEARN_NAME)
    right = sum(s.stroke.shot_type == v.truth[s.shot_id] for s in res.shots)
    assert right / len(res.shots) >= 0.7


def test_json_export_matches_and_loads_without_pickle(trained, tmp_path: Path):
    import json

    model, _ = trained
    path = tmp_path / "m.json"
    path.write_text(json.dumps(L.export_json(model, "base", {"n": 1})))
    blob = L.load(path)
    assert blob["feature_set"] == "base" and blob["n"] == 1
    v = make_rallies(n_rallies=3, seed=42)
    X = L.feature_matrix(shot_features(v.events, v.shuttle, v.players, v.fps))
    # leaf values are stored to 6 decimals (gbt.tree_to_json); over 150 trees that is ~1e-6 in p
    assert np.abs(blob["model"].predict_proba(X) - model.predict_proba(X)).max() < 1e-5
    assert list(blob["model"].predict(X)) == list(model.predict(X))


def test_bundled_model_is_the_default(tmp_path: Path):
    assert L.BUNDLED.is_file()
    blob = L.load(L.BUNDLED)
    assert blob["feature_set"] == "v3" and set(blob["model"].classes) <= set(SHOT_TYPES)
    v = make_rallies(n_rallies=1, seed=6)
    store = Store(tmp_path / "store")
    vid, dest = store.create_video("x.mp4", ".mp4")
    dest.write_bytes(b"")
    meta = VideoMeta(video_id=vid, filename="x.mp4", fps=v.fps, frame_count=len(v.players.frames), width=1920,
                     height=1080, duration_ms=int(len(v.players.frames) * 1000 / v.fps))
    store.save_artifact(vid, "events", v.events.model_copy(update={"video_id": vid}))
    store.save_artifact(vid, "shuttle", v.shuttle)
    store.save_artifact(vid, "players", v.players)
    from rallyiq.strokes.stage import run

    ctx = StageContext(video_id=vid, video_path=dest, work_dir=tmp_path, store=store, meta=meta, config={})
    res = run(ctx)
    assert res.classifier == L.GBT_NAME and len(res.shots) == len(v.events.shots)
    ctx.config["strokes"] = {"classifier": "rules"}
    assert run(ctx).classifier == "rules_v1"


def test_trajectory_features_shape_and_v3_matrix():
    from rallyiq.strokes.trajectory import TRAJ_FEATURES, traj_features

    v = make_rallies(n_rallies=2, seed=7)
    feats = shot_features(v.events, v.shuttle, v.players, v.fps)
    T = traj_features(v.events, v.shuttle, v.players, feats, v.fps)
    assert T.shape == (len(feats), len(TRAJ_FEATURES))
    cov = T[:, TRAJ_FEATURES.index("coverage")]
    assert np.nanmin(cov) >= 0.0 and np.nanmax(cov) <= 1.0
    assert np.isfinite(T[:, TRAJ_FEATURES.index("arc")]).any()  # the synthetic flights are seen
    T2 = traj_features(v.events, v.shuttle, v.players, feats)  # fps estimated from the track
    shape = [i for i, n in enumerate(TRAJ_FEATURES) if n != "coverage"]
    assert np.allclose(T[:, shape], T2[:, shape], equal_nan=True)
    X = L.full_matrix(feats, v.events, v.shuttle, v.players, "v3")
    assert X.shape == (len(feats), len(L.FEATURES_V3))
    assert traj_features(v.events, None, None, feats, v.fps).shape == (len(feats), len(TRAJ_FEATURES))
