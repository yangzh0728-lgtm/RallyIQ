"""Dead-ball filter: on by default (train_v7), model loads, isolated low-confidence hits go first."""
from dataclasses import replace

from rallyiq.events import dead_ball as DB
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from rallyiq.events.synthetic import make_video


def test_default_on_and_bundled_model_loads():
    assert EventsConfig().dead_ball == "auto"
    m = DB.get_model("auto")
    assert m is not None and list(m["features"]) == list(DB.FEATURES) and 0 < m["threshold"] < 1


def test_features_and_probabilities():
    v = make_video(n_rallies=2, seed=5)
    rallies = segment_events(v.shuttle, v.poses, replace(EventsConfig(), hit_scorer="off"))
    r = next(r for r in rallies if len(r.hits) >= 3)
    feats = DB.rally_features(r, lambda pid, f: "near", lambda t, s, w: (0.5, 2.0))
    assert len(feats) == len(r.hits) and set(feats[0]) == set(DB.FEATURES)
    assert feats[0]["first"] == 1.0 and feats[-1]["last"] == 1.0
    m = DB.get_model("auto")
    p = DB.p_dead(m, feats)
    iso = dict(feats[1], gap_prev=5.0, gap_next=5.0, conf=0.2, ps_other=0.0, wv_other=0.0, single=1.0)
    assert DB.p_dead(m, [iso])[0] > p[1]


def test_pipeline_runs_with_filter_on():
    v = make_video(n_rallies=2, seed=6)
    cfg = replace(EventsConfig(), hit_scorer="off")
    a = segment_events(v.shuttle, v.poses, cfg)
    a = segment_events(v.shuttle, v.poses, replace(cfg, dead_ball="off"))
    b = segment_events(v.shuttle, v.poses, replace(cfg, dead_ball="auto"))
    assert sum(len(r.hits) for r in b) <= sum(len(r.hits) for r in a)
