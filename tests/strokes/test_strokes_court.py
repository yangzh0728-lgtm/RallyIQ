import math

import numpy as np

from rallyiq.strokes import court as CM


def test_rally_features_shape_and_neighbours():
    locs = [(3.0, 2.0), (3.5, 11.0), None, (2.0, 3.0)]
    rows = CM.rally_features(locs, [0, 800, 1700, 2500], [0.1, 0.9, math.nan, 0.2], 850.0)
    assert len(rows) == 4 and all(len(r) == len(CM.FEATURES) for r in rows)
    i = {n: k for k, n in enumerate(CM.FEATURES)}
    assert abs(rows[0][i["hd"]] - 4.7) < 1e-9
    assert abs(rows[0][i["rd"]] - 4.3) < 1e-9  # receiver = next hitter
    assert rows[0][i["nhi"]] == 0.9 and math.isnan(rows[0][i["phi"]])
    assert math.isnan(rows[1][i["rd"]])  # next hitter position unknown
    assert rows[0][i["first"]] == 1.0 and rows[3][i["last"]] == 1.0
    assert math.isnan(rows[3][i["fl"]]) and abs(rows[0][i["fl_rel"]] - 800 / 850) < 1e-9


def test_depth_map_aligns_quantiles():
    ref = list(np.linspace(1.0, 6.0, 19))
    far = list(np.linspace(3.0, 8.0, 40))  # one side measured ~2 m too deep
    m = CM.depth_map(ref, far)
    assert abs(m(5.5) - 3.5) < 0.2
    assert CM.depth_map(ref, [4.0, 5.0])(4.0) == 4.0  # too few samples: identity


def test_smooth_uses_transitions():
    # two types: "a" is always followed by "b"; the second shot alone is ambiguous
    start = np.array([0.5, 0.5])
    trans = np.array([[0.05, 0.95], [0.5, 0.5]])
    prior = np.array([0.5, 0.5])
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    S = CM.smooth(P, start, trans, prior, 1.0)
    assert np.allclose(S.sum(1), 1.0)
    assert S[1, 1] > 0.8
    assert np.allclose(CM.smooth(P, start, trans, prior, 0.0)[1], [0.5, 0.5])


def test_bundled_model_loads_and_scores():
    m = CM.get_model()
    assert m is not None and m.classes == sorted(m.classes)
    X = np.asarray(CM.rally_features([(3.0, 4.0), (3.0, 12.5)], [0, 1300], [0.05, 0.95], 900.0))
    P = m.proba(X)
    assert P.shape == (2, len(m.classes)) and np.allclose(P.sum(1), 1.0)
    # a low contact from mid court answered from deep at the back: a lift
    assert m.classes[int(P[0].argmax())] in ("lift", "long_serve", "clear")
