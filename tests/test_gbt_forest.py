"""The packed forest walk (gbt._forest_sum) gives the same raw scores as walking one tree at a time."""
import numpy as np

from rallyiq.gbt import GBT, _tree_values


def _tree(rng, n_feat, depth):
    f, t, m, l, r, leaf, v = [], [], [], [], [], [], []

    def node(d):
        i = len(f)
        f.append(0), t.append(0.0), m.append(0), l.append(0), r.append(0), leaf.append(1), v.append(0.0)
        if d == 0 or rng.random() < 0.2:
            v[i] = float(rng.normal())
            return i
        f[i], t[i], m[i], leaf[i] = int(rng.integers(n_feat)), float(rng.normal()), int(rng.integers(2)), 0
        l[i] = node(d - 1)
        r[i] = node(d - 1)
        return i
    node(depth)
    return {"f": f, "t": t, "m": m, "l": l, "r": r, "leaf": leaf, "v": v}


def test_forest_matches_tree_by_tree():
    rng = np.random.default_rng(0)
    spec = {"classes": [0, 1, 2], "baseline": [0.1, -0.2, 0.3],
            "trees": [[_tree(rng, 5, int(rng.integers(1, 6))) for _ in range(3)] for _ in range(40)]}
    g = GBT(spec)
    X = rng.normal(size=(57, 5))
    X[rng.random(X.shape) < 0.1] = np.nan
    ref = np.tile(g.baseline, (len(X), 1))
    for it in g.trees:
        for k, t in enumerate(it):
            ref[:, k] += _tree_values(t, X)
    assert np.allclose(g.raw(X), ref, atol=1e-12)
    assert g.raw(np.zeros((0, 5))).shape == (0, 3)
