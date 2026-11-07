"""Gradient-boosted trees as plain JSON, evaluated with numpy (no scikit-learn at run time).

``export_hgb`` turns a fitted scikit-learn ``HistGradientBoostingClassifier`` (numerical features,
binary or multiclass) into a JSON-safe dict; ``GBT`` evaluates it. Each tree is a set of parallel node
arrays: ``f`` (feature), ``t`` (threshold, go left when x <= t), ``m`` (NaN goes left), ``l`` / ``r``
(children), ``leaf`` and ``v`` (leaf value). Raw scores are ``baseline`` plus the sum of the trees of
that output; probabilities are the sigmoid (binary) or softmax (multiclass), as in scikit-learn.

JSON instead of pickle/joblib: safe to load, independent of the scikit-learn version, and small
enough to ship in the repository.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tree_to_json(pred) -> dict[str, list]:
    nd = pred.nodes
    return {
        "f": nd["feature_idx"].astype(int).tolist(), "t": [float(v) for v in nd["num_threshold"]],
        "m": nd["missing_go_to_left"].astype(int).tolist(), "l": nd["left"].astype(int).tolist(),
        "r": nd["right"].astype(int).tolist(), "leaf": nd["is_leaf"].astype(int).tolist(),
        "v": [round(float(v), 6) for v in nd["value"]],
    }


def export_hgb(model) -> dict[str, Any]:
    """``{"classes", "baseline": [K], "trees": [[tree] * K per iteration]}``; K = 1 for binary."""
    if getattr(model, "is_categorical_", None) is not None and np.any(model.is_categorical_):
        raise ValueError("categorical features are not supported")
    base = np.asarray(model._baseline_prediction, float).ravel()
    return {
        "classes": [c.item() if hasattr(c, "item") else c for c in model.classes_],
        "baseline": [float(b) for b in base],
        "trees": [[tree_to_json(p) for p in it] for it in model._predictors],
    }


def _tree_values(t: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    node = np.zeros(len(X), dtype=int)
    rows = np.arange(len(X))
    while True:
        live = ~t["leaf"][node]
        if not live.any():
            return t["v"][node]
        n = node[live]
        x = X[rows[live], t["f"][n]]
        left = np.where(np.isnan(x), t["m"][n], x <= t["t"][n])
        node[live] = np.where(left, t["l"][n], t["r"][n])


class GBT:
    def __init__(self, spec: dict[str, Any]):
        self.classes = list(spec["classes"])
        self.classes_ = np.asarray(self.classes, dtype=object)  # scikit-learn-style alias
        self.baseline = np.asarray(spec["baseline"], float)
        self.trees = [[{"f": np.asarray(t["f"], int), "t": np.asarray(t["t"], float), "m": np.asarray(t["m"], bool),
                        "l": np.asarray(t["l"], int), "r": np.asarray(t["r"], int),
                        "leaf": np.asarray(t["leaf"], bool), "v": np.asarray(t["v"], float)} for t in it]
                      for it in spec["trees"]]
        self.n_features = 1 + max((int(np.max(t["f"])) for it in self.trees for t in it), default=0)

    def raw(self, X) -> np.ndarray:
        """(n, K) raw scores."""
        X = np.asarray(X, float)
        X = X.reshape(-1, X.shape[-1]) if X.size else X.reshape(0, max(self.n_features, 1))
        out = np.tile(self.baseline, (len(X), 1))
        for it in self.trees:
            for k, t in enumerate(it):
                out[:, k] += _tree_values(t, X)
        return out

    def predict_proba(self, X) -> np.ndarray:
        """(n, n_classes) probabilities, columns in ``classes`` order."""
        r = self.raw(X)
        if r.shape[1] == 1:
            p = 1.0 / (1.0 + np.exp(-r[:, 0]))
            return np.stack([1.0 - p, p], axis=1)
        e = np.exp(r - r.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        P = self.predict_proba(X)
        return np.asarray(self.classes, dtype=object)[P.argmax(axis=1)]
