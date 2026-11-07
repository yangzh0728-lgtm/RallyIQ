"""Optional backend: scikit-learn stroke-type classifier on the ``features.py`` evidence.

The rules in ``rules.py`` are hand-set ramps; this backend learns the same mapping from
labelled shots instead. The input is exactly the per-shot evidence the rules see
(contact height, hitter / receiver depth, lateral distance, flight time, court speed,
rise, left-frame, position in the rally). A feature that could not be measured is NaN,
which ``HistGradientBoostingClassifier`` handles natively, so nothing is filled with a guess.

* ``train`` fits the model and scores it with grouped cross-validation (folds by match,
  so the same match never sits in training and validation): accuracy, macro F1 and the
  confusion matrix, from out-of-fold predictions.
* ``save`` / ``load`` keep the model as a joblib file. joblib is pickle: only load files
  you trained yourself.
* ``make_model_fn`` plugs a saved model into ``classify`` (``ctx.config["strokes"] =
  {"classifier": "sklearn", "sklearn": {"model": "/path/model.joblib"}}``). The review
  policy (``ok`` / ``needs_review`` / ``undeterminable``) stays the one in ``classify.py``.

``scripts/learn_strokes.py`` builds the training data (synthetic rallies or ShuttleSet
clips) and calls ``train``. scikit-learn is an optional dependency (``pip install -e .[strokes-ml]``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import SHOT_TYPES
from .context import CONTEXT_FEATURES, context_features
from .features import StrokeFeatures

SKLEARN_NAME = "sklearn_hgb"

FEATURES = (
    "contact_h",
    "hitter_depth_m",
    "receiver_depth_m",
    "lateral_m",
    "flight_ms",
    "court_speed_mps",
    "rise",
    "left_frame",
    "first_in_rally",
    "last_in_rally",
    "index_in_rally",
)


# Feature set "v2": the base evidence plus the rally context, direction, pose and early-flight
# features of ``context.py``. On 6 held-out ShuttleSet matches it lifted accuracy from ~0.56 to
# ~0.63 (see rallyiq/strokes/README.md). Models remember which set they were trained on.
FEATURES_V2 = FEATURES + CONTEXT_FEATURES
FEATURE_SETS = {"base": FEATURES, "v2": FEATURES_V2}


class SklearnUnavailable(RuntimeError):
    """scikit-learn is not installed or the model file is missing / unusable."""


def _num(v: Any) -> float:
    if v is None:
        return math.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def feature_row(f: StrokeFeatures) -> list[float]:
    return [_num(getattr(f, name)) for name in FEATURES]


def feature_matrix(feats: Sequence[StrokeFeatures]) -> np.ndarray:
    return np.asarray([feature_row(f) for f in feats], dtype=float).reshape(-1, len(FEATURES))


def full_matrix(feats: Sequence[StrokeFeatures], events=None, shuttle=None, players=None,
                feature_set: str = "v2") -> np.ndarray:
    """The model input for ``feature_set`` ("base" or "v2"; v2 needs the clip's events/shuttle/players)."""
    X = feature_matrix(feats)
    if feature_set == "base":
        return X
    if feature_set != "v2":
        raise ValueError(f"unknown stroke feature set {feature_set!r}")
    if events is None:
        raise ValueError("feature set v2 needs the events of the clip")
    return np.hstack([X, context_features(events, shuttle, players, feats)])


def feature_set_of(names: Sequence[str]) -> str | None:
    for k, v in FEATURE_SETS.items():
        if list(names) == list(v):
            return k
    return None


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise SklearnUnavailable("scikit-learn is not installed (pip install -e .[strokes-ml])") from e


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.08
    max_iter: int = 300
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 10
    l2_regularization: float = 1.0
    class_weight: str | None = "balanced"  # rare types (net kill, block) still count
    folds: int = 5
    seed: int = 0


@dataclass
class TrainReport:
    n: int
    n_groups: int
    folds: int
    classes: list[str]
    accuracy: float | None
    macro_f1: float | None
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)  # rows = truth, cols = predicted (``classes`` order)
    counts: dict[str, int] = field(default_factory=dict)

    def markdown(self) -> str:
        acc = "–" if self.accuracy is None else f"{self.accuracy:.3f}"
        f1 = "–" if self.macro_f1 is None else f"{self.macro_f1:.3f}"
        lines = [f"Shots: {self.n} from {self.n_groups} groups, {self.folds}-fold grouped CV (out-of-fold)",
                 "", f"- accuracy: {acc}", f"- macro F1: {f1}", "",
                 "| type | n | precision | recall | F1 |", "| --- | --- | --- | --- | --- |"]
        for c in self.classes:
            m = self.per_class.get(c, {})
            lines.append(f"| {c} | {self.counts.get(c, 0)} | {m.get('precision', 0):.2f} | "
                         f"{m.get('recall', 0):.2f} | {m.get('f1', 0):.2f} |")
        if self.confusion:
            lines += ["", "Confusion matrix (rows = truth, columns = predicted):", "",
                      "| | " + " | ".join(self.classes) + " |", "|" + " --- |" * (len(self.classes) + 1)]
            for c, row in zip(self.classes, self.confusion):
                lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines) + "\n"


def _model(cfg: TrainConfig):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        class_weight=cfg.class_weight,
        random_state=cfg.seed,
    )


def cross_validate(X: np.ndarray, y: Sequence[str], groups: Sequence[str], cfg: TrainConfig | None = None) -> TrainReport:
    """Out-of-fold accuracy / macro F1 / confusion matrix with ``GroupKFold`` (no group in both halves)."""
    _sklearn()
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold

    cfg = cfg or TrainConfig()
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups, dtype=object)
    classes = [t for t in SHOT_TYPES if t in set(y)]
    counts = {c: int((y == c).sum()) for c in classes}
    n_groups = len(set(groups))
    k = min(cfg.folds, n_groups)
    report = TrainReport(n=len(y), n_groups=n_groups, folds=k, classes=classes, accuracy=None, macro_f1=None, counts=counts)
    if k < 2:
        return report
    pred = np.empty(len(y), dtype=object)
    for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
        m = _model(cfg).fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    report.accuracy = float(accuracy_score(y, pred))
    report.macro_f1 = float(f1_score(y, pred, labels=classes, average="macro", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    report.per_class = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, c in enumerate(classes)}
    report.confusion = confusion_matrix(y, pred, labels=classes).tolist()
    return report


def train(X: np.ndarray, y: Sequence[str], groups: Sequence[str], cfg: TrainConfig | None = None) -> tuple[Any, TrainReport]:
    """Cross-validate, then fit the final model on everything. Returns (model, report)."""
    cfg = cfg or TrainConfig()
    report = cross_validate(X, y, groups, cfg)
    model = _model(cfg).fit(X, np.asarray(y, dtype=object))
    return model, report


def save(model: Any, path: str | Path, meta: dict[str, Any] | None = None, feature_set: str = "base") -> None:
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": model, "features": list(FEATURE_SETS[feature_set]), "feature_set": feature_set,
            "sklearn": sklearn.__version__, **(meta or {})}
    joblib.dump(blob, path)


def load(path: str | Path) -> dict[str, Any]:
    _sklearn()
    import joblib

    p = Path(path).expanduser()
    if not p.is_file():
        raise SklearnUnavailable(f"stroke model not found: {p}")
    blob = joblib.load(p)
    fs = feature_set_of(blob.get("features", []))
    if fs is None:
        raise SklearnUnavailable(f"{p}: trained on different features {blob.get('features')}; retrain it")
    blob["feature_set"] = fs
    return blob


def predict_proba(model: Any, feats: Sequence[StrokeFeatures], X: np.ndarray | None = None) -> list[dict[str, float]]:
    """One {type: p} per shot over all 12 types (types the model never saw get 0).

    ``X`` is the model input when it is not the base feature matrix (feature set v2)."""
    if not feats:
        return []
    P = model.predict_proba(feature_matrix(feats) if X is None else X)
    classes = [str(c) for c in model.classes_]
    out = []
    for row in P:
        d = {t: 0.0 for t in SHOT_TYPES}
        for c, p in zip(classes, row):
            if c in d:
                d[c] = float(p)
        out.append(d)
    return out


def make_model_fn(model_path: str | Path):
    """A ``classify`` model backend from a saved model file."""
    blob = load(model_path)
    model, fs = blob["model"], blob["feature_set"]

    def fn(events, shuttle, players, feats):
        X = full_matrix(feats, events, shuttle, players, fs)
        return {f.shot_id: p for f, p in zip(feats, predict_proba(model, feats, X))}

    return fn
