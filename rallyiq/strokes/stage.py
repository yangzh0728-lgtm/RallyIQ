"""Stage ``strokes``: stroke type for every raw ``events`` shot.

Contract: ``run(ctx) -> StrokesResult`` using ``ctx.events``, ``ctx.shuttle`` and
``ctx.players``. Options in ``ctx.config["strokes"]``:

* ``classifier``: ``"auto"`` (default: bundled ``models/strokes_v3.json``, numpy only, combined with
  the court model ``models/strokes_court_v1.json`` and rally smoothing of ``court.py``; ``court: false``
  or ``sequence: false`` turn those off; the rules if no model is there), ``"rules"``, ``"bst"`` (see ``bst.py``) or ``"sklearn"`` (see ``learned.py``); a
  model backend that cannot load falls back to the rules with a warning unless ``require_model: true``;
* ``sklearn``: ``{"model": "/path/to/strokes.json"}`` (or ``.joblib``; trained by ``scripts/learn_strokes.py``);
* ``bst``: BST settings (``repo``, ``weights``, ``model_name``, ``pose_style``, ``seq_len``);
* ``detail``: ``true`` (default) adds hand / direction / contact / pace / role / intent to every
  classified shot (``detail.py``, bundled ``models/stroke_attrs.json``; rules-only layers if missing);
* any field of :class:`~rallyiq.strokes.classify.StrokesConfig` (thresholds).
"""

from __future__ import annotations

import logging
from dataclasses import fields, replace

from ..contracts import StrokesResult
from ..pipeline.context import StageContext
from .classify import RULES_NAME, StrokesConfig, classify

log = logging.getLogger("rallyiq.strokes")


def _config(opts: dict) -> StrokesConfig:
    known = {f.name for f in fields(StrokesConfig)} - {"features", "rules"}
    return replace(StrokesConfig(), **{k: v for k, v in opts.items() if k in known})


def run(ctx: StageContext) -> StrokesResult:
    meta = ctx.require_meta()
    events = ctx.events
    if events is None or not events.shots:
        return StrokesResult(video_id=meta.video_id, classifier=None, shots=[])
    opts = dict(ctx.config.get("strokes") or {})
    cfg = _config(opts)
    ctx.report(0.1)

    detail, attrs = bool(opts.get("detail", True)), None
    if detail or cfg.classifier == "auto":
        from .detail import get_model as get_attrs

        try:
            attrs = get_attrs()
        except (ValueError, KeyError) as e:
            log.warning("bundled stroke attribute model unusable (%s); rules-only detail", e)

    model = name = None
    if cfg.classifier == "auto":
        from .learned import BUNDLED, GBT_NAME, SklearnUnavailable
        from .learned import make_model_fn as make_learned_fn

        if BUNDLED.is_file():
            try:
                model, name = make_learned_fn(BUNDLED), GBT_NAME
            except (SklearnUnavailable, ValueError, KeyError) as e:
                log.warning("bundled stroke model unusable (%s); using %s", e, RULES_NAME)
        if opts.get("court", True):
            from . import court as CM

            try:
                court = CM.get_model()
            except (ValueError, KeyError) as e:
                court = None
                log.warning("bundled court stroke model unusable (%s)", e)
            if court is not None:
                model = CM.make_model_fn(model, court, attrs, meta.fps, seq=bool(opts.get("sequence", True)))
                name = f"{name}+{CM.NAME}" if name else CM.NAME
    elif cfg.classifier == "bst":
        from .bst import BSTConfig, BSTUnavailable, make_model_fn

        bst_opts = opts.get("bst") or {}
        bcfg = BSTConfig(**{k: v for k, v in bst_opts.items() if k in {f.name for f in fields(BSTConfig)}})
        try:
            model = make_model_fn(bcfg, meta.width, meta.height)
            name = f"bst:{bcfg.model_name}@shuttleset25"
        except BSTUnavailable as e:
            if opts.get("require_model"):
                raise
            log.warning("BST unavailable (%s); using %s", e, RULES_NAME)
    elif cfg.classifier == "sklearn":
        from .learned import SKLEARN_NAME, SklearnUnavailable
        from .learned import make_model_fn as make_sklearn_fn

        path = (opts.get("sklearn") or {}).get("model")
        try:
            if not path:
                raise SklearnUnavailable("strokes.sklearn.model (path to a .joblib file) is not set")
            model = make_sklearn_fn(path)
            name = SKLEARN_NAME
        except SklearnUnavailable as e:
            if opts.get("require_model"):
                raise
            log.warning("sklearn stroke model unavailable (%s); using %s", e, RULES_NAME)
    elif cfg.classifier != "rules":
        raise ValueError(f"unknown strokes classifier {cfg.classifier!r}")

    result = classify(events, ctx.shuttle, ctx.players, meta.fps, cfg, model=model, model_name=name,
                      detail=detail, attrs=attrs)
    ctx.report(0.95)
    return result
