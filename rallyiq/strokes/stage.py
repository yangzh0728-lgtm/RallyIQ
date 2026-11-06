"""Stage ``strokes``: stroke type for every raw ``events`` shot.

Contract: ``run(ctx) -> StrokesResult`` using ``ctx.events``, ``ctx.shuttle`` and
``ctx.players``. Options in ``ctx.config["strokes"]``:

* ``classifier``: ``"rules"`` (default) or ``"bst"`` (see ``bst.py``; falls back to the
  rules with a warning unless ``require_model: true``);
* ``bst``: BST settings (``repo``, ``weights``, ``model_name``, ``pose_style``, ``seq_len``);
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

    model = name = None
    if cfg.classifier == "bst":
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
    elif cfg.classifier != "rules":
        raise ValueError(f"unknown strokes classifier {cfg.classifier!r}")

    result = classify(events, ctx.shuttle, ctx.players, meta.fps, cfg, model=model, model_name=name)
    ctx.report(0.95)
    return result
