"""Stroke-type classification (击球类型) for detected hits. See README.md."""

from .classify import RULES_NAME, StrokesConfig, classify, merge_strokes
from .features import StrokeFeatures, shot_features
from .taxonomy import EN, ZH

__all__ = ["RULES_NAME", "StrokesConfig", "classify", "merge_strokes", "StrokeFeatures", "shot_features", "EN", "ZH"]
