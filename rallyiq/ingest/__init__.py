"""Input handling: video assessment (camera view, frame rate, picture quality)."""

from .quality import assess_quality, quality_for, sample_picture_stats

__all__ = ["assess_quality", "quality_for", "sample_picture_stats"]
