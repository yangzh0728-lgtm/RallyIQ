"""Footwork analysis (脚步分析) from the ankle keypoints projected onto the court.

``analyze_footwork(events, players, calibration, fps)`` returns a ``FootworkResult``: foot contacts,
steps, split steps, jumps, lunges, step patterns and recovery, each linked to the effective hits.
``Footwork`` is the same analysis as an object, used by the match statistics and the shot advice.
"""

from .analyze import BASE, BASE_RADIUS_M, PATTERN_ZH, Footwork, analyze_footwork, summarize

__all__ = ["BASE", "BASE_RADIUS_M", "PATTERN_ZH", "Footwork", "analyze_footwork", "summarize"]
