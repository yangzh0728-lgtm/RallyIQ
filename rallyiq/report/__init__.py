"""Post-match report: statistics computed from events/tracks, and an AI coach on top.

* ``stats.compute_match_stats`` is a pure function of the effective events, player
  tracks and calibration. It never calls a model.
* ``coach.generate_coach_report`` turns ``MatchStats`` into Chinese coaching comments
  through a pluggable provider (Claude API by default, rule-based offline fallback),
  then checks every claim against the stats before it is shown.
* ``export.render_markdown`` renders both for download.
"""

from .coach import CoachUnavailable, generate_coach_report, list_providers, register_provider
from .export import render_markdown
from .stats import compute_match_stats

__all__ = [
    "CoachUnavailable", "compute_match_stats", "generate_coach_report", "list_providers",
    "register_provider", "render_markdown",
]
