"""Tunable thresholds for the rule-based event segmentation.

Distances are expressed relative to the on-screen height of a player
("player heights", H) so the same rules work for near and far players and for
any resolution. Times are milliseconds so the rules do not depend on fps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventsConfig:
    # --- trajectory cleaning ---
    outlier_jump_rel: float = 0.8  # a lone point this far (in H) from both neighbours is dropped
    min_shuttle_conf: float = 0.2

    # --- rally segmentation ---
    end_gap_ms: float = 1200.0  # shuttle unseen this long -> rally over
    static_ms: float = 600.0  # shuttle (nearly) still this long -> rally over
    static_speed_rel: float = 0.35  # H per second below which the shuttle counts as still
    min_rally_ms: float = 700.0
    min_rally_points: int = 8

    # --- hit candidates ---
    window_ms: float = 130.0  # span used for incoming / outgoing velocity
    max_window_ms: float = 450.0  # how far to reach across an occlusion for a velocity
    min_turn_deg: float = 35.0
    full_turn_deg: float = 110.0  # turn angle that saturates the turn score
    speed_jump_ratio: float = 1.8
    min_hit_gap_ms: float = 280.0  # two hits closer than this are one hit
    max_hitter_dist_rel: float = 0.9  # shuttle further than this (in H) from a racket proxy: no hitter
    racket_extension: float = 0.7  # racket head ~ wrist + k * (wrist - elbow)
    min_kp_conf: float = 0.3
    pose_search_frames: int = 3  # look this many frames around a hit for a pose
    landing_window_ms: float = 250.0  # direction change this close to a static end is a landing

    # --- learned hit scorer (learned.py) ---
    # "auto": bundled model (models/hit_scorer.json) if present; "off": hand-weighted rules; or a JSON path.
    # The model's own thresholds (keep / ok / min_candidate) replace the three below unless overridden.
    hit_scorer: str = "auto"

    # --- decisions ---
    min_candidate_conf: float = 0.3  # below this a candidate is not reported at all
    keep_threshold: float = 0.45  # candidates below this survive only if alternation needs them
    low_track_quality: float = 0.35
    ok_threshold: float = 0.6
    hitter_margin: float = 0.15  # proximity gap below which the hitter is ambiguous
    occlusion_gap_ms: float = 150.0  # no observation this close around a hit -> occluded
    long_exchange_ms: float = 2600.0  # no hit for this long inside a rally -> maybe missed one
    default_player_height_px: float = 200.0
