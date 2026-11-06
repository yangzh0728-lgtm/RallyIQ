"""Rule-based rally segmentation and hit-candidate detection (MVP).

Pipeline entry is ``rallyiq.events.stage:run``. For direct use::

    from rallyiq.events import segment_events
    rallies = segment_events(shuttle_obs, pose_frames)
"""

from .config import EventsConfig
from .pipeline import segment_events
from .types import (
    HitEvent,
    Keypoint,
    PlayerPose,
    PoseFrame,
    RallyEndReason,
    RallySegment,
    ReviewStatus,
    ShuttleObs,
)

__all__ = [
    "EventsConfig",
    "HitEvent",
    "Keypoint",
    "PlayerPose",
    "PoseFrame",
    "RallyEndReason",
    "RallySegment",
    "ReviewStatus",
    "ShuttleObs",
    "segment_events",
]
