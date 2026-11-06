"""Player detection, pose keypoints and stable identity tracking (singles and doubles).

Public entry points:

* ``track_players`` - run the full module over an iterable of frames.
* ``RTMOOnnx`` - default multi-person pose backend (ONNX).
* ``decide_format`` - singles vs doubles from the first seconds.
* ``SinglesTracker`` / ``DoublesTracker`` / ``build_candidates`` - the building blocks, for tests
  and for other pose backends.
"""

from .filtering import FilterConfig, build_candidates
from .match_format import FormatConfig, FormatDecision, decide_format
from .geometry import PlayArea, apply_homography, foot_point
from .models import COCO17, PersonDetection, PlayerFrameResult, PlayerState
from .pose import PoseEstimator, RTMOOnnx
from .runner import PlayerTrackingConfig, PlayerTrackingOutput, iter_video, track_players
from .tracker import DoublesTracker, PlayerTracker, SinglesTracker, TrackerConfig

__all__ = [
    "COCO17",
    "DoublesTracker",
    "FilterConfig",
    "FormatConfig",
    "FormatDecision",
    "PersonDetection",
    "PlayArea",
    "PlayerFrameResult",
    "PlayerState",
    "PlayerTracker",
    "PlayerTrackingConfig",
    "PlayerTrackingOutput",
    "PoseEstimator",
    "RTMOOnnx",
    "SinglesTracker",
    "TrackerConfig",
    "apply_homography",
    "build_candidates",
    "decide_format",
    "foot_point",
    "iter_video",
    "track_players",
]
