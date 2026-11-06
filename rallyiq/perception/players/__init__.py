"""Player detection, pose keypoints and stable identity tracking (singles).

Public entry points:

* ``track_players`` - run the full module over an iterable of frames.
* ``RTMOOnnx`` - default multi-person pose backend (ONNX).
* ``SinglesTracker`` / ``build_candidates`` - the building blocks, for tests
  and for other pose backends.
"""

from .filtering import FilterConfig, build_candidates
from .geometry import PlayArea, apply_homography, foot_point
from .models import COCO17, PersonDetection, PlayerFrameResult, PlayerState
from .pose import PoseEstimator, RTMOOnnx
from .runner import PlayerTrackingConfig, iter_video, track_players
from .tracker import SinglesTracker, TrackerConfig

__all__ = [
    "COCO17",
    "FilterConfig",
    "PersonDetection",
    "PlayArea",
    "PlayerFrameResult",
    "PlayerState",
    "PlayerTrackingConfig",
    "PoseEstimator",
    "RTMOOnnx",
    "SinglesTracker",
    "TrackerConfig",
    "apply_homography",
    "build_candidates",
    "foot_point",
    "iter_video",
    "track_players",
]
