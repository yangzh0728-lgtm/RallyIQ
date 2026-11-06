"""Input and output types for the events module.

The input types are a thin local adapter: they carry only the fields the
segmentation rules need. Upstream perception schemas are mapped onto them in
``adapters.py`` so the rules do not depend on how perception stores its data.

Conventions (from the project README):
- time is source-video milliseconds (``t_ms``), frames are 0-based indices;
- image coordinates are pixels with y pointing down;
- court coordinates are metres; ``None`` means "unknown", never a fake default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Sequence

# COCO-17 keypoint indices used by the rules.
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_ANKLE, KP_R_ANKLE = 15, 16


@dataclass(frozen=True)
class ShuttleObs:
    """One frame of shuttle tracking output."""

    frame: int
    t_ms: float
    x: Optional[float] = None  # px; None when not visible
    y: Optional[float] = None
    visible: bool = False
    confidence: float = 0.0
    interpolated: bool = False  # filled in by a tracker, not directly observed
    court_xy: Optional[tuple[float, float]] = None  # metres, if calibrated

    @property
    def usable(self) -> bool:
        return self.visible and self.x is not None and self.y is not None


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PlayerPose:
    """One player in one frame. ``keypoints`` follow COCO-17 order."""

    player_id: str
    keypoints: Sequence[Keypoint]
    side: Optional[str] = None  # "near" / "far" relative to camera, if known
    foot_court_xy: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class PoseFrame:
    frame: int
    t_ms: float
    players: Sequence[PlayerPose] = ()


class ReviewStatus(str, Enum):
    OK = "ok"  # passed all rules; still a model prediction, not ground truth
    NEEDS_REVIEW = "needs_review"
    UNDETERMINABLE = "undeterminable"


class RallyEndReason(str, Enum):
    SHUTTLE_LOST = "shuttle_lost"  # not seen for longer than the end gap
    SHUTTLE_STATIC = "shuttle_static"  # stopped moving (on the floor / in hand)
    VIDEO_END = "video_end"


@dataclass
class HitEvent:
    """A hit candidate (``ShotEvent`` precursor; no shot type in the MVP)."""

    hit_id: str
    rally_id: str
    index_in_rally: int
    frame: int
    t_ms: float
    hitter_id: Optional[str]
    shuttle_xy_px: Optional[tuple[float, float]]
    shuttle_court_xy: Optional[tuple[float, float]]
    confidence: float
    status: ReviewStatus
    reasons: list[str] = field(default_factory=list)
    # Rule scores, kept for debugging and for the review UI.
    scores: dict[str, float] = field(default_factory=dict)
    hitter_court_xy: Optional[tuple[float, float]] = None
    source: str = "rules_v1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class RallySegment:
    rally_id: str
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    end_reason: RallyEndReason
    hits: list[HitEvent] = field(default_factory=list)
    confidence: float = 0.0
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    reasons: list[str] = field(default_factory=list)
    source: str = "rules_v1"

    def to_dict(self) -> dict:
        return {
            "rally_id": self.rally_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "end_reason": self.end_reason.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "source": self.source,
            "hits": [h.to_dict() for h in self.hits],
        }
