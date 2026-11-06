"""Internal data types for the player module.

These mirror the shared contract (``rallyiq.contracts.PlayerDetection`` /
``PlayerFrame``) but stay numpy-friendly for the hot path. ``stage.py``
converts them into the contract models at the boundary.

Court frame (shared with calibration): metres, origin at a court corner,
x across the court 0..6.10, y along the court 0..13.40, y=0 is the baseline
nearer the camera, net at y=6.70.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...contracts.models import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, Side  # noqa: F401

# COCO-17 keypoint order.
COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP = {name: i for i, name in enumerate(COCO17)}


@dataclass
class PersonDetection:
    """One person found by the pose estimator in one frame (image pixels)."""

    bbox: np.ndarray  # (4,) xyxy
    keypoints: np.ndarray  # (17, 3) x, y, conf
    score: float

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(17, 3)
        self.score = float(self.score)

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


@dataclass
class Candidate:
    """A detection enriched with foot / court position and a player-likeness score."""

    det: PersonDetection
    foot_px: np.ndarray  # (2,)
    foot_court: np.ndarray | None  # (2,) metres, None without calibration
    on_court: bool | None  # None without calibration
    player_score: float  # 0..1, how likely this is an active player
    reasons: list[str] = field(default_factory=list)  # why it was penalised
    appearance: np.ndarray | None = None  # normalised colour histogram


@dataclass
class PlayerState:
    """Tracker output for one player in one frame."""

    player_id: str | None  # "P1".."P4", stable across side switches
    track_id: int  # raw tracker slot id
    side: Side | None
    bbox: np.ndarray
    keypoints: np.ndarray
    foot_px: np.ndarray
    foot_court: np.ndarray | None
    on_court: bool | None
    confidence: float
    interpolated: bool = False  # True when filled across a short gap, not observed
    team: str | None = None  # "A" (P1/P3) or "B" (P2/P4)
    appearance: np.ndarray | None = None  # torso colour histogram (internal, not in the contract)


@dataclass
class PlayerFrameResult:
    frame: int
    t_ms: float
    players: list[PlayerState]
