"""Shared data contracts for every RallyIQ module.

Conventions (apply to every model below):

* Time: ``frame`` is the 0-based frame index in the original video and ``t_ms``
  is the presentation time of that frame in milliseconds of the original video.
* Image coordinates: pixels in the original video resolution (after applying the
  container rotation), origin top-left, x right, y down.
* Court coordinates ("court metres"): origin at a court corner, ``x`` across the
  court 0..6.10, ``y`` along the court 0..13.40. ``y = 0`` is the baseline nearer
  the camera when that can be determined; the net is at ``y = 6.70``.
* Nullable fields mean "unknown". Never fill a default value that looks like a
  real measurement.
* ``status`` follows the README levels: ``ok`` (usable), ``needs_review``
  (待复核), ``undeterminable`` (不可判定).
* ``source`` separates model output from interpolation and human edits; raw
  model predictions are never overwritten.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.2.0"

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
NET_Y_M = COURT_LENGTH_M / 2

# COCO-17 keypoint order used by PlayerDetection.keypoints.
COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

Point2 = Annotated[list[float], Field(min_length=2, max_length=2, description="[x, y]")]
Keypoint = Annotated[list[float], Field(min_length=3, max_length=3, description="[x, y, conf]")]
BBox = Annotated[list[float], Field(min_length=4, max_length=4, description="[x1, y1, x2, y2] px")]
Matrix3 = Annotated[
    list[Annotated[list[float], Field(min_length=3, max_length=3)]],
    Field(min_length=3, max_length=3, description="3x3 row-major matrix"),
]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Status(str, Enum):
    ok = "ok"
    needs_review = "needs_review"
    undeterminable = "undeterminable"


class Source(str, Enum):
    model = "model"
    human = "human"


PlayerId = Literal["P1", "P2", "P3", "P4"]
TeamId = Literal["A", "B"]
Side = Literal["near", "far"]
MatchFormat = Literal["singles", "doubles"]

SINGLES_PLAYER_IDS: list[str] = ["P1", "P2"]
DOUBLES_PLAYER_IDS: list[str] = ["P1", "P2", "P3", "P4"]
# Fixed team membership: team A = P1 + P3 (the pair first seen on the near half),
# team B = P2 + P4. In singles each team has one player.
TEAM_OF: dict[str, str] = {"P1": "A", "P3": "A", "P2": "B", "P4": "B"}


def team_of(player_id: str | None) -> str | None:
    return TEAM_OF.get(player_id) if player_id else None


# --------------------------------------------------------------------------- video


class VideoMeta(Contract):
    """Probe result for an uploaded video (stage ``probe``)."""

    video_id: str
    filename: str
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(ge=0)
    codec: str | None = None
    rotation: int = 0
    variable_frame_rate: bool = False

    def t_ms(self, frame: int) -> int:
        return round(frame * 1000.0 / self.fps)

    def frame_at(self, t_ms: float) -> int:
        return int(round(t_ms * self.fps / 1000.0))


# --------------------------------------------------------------------- calibration


class CalibrationSegment(Contract):
    """One stable camera pose. A new segment starts when the camera moves/cuts."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0, description="inclusive")
    homography: Matrix3 | None = Field(
        description="image px -> court metres; null when calibration failed"
    )
    image_corners: list[Point2] | None = Field(
        default=None,
        description="outer court corners in px, order: near-left, near-right, far-right, far-left",
    )
    keypoints: list[CourtKeypoint] = Field(default_factory=list)
    other_courts: list[Annotated[list[Point2], Field(min_length=4, max_length=4)]] = Field(
        default_factory=list,
        description="other courts detected in view (not analysed), outer corners in px, same order as "
        "image_corners; used to reject their players and shuttles and to offer switching court",
    )
    reprojection_error_px: float | None = None
    method: Literal["auto", "manual"] = "auto"
    mirrored: bool = False
    confidence: Confidence = 0.0
    status: Status = Status.needs_review
    notes: list[str] = Field(default_factory=list)


class CourtKeypoint(Contract):
    name: str = Field(description="e.g. 'near_left_corner', 'net_left_post'")
    image: Point2
    court: Point2
    used_for_fit: bool = True


class CourtCalibration(Contract):
    """Stage ``calibration`` output."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    segments: list[CalibrationSegment]

    def segment_for(self, frame: int) -> CalibrationSegment | None:
        for seg in self.segments:
            if seg.start_frame <= frame <= seg.end_frame:
                return seg
        return None

    def homography_for(self, frame: int) -> list[list[float]] | None:
        seg = self.segment_for(frame)
        return seg.homography if seg else None


# ------------------------------------------------------------------------- players


class PlayerDetection(Contract):
    player_id: PlayerId | None = Field(
        default=None, description="stable identity across side switches; null if unresolved"
    )
    track_id: int = Field(description="raw tracker id")
    team: TeamId | None = Field(default=None, description="A = P1/P3, B = P2/P4; null if unresolved")
    side: Side | None = None
    bbox: BBox
    keypoints: list[Keypoint] = Field(
        default_factory=list, description="17 COCO keypoints [x, y, conf] px, or empty"
    )
    foot_px: Point2 | None = None
    foot_court: Point2 | None = None
    on_court: bool | None = None
    confidence: Confidence
    interpolated: bool = Field(default=False, description="true for short-gap fills, not a detection")


class PlayerFrame(Contract):
    frame: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    players: list[PlayerDetection] = Field(default_factory=list)


class PlayerTracks(Contract):
    """Stage ``players`` output. Frames without detections may be omitted."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    model: str | None = Field(default=None, description="pose model name@version")
    keypoint_format: Literal["coco17"] = "coco17"
    match_format: MatchFormat = Field(
        default="singles", description="singles: P1/P2; doubles: P1+P3 (team A) vs P2+P4 (team B)"
    )
    format_confidence: Confidence | None = Field(
        default=None, description="how sure the automatic singles/doubles decision is; null if set by config"
    )
    frames: list[PlayerFrame]
    side_switches: list[int] = Field(default_factory=list, description="frames where the teams swap sides")


# ------------------------------------------------------------------------- shuttle


class ShuttlePoint(Contract):
    frame: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    x: float
    y: float
    confidence: Confidence
    source: Literal["detected", "interpolated"] = "detected"
    court_xy: Point2 | None = Field(
        default=None, description="only if a ground-plane position is actually known"
    )


class FrameGap(Contract):
    start_frame: int
    end_frame: int
    reason: Literal["not_detected", "out_of_frame", "occluded", "unknown"] = "unknown"


class ShuttleTrack(Contract):
    """Stage ``shuttle`` output. Frames with no point = shuttle not visible."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    detector: str | None = Field(default=None, description="detector name@version")
    points: list[ShuttlePoint]
    gaps: list[FrameGap] = Field(default_factory=list)


# -------------------------------------------------------------------------- events


# Stroke types (击球类型). Same 12 classes as the merged ShuttleSet taxonomy (without the
# Top_/Bottom_ side prefix), so a ShuttleSet-trained model maps one to one.
ShotType = Literal[
    "short_serve",  # 发短球
    "long_serve",   # 发高远球
    "clear",        # 高远球
    "smash",        # 杀球
    "drop",         # 吊球 / 切球
    "net_shot",     # 放网
    "cross_net",    # 勾对角
    "push",         # 推球
    "net_kill",     # 扑球
    "drive",        # 平抽
    "lift",         # 挑球
    "block",        # 挡网（接杀挡）
]
SHOT_TYPES: list[str] = list(ShotType.__args__)


class ShotTypeScore(Contract):
    shot_type: ShotType
    p: Confidence


class StrokeAttr(Contract):
    """One layer of a shot's description, and where it comes from."""

    value: str
    p: Confidence | None = Field(default=None, description="model probability; null when measured or a rule")
    source: Literal["model", "measured", "rule"] = Field(
        description="model = learned from ShuttleSet labels; measured = from tracked positions; "
        "rule = a coach's reading of the other layers (no labels exist to learn it)"
    )


class StrokeDetail(Contract):
    """Layered description of a shot on top of its stroke type (see ``rallyiq.strokes.detail``)."""

    family: str | None = Field(default=None, description="clear smash drop drive block push lift net_shot net_kill serve")
    subtype: str | None = Field(default=None, description="cross_court_net_shot, short_serve or long_serve; else null")
    hand: StrokeAttr | None = Field(default=None, description="forehand, backhand or around_head")
    contact: StrokeAttr | None = Field(default=None, description="high (above the net) or low")
    direction: StrokeAttr | None = Field(default=None, description="straight or cross")
    pace: StrokeAttr | None = Field(default=None, description="fast, medium or slow for this family")
    role: StrokeAttr | None = Field(default=None, description="attacking, neutral or defensive")
    intent: StrokeAttr | None = Field(default=None, description="attack, move, control, defend or transition")
    description_zh: str = ""
    source: str | None = Field(default=None, description="attribute model name, e.g. attrs_v1")


class StrokeInfo(Contract):
    """Stroke-type prediction (or human label) for one shot."""

    shot_type: ShotType | None = Field(description="null = undeterminable from this video")
    confidence: Confidence
    status: Status = Status.needs_review
    source: Source = Source.model
    classifier: str | None = Field(default=None, description="e.g. rules_v1, bst@shuttleset25; null for human")
    candidates: list[ShotTypeScore] = Field(default_factory=list, description="top candidates, best first")
    reasons: list[str] = Field(default_factory=list, description="review reason codes")
    evidence: dict[str, float | None] = Field(
        default_factory=dict, description="measured features behind the prediction (null = not measurable)"
    )
    detail: StrokeDetail | None = Field(default=None, description="hand, direction, contact, pace, role, intent")


class ShotStroke(Contract):
    shot_id: str
    frame: int = Field(ge=0)
    stroke: StrokeInfo


class StrokesResult(Contract):
    """Stage ``strokes`` output: one stroke prediction per raw ``events`` shot."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    classifier: str | None = None
    shots: list[ShotStroke]


class ShotEvent(Contract):
    """A hit (击球) candidate. ``shot_type``/``stroke`` come from stage ``strokes``; no landing zone yet."""

    shot_id: str
    rally_id: str | None = None
    frame: int = Field(ge=0)
    t_ms: int = Field(ge=0)
    hitter: PlayerId | None = None
    hitter_confidence: Confidence | None = None
    shuttle_px: Point2 | None = None
    shuttle_court: Point2 | None = None
    hitter_court: Point2 | None = None
    shot_type: ShotType | None = Field(default=None, description="effective stroke type (= stroke.shot_type)")
    stroke: StrokeInfo | None = Field(default=None, description="stroke-type prediction or human label")
    target_zone: str | None = None
    confidence: Confidence
    status: Status = Status.needs_review
    source: Source = Source.model
    reasons: list[str] = Field(default_factory=list, description="why this was proposed")


RallyEndReason = Literal["shuttle_lost", "shuttle_static", "video_end", "manual", "unknown"]


class Rally(Contract):
    rally_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    shot_ids: list[str] = Field(default_factory=list)
    end_reason: RallyEndReason = "unknown"
    winner: PlayerId | TeamId | None = Field(
        default=None, description="singles: P1/P2; doubles: team A/B"
    )
    confidence: Confidence
    status: Status = Status.needs_review
    source: Source = Source.model


class EventsResult(Contract):
    """Stage ``events`` output (raw model prediction; corrections are layered on top)."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    match_format: MatchFormat = "singles"
    rallies: list[Rally]
    shots: list[ShotEvent]


# --------------------------------------------------------------------- corrections


CorrectionOp = Literal[
    "add_shot",       # payload: {frame, hitter?, shuttle_px?}
    "delete_shot",    # payload: {shot_id}
    "set_hitter",     # payload: {shot_id, hitter: "P1".."P4"|null}
    "move_shot",      # payload: {shot_id, frame}
    "add_rally",      # payload: {start_frame, end_frame}
    "delete_rally",   # payload: {rally_id}
    "set_rally_bounds",  # payload: {rally_id, start_frame?, end_frame?}
    "set_rally_winner",  # payload: {rally_id, winner}
    "swap_player_ids",   # payload: {from_frame, a?, b?} swap two ids (default P1/P2) from this frame on
    "confirm_shot",      # payload: {shot_id} mark a model shot as correct
    "confirm_rally",     # payload: {rally_id} mark a model rally as correct
    "set_shot_type",     # payload: {shot_id, shot_type: one of SHOT_TYPES | null}
]


class CorrectionIn(Contract):
    op: CorrectionOp
    payload: dict = Field(default_factory=dict)
    note: str | None = None
    hard_example: bool = Field(default=True, description="collect this edit as a hard example for evaluation")


class Correction(CorrectionIn):
    """Append-only human edit. Effective events = raw events + corrections in order."""

    correction_id: str
    video_id: str
    created_at: datetime
    seq: int


# ------------------------------------------------------------------------- overlay


class OverlayPlayer(Contract):
    player_id: PlayerId | None
    team: TeamId | None = None
    side: Side | None = None
    bbox: BBox
    keypoints: list[Keypoint]
    foot_court: Point2 | None = None
    interpolated: bool = False


class OverlayShuttle(Contract):
    x: float
    y: float
    source: Literal["detected", "interpolated"]
    confidence: Confidence


class OverlayFrame(Contract):
    """Everything the web player draws on one frame.

    Hits/rallies are not repeated here: take them from the effective events
    (``GET /api/videos/{id}/events``) so human corrections show up immediately.
    """

    frame: int
    t_ms: int
    players: list[OverlayPlayer] = Field(default_factory=list)
    shuttle: OverlayShuttle | None = None


class CourtLines(Contract):
    """Court line segments projected into image px for one calibration segment."""

    start_frame: int
    end_frame: int
    lines: list[Annotated[list[Point2], Field(min_length=2, max_length=2)]]
    boundary: list[Annotated[list[Point2], Field(min_length=2, max_length=2)]] = Field(
        default_factory=list, description="in-play rally boundary for the match format (singles or doubles sidelines)"
    )
    status: Status
    image_corners: list[Point2] | None = Field(
        default=None, description="outer corners of the analysed court, near-left, near-right, far-right, far-left")
    other_courts: list[Annotated[list[Point2], Field(min_length=4, max_length=4)]] = Field(
        default_factory=list, description="other courts in view (ignored); the UI lets the user switch to one")


class OverlayData(Contract):
    """Stage ``overlay`` output, served as-is to the web UI."""

    schema_version: str = SCHEMA_VERSION
    video: VideoMeta
    match_format: MatchFormat = "singles"
    court: list[CourtLines] = Field(default_factory=list)
    keypoint_format: Literal["coco17"] = "coco17"
    skeleton: list[Annotated[list[int], Field(min_length=2, max_length=2)]] = Field(
        default_factory=list, description="keypoint index pairs to draw as bones"
    )
    frames: list[OverlayFrame]


# ------------------------------------------------------------------- jobs & videos


StageName = Literal["probe", "calibration", "players", "shuttle", "events", "strokes", "overlay"]
STAGES: list[str] = ["probe", "calibration", "players", "shuttle", "events", "strokes", "overlay"]

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class StageRun(Contract):
    name: StageName
    state: Literal["pending", "running", "succeeded", "failed", "skipped"] = "pending"
    progress: Confidence = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class Job(Contract):
    job_id: str
    video_id: str
    state: JobState
    stages: list[StageRun]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class Video(Contract):
    video_id: str
    filename: str
    size_bytes: int
    created_at: datetime
    meta: VideoMeta | None = None
    latest_job: Job | None = None
    artifacts: list[StageName] = Field(default_factory=list)


class MatchFormatSetting(Contract):
    """Per-video choice of singles / doubles. ``auto`` lets the players stage decide."""

    match_format: Literal["auto", "singles", "doubles"] = "auto"


class HeatmapData(Contract):
    """Court occupancy grid in court metres for one player, one team, or everyone."""

    video_id: str
    player_id: PlayerId | None
    team: TeamId | None = None
    cell_m: float
    nx: int
    ny: int
    counts: list[list[int]] = Field(description="counts[iy][ix]; cell (ix, iy) covers x in [ix*cell, (ix+1)*cell)")
    samples: int



# -------------------------------------------------------------------------- report


MetricConfidence = Literal["high", "medium", "low"]


class Metric(Contract):
    """One computed number with its evidence size and reliability.

    ``value`` is null when the metric cannot be computed (e.g. no calibration, no
    winners entered). ``status`` follows the README levels; ``confidence`` is the
    plain-language reliability used by the coach to phrase caveats.
    """

    key: str = Field(description="stable dotted key, e.g. 'P1.movement.distance_m'")
    label: str = Field(description="Chinese display label")
    value: float | None
    unit: str | None = None
    n: int = Field(ge=0, description="sample size the value is computed from")
    status: Status
    confidence: MetricConfidence
    note: str | None = None


class Distribution(Contract):
    """Counts over named categories (e.g. shot types, court zones, rally lengths)."""

    key: str
    label: str
    counts: dict[str, int]
    n: int = Field(ge=0)
    status: Status
    confidence: MetricConfidence
    note: str | None = None


class PlayerStats(Contract):
    player_id: PlayerId
    metrics: list[Metric]
    distributions: list[Distribution] = Field(default_factory=list)


GamePhase = Literal["attack", "neutral", "defense"]


class StyleScore(Contract):
    """How well one player's numbers match one playing-style prototype (rules, no trained model)."""

    code: str = Field(description="style / system code, e.g. 'power_attack', 'rear_attack'")
    name: str = Field(description="Chinese name, e.g. '强攻压制型'")
    name_en: str
    score: float | None = Field(description="0-100 match to the prototype; null when too few features were measurable")
    coverage: float = Field(ge=0, le=1, description="share of the prototype's feature weight that could be measured")
    reasons: list[str] = Field(default_factory=list, description="the features that matched best, with their values")
    representatives: list[str] = Field(default_factory=list, description="well-known players of this style")


class TacticUse(Contract):
    """How often a player ran one tactical pattern, and how it went."""

    code: str = Field(description="space_control | net_to_attack | rally_to_attack | defence_to_attack | "
                                  "pattern_breaking | tempo_disruption")
    name: str
    uses: int = Field(ge=0)
    decided: int = Field(ge=0, description="uses in rallies with a known winner")
    won: int = Field(ge=0, description="of ``decided``, rallies the player (team) won")
    success_pct: float | None = Field(default=None, description="won / decided")
    forced_pct: float | None = Field(
        default=None, description="uses after which the opponent had to defend (lift / block) or lost the rally")
    examples_ms: list[int] = Field(default_factory=list, description="video times of up to 3 uses, to replay")


class ReplyHabit(Contract):
    """Next-shot tendency: after the opponent plays ``after``, this player most often replies ``reply``."""

    after: ShotType
    reply: ShotType
    n: int = Field(ge=0)
    of: int = Field(ge=0)
    pct: float


class PhaseChoice(Contract):
    """What a player did in one attack / neutral / defense state."""

    phase: GamePhase
    shots: int = Field(ge=0)
    counts: dict[str, int] = Field(default_factory=dict, description="stroke code -> count")
    recommended_pct: float | None = Field(
        default=None, description="share of these shots that were a recommended choice for the state")


class PlayerStyle(Contract):
    """Playing style of one singles player or one doubles team, from rule-based prototype matching.

    Features are ``Metric``s under ``<Pn|team>.style.*`` in ``MatchStats`` (cite those as evidence);
    this object holds the style scores, tactics, attack/neutral/defense choices and next-shot habits.
    """

    subject: str = Field(description="P1/P2 (singles) or A/B (doubles team)")
    players: list[PlayerId] = Field(default_factory=list)
    kind: Literal["singles", "doubles"]
    primary: StyleScore | None = None
    secondary: StyleScore | None = Field(default=None, description="set when it is within a few points of primary")
    scores: list[StyleScore] = Field(default_factory=list, description="every prototype, best first")
    label: str = Field(description="one-line Chinese verdict, e.g. '拉吊突击型（兼速度抢攻型）' or '证据不足'")
    phases: list[PhaseChoice] = Field(default_factory=list)
    tactics: list[TacticUse] = Field(default_factory=list)
    habits: list[ReplyHabit] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list, description="feature metric keys used")
    advice: list[str] = Field(default_factory=list, description="rule advice: execution, state choices, vs this opponent")
    n_shots: int = Field(ge=0, description="classified non-serve shots the style is based on")
    status: Status
    confidence: MetricConfidence
    note: str | None = None


class ShotMoment(Contract):
    """One shot whose choice should change, found by rules from stroke type + both players' positions.

    Positions are in each player's own-half frame (``lateral`` from their own left sideline as they
    face the net, ``depth`` from their own baseline: 0 = baseline, 6.70 = net). Zone names for the
    opponent and the target are given from the hitter's point of view (the hitter's left/right).
    """

    moment_id: str
    shot_id: str
    rally_id: str | None = None
    rally_no: int | None = Field(default=None, description="1-based rally number in the video")
    shot_no: int = Field(ge=1, description="1-based shot number in the rally")
    t_ms: int = Field(ge=0)
    player_id: PlayerId
    shot_type: ShotType
    shot_label_zh: str | None = Field(default=None, description="stroke with hand/direction, e.g. '正手直线杀球'")
    shot_detail_zh: str | None = Field(default=None, description="the layered description (StrokeDetail)")
    hitter_zone: str = Field(description="where the hitter stood, e.g. '后场左侧' (own view)")
    hitter_pos: Point2 = Field(description="hitter (lateral, depth) m, own-half frame")
    opponent_id: PlayerId
    opponent_zone: str = Field(description="where the opponent stood, from the hitter's view, e.g. '网前右侧'")
    opponent_pos: Point2 = Field(description="opponent (lateral, depth) m, opponent's own-half frame")
    opponent_reach_m: float | None = Field(default=None, description="how far the opponent moved to reach this shot")
    next_shot_type: ShotType | None = Field(default=None, description="the opponent's reply")
    outcome: Literal["won", "lost", "continued", "unknown"] = Field(
        description="won/lost = the hitter's side won/lost the rally within the next two shots"
    )
    phase: GamePhase | None = Field(
        default=None, description="the hitter's state at this shot: attack / neutral / defense (from the incoming ball)")
    issue: str = Field(description="rule code, e.g. 'to_opponent', 'clear_to_rear', 'short_to_front'")
    problem: str
    suggested_shot: ShotType | None = None
    target_zone: str | None = Field(default=None, description="where to send it instead, from the hitter's view")
    suggestion: str
    why: str
    severity: float = Field(ge=0)
    confidence: MetricConfidence


class RallyLine(Contract):
    """One rally of the match as the report lists it (rally log, score flow, tempo)."""

    rally_no: int = Field(ge=1, description="1-based, in time order")
    rally_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    shots: int = Field(ge=0)
    duration_s: float
    server: PlayerId | TeamId | None = Field(default=None, description="side of the first shot's hitter (team in doubles)")
    winner: PlayerId | TeamId | None = None
    last_hitter: PlayerId | None = None
    last_shot_type: str | None = Field(default=None, description="Chinese label of the last shot's stroke type")
    mean_interval_s: float | None = Field(default=None, description="mean time between consecutive hits, s")
    game_no: int | None = Field(default=None, description="game (局) this rally belongs to, from the running score")
    score: dict[str, int] | None = Field(
        default=None, description="game score after this rally, by side; null when the winner is not entered"
    )
    status: Status = Status.needs_review


class GameScore(Contract):
    """One game (局) reconstructed from the entered rally winners (21 points, win by 2, cap 30)."""

    game_no: int = Field(ge=1)
    score: dict[str, int]
    winner: PlayerId | TeamId | None = Field(default=None, description="null while the game is unfinished")
    rallies: int = Field(ge=0, description="rallies with a known winner in this game")


class MatchStats(Contract):
    """Post-match statistics computed from effective events + player tracks.

    Pure function of the inputs (no model is called). ``fingerprint`` changes
    whenever the inputs change, so a cached coach report can be marked stale.
    """

    schema_version: str = SCHEMA_VERSION
    video_id: str
    fingerprint: str
    duration_ms: int | None = None
    calibration_status: Status | None = None
    shot_types_available: bool = Field(description="false until a stroke classifier fills ShotEvent.shot_type")
    metrics: list[Metric]
    distributions: list[Distribution] = Field(default_factory=list)
    players: list[PlayerStats]
    moments: list[ShotMoment] = Field(
        default_factory=list, description="shots whose choice should change (singles, needs stroke types + tracks)"
    )
    styles: list[PlayerStyle] = Field(
        default_factory=list, description="playing style per player (singles) or team (doubles); needs stroke types")
    rallies: list[RallyLine] = Field(default_factory=list, description="rally log in time order")
    games: list[GameScore] = Field(default_factory=list, description="games reconstructed from entered winners")
    caveats: list[str] = Field(default_factory=list)


InsightKind = Literal["strength", "weakness", "pattern", "tactic", "data_quality"]


class CoachInsight(Contract):
    kind: InsightKind
    player_id: PlayerId | None = None
    title: str
    detail: str
    evidence: list[str] = Field(description="Metric/Distribution keys from MatchStats that support this insight")
    confidence: MetricConfidence
    grounded: bool = Field(default=True, description="false if a number in the text was not found in the cited evidence")
    warnings: list[str] = Field(default_factory=list)


class NoteRef(Contract):
    """A coaching-note citation (``knowledge/coaching/notes``). Link and title come from the
    note file, never from the model, so a citation always points at a real video."""

    note_id: str
    coach: str
    title: str
    url: str = Field(description="source video; with ``t_s`` it opens at that moment")
    t_s: int | None = Field(default=None, description="seconds into the video where the cited part starts, when known")
    drill: str | None = Field(default=None, description="name of the note's drill this advice uses, if any")


class TrainingItem(Contract):
    player_id: PlayerId | None = None
    drill: str
    why: str
    evidence: list[str] = Field(default_factory=list)
    retest_metric: str | None = Field(default=None, description="metric key to compare on the next upload")
    focus: str | None = Field(default=None, description="the weakness this item addresses, e.g. '失分多在网前球'")
    key_points: list[str] = Field(default_factory=list, description="technique points / cues taken from the cited notes")
    sources: list[NoteRef] = Field(default_factory=list, description="coaching notes this advice is based on")
    warnings: list[str] = Field(default_factory=list)


class ShotAdvice(Contract):
    """Coach advice on one ``ShotMoment``: time, positions and target come from the moment, never the model."""

    moment_id: str
    shot_id: str
    player_id: PlayerId
    rally_no: int | None = None
    shot_no: int
    t_ms: int = Field(ge=0)
    timestamp: str = Field(description="mm:ss (h:mm:ss over an hour) into the video")
    shot_type: ShotType
    suggested_shot: ShotType | None = None
    situation: str = Field(description="what happened: shot, both players' zones (from the moment)")
    advice: str
    why: str
    confidence: MetricConfidence
    source: Literal["model", "rules"] = Field(description="model = written by the coach provider; rules = rule text")
    warnings: list[str] = Field(default_factory=list)


class CoachReport(Contract):
    """LLM (or rule-based) coaching comments. Never a source of truth: every claim
    cites ``MatchStats`` keys and is validated against them."""

    schema_version: str = SCHEMA_VERSION
    video_id: str
    stats_fingerprint: str
    provider: str
    model: str | None = None
    created_at: datetime
    language: Literal["zh"] = "zh"
    summary: str
    insights: list[CoachInsight]
    training: list[TrainingItem] = Field(default_factory=list)
    shot_advice: list[ShotAdvice] = Field(default_factory=list, description="shot-by-shot changes, in video order")
    self_player: PlayerId | None = Field(default=None, description="the user (我), when known; advice is about them")
    caveats: list[str] = Field(default_factory=list)
    fallback_reason: str | None = Field(default=None, description="why the requested provider was not used")
    dropped: int = Field(default=0, description="insights removed because they cited no valid evidence")


class MatchReport(Contract):
    """``GET /api/videos/{id}/report``: stats plus the latest coach report, if any."""

    stats: MatchStats
    coach: CoachReport | None = None
    coach_stale: bool = Field(default=False, description="coach was generated from different stats (e.g. before new corrections)")

CalibrationSegment.model_rebuild()


# ------------------------------------------------------------------ input quality


CameraView = Literal["end_center", "end_oblique", "side", "unknown"]
CameraHeight = Literal["high", "medium", "low", "unknown"]
Capability = Literal["court_metrics", "players", "shuttle", "rallies", "hits"]


class QualityCheck(Contract):
    """One measured property of the input video (frame rate, camera view, blur, ...)."""

    key: str = Field(description="stable key, e.g. 'fps', 'camera_view', 'far_player_rate'")
    label: str = Field(description="Chinese display label")
    value: str | None = Field(description="human-readable value; null if it could not be measured")
    score: float | None = Field(default=None, description="raw number behind ``value``, when there is one")
    status: Status
    note: str | None = None


class CapabilityReliability(Contract):
    """How far one kind of output can be trusted on this particular video."""

    capability: Capability
    label: str
    status: Status
    reasons: list[str] = Field(default_factory=list, description="check keys / plain reasons behind the status")


class VideoQuality(Contract):
    """``GET /api/videos/{id}/quality``: input assessment derived from the probe and stage outputs.

    Says what kind of footage this is and, per output, whether it is usable,
    needs review, or cannot be determined here. Never blocks analysis.
    """

    schema_version: str = SCHEMA_VERSION
    video_id: str
    fingerprint: str = Field(description="changes when the inputs (artifacts) change")
    camera_view: CameraView = "unknown"
    camera_height: CameraHeight = "unknown"
    checks: list[QualityCheck]
    capabilities: list[CapabilityReliability]
    summary: str


# ------------------------------------------------------------------- hard examples


HardExampleModule = Literal["hits", "rallies", "players", "strokes", "labels"]


class HardExample(Contract):
    """One human correction marked ``hard_example``, with the model output it fixed and the
    video conditions it happened under. Collected across videos to build the evaluation set."""

    video_id: str
    filename: str | None = None
    correction_id: str
    op: str
    kind: str = Field(description="e.g. missed_hit, false_hit, hit_timing, wrong_hitter, missed_rally, "
                      "false_rally, rally_bounds, wrong_shot_type, id_swap, confirmed, label")
    module: HardExampleModule
    is_error: bool = Field(description="false for confirmations and plain labels (e.g. rally winner)")
    frame: int | None = None
    t_ms: int | None = None
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] | None = Field(default=None, description="the raw model item the correction refers to")
    conditions: dict[str, str] = Field(default_factory=dict, description="camera_view, camera_height, fps, resolution, ...")
    note: str | None = None


class ConditionGroup(Contract):
    """Error counts for one value of one condition (e.g. camera_view = side)."""

    condition: str
    value: str
    videos: int
    model_shots: int
    model_rallies: int
    errors: dict[str, int] = Field(description="module -> number of corrections that fixed a model error")
    confirmed: int
    hit_error_rate: float | None = Field(description="hit errors / model shots; null without shots")


class HardExampleSummary(Contract):
    """``GET /api/hard_examples/summary``: where the model fails most, per condition."""

    total: int
    errors: int
    videos: int
    by_kind: dict[str, int]
    groups: list[ConditionGroup]
    weakest: list[ConditionGroup] = Field(description="conditions with the highest hit error rate (enough shots)")
