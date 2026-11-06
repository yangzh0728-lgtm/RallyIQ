"""Personal long-term memory (README M5) and opponent reports (M6).

* ``PeopleAssignment``: who P1..P4 are in one video (names), and which one is "me".
* ``AnalysisEvent``: append-only log entry. Committing a reviewed match writes one
  event per named player; withdrawing writes a ``match_withdrawn`` event. Nothing
  in the log is ever edited, so a new upload never overwrites old evidence.
* ``PlayerProfile``: recomputed from the log on read (series per metric, trend,
  changes, and retests of the training items suggested last time).
* ``OpponentReport``: conditional shot habits of one person across their videos,
  each with sample size, confidence, timestamped evidence and counter-examples.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .models import (
    SCHEMA_VERSION,
    Contract,
    Distribution,
    Metric,
    MetricConfidence,
    NoteRef,
    PlayerId,
    Status,
    TrainingItem,
)


class PeopleAssignment(Contract):
    """``PUT /api/videos/{id}/people``. Unnamed player ids are left out of the profile."""

    video_id: str
    people: dict[PlayerId, str] = Field(default_factory=dict, description="player id -> person name")
    self_player: PlayerId | None = Field(default=None, description="which player is the user (我)")
    updated_at: datetime | None = None


class ShotRecord(Contract):
    """Snapshot of one of this player's shots at commit time (evidence for opponent habits)."""

    shot_id: str
    rally_id: str | None = None
    t_ms: int
    index_in_rally: int | None = Field(default=None, description="0 = serve")
    shot_type: str | None = None
    shot_type_status: Status | None = None
    zone: Literal["后场", "中场", "前场"] | None = Field(default=None, description="hitter depth in own half")
    prev_shot_type: str | None = Field(default=None, description="the opponent's shot just before, in the same rally")
    status: Status


AnalysisEventKind = Literal["match_committed", "match_withdrawn"]


class AnalysisEvent(Contract):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    seq: int
    kind: AnalysisEventKind
    recorded_at: datetime
    video_id: str
    filename: str | None = None
    video_created_at: datetime | None = None
    person: str | None = None
    player_id: PlayerId | None = None
    is_self: bool = False
    opponents: list[str] = Field(default_factory=list)
    partner: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    stats_fingerprint: str | None = None
    metrics: list[Metric] = Field(default_factory=list, description="this player's metrics, keys without the 'P1.' prefix")
    distributions: list[Distribution] = Field(default_factory=list)
    match_metrics: list[Metric] = Field(default_factory=list)
    training: list[TrainingItem] = Field(default_factory=list, description="coach suggestions for this player at commit time")
    shots: list[ShotRecord] = Field(default_factory=list)
    corrections: int = 0
    confirmation: Literal["human_committed"] = "human_committed"
    model_versions: dict[str, str] = Field(default_factory=dict)


Trend = Literal["up", "down", "flat", "new", "insufficient"]


class ProfilePoint(Contract):
    video_id: str
    filename: str | None = None
    recorded_at: datetime
    value: float | None
    n: int
    status: Status


class ProfileMetric(Contract):
    key: str
    label: str
    unit: str | None = None
    points: list[ProfilePoint]
    latest: float | None = None
    previous: float | None = None
    delta: float | None = None
    trend: Trend
    confidence: MetricConfidence
    note: str | None = None


class RetestResult(Contract):
    """A training item suggested after one match, measured again on the next one."""

    drill: str
    why: str
    metric: str
    label: str | None = None
    unit: str | None = None
    from_video_id: str
    to_video_id: str | None = None
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    focus: str | None = None
    sources: list[NoteRef] = Field(default_factory=list, description="coaching notes the drill came from")
    same_conditions: bool | None = Field(default=None, description="camera view and frame rate match (README: 同条件下复测)")
    status: Literal["improved", "worse", "changed", "unchanged", "pending", "not_measurable"] = Field(
        description="improved/worse only where higher is clearly better (win and coverage rates); "
        "otherwise 'changed' with the delta")


class ProfileVideo(Contract):
    video_id: str
    filename: str | None = None
    recorded_at: datetime
    player_id: PlayerId | None = None
    opponents: list[str] = Field(default_factory=list)
    conditions: dict[str, str] = Field(default_factory=dict)


class PlayerProfile(Contract):
    """``GET /api/profile``: recomputed from the append-only log."""

    schema_version: str = SCHEMA_VERSION
    person: str
    is_self: bool
    videos: list[ProfileVideo]
    metrics: list[ProfileMetric]
    retests: list[RetestResult] = Field(default_factory=list)
    changes: list[str] = Field(default_factory=list, description="新增 / 冲突 / 趋势, plain Chinese lines")
    log_events: int = 0


class HabitEvidence(Contract):
    video_id: str
    shot_id: str
    t_ms: int


class Habit(Contract):
    """One conditional tendency: "when <condition>, plays <outcome> p of the time"."""

    condition: str = Field(description="Chinese, e.g. '在后场击球时'")
    outcome: str = Field(description="Chinese, e.g. '杀球'")
    p: float
    n: int = Field(description="shots matching the condition")
    hits: int
    confidence: MetricConfidence
    evidence: list[HabitEvidence] = Field(default_factory=list)
    counter_examples: list[HabitEvidence] = Field(default_factory=list)


class OpponentReport(Contract):
    """``GET /api/opponents/{name}``: conditional habits over every committed video of one person."""

    schema_version: str = SCHEMA_VERSION
    person: str
    videos: list[ProfileVideo]
    shots: int
    typed_shots: int
    distributions: list[Distribution] = Field(default_factory=list)
    habits: list[Habit] = Field(default_factory=list, description="top tendencies, strongest first")
    next_shot: list[Habit] = Field(default_factory=list, description="reply to my previous shot type -> their next shot")
    caveats: list[str] = Field(default_factory=list)
