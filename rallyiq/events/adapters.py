"""Map shared contracts (``rallyiq.contracts``) to and from the rule inputs/outputs."""

from __future__ import annotations

from typing import Optional

from ..contracts import EventsResult, PlayerTracks, Rally, ShotEvent, ShuttleTrack, team_of
from .types import HitEvent, Keypoint, PlayerPose, PoseFrame, RallySegment, ShuttleObs

_PLAYER_IDS = ("P1", "P2", "P3", "P4")


def calibration_segments(cal) -> list[tuple[int, int, bool]]:
    """(start_frame, end_frame, usable) per camera segment of a ``CourtCalibration`` (None -> [])."""
    if cal is None:
        return []
    return [(sg.start_frame, sg.end_frame, sg.homography is not None and str(getattr(sg.status, "value", sg.status))
             != "undeterminable") for sg in cal.segments]


def shuttle_from_contract(track: Optional[ShuttleTrack]) -> list[ShuttleObs]:
    if track is None:
        return []
    return [
        ShuttleObs(
            frame=p.frame,
            t_ms=float(p.t_ms),
            x=p.x,
            y=p.y,
            visible=True,
            confidence=p.confidence,
            interpolated=p.source == "interpolated",
            court_xy=tuple(p.court_xy) if p.court_xy else None,
        )
        for p in track.points
    ]


def poses_from_contract(tracks: Optional[PlayerTracks]) -> list[PoseFrame]:
    if tracks is None:
        return []
    doubles = tracks.match_format == "doubles"
    frames = []
    for pf in tracks.frames:
        players = []
        for d in pf.players:
            if d.on_court is False:
                continue  # spectators, umpire, players on the next court
            players.append(
                PlayerPose(
                    # Unresolved identities still count for proximity, but are
                    # never reported as a hitter (see ``events_to_contract``).
                    player_id=d.player_id or f"track{d.track_id}",
                    keypoints=[Keypoint(k[0], k[1], k[2]) for k in d.keypoints],
                    side=d.side,
                    foot_court_xy=tuple(d.foot_court) if d.foot_court else None,
                    # Singles alternates by player, doubles by team.
                    team=(d.team or team_of(d.player_id)) if doubles else None,
                )
            )
        frames.append(PoseFrame(pf.frame, float(pf.t_ms), players))
    return frames


def _pair(v) -> Optional[list[float]]:
    return [round(float(v[0]), 3), round(float(v[1]), 3)] if v is not None else None


def _shot(h: HitEvent, shot_id: str, rally_id: str) -> ShotEvent:
    hitter = h.hitter_id if h.hitter_id in _PLAYER_IDS else None
    reasons = list(h.reasons)
    if h.hitter_id is not None and hitter is None:
        reasons.append("hitter_identity_unresolved")
    return ShotEvent(
        shot_id=shot_id,
        rally_id=rally_id,
        frame=h.frame,
        t_ms=int(round(h.t_ms)),
        hitter=hitter,
        hitter_confidence=(round(h.scores.get("proximity", 0.0), 3) if hitter else None),
        shuttle_px=_pair(h.shuttle_xy_px),
        shuttle_court=_pair(h.shuttle_court_xy),
        hitter_court=_pair(h.hitter_court_xy) if hitter else None,
        confidence=max(0.0, min(1.0, h.confidence)),
        status="needs_review" if "hitter_identity_unresolved" in reasons else h.status.value,
        source="model",
        reasons=reasons,
    )


def events_to_contract(video_id: str, rallies: list[RallySegment], match_format: str = "singles") -> EventsResult:
    out_rallies: list[Rally] = []
    shots: list[ShotEvent] = []
    for n, r in enumerate(rallies, start=1):
        rally_id = f"r{n:03d}"
        ids = []
        for h in r.hits:
            sid = f"s{len(shots) + 1:04d}"
            shots.append(_shot(h, sid, rally_id))
            ids.append(sid)
        out_rallies.append(
            Rally(
                rally_id=rally_id,
                start_frame=r.start_frame,
                end_frame=r.end_frame,
                start_ms=int(round(r.start_ms)),
                end_ms=int(round(r.end_ms)),
                shot_ids=ids,
                end_reason=r.end_reason.value,
                winner=None,  # point attribution is not decided by these rules
                confidence=max(0.0, min(1.0, r.confidence)),
                status=r.status.value,
                source="model",
            )
        )
    return EventsResult(video_id=video_id, match_format=match_format, rallies=out_rallies, shots=shots)
