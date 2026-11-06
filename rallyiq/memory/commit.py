"""Build the ``AnalysisEvent``s written when the user commits a reviewed match to memory."""

from __future__ import annotations

from ..contracts import (
    CoachReport,
    EventsResult,
    MatchStats,
    PeopleAssignment,
    ShotRecord,
    Video,
    team_of,
)
from ..report.stats import DEPTH_ZONES, _zone, own_half


def _strip(key: str, pid: str) -> str:
    return key[len(pid) + 1:] if key.startswith(pid + ".") else key


def shot_records(events: EventsResult, pid: str) -> list[ShotRecord]:
    by_rally: dict[str | None, list] = {}
    for s in sorted(events.shots, key=lambda s: s.frame):
        by_rally.setdefault(s.rally_id, []).append(s)
    out = []
    for rid, shots in by_rally.items():
        for i, s in enumerate(shots):
            if s.hitter != pid:
                continue
            prev = shots[i - 1] if i > 0 and rid is not None else None
            zone = _zone(own_half(s.hitter_court)[1], DEPTH_ZONES) if s.hitter_court else None
            out.append(ShotRecord(
                shot_id=s.shot_id, rally_id=rid, t_ms=s.t_ms, index_in_rally=i if rid is not None else None,
                shot_type=s.shot_type, shot_type_status=s.stroke.status if s.stroke else None, zone=zone,
                prev_shot_type=prev.shot_type if prev is not None and prev.hitter != pid else None,
                status=s.status,
            ))
    return out


def build_commit(video: Video, people: PeopleAssignment, stats: MatchStats, events: EventsResult,
                 coach: CoachReport | None, conditions: dict[str, str], corrections: int) -> list[dict]:
    """One event dict per named player (MemoryLog.append fills ids and timestamps)."""
    named = {pid: name.strip() for pid, name in people.people.items() if name and name.strip()}
    if not named:
        raise ValueError("name at least one player before committing")
    per_player = {p.player_id: p for p in stats.players}
    out = []
    for pid, name in named.items():
        ps = per_player.get(pid)
        team = team_of(pid)
        opponents = [n for p, n in named.items() if team_of(p) != team]
        partner = next((n for p, n in named.items() if p != pid and team_of(p) == team), None)
        out.append(dict(
            kind="match_committed", video_id=video.video_id, filename=video.filename,
            video_created_at=video.created_at, person=name, player_id=pid, is_self=people.self_player == pid,
            opponents=opponents, partner=partner, conditions=conditions, stats_fingerprint=stats.fingerprint,
            metrics=[m.model_copy(update={"key": _strip(m.key, pid)}) for m in (ps.metrics if ps else [])],
            distributions=[d.model_copy(update={"key": _strip(d.key, pid)}) for d in (ps.distributions if ps else [])],
            match_metrics=stats.metrics,
            training=[t.model_copy(update={"retest_metric": _strip(t.retest_metric, pid) if t.retest_metric else None})
                      for t in (coach.training if coach else []) if t.player_id == pid],
            shots=shot_records(events, pid), corrections=corrections,
            model_versions={k: v for k, v in (("coach", coach.provider if coach else None),) if v},
        ))
    return out
