"""Match flow: rally log, score progression and tempo (no model calls).

Everything here is derived from the *effective* events (after corrections):

- **Rally log** (``RallyLine``): one row per rally with shot count, duration, server, winner, the last shot and
  the mean time between hits, so the report can list rallies and jump back to the video.
- **Score flow** (``GameScore`` + ``RallyLine.score``): the running score rebuilt from the entered rally winners.
  Games follow the badminton rules (21 points, win by 2, capped at 30). Rallies without a winner do not
  move the score. From the flow come lead changes, the biggest lead and each side's longest scoring run.
- **Tempo**: the time between consecutive hits by opposite sides. For a player, *preparation time* is the time
  they had between the opponent's shot and their own (low = they were rushed), *time given* is the time their
  shot left the opponent (low = they put pressure on). Split into the first and second half of the match, the
  same numbers show whether play slowed down.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..contracts import EventsResult, GameScore, Metric, RallyLine, ShotEvent, team_of

GAME_POINTS = 21
GAME_CAP = 30
# Plausible time between two hits by opposite sides (s). Shorter is a duplicate detection, longer a missed hit.
MIN_INTERVAL_S = 0.15
MAX_INTERVAL_S = 3.0


def _side_fn(events: EventsResult) -> Callable[[str | None], str | None]:
    doubles = events.match_format == "doubles"
    return lambda x: ((team_of(x) or x) if doubles else x) if x else None


def sides_of(events: EventsResult) -> tuple[str, str]:
    return ("A", "B") if events.match_format == "doubles" else ("P1", "P2")


def _intervals(shots: list[ShotEvent], side) -> list[tuple[ShotEvent, ShotEvent, float]]:
    """(previous shot, next shot, seconds) for consecutive hits by opposite known sides."""
    out = []
    for a, b in zip(shots, shots[1:]):
        sa, sb = side(a.hitter), side(b.hitter)
        if sa is None or sb is None or sa == sb:
            continue
        dt = (b.t_ms - a.t_ms) / 1000
        if MIN_INTERVAL_S <= dt <= MAX_INTERVAL_S:
            out.append((a, b, dt))
    return out


def _game_over(score: dict[str, int]) -> bool:
    hi, lo = max(score.values()), min(score.values())
    return hi >= GAME_CAP or (hi >= GAME_POINTS and hi - lo >= 2)


def rally_log(events: EventsResult, shot_label: Callable[[str | None], str | None]) -> tuple[list[RallyLine], list[GameScore]]:
    side = _side_fn(events)
    sides = sides_of(events)
    shots_by_id = {s.shot_id: s for s in events.shots}
    lines: list[RallyLine] = []
    games: list[GameScore] = []
    score = {s: 0 for s in sides}
    game_no, game_rallies = 1, 0
    for i, r in enumerate(sorted(events.rallies, key=lambda r: r.start_frame)):
        ss = sorted((shots_by_id[x] for x in r.shot_ids if x in shots_by_id), key=lambda s: s.frame)
        iv = [dt for _, _, dt in _intervals(ss, side)]
        line_score, line_game = None, None
        winner = side(r.winner)
        if winner in score:
            score[winner] += 1
            game_rallies += 1
            line_score, line_game = dict(score), game_no
            if _game_over(score):
                games.append(GameScore(game_no=game_no, score=dict(score), winner=winner, rallies=game_rallies))
                score = {s: 0 for s in sides}
                game_no, game_rallies = game_no + 1, 0
        last = ss[-1] if ss else None
        lines.append(RallyLine(
            rally_no=i + 1, rally_id=r.rally_id, start_ms=r.start_ms, end_ms=r.end_ms, shots=len(r.shot_ids),
            duration_s=round((r.end_ms - r.start_ms) / 1000, 2), server=side(ss[0].hitter) if ss else None,
            winner=winner, last_hitter=last.hitter if last else None,
            last_shot_type=shot_label(last.shot_type) if last and last.shot_type else None,
            mean_interval_s=round(float(np.mean(iv)), 2) if iv else None, game_no=line_game, score=line_score,
            status=r.status,
        ))
    if game_rallies:
        games.append(GameScore(game_no=game_no, score=dict(score), winner=None, rallies=game_rallies))
    return lines, games


def _runs(lines: list[RallyLine]) -> tuple[dict[str, int], int, dict[str, int]]:
    """(longest scoring run per side, lead changes, biggest lead per side) over decided rallies, per game."""
    longest: dict[str, int] = {}
    biggest: dict[str, int] = {}
    changes = 0
    cur_side, cur_len, leader, game = None, 0, None, None
    for ln in lines:
        if ln.score is None or ln.winner is None:
            continue
        if ln.game_no != game:  # new game: runs and leads restart
            cur_side, cur_len, leader, game = None, 0, None, ln.game_no
        cur_len = cur_len + 1 if ln.winner == cur_side else 1
        cur_side = ln.winner
        longest[cur_side] = max(longest.get(cur_side, 0), cur_len)
        (a, sa), (b, sb) = ln.score.items()
        now = a if sa > sb else b if sb > sa else None
        if now is not None:
            biggest[now] = max(biggest.get(now, 0), abs(sa - sb))
            if leader is not None and now != leader:
                changes += 1
            leader = now
    return longest, changes, biggest


def _conf(n: int, ok_n: int, review_n: int) -> tuple[str, str]:
    if n >= ok_n:
        return "ok", "high"
    if n >= review_n:
        return "needs_review", "medium"
    if n > 0:
        return "needs_review", "low"
    return "undeterminable", "low"


def _m(key: str, label: str, value: float | None, unit: str | None, n: int, ok_n: int, review_n: int,
       note: str | None = None, digits: int = 2) -> Metric:
    status, conf = _conf(n if value is not None else 0, ok_n, review_n)
    return Metric(key=key, label=label, value=None if value is None else round(float(value), digits), unit=unit,
                  n=n, status=status, confidence=conf, note=note)


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def flow_metrics(events: EventsResult, lines: list[RallyLine], games: list[GameScore],
                 player_ids: tuple[str, ...]) -> tuple[list[Metric], dict[str, list[Metric]]]:
    """Match-level and per-player metrics for score flow and tempo."""
    side = _side_fn(events)
    shots_by_id = {s.shot_id: s for s in events.shots}
    rallies = sorted(events.rallies, key=lambda r: r.start_frame)
    per_rally = [_intervals(sorted((shots_by_id[x] for x in r.shot_ids if x in shots_by_id), key=lambda s: s.frame), side)
                 for r in rallies]
    all_iv = [dt for iv in per_rally for _, _, dt in iv]
    half = len(rallies) // 2
    first = [dt for iv in per_rally[:half] for _, _, dt in iv]
    second = [dt for iv in per_rally[half:] for _, _, dt in iv]
    lens = [ln.shots for ln in lines]
    note_iv = f"相邻两拍由双方击出、间隔在 {MIN_INTERVAL_S}–{MAX_INTERVAL_S} 秒之间的才计入"
    decided = [ln for ln in lines if ln.score is not None]
    longest, changes, biggest = _runs(lines)

    match = [
        _m("match.tempo.interval_mean_s", "平均拍间间隔", _mean(all_iv), "s", len(all_iv), 60, 15, note_iv),
        _m("match.tempo.interval_first_half_s", "前半段平均拍间间隔", _mean(first), "s", len(first), 30, 8,
           "按回合顺序分成前后两半"),
        _m("match.tempo.interval_second_half_s", "后半段平均拍间间隔", _mean(second), "s", len(second), 30, 8,
           "比前半段明显变长，通常是节奏放慢或体能下降"),
        _m("match.rally_len.first_half_mean", "前半段平均每回合拍数", _mean(lens[:half]), "拍", half, 8, 3, digits=1),
        _m("match.rally_len.second_half_mean", "后半段平均每回合拍数", _mean(lens[half:]), "拍", len(lens) - half, 8, 3,
           digits=1),
        _m("match.score.games", "局数（按已标注胜负推算）", float(len(games)) if games else None, "局", len(decided),
           15, 5, "21 分制、领先 2 分获胜、30 分封顶；有回合未标注胜负时比分会偏低", digits=0),
        _m("match.score.lead_changes", "领先易手次数", float(changes) if decided else None, "次", len(decided), 15, 5,
           digits=0),
    ]

    per: dict[str, list[Metric]] = {}
    for pid in player_ids:
        me = side(pid)
        prep = [dt for iv in per_rally for _, b, dt in iv if b.hitter == pid]
        given = [dt for iv in per_rally for a, _, dt in iv if a.hitter == pid]
        per[pid] = [
            _m(f"{pid}.tempo.prep_s", "回球准备时间（对手出球到本人击球）", _mean(prep), "s", len(prep), 30, 8,
               "越短说明越常被对手压着打；" + note_iv),
            _m(f"{pid}.tempo.given_s", "留给对手的时间（本人出球到对手击球）", _mean(given), "s", len(given), 30, 8,
               "越短说明本人的球越快、越有压迫性"),
            _m(f"{pid}.win.longest_run", "最长连续得分", float(longest.get(me, 0)) if decided else None, "分",
               len(decided), 15, 5, "双打按所在队伍计" if events.match_format == "doubles" else None, digits=0),
            _m(f"{pid}.win.biggest_lead", "单局最大领先", float(biggest.get(me, 0)) if decided else None, "分",
               len(decided), 15, 5, digits=0),
        ]
    return match, per
