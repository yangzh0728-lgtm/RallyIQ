"""Apply append-only human corrections on top of the raw ``events`` artifact.

The raw prediction is never modified; ``apply_corrections`` returns a new
``EventsResult`` in which edited items carry ``source="human"``.
Corrections that refer to unknown ids are ignored (reported in ``skipped``).
"""

from __future__ import annotations

from ..contracts import Correction, EventsResult, Rally, ShotEvent, VideoMeta

_SWAP = {"P1": "P2", "P2": "P1", None: None}


class CorrectionError(ValueError):
    pass


def validate(op: str, payload: dict) -> None:
    need = {
        "add_shot": ["frame"],
        "delete_shot": ["shot_id"],
        "set_hitter": ["shot_id", "hitter"],
        "move_shot": ["shot_id", "frame"],
        "add_rally": ["start_frame", "end_frame"],
        "delete_rally": ["rally_id"],
        "set_rally_bounds": ["rally_id"],
        "set_rally_winner": ["rally_id", "winner"],
        "swap_player_ids": ["from_frame"],
    }[op]
    missing = [k for k in need if k not in payload]
    if missing:
        raise CorrectionError(f"{op} needs {', '.join(missing)}")
    for k in ("hitter", "winner"):
        if k in payload and payload[k] not in ("P1", "P2", None):
            raise CorrectionError(f"{k} must be 'P1', 'P2' or null")


def apply_corrections(
    raw: EventsResult, corrections: list[Correction], meta: VideoMeta | None
) -> tuple[EventsResult, list[str]]:
    shots: dict[str, ShotEvent] = {s.shot_id: s.model_copy(deep=True) for s in raw.shots}
    rallies: dict[str, Rally] = {r.rally_id: r.model_copy(deep=True) for r in raw.rallies}
    skipped: list[str] = []

    def t_ms(frame: int) -> int:
        return meta.t_ms(frame) if meta else 0

    for c in corrections:
        p = c.payload
        if c.op == "add_shot":
            sid = f"h{c.seq:04d}"
            shots[sid] = ShotEvent(
                shot_id=sid, frame=int(p["frame"]), t_ms=t_ms(int(p["frame"])), hitter=p.get("hitter"),
                shuttle_px=p.get("shuttle_px"), confidence=1.0, status="ok", source="human",
                reasons=[f"added by {c.correction_id}"],
            )
        elif c.op in ("delete_shot", "set_hitter", "move_shot"):
            s = shots.get(p["shot_id"])
            if s is None:
                skipped.append(c.correction_id)
                continue
            if c.op == "delete_shot":
                del shots[s.shot_id]
                continue
            if c.op == "set_hitter":
                s.hitter, s.hitter_confidence = p["hitter"], 1.0
            else:
                s.frame, s.t_ms = int(p["frame"]), t_ms(int(p["frame"]))
            s.source, s.status = "human", "ok"
        elif c.op == "add_rally":
            rid = f"hr{c.seq:04d}"
            sf, ef = int(p["start_frame"]), int(p["end_frame"])
            rallies[rid] = Rally(rally_id=rid, start_frame=sf, end_frame=ef, start_ms=t_ms(sf), end_ms=t_ms(ef),
                                 end_reason="manual", confidence=1.0, status="ok", source="human")
        elif c.op in ("delete_rally", "set_rally_bounds", "set_rally_winner"):
            r = rallies.get(p["rally_id"])
            if r is None:
                skipped.append(c.correction_id)
                continue
            if c.op == "delete_rally":
                del rallies[r.rally_id]
                continue
            if c.op == "set_rally_winner":
                r.winner = p["winner"]
            else:
                if "start_frame" in p:
                    r.start_frame, r.start_ms = int(p["start_frame"]), t_ms(int(p["start_frame"]))
                if "end_frame" in p:
                    r.end_frame, r.end_ms = int(p["end_frame"]), t_ms(int(p["end_frame"]))
                    r.end_reason = "manual"
            r.source, r.status = "human", "ok"
        elif c.op == "swap_player_ids":
            f0 = int(p["from_frame"])
            for s in shots.values():
                if s.frame >= f0:
                    s.hitter = _SWAP[s.hitter]
            for r in rallies.values():
                if r.start_frame >= f0:
                    r.winner = _SWAP[r.winner]

    # Re-derive rally membership from frames so added/moved shots land in the right rally.
    ordered_rallies = sorted(rallies.values(), key=lambda r: r.start_frame)
    for r in ordered_rallies:
        r.shot_ids = []
    ordered_shots = sorted(shots.values(), key=lambda s: s.frame)
    for s in ordered_shots:
        s.rally_id = None
        for r in ordered_rallies:
            if r.start_frame <= s.frame <= r.end_frame:
                s.rally_id = r.rally_id
                r.shot_ids.append(s.shot_id)
                break
    return EventsResult(video_id=raw.video_id, rallies=ordered_rallies, shots=ordered_shots), skipped
