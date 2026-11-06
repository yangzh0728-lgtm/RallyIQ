"""Turn human corrections into a hard-example set, grouped by video conditions.

README (推出后): "每次使用时的人工修正自动记录为难例，积累成评估集，按机位、帧率、场馆和光照分组。
每周挑出最弱的场景……针对性改进，并用同一评估集对比前后。"

Every correction posted with ``hard_example: true`` (the UI default) becomes a
``HardExample``: what kind of model error it fixed, the raw model item it refers
to, where it is in the video, and the conditions of that video taken from the
input assessment (camera view and height, frame rate, resolution, lighting,
singles/doubles). ``summarize`` turns them into per-condition error rates so the
weakest scenario is visible at a glance.

Nothing here edits corrections or events; it only reads the store.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from ..contracts import (
    ConditionGroup,
    Correction,
    EventsResult,
    HardExample,
    HardExampleSummary,
    StrokesResult,
    VideoMeta,
    VideoQuality,
)

# op -> (kind, module, is_error) when the correction refers to a model item.
OPS: dict[str, tuple[str, str, bool]] = {
    "add_shot": ("missed_hit", "hits", True),
    "delete_shot": ("false_hit", "hits", True),
    "move_shot": ("hit_timing", "hits", True),
    "set_hitter": ("wrong_hitter", "hits", True),
    "add_rally": ("missed_rally", "rallies", True),
    "delete_rally": ("false_rally", "rallies", True),
    "set_rally_bounds": ("rally_bounds", "rallies", True),
    "swap_player_ids": ("id_swap", "players", True),
    "set_shot_type": ("wrong_shot_type", "strokes", True),
    "confirm_shot": ("confirmed", "hits", False),
    "confirm_rally": ("confirmed", "rallies", False),
    "set_rally_winner": ("label", "labels", False),  # the model does not predict winners
}

CONDITIONS = ("camera_view", "camera_height", "fps", "resolution", "light", "match_format")
MIN_SHOTS_FOR_WEAKEST = 20


def fps_bucket(fps: float) -> str:
    return "<25" if fps < 24.5 else "25-35" if fps < 36 else "36-49" if fps < 49 else "50+"


def resolution_bucket(meta: VideoMeta) -> str:
    s = min(meta.width, meta.height)
    return "<720p" if s < 700 else "720p" if s < 1000 else "1080p" if s < 1400 else "1440p+"


def video_conditions(meta: VideoMeta | None, quality: VideoQuality | None, match_format: str | None) -> dict[str, str]:
    c: dict[str, str] = {}
    if quality is not None:
        c["camera_view"] = quality.camera_view
        c["camera_height"] = quality.camera_height
        b = next((x for x in quality.checks if x.key == "brightness"), None)
        if b is not None and b.value:
            c["light"] = {"偏暗": "dark", "过曝": "bright", "正常": "normal"}.get(b.value, b.value)
    if meta is not None:
        c["fps"] = fps_bucket(meta.fps)
        c["resolution"] = resolution_bucket(meta)
    if match_format:
        c["match_format"] = match_format
    return c


def _model_lookup(raw: EventsResult | None) -> tuple[dict, dict]:
    if raw is None:
        return {}, {}
    return ({s.shot_id: s for s in raw.shots if s.source == "model"},
            {r.rally_id: r for r in raw.rallies if r.source == "model"})


def examples_for_video(
    video_id: str,
    corrections: Iterable[Correction],
    raw: EventsResult | None,
    meta: VideoMeta | None,
    conditions: dict[str, str],
    filename: str | None = None,
) -> list[HardExample]:
    shots, rallies = _model_lookup(raw)
    out: list[HardExample] = []
    for c in corrections:
        if not c.hard_example or c.op not in OPS:
            continue
        kind, module, is_error = OPS[c.op]
        p = c.payload
        frame: int | None = None
        model_item = None
        if "shot_id" in p:
            s = shots.get(p["shot_id"])
            if s is None:  # refers to a shot a human added: a label, not a model error
                kind, is_error = "label", False
            else:
                frame = s.frame
                model_item = s.model_dump(mode="json")
                if c.op == "set_shot_type" and s.shot_type == p.get("shot_type"):
                    kind, is_error = "confirmed", False
                if c.op == "set_hitter" and s.hitter == p.get("hitter"):
                    kind, is_error = "confirmed", False
        elif "rally_id" in p:
            r = rallies.get(p["rally_id"])
            if r is None:
                kind, is_error = "label", False
            else:
                frame = r.start_frame
                model_item = r.model_dump(mode="json")
        if c.op == "add_shot":
            frame = int(p["frame"])
        elif c.op == "add_rally":
            frame = int(p["start_frame"])
        elif c.op == "swap_player_ids":
            frame = int(p["from_frame"])
        if c.op == "move_shot" and model_item is not None:
            model_item = {**model_item, "moved_by_frames": int(p["frame"]) - model_item["frame"]}
        out.append(HardExample(
            video_id=video_id, filename=filename, correction_id=c.correction_id, op=c.op, kind=kind,
            module=module if kind != "label" else "labels", is_error=is_error, frame=frame,
            t_ms=meta.t_ms(frame) if (meta is not None and frame is not None) else None,
            created_at=c.created_at, payload=dict(p), model=model_item, conditions=conditions, note=c.note,
        ))
    return out


def collect(store, owner_id: str | None = None) -> tuple[list[HardExample], dict[str, dict]]:
    """All hard examples in the store (or one account's videos), plus per-video totals used to
    normalise error rates."""
    from ..ingest.quality import QUALITY_ARTIFACT, quality_for  # noqa: PLC0415
    from ..strokes.classify import merge_strokes  # noqa: PLC0415

    examples: list[HardExample] = []
    videos: dict[str, dict] = {}
    for v in store.list_videos(owner_id=owner_id):
        corrections = store.list_corrections(v.video_id)
        if not any(c.hard_example for c in corrections):
            continue
        meta = store.get_video_meta(v.video_id)
        raw = store.load_artifact(v.video_id, "events", EventsResult)
        if raw is not None:
            raw = merge_strokes(raw, store.load_artifact(v.video_id, "strokes", StrokesResult))
        # Reuse the cached assessment; compute it (without frame sampling) only when missing.
        quality = store.load_artifact(v.video_id, QUALITY_ARTIFACT, VideoQuality)
        if quality is None and meta is not None:
            quality = quality_for(store, v.video_id, sample_pictures=False)
        cond = video_conditions(meta, quality, raw.match_format if raw is not None else None)
        examples += examples_for_video(v.video_id, corrections, raw, meta, cond, v.filename)
        videos[v.video_id] = {
            "conditions": cond,
            "model_shots": sum(1 for s in raw.shots if s.source == "model") if raw else 0,
            "model_rallies": sum(1 for r in raw.rallies if r.source == "model") if raw else 0,
        }
    return examples, videos


def summarize(examples: list[HardExample], videos: dict[str, dict]) -> HardExampleSummary:
    groups: list[ConditionGroup] = []
    for cond in CONDITIONS:
        by_value: dict[str, list[str]] = defaultdict(list)
        for vid, info in videos.items():
            if cond in info["conditions"]:
                by_value[info["conditions"][cond]].append(vid)
        for value, vids in sorted(by_value.items()):
            vs = set(vids)
            ex = [e for e in examples if e.video_id in vs]
            errors = Counter(e.module for e in ex if e.is_error)
            shots = sum(videos[v]["model_shots"] for v in vids)
            groups.append(ConditionGroup(
                condition=cond, value=value, videos=len(vids), model_shots=shots,
                model_rallies=sum(videos[v]["model_rallies"] for v in vids),
                errors=dict(errors), confirmed=sum(1 for e in ex if e.kind == "confirmed"),
                hit_error_rate=round(errors.get("hits", 0) / shots, 4) if shots else None,
            ))
    ranked = sorted((g for g in groups if g.hit_error_rate is not None and g.model_shots >= MIN_SHOTS_FOR_WEAKEST),
                    key=lambda g: g.hit_error_rate, reverse=True)
    return HardExampleSummary(
        total=len(examples), errors=sum(e.is_error for e in examples), videos=len(videos),
        by_kind=dict(Counter(e.kind for e in examples)), groups=groups, weakest=ranked[:3],
    )
