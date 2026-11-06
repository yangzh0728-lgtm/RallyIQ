from datetime import datetime, timezone

from rallyiq.contracts import Correction, EventsResult, Rally, ShotEvent, VideoMeta
from rallyiq.review import examples_for_video, summarize

TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
META = VideoMeta(video_id="v", filename="x.mp4", fps=30, frame_count=900, width=1920, height=1080, duration_ms=30000)


def raw():
    shots = [ShotEvent(shot_id=f"s{i}", rally_id="r1", frame=30 * i, t_ms=1000 * i, hitter="P1", confidence=0.7,
                       shot_type="clear") for i in range(1, 4)]
    return EventsResult(video_id="v", rallies=[Rally(rally_id="r1", start_frame=0, end_frame=200, start_ms=0,
                                                     end_ms=6666, confidence=0.8)], shots=shots)


def corr(i, op, payload, hard=True):
    return Correction(correction_id=f"c{i}", video_id="v", seq=i, created_at=TS, op=op, payload=payload, hard_example=hard)


def test_kinds_and_confirmations():
    cs = [
        corr(1, "set_shot_type", {"shot_id": "s1", "shot_type": "clear"}),  # same as model -> confirmation
        corr(2, "set_shot_type", {"shot_id": "s2", "shot_type": "smash"}),
        corr(3, "move_shot", {"shot_id": "s3", "frame": 95}),
        corr(4, "set_rally_winner", {"rally_id": "r1", "winner": "P1"}),
        corr(5, "swap_player_ids", {"from_frame": 120}),
        corr(6, "delete_shot", {"shot_id": "h0001"}),  # a human-added shot: label, not a model error
    ]
    ex = examples_for_video("v", cs, raw(), META, {"fps": "25-35"})
    assert [(e.kind, e.is_error) for e in ex] == [
        ("confirmed", False), ("wrong_shot_type", True), ("hit_timing", True), ("label", False),
        ("id_swap", True), ("label", False)]
    assert ex[2].model["moved_by_frames"] == 5 and ex[2].frame == 90 and ex[2].t_ms == 3000
    assert ex[4].frame == 120 and ex[4].module == "players"


def test_summary_ranks_weakest_condition():
    def vid(name, view, shots, errors):
        cs = [corr(i, "add_shot", {"frame": i}) for i in range(errors)]
        return examples_for_video(name, cs, raw(), META, {"camera_view": view}), {
            "conditions": {"camera_view": view}, "model_shots": shots, "model_rallies": 1}

    e1, v1 = vid("a", "end_center", 100, 5)
    e2, v2 = vid("b", "side", 50, 10)
    s = summarize(e1 + e2, {"a": v1, "b": v2})
    assert s.weakest[0].value == "side" and s.weakest[0].hit_error_rate == 0.2
    assert s.by_kind == {"missed_hit": 15}
