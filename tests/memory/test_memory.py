from datetime import datetime, timedelta, timezone

from rallyiq.contracts import Metric, ShotRecord, TrainingItem
from rallyiq.memory import MemoryLog, build_opponent_report, build_profile, current_events

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def m(key, value, status="ok", n=50):
    return Metric(key=key, label=key, value=value, unit="%", n=n, status=status, confidence="medium")


def commit(i, metrics, training=(), shots=(), view="end_center", person="我", self_=True, pid="P1"):
    return dict(kind="match_committed", video_id=f"v{i}", filename=f"{i}.mp4", video_created_at=T0 + timedelta(days=i),
                person=person, player_id=pid, is_self=self_, opponents=["小李"], conditions={"camera_view": view, "fps": "25-35"},
                metrics=metrics, training=list(training), shots=list(shots))


def test_trend_conflict_and_retest(tmp_path):
    log = MemoryLog(tmp_path)
    t = TrainingItem(player_id="P1", drill="步法", why="覆盖不足", retest_metric="cover.ratio_pct")
    log.append([commit(1, [m("win.rate_pct", 40), m("cover.ratio_pct", 50)], [t])])
    log.append([commit(2, [m("win.rate_pct", 60), m("cover.ratio_pct", 52)], view="side")])
    p = build_profile(log.read())
    win = next(x for x in p.metrics if x.key == "win.rate_pct")
    assert win.trend == "up" and win.delta == 20 and win.note  # 50% jump between two ok matches
    cover = next(x for x in p.metrics if x.key == "cover.ratio_pct")
    assert cover.trend == "flat"
    assert p.retests[0].status == "unchanged" and p.retests[0].same_conditions is False
    assert any(c.startswith("趋势：win.rate_pct 上升") for c in p.changes)
    assert any(c.startswith("冲突提醒") for c in p.changes)


def test_recommit_and_withdraw_never_delete(tmp_path):
    log = MemoryLog(tmp_path)
    log.append([commit(1, [m("shots", 10)])])
    log.append([commit(1, [m("shots", 12)])])  # re-commit after more corrections
    cur = current_events(log.read())
    assert len(cur) == 1 and cur[0].metrics[0].value == 12
    log.append([dict(kind="match_withdrawn", video_id="v1")])
    assert current_events(log.read()) == [] and len(log.read()) == 3
    log.append([commit(1, [m("shots", 13)])])  # committed again after the withdrawal
    assert current_events(log.read())[0].metrics[0].value == 13


def test_opponent_habits_with_evidence_and_counter_examples(tmp_path):
    log = MemoryLog(tmp_path)
    shots = []
    for k in range(10):
        shots.append(ShotRecord(shot_id=f"a{k}", t_ms=1000 * k, index_in_rally=2, zone="后场",
                                shot_type="smash" if k < 7 else "clear", shot_type_status="needs_review",
                                prev_shot_type="lift", status="ok"))
    log.append([commit(1, [], shots=shots, person="小李", self_=False, pid="P2")])
    r = build_opponent_report(log.read(), "小李")
    h = r.habits[0]
    assert (h.condition, h.outcome, h.n, h.hits) == ("在后场击球时", "杀球", 10, 7)
    assert len(h.evidence) == 5 and len(h.counter_examples) == 3 and h.confidence == "low"
    assert r.next_shot[0].condition == "我方挑球之后"
    assert any("待复核" in c for c in r.caveats) and any("1 场" in c for c in r.caveats)
