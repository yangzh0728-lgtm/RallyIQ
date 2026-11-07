"""Write a shareable match report (one self-contained HTML file) for an analysed video.

    # a video analysed by the local app (data dir: $RALLYIQ_DATA or ./data)
    python scripts/match_report_html.py vid_abc123 -o report.html
    # a folder of artifacts (events.json, strokes.json, players.json, calibration.json, video_meta.json, ...)
    python scripts/match_report_html.py ~/rallyiq-data/eval_dji/run41m -o dji0009.html
    # the synthetic sample in docs/sample-report/ (no video, no models)
    python scripts/match_report_html.py --sample

Everything runs locally and offline: the AI coach section uses a saved ``coach.json`` when there is one,
otherwise the rule-based offline coach (``--no-coach`` leaves it out). The page holds no video and loads
nothing from the network, so it can be emailed, opened on a phone or printed to PDF.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from datetime import UTC, datetime
from pathlib import Path

from rallyiq import contracts as c
from rallyiq.api.corrections import apply_corrections
from rallyiq.report import compute_match_stats, render_html
from rallyiq.report.coach import OfflineProvider, build_report
from rallyiq.strokes import classify, merge_strokes

SAMPLE_OUT = Path(__file__).resolve().parents[1] / "docs" / "sample-report" / "match_report.html"


def _load(path: Path, model: type[c.Contract]):
    return model.model_validate_json(path.read_text()) if path.exists() else None


def _coach(stats: c.MatchStats, saved: c.CoachReport | None, use: bool) -> tuple[c.CoachReport | None, bool]:
    if not use:
        return None, False
    if saved is not None:
        return saved, saved.stats_fingerprint != stats.fingerprint
    return build_report(OfflineProvider().generate(stats), stats, OfflineProvider()), False


def from_dir(d: Path, use_coach: bool) -> tuple[c.MatchReport, c.EventsResult, c.PlayerTracks | None, str]:
    events = _load(d / "events.json", c.EventsResult)
    if events is None:
        raise SystemExit(f"{d}: no events.json")
    meta = _load(d / "video_meta.json", c.VideoMeta)
    players = _load(d / "players.json", c.PlayerTracks)
    cal = _load(d / "calibration.json", c.CourtCalibration)
    events = merge_strokes(events, _load(d / "strokes.json", c.StrokesResult))
    corr = d / "corrections.json"
    if corr.exists():
        from pydantic import TypeAdapter

        events, _ = apply_corrections(events, TypeAdapter(list[c.Correction]).validate_json(corr.read_text()), meta)
    stats = compute_match_stats(events, players, cal, meta)
    coach, stale = _coach(stats, _load(d / "coach.json", c.CoachReport), use_coach)
    name = meta.filename if meta else d.name
    return c.MatchReport(stats=stats, coach=coach, coach_stale=stale), events, players, name


def from_store(video_id: str, data: str | None, use_coach: bool):
    from rallyiq.store import Store

    store = Store(data)
    raw = store.load_artifact(video_id, "events", c.EventsResult)
    if raw is None:
        raise SystemExit(f"{video_id}: no events artifact (not analysed yet?)")
    meta = store.get_video_meta(video_id)
    events, _ = apply_corrections(merge_strokes(raw, store.load_artifact(video_id, "strokes", c.StrokesResult)),
                                  store.list_corrections(video_id), meta)
    players = store.load_artifact(video_id, "players", c.PlayerTracks)
    stats = compute_match_stats(events, players, store.load_artifact(video_id, "calibration", c.CourtCalibration), meta)
    coach, stale = _coach(stats, store.load_artifact(video_id, "coach", c.CoachReport), use_coach)
    v = store.get_video(video_id)
    return c.MatchReport(stats=stats, coach=coach, coach_stale=stale), events, players, v.filename if v else video_id


# A believable shot-to-shot flow for the synthetic sample: reply options after each stroke type.
_REPLIES = {
    "short_serve": ["push", "net_shot", "lift"], "long_serve": ["clear", "smash", "drop"],
    "clear": ["clear", "drop", "smash"], "lift": ["smash", "drop", "clear"], "smash": ["block", "lift", "drive"],
    "drop": ["net_shot", "lift", "push"], "net_shot": ["lift", "net_shot", "net_kill"], "block": ["lift", "net_shot"],
    "push": ["drive", "lift", "block"], "drive": ["drive", "block", "lift"], "net_kill": ["block"],
}


def _sample_events(events: c.EventsResult) -> tuple[c.EventsResult, dict[str, str]]:
    """Vary the demo rallies (3-12 shots), script a stroke for every shot, and decide each rally from its
    last shot: a smash / net kill wins it for the hitter, anything else loses it."""
    seed = 7
    by_id = {s.shot_id: s for s in events.shots}
    rallies, shots, script = [], [], {}
    for i, r in enumerate(events.rallies):
        seed = (seed * 1103515245 + 12345) % 2**31
        k = min(len(r.shot_ids), 3 + seed % 10)
        ids = r.shot_ids[:k]
        prev = None
        for j, sid in enumerate(ids):
            seed = (seed * 1103515245 + 12345) % 2**31
            hitter = by_id[sid].hitter
            if j == 0:
                t = "short_serve" if seed % 4 else "long_serve"
            else:
                opts = _REPLIES[prev]
                if hitter == "P1" and "smash" in opts and seed % 3 == 0:
                    t = "smash"  # P1 attacks more often
                else:
                    t = opts[seed % len(opts)]
            script[sid] = prev = t
        last = by_id[ids[-1]]
        won = script[ids[-1]] in ("smash", "net_kill")
        other = "P2" if last.hitter == "P1" else "P1"
        rallies.append(r.model_copy(update={"shot_ids": ids, "end_ms": last.t_ms + 900,
                                            "end_frame": last.frame + 27, "winner": last.hitter if won else other}))
        shots += [by_id[x] for x in ids]
    return events.model_copy(update={"rallies": rallies, "shots": shots}), script


def sample():
    """A synthetic 3-minute singles match (``rallyiq.pipeline.demo`` tracks, scripted strokes): fake but complete,
    with winners entered. Only for showing what the report looks like."""
    from rallyiq.pipeline import demo

    meta = c.VideoMeta(video_id="vid_sample", filename="sample_match.mp4", fps=30.0, frame_count=5400, width=1920,
                       height=1080, duration_ms=180_000, codec="h264")
    cal = demo.demo_calibration(meta)
    players = demo.demo_players(meta, cal)
    shuttle = demo.demo_shuttle(meta, players)
    raw, script = _sample_events(demo.demo_events(meta, players))

    def scripted(_events, _shuttle, _players, _feats):
        return {sid: {t: 0.82, **{o: 0.06 for o in ("clear", "lift", "drop") if o != t}} for sid, t in script.items()}

    # hand / direction / pace ... layers when the stroke classifier offers them
    extra = {"detail": True} if "detail" in inspect.signature(classify).parameters else {}
    events = merge_strokes(raw, classify(raw, shuttle, players, meta.fps, model=scripted, model_name="sample_script",
                                         **extra))
    stats = compute_match_stats(events, players, cal, meta)
    coach = build_report(OfflineProvider().generate(stats), stats, OfflineProvider())
    return c.MatchReport(stats=stats, coach=coach), events, players, meta.filename


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="video id in the data dir, or a folder of artifact JSON files")
    ap.add_argument("-o", "--out", type=Path, help="output .html (default <name>_report.html)")
    ap.add_argument("--data", help="data dir for a video id (default $RALLYIQ_DATA or ./data)")
    ap.add_argument("--title", help="page title (default 赛后报告 · <video file name>)")
    ap.add_argument("--no-coach", action="store_true", help="leave out the AI coach section")
    ap.add_argument("--sample", action="store_true", help=f"write the synthetic sample to {SAMPLE_OUT}")
    args = ap.parse_args(argv)

    generated = None
    if args.sample:
        report, events, players, name = sample()
        out = args.out or SAMPLE_OUT
        generated = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)  # stable file for the repo
    elif args.source and Path(args.source).expanduser().is_dir():
        report, events, players, name = from_dir(Path(args.source).expanduser(), not args.no_coach)
        out = args.out or Path(f"{Path(name).stem}_report.html")
    elif args.source:
        report, events, players, name = from_store(args.source, args.data, not args.no_coach)
        out = args.out or Path(f"{args.source}_report.html")
    else:
        ap.error("give a video id, an artifacts folder, or --sample")
    page = render_html(report, events, players, title=args.title or f"赛后报告 · {name}", video_name=name,
                       generated=generated)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
