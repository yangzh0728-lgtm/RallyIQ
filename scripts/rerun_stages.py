"""Re-run the later stages of a finished ``real_run.py`` run, without redoing the slow ones.

    python scripts/rerun_stages.py --out ~/rallyiq-data/own/run_0529_court2 --only own0529_0006
    python scripts/rerun_stages.py --out RUN --from events          # only events + strokes

For each clip in ``<out>/results.json`` the clip's ``.meta.json`` is read again, so court
points and ``neighbour_courts`` added to ``evaluation/own_clips.yaml`` since the run (copied
into the meta by ``extract_clips``) replace the stored manual calibration. Then the stages
from ``--from`` on are re-run in the same store (with the calibration stage too whenever
the meta has court points, since later stages read the calibration it writes). The shuttle stage re-tracks the raw peaks it
saved last time instead of running the network again, so a court-lock or tracker change
costs seconds, not the hour a full run takes. Players are never re-run (use real_run).

Rewrites ``<out>/preds/<clip>.json``, ``results.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import from_contracts, schema
from rallyiq.contracts import STAGES
from rallyiq.pipeline.runner import MANUAL_CALIBRATION, run_job
from rallyiq.store import Store
from scripts.diag_tracklets import parse_settings
from scripts.real_run import (
    DEFAULT_CLIPS,
    clip_stats,
    manual_calibration,
    render,
)

RERUNNABLE = ("calibration", "shuttle", "events", "strokes", "overlay")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="a finished real_run.py output directory")
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--from", dest="first", choices=RERUNNABLE, default="calibration")
    ap.add_argument("--no-overlay", action="store_true", help="skip the overlay video stage")
    ap.add_argument("--tracker", nargs="*", default=[], metavar="KEY=VALUE",
                    help="shuttle tracker settings, e.g. flight_split=true contact_gate=ends")
    ap.add_argument("--contact", choices=("box", "wrist"), help="shuttle racket-contact test")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    out = a.out.expanduser()
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    store = Store(out / "data")
    stages = [s for s in STAGES[STAGES.index(a.first):]
              if s != "players" and not (a.no_overlay and s == "overlay")]  # players are kept as they are
    shuttle: dict = {"reuse_candidates": True}
    if a.tracker:
        shuttle["tracker"] = parse_settings(a.tracker)
    if a.contact:
        shuttle["contact"] = a.contact
    config = {"shuttle": shuttle}
    for r in results:
        if a.only and r["clip_id"] not in a.only:
            continue
        clip = a.clips.expanduser() / f"{r['clip_id']}.mp4"
        cmeta = json.loads(clip.with_suffix(".meta.json").read_text(encoding="utf-8"))
        vid = r["video_id"]
        todo = list(stages)
        if cmeta.get("court_points"):
            n = int(cmeta.get("frame_count") or round(cmeta["duration_ms"] / 1000 * cmeta["fps"]))
            cal = manual_calibration(vid, cmeta["court_points"], n, cmeta.get("neighbour_courts"))
            store.save_artifact(vid, MANUAL_CALIBRATION, cal)
            print(f"{r['clip_id']}: manual court, {len(cal.segments[0].other_courts)} neighbour courts")
            if "calibration" not in todo:
                # Later stages read calibration.json, which only the calibration stage
                # rebuilds from the manual court; it is quick.
                todo.insert(0, "calibration")
        print(f"=== {r['clip_id']} ({vid}): {', '.join(todo)}", flush=True)
        t0 = time.monotonic()
        job = run_job(store, store.create_job(vid, todo), config)
        art = store.video_dir(vid) / "artifacts"  # this store, also when the run dir was copied
        r.update({"artifacts": str(art), "state": job.state, "error": job.error, "rerun_from": a.first,
                  "rerun_s": round(time.monotonic() - t0, 1)})
        r["stages"].update({s.name: {"state": s.state, "error": s.error,
                                     "s": round((s.finished_at - s.started_at).total_seconds(), 1)
                                     if s.started_at and s.finished_at else None}
                            for s in job.stages if s.state != "skipped"})
        r.update(clip_stats(art))
        doc = from_contracts.convert(art, r["clip_id"], int(cmeta["start_ms"]))
        schema.save(doc, out / "preds" / f"{r['clip_id']}.json")
        print(json.dumps({k: r.get(k) for k in ("clip_id", "state", "rerun_s", "shuttle_detected_frac",
                                                "rallies", "hits")}, ensure_ascii=False), flush=True)

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render(results, None), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
