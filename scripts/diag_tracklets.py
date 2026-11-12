"""Why did the shuttle tracker keep or drop the trajectories around some frames?

    python scripts/diag_tracklets.py --run ~/rallyiq-data/own/run_0529_court4 --clip own0529_0006 \\
        --frames 0 39 408 1307 3356 7191

Re-tracks the saved raw peaks of a finished run (no network) with the run's calibration
and players, the same way the shuttle stage does, and prints every linked trajectory that
covers or comes within ``--window`` frames of the given frames: span, length, mean
target-court weight (1 = only over our court, 0.6 = also over another court), whether it
reaches one of our players, and the verdict (kept / foreign / ambiguous / idle). For
every candidate on the given frames it prints the court weight, the loose and the strict
(swinging-wrist racket contact) player support, whether it is in a hand, the rest flag and whether it sits in a fixed-feature cell
(dropped before linking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import CourtCalibration, PlayerTracks, VideoMeta
from rallyiq.perception.shuttle.stage import (
    CANDIDATES_FILE,
    court_lock,
    load_candidates,
    contact_tests,
    rest_area,
    tracker_config,
)
from rallyiq.perception.shuttle.tracker import ShuttleTracker


def parse_settings(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--peak-threshold", type=float, default=0.2)
    ap.add_argument("--contact", default="box", choices=("box", "wrist"))
    ap.add_argument("--tracker", nargs="*", default=[], metavar="KEY=VALUE", help="tracker settings")
    a = ap.parse_args(argv)

    run = a.run.expanduser()
    res = next(r for r in json.loads((run / "results.json").read_text()) if r["clip_id"] == a.clip)
    vdir = run / "data" / "videos" / res["video_id"]
    art = vdir / "artifacts"
    meta = VideoMeta.model_validate_json((art / "probe.json").read_text())
    cal = CourtCalibration.model_validate_json((art / "calibration.json").read_text())
    players = PlayerTracks.model_validate_json((art / "players.json").read_text())
    cands, (w, h), _ = load_candidates(vdir / "work" / CANDIDATES_FILE)
    region, support, mode = court_lock(cal, players)
    rest = rest_area(cal, players)
    strict, held = contact_tests(players, meta, a.contact)
    sx, sy = meta.width / w, meta.height / h
    scaled = {f: [(x * sx, y * sy, s) for x, y, s in cs] for f, cs in cands.items()}
    tr = ShuttleTracker(meta.fps, meta.width, meta.height, tracker_config(parse_settings(a.tracker), a.peak_threshold),
                        region=region, support=support, rest=rest, strict_support=strict, held=held)
    tr.track(scaled, meta.frame_count)
    print(f"court lock: {mode}; {len(tr.log)} trajectories judged; {len(tr.hot_cells)} fixed-feature cells")
    for f in a.frames:
        print(f"\n== frame {f}: candidates {[tuple(round(v, 2) for v in c) for c in scaled.get(f, [])]}")
        for x, y, s in scaled.get(f, []):
            weight = f"{region(f, x, y):.2f}" if region else None
            print(f"   ({x:.0f},{y:.0f}) score {s:.2f}: court weight {weight}, "
                  f"near our player {support(f, x, y) if support else None}, "
                  f"racket contact {strict(f, x, y) if strict else None}, held {held(f, x, y) if held else None}, rest {rest(f, x, y) if rest else None}, "
                  f"fixed feature {tr.hot_cell(x, y) in tr.hot_cells}")
        for start, end, n, m, sup, verdict in tr.log:
            if start - a.window <= f <= end + a.window:
                print(f"   track {start}-{end} ({n} pts): mean weight {m}, supported {sup} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
