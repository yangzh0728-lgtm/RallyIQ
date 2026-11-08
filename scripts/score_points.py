"""Score shuttle and hit output against a small hand-checked list of frames.

    python scripts/score_points.py --truth qa_0529_points.json \\
        --run ~/rallyiq-data/own/run_0529_court7 --run ~/rallyiq-data/own/run_0529_court11

The truth file is a list of checked frames, written while looking at review stills:

    [{"clip": "own0529_0006", "frame": 39, "real": [550, 52], "false": [[212, 613]],
      "hit": null},
     {"clip": "own0529_0008", "frame": 144, "real": [634, 467], "false": [], "hit": "near"},
     {"clip": "own0529_0006", "frame": 2390, "real": null, "false": [], "hit": null}]

``real`` is where our shuttle is (null: not in view / not in play, so nothing should be
shown), ``false`` are things that must not be shown (a neighbour's shuttle, a duct tip),
``hit`` (optional) is "near" / "far" when one of our players hits within 3 frames, null
when nobody does. Each run is scored per frame: real shuttle found, false thing shown,
something shown where nothing should be, hit found with the right side, false hit.
Prints one table row per run and, with ``--frames``, the frames whose verdict differs
between runs, so a change can be judged on the frames it touched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TOL_PX = 25.0
HIT_FRAMES = 3
NET_Y_M = 6.7


def _artifacts(run: Path, clip: str) -> Path:
    res = next(r for r in json.loads((run / "results.json").read_text()) if r["clip_id"] == clip)
    return run / "data" / "videos" / res["video_id"] / "artifacts"


def _side(shot: dict) -> str | None:
    """near / far from where the hitter stood (court y < net = near half)."""
    hc = shot.get("hitter_court")
    return None if hc is None else ("near" if hc[1] < NET_Y_M else "far")


def load_run(run: Path, clips: set[str]) -> dict[str, dict]:
    out = {}
    for clip in clips:
        art = _artifacts(run, clip)
        sh = json.loads((art / "shuttle.json").read_text())
        ev = json.loads((art / "events.json").read_text()) if (art / "events.json").exists() else {}
        out[clip] = {
            "shuttle": {p["frame"]: (p["x"], p["y"]) for p in sh.get("points", [])},
            "hits": [(s["frame"], _side(s)) for s in ev.get("shots", [])],
        }
    return out


def verdict(entry: dict, run: dict) -> dict[str, bool]:
    shown = run[entry["clip"]]["shuttle"].get(entry["frame"])
    real, false = entry.get("real"), entry.get("false") or []
    v = {
        "real_found": bool(real and shown and math.dist(shown, real) <= TOL_PX),
        "real_missed": bool(real and not (shown and math.dist(shown, real) <= TOL_PX)),
        "false_shown": bool(shown and any(math.dist(shown, q) <= TOL_PX for q in false)),
        "shown_when_none": bool(real is None and shown is not None),
    }
    if "hit" in entry:
        near = [side for f, side in run[entry["clip"]]["hits"] if abs(f - entry["frame"]) <= HIT_FRAMES]
        if entry["hit"]:
            v["hit_found"] = entry["hit"] in near
            v["hit_missed"] = not near
            v["hit_wrong_side"] = bool(near) and entry["hit"] not in near
        else:
            v["false_hit"] = bool(near)
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--frames", action="store_true", help="list frames whose verdict differs between runs")
    a = ap.parse_args(argv)
    truth = json.loads(a.truth.expanduser().read_text())
    clips = {e["clip"] for e in truth}
    runs = {r: load_run(r.expanduser(), clips) for r in a.run}
    keys = ("real_found", "real_missed", "false_shown", "shown_when_none",
            "hit_found", "hit_missed", "hit_wrong_side", "false_hit")
    per_run = {r: [verdict(e, data) for e in truth] for r, data in runs.items()}
    print(f"{len(truth)} checked frames")
    print("run".ljust(40) + "".join(k.rjust(16) for k in keys))
    for r, vs in per_run.items():
        print(r.name.ljust(40) + "".join(str(sum(v.get(k, False) for v in vs)).rjust(16) for k in keys))
    if a.frames and len(per_run) > 1:
        print("\nframes that differ:")
        for i, e in enumerate(truth):
            vs = [per_run[r][i] for r in per_run]
            if any(v != vs[0] for v in vs[1:]):
                tags = ["|".join(k for k, b in v.items() if b) or "-" for v in vs]
                print(f"  {e['clip']} f{e['frame']}: " + "  vs  ".join(tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
