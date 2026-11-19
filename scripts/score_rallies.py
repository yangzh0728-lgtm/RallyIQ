"""Score rally segmentation against hand-labelled rally start/end times.

    python scripts/score_rallies.py --truth qa_rallies/rallies_truth.json \\
        --run ~/rallyiq-data/runs/own-v12 --run /path/to/other_run [--list]

The truth file is a list of labelled windows of a clip:

    [{"clip": "own_20260529_0006", "window_s": [204, 330],
      "rallies": [{"start_s": 218.0, "end_s": 223.0, "sure": true, "why": "..."}],
      "unsure": [[611.7, 612.7]]}]

Everything inside a window that is not a rally is dead time. Predicted rallies (``events.json``) are
clipped to the windows; a predicted rally that is mostly inside an ``unsure`` span is ignored.

Per run: ``found`` truth rallies (predicted rallies cover at least half of it), ``missed``,
``fragments`` (extra predicted rallies inside a found truth rally), ``merged`` (predicted rallies
spanning two or more truth rallies), ``false`` (predicted rallies in dead time), the median start /
end error in seconds over found ``sure`` rallies, and how many of the false rallies have no hit.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MIN_COVER = 0.5  # share of a truth rally that predicted rallies must cover for it to count as found
FALSE_MAX_OVERLAP_S = 0.3  # a predicted rally overlapping real play less than this is a false rally
COLUMNS = ("truth", "found", "missed", "fragments", "merged", "false", "start_err", "end_err", "false_nohit")


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _pred_spans(windows: list[dict], pred: list[dict]) -> list[tuple[float, float, int]]:
    """(start_s, end_s, n_hits) of predicted rallies clipped to the windows, minus unsure ones."""
    out = []
    for r in pred:
        span = (r["start_ms"] / 1000.0, r["end_ms"] / 1000.0)
        for w in windows:
            ws = tuple(w["window_s"])
            if _overlap(span, ws) <= 0:
                continue
            clipped = (max(span[0], ws[0]), min(span[1], ws[1]))
            dur = max(clipped[1] - clipped[0], 1e-6)
            if sum(_overlap(clipped, tuple(u[:2])) for u in w.get("unsure", [])) >= 0.5 * dur:
                continue
            out.append((clipped[0], clipped[1], len(r.get("shot_ids") or [])))
    return out


def score_clip(windows: list[dict], pred: list[dict], listing: list[str] | None = None) -> dict:
    truth = [(t["start_s"], t["end_s"], bool(t.get("sure", True))) for w in windows for t in w["rallies"]]
    spans = _pred_spans(windows, pred)
    s = {"truth": len(truth), "found": 0, "missed": 0, "fragments": 0, "merged": 0, "false": 0, "false_nohit": 0}
    start_err, end_err = [], []
    for a, b, sure in truth:
        hit = [p for p in spans if _overlap(p[:2], (a, b)) > 0]
        if sum(_overlap(p[:2], (a, b)) for p in hit) < MIN_COVER * (b - a):
            s["missed"] += 1
            if listing is not None:
                listing.append(f"missed {a:.1f}-{b:.1f}")
            continue
        s["found"] += 1
        s["fragments"] += len(hit) - 1
        if listing is not None and len(hit) > 1:
            listing.append(f"split  {a:.1f}-{b:.1f} into " + ", ".join(f"{p[0]:.1f}-{p[1]:.1f}" for p in hit))
        if sure:
            start_err.append(abs(min(p[0] for p in hit) - a))
            end_err.append(abs(max(p[1] for p in hit) - b))
    for p in spans:
        touched = [t for t in truth if _overlap(p[:2], t[:2]) > 0]
        if len(touched) >= 2:
            s["merged"] += 1
            if listing is not None:
                listing.append(f"merged {p[0]:.1f}-{p[1]:.1f} spans {len(touched)} rallies")
        if sum(_overlap(p[:2], t[:2]) for t in truth) < FALSE_MAX_OVERLAP_S:
            s["false"] += 1
            s["false_nohit"] += p[2] == 0
            if listing is not None:
                listing.append(f"false  {p[0]:.1f}-{p[1]:.1f} ({p[2]} hits)")
    s["start_err_s"] = round(statistics.median(start_err), 3) if start_err else 0.0
    s["end_err_s"] = round(statistics.median(end_err), 3) if end_err else 0.0
    s["_start"], s["_end"] = start_err, end_err
    return s


def _events(run: Path, clip: str) -> list[dict]:
    res = next(r for r in json.loads((run / "results.json").read_text()) if r["clip_id"] == clip)
    ev = run / "data" / "videos" / res["video_id"] / "artifacts" / "events.json"
    return json.loads(ev.read_text()).get("rallies", [])


def score_run(truth: list[dict], run: Path, listing: list[str] | None = None) -> dict:
    total = {k: 0 for k in COLUMNS if not k.endswith("_err")}
    starts, ends = [], []
    for clip in dict.fromkeys(w["clip"] for w in truth):
        sub = [] if listing is not None else None
        s = score_clip([w for w in truth if w["clip"] == clip], _events(run, clip), sub)
        for k in total:
            total[k] += s[k]
        starts += s["_start"]
        ends += s["_end"]
        if listing is not None:
            listing += [f"  {clip} {line}" for line in sub]
    total["start_err"] = statistics.median(starts) if starts else 0.0
    total["end_err"] = statistics.median(ends) if ends else 0.0
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--list", action="store_true", help="list missed, split, merged and false rallies per run")
    a = ap.parse_args(argv)
    truth = json.loads(a.truth.expanduser().read_text())
    print("run".ljust(40) + "".join(k.rjust(12) for k in COLUMNS))
    details = []
    for run in a.run:
        listing: list[str] | None = [] if a.list else None
        t = score_run(truth, run.expanduser(), listing)
        cells = [f"{t[k]:.2f}" if k.endswith("_err") else str(t[k]) for k in COLUMNS]
        print(run.name.ljust(40) + "".join(c.rjust(12) for c in cells))
        if listing:
            details += [run.name] + listing
    if details:
        print("\n" + "\n".join(details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
