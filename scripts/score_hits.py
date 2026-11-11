"""Score hit detection against hits labelled in whole rallies (recall per side, false hits).

    python scripts/score_hits.py --truth qa_rallies/hits_truth.json --run RUN [--run RUN2] [--list]

Truth: {"windows": {clip: [[t0, t1], ...]}, "hits": {clip: [{"t": s, "side": "near"|"far", "sure": bool}]}}.
Every hit inside the windows is labelled, so a detected hit farther than ``TOL_S`` from all labelled
hits (sure or not) is false. Recall counts ``sure`` hits only; a found hit whose side (from
``hitter_court``) differs from the label is also counted as ``wrong_side``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_S = 0.25
NET_Y_M = 6.7
COLUMNS = ("sure", "found", "missed", "wrong_side", "false", "near_found", "near_sure", "far_found", "far_sure")


def _detected(run: Path, clip: str) -> list[tuple[float, str | None]]:
    res = next(r for r in json.loads((run / "results.json").read_text()) if r["clip_id"] == clip)
    art = run / "data" / "videos" / res["video_id"] / "artifacts"
    fps = json.loads((art / "probe.json").read_text())["fps"]
    out = []
    for s in json.loads((art / "events.json").read_text()).get("shots", []):
        hc = s.get("hitter_court")
        out.append((s["frame"] / fps, None if hc is None else ("near" if hc[1] < NET_Y_M else "far")))
    return out


def score_run(truth: dict, run: Path, listing: list[str] | None = None) -> dict:
    s = {k: 0 for k in COLUMNS}
    for clip, hs in truth["hits"].items():
        wins = truth["windows"][clip]
        det = [d for d in _detected(run, clip) if any(a <= d[0] <= b for a, b in wins)]
        for h in hs:
            if not h["sure"]:
                continue
            s["sure"] += 1
            s[h["side"] + "_sure"] += 1
            near = [side for t, side in det if abs(t - h["t"]) <= TOL_S]
            if near:
                s["found"] += 1
                s[h["side"] + "_found"] += 1
                if h["side"] not in near:
                    s["wrong_side"] += 1
            else:
                s["missed"] += 1
                if listing is not None:
                    listing.append(f"  {clip} missed {h['t']:.2f} {h['side']}")
        for t, side in det:
            if not any(abs(t - h["t"]) <= TOL_S for h in hs):
                s["false"] += 1
                if listing is not None:
                    listing.append(f"  {clip} false  {t:.2f} {side}")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--list", action="store_true", help="list missed and false hits per run")
    a = ap.parse_args(argv)
    truth = json.loads(a.truth.expanduser().read_text())
    print("run".ljust(32) + "".join(k.rjust(11) for k in COLUMNS))
    details = []
    for run in a.run:
        listing: list[str] | None = [] if a.list else None
        s = score_run(truth, run.expanduser(), listing)
        print(run.name.ljust(32) + "".join(str(s[k]).rjust(11) for k in COLUMNS))
        if listing:
            details += [run.name] + listing
    if details:
        print("\n" + "\n".join(details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
