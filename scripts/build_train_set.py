"""Build a labelled training set from ShuttleSet: one clip per set (game) of every non-held-out match.

    python scripts/build_train_set.py                   # ~90 clips (38 matches x their sets), up to 60 s each
    python scripts/build_train_set.py --limit 4         # quick check
    python scripts/build_train_set.py --window 60 --stride 2

ShuttleSet (KDD 2023, CoachAI-Projects, MIT) has 44 BWF singles matches / 104 sets with every
stroke annotated (hit frame, hitter position, court homography). The 6 matches that
scripts/public_eval.py scores are held out by match (evaluation/public/shuttleset.py:
eval_matches), so nothing here overlaps the before/after test set.

For every clip this downloads only the window (720p, yt-dlp), writes the label, and runs the
pipeline (probe, calibration, players, shuttle, events) so later learning steps
(scripts/learn_hits.py) can re-run the event stage offline on the cached artifacts.
Resumable: clips already in index.json with state "succeeded" are skipped.

Output: ~/rallyiq-data/public/shuttleset_train/{clips,labels,preds,data,index.json}. Videos are
BWF broadcasts: personal research use only, same terms as evaluation/README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import schema  # noqa: E402
from evaluation.public import shuttleset as SS  # noqa: E402
from evaluation.tools.extract_clips import cut, probe  # noqa: E402

OUT = Path.home() / "rallyiq-data" / "public" / "shuttleset_train"


def plan(set_dir: Path, window: float) -> list[dict[str, Any]]:
    """(clip id, match, set csv, window) for every set of every training match."""
    items = []
    for m in SS.train_matches(SS.load_matches(set_dir)):
        for set_csv in sorted((set_dir / m["video"]).glob("set*.csv")):
            strokes = SS.load_strokes(set_csv)
            if not strokes:
                continue
            fps = SS.video_fps(strokes)
            rallies = sorted({s["rally"] for s in strokes})
            # start a third of the way into the set: mid-game play, different scores per set
            first = rallies[len(rallies) // 3] if len(rallies) > 3 else rallies[0]
            win = SS.pick_window(strokes, fps, window, first_rally=first)
            items.append({"id": f"st{int(m['id']):02d}_{set_csv.stem}", "match": m, "strokes": strokes,
                          "fps": fps, "window": win})
    return items


def main(argv: list[str] | None = None) -> int:
    import public_eval
    import real_run
    from rallyiq.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuttleset", type=Path, help="ShuttleSet/set dir (default: clone next to public_eval's)")
    ap.add_argument("--window", type=float, default=60.0, help="seconds per clip (whole rallies)")
    ap.add_argument("--stride", type=int, default=2, help="players: pose every n-th frame")
    ap.add_argument("--limit", type=int, default=0, help="only the first n clips")
    ap.add_argument("--only", nargs="*", help="clip ids")
    ap.add_argument("--no-run", action="store_true", help="download + label only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    set_dir = public_eval.shuttleset_dir(a.shuttleset)
    clips, labels, preds = OUT / "clips", OUT / "labels", OUT / "preds"
    for d in (clips, labels, preds):
        d.mkdir(parents=True, exist_ok=True)
    idx_p = OUT / "index.json"
    index: dict[str, Any] = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    store = Store(OUT / "data")

    items = plan(set_dir, a.window)
    if a.only:
        items = [i for i in items if i["id"] in a.only]
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} clips from {len({i['match']['id'] for i in items})} training matches", flush=True)
    t_all = time.monotonic()
    for k, it in enumerate(items, 1):
        cid, m, win = it["id"], it["match"], it["window"]
        if index.get(cid, {}).get("state") == "succeeded":
            continue
        clip = clips / f"{cid}.mp4"
        if not clip.exists():
            try:
                lo = max(0.0, win[0] - 2)
                src = public_eval.download_720(m["youtube_id"], lo, win[1] + 2, OUT / "raw")
                cut(src, win[0] - lo, win[1] - win[0], clip)
                src.unlink(missing_ok=True)
            except subprocess.CalledProcessError as e:
                print(f"[skip] {cid}: download/cut failed {e}", flush=True)
                index[cid] = {"state": "download_failed", "match_id": m["id"]}
                idx_p.write_text(json.dumps(index, indent=1), encoding="utf-8")
                continue
        info = probe(clip)
        clip.with_suffix(".meta.json").write_text(json.dumps(
            {"clip_id": cid, "video_id": m["youtube_id"], "start_ms": int(round(win[0] * 1000)),
             "duration_ms": int(round((win[1] - win[0]) * 1000)), **info}), encoding="utf-8")
        doc = SS.build_label(cid, m, it["strokes"], it["fps"], win, info["width"], info["height"])
        doc["split"] = "train"
        schema.save(doc, labels / f"{cid}.json")
        entry = {"match_id": m["id"], "video": m["video"], "window": [round(w, 3) for w in win],
                 "hits": len(doc["hits"]), "fps": info["fps"], "label_fps": it["fps"]}
        if a.no_run:
            index[cid] = {**entry, "state": "labelled"}
        else:
            print(f"=== [{k}/{len(items)}] {cid} {m['video']} [{win[0]:.0f}s, {win[1]:.0f}s] "
                  f"{len(doc['hits'])} hits  (elapsed {time.monotonic() - t_all:.0f}s)", flush=True)
            r = real_run.run_clip(store, clip, {"players_stride": a.stride}, preds)
            index[cid] = {**entry, "state": r["state"], "error": r["error"], "artifacts": r["artifacts"],
                          "wall_s": r["wall_s"], "stages": {n: s["s"] for n, s in r["stages"].items()}}
            print(json.dumps({"clip": cid, "state": r["state"], "wall_s": r["wall_s"]}), flush=True)
        idx_p.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in index.values() if v.get("state") == "succeeded")
    print(f"\ndone: {ok} clips succeeded, index {idx_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
