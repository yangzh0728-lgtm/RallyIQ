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

DATA = Path.home() / "rallyiq-data" / "public"
OUT = DATA / "shuttleset_train"
STAGE_ZH = {"probe": "读取视频", "calibration": "球场标定", "players": "球员姿态", "shuttle": "羽毛球检测",
            "events": "回合/击球", "overlay": "叠加层"}


class StagePrinter:
    """Prints one line per stage every 10% of progress, with seconds since the stage was first seen.
    Stages can overlap (prefetch), so the state is kept per stage."""

    def __init__(self) -> None:
        self.t0: dict[str, float] = {}
        self.next: dict[str, float] = {}

    def __call__(self, stage: str, p: float) -> None:
        now = time.monotonic()
        if stage not in self.t0:
            self.t0[stage], self.next[stage] = now, 0.1
        if self.next[stage] > 1.0:
            return  # already reported as done
        if p >= self.next[stage]:
            done = "完成" if p >= 1.0 else f"{p:4.0%}"
            print(f"    [{STAGE_ZH.get(stage, stage)}] {done}  {now - self.t0[stage]:5.0f}s", flush=True)
            self.next[stage] = 1.1 if p >= 1.0 else min((int(p * 10) + 1) / 10, 1.0)


def _pct(v: Any) -> str:
    return "–" if v is None else f"{v:.0%}"


def clip_summary(r: dict[str, Any], label: dict[str, Any], pred_path: Path) -> str:
    """Per-clip result: what each module produced, and hit F1 against the ShuttleSet label."""
    from evaluation import metrics as M

    f1 = "–"
    if pred_path.exists():
        st = M.eval_hits(label, schema.load(pred_path)).summary()
        if st["f1"] is not None:
            f1 = f"{st['f1']:.2f}（不分方 {st['f1_ignore_side']:.2f}）"
    stages = "  ".join(f"{STAGE_ZH.get(n, n)} {s['s']}s" for n, s in r["stages"].items() if s.get("s") is not None)
    return (f"    结果：标定 {_pct(r.get('court_calibrated_frac'))}  两名球员 {_pct(r.get('players_both_frac'))}  "
            f"球检出 {_pct(r.get('shuttle_detected_frac'))}  回合 {r.get('rallies', '–')}  "
            f"击球 预测 {r.get('hits', '–')} / 标注 {len(label['hits'])}  击球F1 {f1}\n    耗时：{stages}")


def plan(set_dir: Path, window: float, split: str = "train") -> list[dict[str, Any]]:
    """(clip id, match, set csv, window) for every set of every training match, or with
    ``split="test"`` of every held-out match (a larger test set than public_eval's one clip per match)."""
    items = []
    matches = SS.load_matches(set_dir)
    chosen = SS.train_matches(matches) if split == "train" else SS.eval_matches(matches)
    prefix = "st" if split == "train" else "sx"
    for m in chosen:
        for set_csv in sorted((set_dir / m["video"]).glob("set*.csv")):
            strokes = SS.load_strokes(set_csv)
            if not strokes:
                continue
            fps = SS.video_fps(strokes)
            rallies = sorted({s["rally"] for s in strokes})
            # start a third of the way into the set: mid-game play, different scores per set
            first = rallies[len(rallies) // 3] if len(rallies) > 3 else rallies[0]
            win = SS.pick_window(strokes, fps, window, first_rally=first)
            items.append({"id": f"{prefix}{int(m['id']):02d}_{set_csv.stem}", "match": m, "strokes": strokes,
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
    ap.add_argument("--split", choices=("train", "test"), default="train",
                    help="test: every set of the 6 held-out matches -> ~/rallyiq-data/public/shuttleset_test")
    ap.add_argument("--out", type=Path, help="output folder (default shuttleset_train / shuttleset_test); "
                    "point a new folder's clips/ at an old one to re-run the pipeline without downloading again")
    ap.add_argument("--quiet", action="store_true", help="no per-stage progress lines")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    set_dir = public_eval.shuttleset_dir(a.shuttleset)
    out = a.out.expanduser() if a.out else (OUT if a.split == "train" else DATA / "shuttleset_test")
    clips, labels, preds = out / "clips", out / "labels", out / "preds"
    for d in (clips, labels, preds):
        d.mkdir(parents=True, exist_ok=True)
    idx_p = out / "index.json"
    index: dict[str, Any] = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {}
    store = Store(out / "data")

    items = plan(set_dir, a.window, a.split)
    if a.only:
        items = [i for i in items if i["id"] in a.only]
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} clips from {len({i['match']['id'] for i in items})} training matches", flush=True)
    t_all = time.monotonic()
    run_times: list[float] = []
    todo = sum(1 for i in items if index.get(i["id"], {}).get("state") != "succeeded")
    for k, it in enumerate(items, 1):
        cid, m, win = it["id"], it["match"], it["window"]
        if index.get(cid, {}).get("state") == "succeeded":
            continue
        clip = clips / f"{cid}.mp4"
        if not clip.exists():
            try:
                lo = max(0.0, win[0] - 2)
                src = public_eval.download_720(m["youtube_id"], lo, win[1] + 2, out / "raw")
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
        doc["split"] = a.split
        schema.save(doc, labels / f"{cid}.json")
        entry = {"match_id": m["id"], "video": m["video"], "window": [round(w, 3) for w in win],
                 "hits": len(doc["hits"]), "fps": info["fps"], "label_fps": it["fps"]}
        if a.no_run:
            index[cid] = {**entry, "state": "labelled"}
        else:
            print(f"=== [{k}/{len(items)}] {cid} {m['video']} [{win[0]:.0f}s, {win[1]:.0f}s] "
                  f"{len(doc['hits'])} hits  (elapsed {time.monotonic() - t_all:.0f}s)", flush=True)
            r = real_run.run_clip(store, clip, {"players_stride": a.stride}, preds,
                                  None if a.quiet else StagePrinter())
            index[cid] = {**entry, "state": r["state"], "error": r["error"], "artifacts": r["artifacts"],
                          "wall_s": r["wall_s"], "stages": {n: s["s"] for n, s in r["stages"].items()}}
            run_times.append(r["wall_s"])
            todo -= 1
            avg = sum(run_times) / len(run_times)
            done = sum(1 for v in index.values() if v.get("state") == "succeeded") + (r["state"] == "succeeded")
            print(f"    {cid}: {r['state']}{'（' + str(r['error']) + '）' if r['error'] else ''}  用时 {r['wall_s']:.0f}s", flush=True)
            if r["state"] == "succeeded":
                print(clip_summary(r, doc, preds / f"{cid}.json"), flush=True)
            print(f"    进度：已完成 {done} 段，本轮剩余 {todo} 段，平均 {avg / 60:.1f} 分钟/段，"
                  f"预计还需 {todo * avg / 3600:.1f} 小时", flush=True)
        idx_p.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in index.values() if v.get("state") == "succeeded")
    print(f"\ndone: {ok} clips succeeded, index {idx_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
