"""Measure accuracy on public labelled datasets, without hand labelling.

    # ShuttleSet: court corners, hits (+ near/far), rallies on BWF broadcast singles
    python scripts/public_eval.py shuttleset --matches 6
    # Shuttlecock Trajectory Dataset (TrackNet): per-frame shuttle position
    python scripts/public_eval.py tracknet --root ~/rallyiq-data/public/tracknet --max-rallies 40

Outputs go to ~/rallyiq-data/public/<dataset>/ (labels/, preds/, report.md). Nothing is
written into the repository. Coverage and caveats:

* ShuttleSet has no per-frame shuttle positions, so it scores court / hits / rallies only.
  Its rally end is approximate (last stroke + 1 s). Videos are BWF YouTube broadcasts
  (personal research use only, same terms as evaluation/README.md).
* The TrackNet dataset scores the shuttle only. Our shuttle checkpoint was trained on its
  train split, so the test split is an in-domain upper bound (same broadcast style).
* Neither covers side/fixed cameras or phone footage, nor doubles.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import benchmark, schema  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from evaluation.public import shuttleset as SS  # noqa: E402
from evaluation.public import tracknet as TN  # noqa: E402
from evaluation.tools.extract_clips import cut, probe  # noqa: E402

DATA = Path.home() / "rallyiq-data" / "public"
COACHAI = "https://github.com/wywyWang/CoachAI-Projects.git"


def _bench(labels: Path, preds: Path, manifest: dict[str, dict], out: Path) -> str:
    rep = benchmark.run(labels, preds, manifest, None, False)
    md = benchmark.to_markdown(rep)
    old = M.HIT_TOL_FRAMES
    try:  # ShuttleSet hit frames are hand-picked; also show a looser tolerance
        M.HIT_TOL_FRAMES = 6
        loose = benchmark.run(labels, preds, manifest, None, False)["overall"]["hits"]
    finally:
        M.HIT_TOL_FRAMES = old
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md += f"\n\n击球（±6 帧容差）：{json.dumps(loose, ensure_ascii=False, default=str)}\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    return md


# ------------------------------------------------------------------ ShuttleSet
def shuttleset_dir(arg: Path | None) -> Path:
    if arg:
        return arg.expanduser()
    repo = DATA / "CoachAI-Projects"
    if not (repo / "ShuttleSet" / "set" / "match.csv").exists():
        DATA.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", COACHAI, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "ShuttleSet"], check=True)
    return repo / "ShuttleSet" / "set"


def download_720(youtube_id: str, a: float, b: float, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{youtube_id}__{int(a)}-{int(b)}.mp4"
    if not out.exists():
        subprocess.run(["yt-dlp", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]",
                        "--merge-output-format", "mp4", "--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts",
                        "-o", str(out), f"https://www.youtube.com/watch?v={youtube_id}"], check=True)
    return out


def run_shuttleset(a: argparse.Namespace) -> int:
    import real_run
    from rallyiq.store import Store

    set_dir = shuttleset_dir(a.shuttleset)
    out = DATA / "shuttleset"
    labels, preds, clips = out / "labels", out / "preds", out / "clips"
    for d in (labels, preds, clips):
        d.mkdir(parents=True, exist_ok=True)
    matches = SS.load_matches(set_dir)
    chosen = SS.eval_matches(matches, a.matches)
    manifest, results = {}, []
    store = Store(out / "data")
    for m in chosen:
        cid = f"ss{int(m['id']):02d}"
        set_csv = set_dir / m["video"] / "set1.csv"
        if not set_csv.exists():
            print(f"[跳过] {cid}: 没有 set1.csv")
            continue
        strokes = SS.load_strokes(set_csv)
        fps = SS.video_fps(strokes)
        win = SS.pick_window(strokes, fps, a.window)
        clip = clips / f"{cid}.mp4"
        if not clip.exists():
            try:
                lo = max(0.0, win[0] - 2)
                src = download_720(m["youtube_id"], lo, win[1] + 2, out / "raw")
                cut(src, win[0] - lo, win[1] - win[0], clip)
            except subprocess.CalledProcessError as e:
                print(f"[跳过] {cid}: 下载/切片失败 {e}")
                continue
        info = probe(clip)
        if abs(info["fps"] - fps) > 0.5:
            print(f"[注意] {cid}: 视频 {info['fps']} fps，ShuttleSet 帧号按 {fps} fps 换算")
        clip.with_suffix(".meta.json").write_text(json.dumps(
            {"clip_id": cid, "video_id": m["youtube_id"], "start_ms": int(round(win[0] * 1000)),
             "duration_ms": int(round((win[1] - win[0]) * 1000)), **info}), encoding="utf-8")
        doc = SS.build_label(cid, m, strokes, fps, win, info["width"], info["height"])
        schema.save(doc, labels / f"{cid}.json")
        manifest[cid] = SS.manifest_entry(cid, m, win)
        if a.skip_run and (preds / f"{cid}.json").exists():
            continue
        print(f"=== {cid} {m['video']} [{win[0]:.0f}s, {win[1]:.0f}s] {len(doc['hits'])} hits", flush=True)
        results.append(real_run.run_clip(store, clip, {"players_stride": 1}, preds))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_bench(labels, preds, manifest, out))
    print(f"\n报告：{out / 'report.md'}")
    return 0


# ------------------------------------------------------------------ TrackNet
def run_tracknet(a: argparse.Namespace) -> int:
    from rallyiq.perception.shuttle.__main__ import main as shuttle_main

    root = a.root.expanduser()
    out = DATA / "tracknet"
    labels, preds, work = out / "labels", out / "preds", out / "work"
    for d in (labels, preds, work):
        d.mkdir(parents=True, exist_ok=True)
    items = TN.rallies(root, a.split)
    if not items:
        print(f"{root}/{a.split} 下没有找到 <match>/csv/*_ball.csv + video/*.mp4", file=sys.stderr)
        return 1
    step = max(1, len(items) // a.max_rallies)
    manifest = {}
    for cid, video, csv_path in items[::step][: a.max_rallies]:
        info = probe(video)
        n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True, check=True).stdout.strip() or 0)
        schema.save(TN.build_label(cid, csv_path, info["width"], info["height"], info["fps"], n), labels / f"{cid}.json")
        manifest[cid] = {"id": cid, "split": "eval", "conditions": {"source": "tracknet_" + a.split,
                                                                     "camera": "broadcast_main", "discipline": "singles",
                                                                     "venue": video.parent.parent.name}}
        if not (preds / f"{cid}.json").exists():
            rc = shuttle_main([str(video), "--out", str(work / f"{cid}.json"), "--benchmark-clip-id", cid,
                               "--benchmark-start-ms", "0", "--benchmark-out", str(preds / f"{cid}.json")])
            if rc != 0:
                print("羽毛球检测失败（没有权重？见 rallyiq/perception/shuttle/README.md）", file=sys.stderr)
                return rc
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_bench(labels, preds, manifest, out))
    print(f"\n报告：{out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shuttleset")
    s.add_argument("--shuttleset", type=Path, help="ShuttleSet/set 目录（默认自动 clone 到 ~/rallyiq-data/public）")
    s.add_argument("--matches", type=int, default=6)
    s.add_argument("--window", type=float, default=90.0, help="每场取多少秒（整回合）")
    s.add_argument("--skip-run", action="store_true", help="已有预测的片段不重跑")
    t = sub.add_parser("tracknet")
    t.add_argument("--root", type=Path, required=True, help="解压后的 Shuttlecock Trajectory Dataset 根目录")
    t.add_argument("--split", default="test")
    t.add_argument("--max-rallies", type=int, default=40)
    a = ap.parse_args(argv)
    return run_shuttleset(a) if a.cmd == "shuttleset" else run_tracknet(a)


if __name__ == "__main__":
    raise SystemExit(main())
