"""Find out why shuttle detection is weak on a clip, and which setting fixes it.

    python scripts/shuttle_sweep.py --clip ko26_court3                 # uses ~/rallyiq-data/runs/latest
    python scripts/shuttle_sweep.py --clip ko26_court3 --clip ae26_ms_f
    python scripts/shuttle_sweep.py ~/rallyiq-data/runs/latest/data/videos/<video_id>
    python scripts/shuttle_sweep.py --clip ko26_court3 --no-detect     # only re-analyse saved candidates

Works on a finished ``scripts/real_run.py`` run (it needs the clip's calibration, players
and events artifacts). For each clip:

1. ROI report: how large the target court is in the frame, and the crop/zoom ``roi=auto``
   would use per camera segment.
2. Re-runs only the shuttle detector with ``roi=off`` and ``roi=auto`` at a low peak
   threshold (0.05, top 5 peaks), caching the candidates in ``work/sweep/``.
3. For the saved run and each re-run, at peak thresholds 0.2 / 0.1 / 0.05, prints how many
   frames have any raw peak, a peak over the target court, a tracked point, and the
   fraction of rally frames covered. All label-free counts, not accuracy.

Reading the table: low "raw peak" means the network does not see the shuttle (try the ROI
crop or fine-tuning); high "raw peak" but low "on target court" means the peaks are on
another court or in the stands; high "on target court" but low "tracked" means the tracker
or court lock throws them away.

Writes ``shuttle_sweep.md`` and ``shuttle_sweep.json`` next to the run's report.
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

from rallyiq.contracts import (
    CourtCalibration,
    EventsResult,
    PlayerTracks,
    VideoMeta,
)
from rallyiq.perception.shuttle.diagnose import (
    coverage,
    format_table,
    roi_report,
)
from rallyiq.perception.shuttle.stage import (
    CANDIDATES_FILE,
    save_candidates,
    segments_for,
)

DEFAULT_RUN = Path.home() / "rallyiq-data" / "runs" / "latest"
THRESHOLDS = (0.2, 0.1, 0.05)
SWEEP_THRESHOLD = 0.05
SWEEP_TOP_K = 5


def _load(path: Path, model):
    return model.model_validate_json(path.read_text()) if path.exists() else None


def _read_candidates(path: Path) -> tuple[dict[int, list[tuple]], tuple[int, int]]:
    doc = json.loads(path.read_text())
    w, h = doc["frame_size"]
    return {int(f): [tuple(c) for c in v["candidates"]] for f, v in doc["frames"].items()}, (w, h)


def video_dirs_for(run: Path, clips: list[str]) -> list[tuple[str, Path]]:
    res = json.loads((run / "results.json").read_text())
    by_clip = {r["clip_id"]: Path(r["artifacts"]).parent for r in res}
    missing = [c for c in clips if c not in by_clip]
    if missing:
        raise SystemExit(f"{missing} not in {run / 'results.json'} (have {sorted(by_clip)})")
    return [(c, by_clip[c]) for c in clips]


def source_video(video_dir: Path) -> Path | None:
    return next((p for p in sorted(video_dir.glob("source.*"))), None)


def detect_variant(video: Path, meta: VideoMeta, calib: CourtCalibration | None, roi: str, out: Path,
                   device: str | None) -> float:
    from rallyiq.perception.shuttle.detector import (
        DetectorConfig,
        HeatmapShuttleDetector,
        find_weights,
    )

    weights = find_weights()
    if weights is None:
        raise SystemExit("no shuttle weights found (see rallyiq/perception/shuttle/README.md)")
    det = HeatmapShuttleDetector(weights, DetectorConfig(device=device, peak_threshold=SWEEP_THRESHOLD,
                                                         top_k=SWEEP_TOP_K))
    t0 = time.monotonic()
    cands, size = det.detect(video, meta.fps, meta.frame_count, segments_for(meta, calib, roi))
    elapsed = time.monotonic() - t0
    save_candidates(out, cands, {}, det.name, size)
    return elapsed


def sweep(clip: str, video_dir: Path, detect: bool, force: bool, device: str | None) -> dict:
    art = video_dir / "artifacts"
    meta = _load(art / "probe.json", VideoMeta)
    if meta is None:
        raise SystemExit(f"{art / 'probe.json'} missing")
    calib = _load(art / "calibration.json", CourtCalibration)
    players = _load(art / "players.json", PlayerTracks)
    events = _load(art / "events.json", EventsResult)
    out: dict = {"clip": clip, "video_dir": str(video_dir), "size": [meta.width, meta.height], "fps": meta.fps,
                 "roi": roi_report(meta, calib), "variants": [], "timing_s": {}}

    variants: list[tuple[str, Path]] = []
    saved = video_dir / "work" / CANDIDATES_FILE
    if saved.exists():
        variants.append(("saved run", saved))
    if detect:
        video = source_video(video_dir)
        if video is None:
            raise SystemExit(f"no source video in {video_dir}")
        sweep_dir = video_dir / "work" / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for roi in ("off", "auto"):
            if roi == "auto" and not any(r.get("roi") for r in out["roi"]):
                print(f"  {clip}: roi=auto would not crop any segment, skipped", flush=True)
                continue
            path = sweep_dir / f"roi-{roi}.json"
            if force or not path.exists():
                print(f"  {clip}: detecting with roi={roi} ...", flush=True)
                out["timing_s"][f"roi={roi}"] = round(detect_variant(video, meta, calib, roi, path, device), 1)
            variants.append((f"roi={roi}", path))
    for name, path in variants:
        cands, size = _read_candidates(path)
        for thr in THRESHOLDS:
            out["variants"].append(coverage(name, meta, cands, size, thr, calib, players, events).as_dict())
    return out


def render(results: list[dict]) -> str:
    from rallyiq.perception.shuttle.diagnose import Coverage

    L = ["# 羽毛球检测诊断（shuttle_sweep）", "",
         ("> 无标注统计，不是精度。raw peak = 有任意热图峰值的帧；on target court = 峰值在目标球场上空；"
          "tracked = 跟踪后保留的检测点；in rallies = 回合内被覆盖的帧比例。"), ""]
    for r in results:
        L += [f"## {r['clip']}  ({r['size'][0]}x{r['size'][1]}, {r['fps']:.0f} fps)", ""]
        if r["roi"]:
            L += ["| 段 | 状态 | 其他球场 | 球场宽/帧宽 | 球场高/帧高 | auto ROI | 放大 |", "|---|---|---|---|---|---|---|"]
            for s in r["roi"]:
                L.append(f"| {s['frames'][0]}–{s['frames'][1]} | {s['status']} | {s['other_courts']} | "
                         f"{s.get('court_w_frac', '–')} | {s.get('court_h_frac', '–')} | {s.get('roi') or '不裁剪'} | "
                         f"{s.get('roi_zoom', '–')} |")
            L.append("")
        if r["timing_s"]:
            L += ["检测耗时：" + "，".join(f"{k} {v}s" for k, v in r["timing_s"].items()), ""]
        L += [format_table([Coverage(**v) for v in r["variants"]]), ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path, nargs="*", help="store video folders (alternative to --clip)")
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="real_run output folder")
    ap.add_argument("--clip", action="append", default=[], help="clip id from the run's results.json")
    ap.add_argument("--no-detect", action="store_true", help="only analyse the saved candidates")
    ap.add_argument("--force", action="store_true", help="re-run detection even if cached")
    ap.add_argument("--device", help="cuda | mps | cpu")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    run = a.run.expanduser().resolve()
    targets = [(d.name, d.expanduser()) for d in a.video_dir]
    if a.clip:
        targets += video_dirs_for(run, a.clip)
    if not targets:
        ap.error("give --clip or a video folder")
    results = [sweep(c, d, not a.no_detect, a.force, a.device) for c, d in targets]
    md = render(results)
    out_dir = run if a.clip else targets[0][1]
    (out_dir / "shuttle_sweep.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (out_dir / "shuttle_sweep.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n-> {out_dir / 'shuttle_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
