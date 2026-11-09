"""Run the full pipeline on real clips and write pre-annotations + a first report.

    python scripts/real_run.py                       # every clip in ~/rallyiq-data/bwf/clips
    python scripts/real_run.py --only ae26_ms_f      # a subset

For each ``<clip_id>.mp4`` (+ ``.meta.json`` written by ``evaluation.tools.extract_clips``):

1. import it into the store and run every pipeline stage, timing each one;
2. convert the artifacts with ``evaluation.from_contracts`` into ``<out>/preds/<clip_id>.json``,
   which ``evaluation.tools.label_tool --prefill`` uses as the starting point for labeling;
3. compute label-free statistics per module (coverage, detection rate, counts, speed).

Fixed-camera clips can carry ``court_points`` in their meta.json (from ``court_points`` in
``evaluation/own_clips.yaml``): named court-line intersections in clip pixels. They are fitted
into a manual calibration for the whole clip, which replaces automatic calibration, the same way
the web app's 4-corner override does. Use this when auto calibration locks onto the wrong court.

When reviewed labels exist in ``evaluation/labels/`` the real benchmark is run as well and
its markdown is appended. Without labels the report only shows how much each module
produced, which is NOT accuracy; the report says so.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import benchmark, from_contracts, schema
from rallyiq.contracts import (
    STAGES,
    CalibrationSegment,
    CourtCalibration,
    CourtKeypoint,
)
from rallyiq.contracts.geometry import (
    COURT_CORNERS_M,
    apply_homography,
    fit_homography,
    invert_homography,
)
from rallyiq.pipeline.runner import (
    MANUAL_CALIBRATION,
    run_job,
)
from rallyiq.store import Store

DEFAULT_CLIPS = Path.home() / "rallyiq-data" / "bwf" / "clips"
DEFAULT_OUT = Path.home() / "rallyiq-data" / "runs" / "latest"


def _load(art: Path, name: str) -> dict[str, Any] | None:
    p = art / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _stride(frames: list[dict[str, Any]]) -> int:
    """The frame step pose ran at (players_stride), read back from the frame numbers."""
    step = 0
    for a, b in itertools.pairwise(frames):
        step = math.gcd(step, b["frame"] - a["frame"])
    return max(step, 1)


def clip_stats(art: Path) -> dict[str, Any]:
    """Label-free numbers: what fraction of frames each module produced output for."""
    meta = _load(art, "probe") or {}
    n = max(int(meta.get("frame_count") or 0), 1)
    out: dict[str, Any] = {"frames": n, "fps": meta.get("fps"), "size": f"{meta.get('width')}x{meta.get('height')}"}

    cal = _load(art, "calibration")
    cal_frames: set[int] | None = None
    if cal:
        good = [s for s in cal["segments"] if s.get("homography") and s.get("status") != "undeterminable"]
        ok = sum(s["end_frame"] - s["start_frame"] + 1 for s in good)
        cal_frames = {i for s in good for i in range(s["start_frame"], s["end_frame"] + 1)}
        errs = [s["reprojection_error_px"] for s in cal["segments"] if s.get("reprojection_error_px") is not None]
        out["court_calibrated_frac"] = round(min(ok / n, 1.0), 3)
        out["court_segments"] = len(cal["segments"])
        out["court_reproj_px_median"] = round(statistics.median(errs), 2) if errs else None

    pl = _load(art, "players")
    if pl:
        frames = pl["frames"]
        need = 4 if pl.get("match_format") == "doubles" else 2

        def all_in(f: dict[str, Any]) -> bool:
            return sum(1 for p in f["players"] if not p.get("interpolated")) >= need

        both = sum(1 for f in frames if all_in(f))
        far = sum(1 for f in frames if any(p.get("side") == "far" and not p.get("interpolated") for p in f["players"]))
        # With players_stride > 1 pose runs on every n-th frame only, so the share is over those frames.
        n_pose = (n - 1) // _stride(frames) + 1
        out["players_model"] = pl.get("model")
        out["players_both_frac"] = round(min(both / n_pose, 1.0), 3)
        out["players_far_frac"] = round(min(far / n_pose, 1.0), 3)
        if cal_frames:
            # Close-ups and replays have no court, so nobody can be placed there:
            # the share over calibrated frames is the number that shows misses.
            on_cal = [f for f in frames if f["frame"] in cal_frames]
            out["players_both_cal_frac"] = round(sum(1 for f in on_cal if all_in(f)) / max(len(on_cal), 1), 3)
        out["players_side_switches"] = len(pl.get("side_switches", []))

    sh = _load(art, "shuttle")
    if sh:
        det = [p for p in sh["points"] if p.get("source", "detected") == "detected"]
        out["shuttle_detector"] = sh.get("detector")
        out["shuttle_detected_frac"] = round(len(det) / n, 3)
        out["shuttle_interp_points"] = len(sh["points"]) - len(det)

    ev = _load(art, "events")
    if ev:
        shots = ev.get("shots", [])
        out["rallies"] = len(ev.get("rallies", []))
        out["hits"] = len(shots)
        out["hits_ok"] = sum(1 for s in shots if s.get("status") == "ok")

    st = _load(art, "strokes")
    if st:
        strokes = [s["stroke"] for s in st.get("shots", [])]
        out["stroke_classifier"] = st.get("classifier")
        out["strokes_typed"] = sum(1 for s in strokes if s.get("shot_type"))
        out["strokes_ok"] = sum(1 for s in strokes if s.get("status") == "ok")
    return out


def court_lookup() -> dict[str, tuple[float, float]]:
    """Court coordinates (m) of every name usable in ``court_points``: line intersections + outer corners."""
    from rallyiq.perception.calibration import court_model as cm

    names = {k.name: k.xy for k in cm.keypoints()}
    return names | {f"{k}_corner": v for k, v in cm.CORNERS.items()}


def manual_calibration(video_id: str, points: dict[str, list[float]], frame_count: int,
                       neighbours: list[list[float]] | None = None) -> CourtCalibration:
    """One manual segment for the whole clip from >= 4 named image points (fixed camera).

    ``neighbours``: other courts in the same hall as [dx, dy] offsets (metres, our court
    frame) of their near-left corner. They become ``other_courts`` so the court lock can
    reject shuttles and people on them; without them a low camera puts much of the hall
    inside our court's air box.
    """
    lookup = court_lookup()
    unknown = sorted(set(points) - set(lookup))
    if unknown:
        raise ValueError(f"unknown court point names: {unknown}")
    if len(points) < 4:
        raise ValueError("court_points needs at least 4 points")
    names = list(points)
    img = [list(map(float, points[n])) for n in names]
    court = [list(lookup[n]) for n in names]
    # fit court -> image so the least-squares error is in pixels, where the points were measured;
    # fitting image -> court lets a 1 px slip on the squashed far court bend the whole near court
    to_img = fit_homography(court, img)
    h = invert_homography(to_img)
    resid = [((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
             for (cx, cy), (px, py) in zip(img, apply_homography(to_img, court))]
    corners = apply_homography(to_img, COURT_CORNERS_M)
    others = [apply_homography(to_img, [[x + dx, y + dy] for x, y in COURT_CORNERS_M])
              for dx, dy in (neighbours or [])]
    seg = CalibrationSegment(
        start_frame=0, end_frame=max(frame_count - 1, 0), homography=h, image_corners=corners,
        other_courts=others,
        keypoints=[CourtKeypoint(name=n, image=i, court=c) for n, i, c in zip(names, img, court)],
        reprojection_error_px=round(max(resid), 2), method="manual", confidence=1.0, status="ok",
        notes=[f"fixed camera: {len(names)} court points from own_clips.yaml"],
    )
    return CourtCalibration(video_id=video_id, segments=[seg])


def run_clip(store: Store, clip: Path, config: dict[str, Any], preds: Path, on_progress=None) -> dict[str, Any]:
    meta_p = clip.with_suffix(".meta.json")
    cmeta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    clip_id = cmeta.get("clip_id", clip.stem)
    video_id, dest = store.create_video(clip.name, clip.suffix)
    shutil.copyfile(clip, dest)
    store.register_video(video_id, clip.name, dest)
    if cmeta.get("court_points"):
        n = int(cmeta.get("frame_count") or round(cmeta["duration_ms"] / 1000 * cmeta["fps"]))
        store.save_artifact(video_id, MANUAL_CALIBRATION, manual_calibration(video_id, cmeta["court_points"], n,
                                                                       cmeta.get("neighbour_courts")))
    job = store.create_job(video_id, list(STAGES))

    t0 = time.monotonic()
    job = run_job(store, job, config, on_progress=on_progress)
    wall = time.monotonic() - t0
    art = store.video_dir(video_id) / "artifacts"
    res: dict[str, Any] = {
        "clip_id": clip_id, "video_id": video_id, "state": job.state, "error": job.error,
        "wall_s": round(wall, 1), "artifacts": str(art),
        "stages": {s.name: {"state": s.state, "error": s.error,
                            "s": round((s.finished_at - s.started_at).total_seconds(), 1)
                            if s.started_at and s.finished_at else None} for s in job.stages},
    }
    res.update(clip_stats(art))
    if cmeta:
        doc = from_contracts.convert(art, clip_id, int(cmeta["start_ms"]))
        schema.save(doc, preds / f"{clip_id}.json")
        res["prefill"] = str(preds / f"{clip_id}.json")
    return res


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{v:.0%}" if pct else str(v)


def render(results: list[dict[str, Any]], bench_md: str | None) -> str:
    L = ["# RallyIQ 真实片段运行报告", ""]
    L.append("> 下表是**无标注统计**：各模块在多少帧上给出了结果，不是精度。"
             "精度要等 `evaluation/labels/` 里有已复核（reviewed）的标注后由基准给出。")
    L.append("")
    cols = [("片段", "clip_id"), ("状态", "state"), ("耗时 s", "wall_s"), ("分辨率", "size"), ("fps", "fps"),
            ("球场已标定", "court_calibrated_frac"), ("标定段", "court_segments"),
            ("重投影 px", "court_reproj_px_median"), ("两名球员", "players_both_frac"), ("两名球员/已标定帧", "players_both_cal_frac"),
            ("远端球员", "players_far_frac"), ("换边", "players_side_switches"),
            ("球检出", "shuttle_detected_frac"), ("回合", "rallies"), ("击球", "hits"), ("击球 ok", "hits_ok")]
    L.append("| " + " | ".join(c for c, _ in cols) + " |")
    L.append("|" + "---|" * len(cols))
    for r in results:
        L.append("| " + " | ".join(_fmt(r.get(k), k.endswith("_frac")) for _, k in cols) + " |")
    L += ["", "## 各阶段耗时（秒）与错误", ""]
    L.append("| 片段 | " + " | ".join(STAGES) + " |")
    L.append("|" + "---|" * (len(STAGES) + 1))
    for r in results:
        cells = []
        for s in STAGES:
            st = r["stages"].get(s, {})
            cells.append(_fmt(st.get("s")) if st.get("state") == "succeeded" else (st.get("state") or "–"))
        L.append(f"| {r['clip_id']} | " + " | ".join(cells) + " |")
    errs = [(r["clip_id"], n, s["error"]) for r in results for n, s in r["stages"].items() if s.get("error")]
    if errs:
        L += ["", "错误："] + [f"- `{c}` {n}: {e}" for c, n, e in errs]
    models = {k: sorted({str(r.get(k)) for r in results}) for k in ("players_model", "shuttle_detector")}
    L += ["", f"模型：players={models['players_model']}，shuttle={models['shuttle_detector']}"]
    if bench_md:
        L += ["", "## 基准（已复核标注）", "", bench_md]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--far-boost", choices=("auto", "on", "off"), default="auto",
                    help="players: second pose pass on the far half of the court (auto: only when the far player is small)")
    ap.add_argument("--stride", type=int, default=1, help="players: pose every n-th frame")
    ap.add_argument("--parallel", choices=("auto", "on", "off"), default="auto",
                    help="run shuttle detection alongside players (auto: only when the shuttle model has a GPU)")
    ap.add_argument("--match-format", choices=("auto", "singles", "doubles"), help="players: match format")
    ap.add_argument("--events", nargs="*", default=[], metavar="KEY=VALUE",
                    help="events settings, e.g. swing_veto=0.1 swing_veto_keep_exchange_ms=2700")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    out = a.out.expanduser()
    preds = out / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    store = Store(out / "data")
    config: dict[str, Any] = {"players_stride": a.stride, "players_far_boost": a.far_boost,
                              "parallel_stages": a.parallel}
    if a.match_format:
        config["match_format"] = a.match_format
    if a.events:
        from scripts.diag_tracklets import parse_settings

        config["events"] = parse_settings(a.events)

    clips = sorted(p for p in a.clips.expanduser().glob("*.mp4")
                   if not p.name.startswith("._") and (not a.only or p.stem in a.only))
    if not clips:
        print(f"{a.clips} 里没有片段，先运行 python -m evaluation.tools.extract_clips", file=sys.stderr)
        return 1
    results = []
    for c in clips:
        print(f"=== {c.stem}", flush=True)
        r = run_clip(store, c, config, preds)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)

    bench_md = None
    labels = ROOT / "evaluation" / "labels"
    if any(labels.glob("*.json")):
        md = out / "bench.md"
        benchmark.main(["--pred", str(preds), "--all-splits", "--markdown", str(md), "--json", str(out / "bench.json")])
        bench_md = md.read_text(encoding="utf-8") if md.exists() else None

    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report = out / "report.md"
    report.write_text(render(results, bench_md), encoding="utf-8")
    print(f"\n报告：{report}\n预标注：{preds}")
    return 0 if all(r["state"] == "succeeded" for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
