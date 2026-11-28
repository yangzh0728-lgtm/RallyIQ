"""train_v11 pre-label batch (from own-v14/batch_v14.py). Code = train_v11/repo (claude/train-v11), profile own + own batch config.
usage: RALLYIQ_BATCH_OUT=<dir with queue.json> python batch_v11.py"""
import json, shutil, subprocess, sys, time, os
from pathlib import Path
WT = Path.home() / "Desktop/RallyIQ/train_v11/repo"; sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT / "scripts"))
import real_run, manual_court
from rallyiq.store import Store
from rallyiq.pipeline.runner import MANUAL_CALIBRATION, run_job
from rallyiq.pipeline.profiles import config_from_env
from rallyiq.contracts import STAGES
O = Path(os.environ["RALLYIQ_BATCH_OUT"]).expanduser(); SRC = Path.home() / "Desktop/RallyIQ/train_v11/src"
BASE = {"players_stride": 2, "players_far_boost": "auto", "parallel_stages": "auto", "players_min_inside_frac": 0.25,
        "events": {"swing_veto": 0.1, "swing_veto_keep_exchange_ms": 2700, "min_no_hit_rally_ms": 2500, "landed_ms": 0}}
CFG = config_from_env(BASE, "own")
store = Store(O / "data"); res_p = O / "results.json"
results = json.loads(res_p.read_text()) if res_p.exists() else []
print("code", subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), CFG, flush=True)
for q in json.loads((O / "queue.json").read_text()):
    cid = q["video"]
    if cid in {r["clip_id"] for r in results}: continue
    src = SRC / f"{cid}.mp4"; print(f"=== {cid} {time.strftime('%H:%M:%S')} court {q['court']['kind']}", flush=True)
    r = {"clip_id": cid, "court": q["court"]}
    try:
        vid, dest = store.create_video(src.name, ".mp4"); shutil.copyfile(src, dest); store.register_video(vid, src.name, dest)
        import cv2; cap = cv2.VideoCapture(str(src)); n = int(cap.get(7)); cap.release()
        c = q["court"]
        if c["kind"] == "points":
            cal = real_run.manual_calibration(vid, json.loads(Path(c["file"]).read_text()), n, c.get("neighbours"))
        else:
            size, poses = manual_court.load_labels(Path(c["file"]))
            cal = manual_court.build(vid, size, poses, (1920, 1080), n, source=Path(c["file"]).name)
        if c["kind"] == "label" and c.get("neighbours"):
            from rallyiq.contracts.geometry import COURT_CORNERS_M, apply_homography, invert_homography
            cal = cal.model_copy(update={"segments": [s.model_copy(update={"other_courts": [apply_homography(invert_homography(s.homography), [[x + dx, y + dy] for x, y in COURT_CORNERS_M]) for dx, dy in c["neighbours"]]}) for s in cal.segments]})
        store.save_artifact(vid, MANUAL_CALIBRATION, cal)
        cfg = dict(CFG); cfg["match_format"] = q.get("match_format", "auto"); r["match_format"] = cfg["match_format"]
        t0 = time.monotonic(); job = run_job(store, store.create_job(vid, list(STAGES)), cfg)
        art = store.video_dir(vid) / "artifacts"
        r.update({"video_id": vid, "state": job.state, "error": job.error, "wall_s": round(time.monotonic() - t0, 1), "artifacts": str(art),
                  "stages": {s.name: {"state": s.state, "error": s.error} for s in job.stages}})
        r.update(real_run.clip_stats(art))
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); r.update({"state": "crashed", "error": repr(e)})
    results.append(r); res_p.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "stages"}, ensure_ascii=False), flush=True)
print("BATCH DONE", time.strftime('%H:%M:%S'), flush=True)
