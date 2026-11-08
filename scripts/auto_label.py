"""Label a clip's shuttle positions automatically; a person only spot-checks the result.

    python scripts/auto_label.py ko26_court2              # 1. propose (writes numbered sheets)
    python scripts/auto_label.py ko26_court2 --apply --not-shuttle 3,17 --unsure 5
                                                          # 2. apply, after looking at the sheets
    python scripts/auto_label.py ko26_court2 --preview    # 3. video with the labels drawn

Starting point: the existing label file, or a new one from the pipeline's prediction
(``preds/<clip>.json`` of the latest ``scripts/real_run.py`` run). Those predictions mark every
frame the tracker did not keep as "not visible", including frames where the shuttle is
plainly there (in the server's hand, on the floor, a missed fast shot).

``propose`` runs a fine-tuned checkpoint (default: the last epoch of the latest
``shuttle_loop.py`` round) over every frame and decides per frame:

* the label was edited by hand (differs from the prediction it started from): keep it;
* label and model agree (same place, or both see nothing): keep it;
* labeled "not visible", but the model fires with score >= ``--min-score`` (0.4) and a
  neighbouring frame (±2) has a model peak or a label close by: **proposed** as visible at
  the model's point. Every proposal gets a number on a contact sheet (2x crops, magenta
  circle = proposed point) and must be looked at;
* anything else where they disagree (weak or isolated peak, model elsewhere): the label is
  **removed**, so training ignores the frame instead of learning a guess.

``--apply`` writes the label file (status ``reviewed``): proposals become visible, except
sheet numbers given with ``--not-shuttle`` (set "not visible") or ``--unsure`` (removed).
An ``auto_label`` block in the file records the model, the counts and the accepted frames.

``--preview`` renders ``<out>/<clip>-preview.mp4`` (half size): green circle = label,
cyan = accepted proposal, frame number top left, so a screenshot names the frame.

Outputs go to ``~/rallyiq-data/autolabel/<clip>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import schema

LABELS = ROOT / "evaluation" / "labels"
WEIGHTS = ROOT / "weights" / "shuttle"
DEFAULT_CLIPS = Path.home() / "rallyiq-data" / "bwf" / "clips"
RUNS = Path.home() / "rallyiq-data" / "runs"
OUT = Path.home() / "rallyiq-data" / "autolabel"

AGREE_PX = 15.0  # at 1080p: label and model peak count as the same point
STEP_PX = 60.0  # at 1080p, per frame: how far a shuttle can move between neighbouring frames


# ------------------------------------------------------------------ frames <-> ms


def frame_of(t_ms: float, start_ms: int, fps: float) -> int:
    return round((t_ms - start_ms) * fps / 1000.0)


def t_of(frame: int, start_ms: int, fps: float) -> int:
    return round(start_ms + frame * 1000.0 / fps)


def shuttle_by_frame(doc: dict, start_ms: int) -> dict[int, tuple[float, float] | None]:
    fps = float(doc["video"]["fps"])
    out = {}
    for s in doc.get("shuttle", []):
        f = frame_of(s["t_ms"], start_ms, fps)
        if f >= 0:
            out[f] = (float(s["xy"][0]), float(s["xy"][1])) if s.get("visible") and s.get("xy") else None
    return out


def labels_sha(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc.get("shuttle", []), sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------- decisions


def _near(a, b, px: float) -> bool:
    return a is not None and b is not None and math.hypot(a[0] - b[0], a[1] - b[1]) <= px


def consistent(f: int, xy, peaks: dict, labels: dict, scale: float, reach: int = 2) -> bool:
    """A neighbouring frame has a model peak or a visible label within plausible reach."""
    for d in range(1, reach + 1):
        for g in (f - d, f + d):
            r = STEP_PX * scale * d
            p = peaks.get(g)
            if (p is not None and _near(xy, p[:2], r)) or _near(xy, labels.get(g), r):
                return True
    return False


def decide(labels: dict[int, tuple | None], human: set[int], peaks: dict[int, tuple | None],
           height: int, min_score: float = 0.4) -> list[dict]:
    """Frames whose label should change: action ``propose`` (visible at the model's point,
    to be looked at), ``ignore`` (remove the label) or ``invisible`` (new frame, model sees nothing)."""
    scale = height / 1080.0
    out = []
    for f in sorted(set(labels) | set(peaks)):
        if f in human:
            continue
        absent = f not in labels
        lab, p = labels.get(f), peaks.get(f)
        xy = None if p is None else (p[0], p[1])
        score = None if p is None else p[2]
        if lab is None:
            if p is None:
                if absent:
                    out.append({"frame": f, "action": "invisible"})
                continue  # both see nothing
            if score >= min_score and consistent(f, xy, peaks, labels, scale):
                out.append({"frame": f, "action": "propose", "xy": [round(xy[0], 1), round(xy[1], 1)],
                            "score": round(score, 3)})
            elif not absent:
                out.append({"frame": f, "action": "ignore", "why": "weak or isolated model peak",
                            "score": round(score, 3)})
            continue
        if p is None or _near(lab, xy, AGREE_PX * scale):
            continue  # labeled visible: the model agrees, or simply misses it
        out.append({"frame": f, "action": "ignore", "why": "model peak elsewhere", "score": round(score, 3),
                    "xy": [round(xy[0], 1), round(xy[1], 1)]})
    return out


def human_frames(doc: dict, prefill: dict | None, start_ms: int) -> set[int]:
    """Frames whose label differs from the prediction the file started from."""
    if not prefill:
        return set()
    before = shuttle_by_frame(prefill, start_ms)
    now = shuttle_by_frame(doc, start_ms)
    return {f for f, v in now.items() if f not in before or before[f] != v}


def find_prefill(clip_id: str, doc: dict | None, start_ms: int, runs: Path) -> Path | None:
    """The prediction file a label file started from: the one it differs least from
    (for a new label file: the latest one)."""
    cands = sorted(runs.glob(f"*/preds/{clip_id}.json"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
    if not cands or doc is None:
        return cands[-1] if cands else None
    now = shuttle_by_frame(doc, start_ms)

    def diff(p: Path) -> int:
        before = shuttle_by_frame(schema.load(p), start_ms)
        return sum(1 for f, v in now.items() if before.get(f, "x") != v)

    return min(reversed(cands), key=diff)


# -------------------------------------------------------------------- images


def numbered_sheets(video: Path, props: list[dict], out_dir: Path, clip_id: str, per: int = 48, cols: int = 6,
                    region: tuple[int, int] = (160, 120), zoom: int = 2) -> list[Path]:
    """Contact sheets of the proposals: tile ``n`` = ``props[n-1]``, magenta circle = proposed point."""
    import cv2
    import numpy as np

    rw, rh = region
    tw, th = rw * zoom, rh * zoom
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for i, m in enumerate(props, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"])
        ok, fr = cap.read()
        img = np.zeros((th, tw, 3), np.uint8)
        if ok:
            cx, cy = m["xy"]
            x0 = int(min(max(cx - rw / 2, 0), max(fr.shape[1] - rw, 0)))
            y0 = int(min(max(cy - rh / 2, 0), max(fr.shape[0] - rh, 0)))
            crop = fr[y0:y0 + rh, x0:x0 + rw]
            img[:crop.shape[0] * zoom, :crop.shape[1] * zoom] = cv2.resize(
                crop, (crop.shape[1] * zoom, crop.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
            cv2.circle(img, (int((cx - x0) * zoom), int((cy - y0) * zoom)), 14, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (62, 26), (0, 0, 0), -1)
        cv2.putText(img, str(i), (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, th - 18), (tw, th), (0, 0, 0), -1)
        cv2.putText(img, f"frame {m['frame']}  score {m['score']:.2f}", (4, th - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (tw - 1, th - 1), (90, 90, 90), 1)
        tiles.append(img)
    cap.release()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{clip_id}-sheet-*.jpg"):
        old.unlink()
    paths = []
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        chunk += [np.zeros((th, tw, 3), np.uint8)] * ((-len(chunk)) % cols)
        rows = [np.hstack(chunk[r:r + cols]) for r in range(0, len(chunk), cols)]
        p = out_dir / f"{clip_id}-sheet-{k // per + 1:02d}.jpg"
        cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


def render_preview(video: Path, doc: dict, start_ms: int, out: Path, scale: float = 0.5) -> Path:
    import cv2

    labels = shuttle_by_frame(doc, start_ms)
    accepted = set((doc.get("auto_label") or {}).get("accepted_frames", []))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or float(doc["video"]["fps"])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        img = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        lab = labels.get(f, "absent")
        if lab not in ("absent", None):
            color = (255, 255, 0) if f in accepted else (0, 255, 0)
            cv2.circle(img, (int(lab[0] * scale), int(lab[1] * scale)), 12, color, 2, cv2.LINE_AA)
        if lab == "absent":
            state = "ignored"
        elif lab is None:
            state = "no shuttle"
        else:
            state = "auto" if f in accepted else "label"
        cv2.rectangle(img, (0, 0), (330, 30), (0, 0, 0), -1)
        cv2.putText(img, f"frame {f}  {state}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        vw.write(img)
        f += 1
    cap.release()
    vw.release()
    return out


# ------------------------------------------------------------------- steps


def load_clip(clip_id: str, clips_dir: Path) -> tuple[Path, dict]:
    video, meta_p = clips_dir / f"{clip_id}.mp4", clips_dir / f"{clip_id}.meta.json"
    if not video.exists() or not meta_p.exists():
        raise SystemExit(f"找不到 {video}：先运行 python -m evaluation.tools.extract_clips --download --only {clip_id}")
    return video, json.loads(meta_p.read_text(encoding="utf-8"))


def latest_loop_model(weights_dir: Path) -> Path | None:
    lasts = sorted((weights_dir / "loop").glob("round-*-last.pt"))
    if lasts:
        return lasts[-1]
    from rallyiq.perception.shuttle.detector import find_weights

    return find_weights(None)


def propose(clip_id: str, a) -> dict:
    from evaluation.tools.label_tool import load_or_create
    from rallyiq.perception.shuttle.training import (
        TrainClip,
        TrainConfig,
        disagreements,
    )

    video, meta = load_clip(clip_id, a.clips)
    start_ms, fps = int(meta["start_ms"]), float(meta["fps"])
    label_path = a.labels / f"{clip_id}.json"
    existing = schema.load(label_path) if label_path.exists() else None
    pre_p = Path(a.prefill).expanduser() if a.prefill else find_prefill(clip_id, existing, start_ms, a.runs)
    if existing is None and pre_p is None:
        raise SystemExit(f"{clip_id} 还没有标注，也没有预测文件：先运行 python scripts/real_run.py --only {clip_id}")
    doc = existing or load_or_create(clip_id, meta, a.labels, pre_p)
    prefill = schema.load(pre_p) if pre_p else None
    labels = shuttle_by_frame(doc, start_ms)
    import cv2

    n = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    every = {f: labels.get(f) for f in range(n)}  # run the model over every frame of the clip
    model = Path(a.model).expanduser() if a.model else latest_loop_model(a.weights_dir)
    if model is None:
        raise SystemExit("找不到模型：先跑一轮 scripts/shuttle_loop.py，或用 --model 指定")
    print(f"{clip_id}: {n} 帧，标注起点 {pre_p.name if pre_p else '已有标注'}，模型 {model}")
    peaks: dict = {}
    disagreements(model, [TrainClip(clip_id, video, fps, every)], TrainConfig(device=a.device), peaks=peaks)
    human = human_frames(doc, prefill, start_ms)
    labels_now = {f: v for f, v in labels.items() if 0 <= f < n}
    changes = decide(labels_now, human, peaks.get(clip_id, {}), int(meta["height"]), a.min_score)
    props = [c for c in changes if c["action"] == "propose"]
    for i, c in enumerate(props, 1):
        c["tile"] = i
    out_dir = a.out / clip_id
    sheets = numbered_sheets(video, props, out_dir, clip_id) if props else []
    plan = {
        "clip": clip_id, "created": time.strftime("%Y-%m-%d %H:%M"), "model": str(model),
        "prefill": str(pre_p) if pre_p else None, "labels_sha": labels_sha(doc), "frames": n,
        "start_ms": start_ms, "fps": fps, "human_frames": len(human), "min_score": a.min_score,
        "new_doc": None if existing else doc, "changes": changes, "sheets": [str(p) for p in sheets],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{clip_id}-plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    count = {k: sum(1 for c in changes if c["action"] == k) for k in ("propose", "ignore", "invisible")}
    print(f"手工改过、保持不变：{len(human)} 帧")
    print(f"模型提议为可见（需要看拼图确认）：{count['propose']} 帧")
    print(f"模型和标注矛盾、改为不参与训练：{count['ignore']} 帧")
    if count["invisible"]:
        print(f"新增“看不见”：{count['invisible']} 帧")
    for p in sheets:
        print(f"拼图：{p}")
    print(f"看完拼图后：python scripts/auto_label.py {clip_id} --apply --not-shuttle 编号,… --unsure 编号,…")
    return plan


def parse_ids(s: str | None) -> set[int]:
    return {int(x) for x in s.replace("，", ",").split(",") if x.strip()} if s else set()


def apply_plan(plan: dict, doc: dict, not_shuttle: set[int], unsure: set[int]) -> dict:
    start_ms, fps = plan["start_ms"], plan["fps"]
    tiles = {c["tile"] for c in plan["changes"] if c["action"] == "propose"}
    bad = (not_shuttle | unsure) - tiles
    if bad:
        raise SystemExit(f"拼图里没有这些编号：{sorted(bad)}（共 {len(tiles)} 个）")
    by_t = {s["t_ms"]: s for s in doc["shuttle"]}
    accepted, counts = [], {"accepted": 0, "not_shuttle": 0, "unsure": 0, "ignored": 0, "invisible": 0}
    for c in plan["changes"]:
        t = t_of(c["frame"], start_ms, fps)
        act = c["action"]
        if act == "propose":
            if c["tile"] in not_shuttle:
                by_t[t] = {"t_ms": t, "visible": False, "xy": None}
                counts["not_shuttle"] += 1
            elif c["tile"] in unsure:
                by_t.pop(t, None)
                counts["unsure"] += 1
            else:
                by_t[t] = {"t_ms": t, "visible": True, "xy": c["xy"]}
                accepted.append(c["frame"])
                counts["accepted"] += 1
        elif act == "ignore":
            by_t.pop(t, None)
            counts["ignored"] += 1
        elif act == "invisible":
            by_t[t] = {"t_ms": t, "visible": False, "xy": None}
            counts["invisible"] += 1
    doc["shuttle"] = sorted(by_t.values(), key=lambda s: s["t_ms"])
    doc["status"] = "reviewed"
    doc["auto_label"] = {"model": plan["model"], "date": time.strftime("%Y-%m-%d %H:%M"),
                         "min_score": plan["min_score"], "human_frames_kept": plan["human_frames"], **counts,
                         "accepted_frames": accepted}
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_id")
    ap.add_argument("--apply", action="store_true", help="write the label file from the last proposal")
    ap.add_argument("--not-shuttle", help="sheet numbers whose magenta point is not the shuttle: 3,17")
    ap.add_argument("--unsure", help="sheet numbers that cannot be told: these frames are left out")
    ap.add_argument("--force", action="store_true", help="apply even if the label file changed since proposing")
    ap.add_argument("--preview", action="store_true", help="render a video with the labels drawn")
    ap.add_argument("--model", help="checkpoint that proposes points (default: last epoch of the latest loop round)")
    ap.add_argument("--prefill", help="prediction file the labels start from (default: found in --runs)")
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--device", help="cuda | mps | cpu")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--weights-dir", type=Path, default=WEIGHTS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.clips, a.runs, a.out = a.clips.expanduser(), a.runs.expanduser(), a.out.expanduser()
    cid = a.clip_id
    label_path = a.labels / f"{cid}.json"

    if a.preview:
        video, meta = load_clip(cid, a.clips)
        if not label_path.exists():
            raise SystemExit(f"没有 {label_path}")
        out = render_preview(video, schema.load(label_path), int(meta["start_ms"]), a.out / cid / f"{cid}-preview.mp4")
        print(f"预览视频：{out}（绿圈=原有标注，青色圈=自动加的，左上角是帧号）")
        return 0

    if a.apply:
        plan_p = a.out / cid / f"{cid}-plan.json"
        if not plan_p.exists():
            raise SystemExit(f"没有 {plan_p}：先运行 python scripts/auto_label.py {cid}")
        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        doc = schema.load(label_path) if label_path.exists() else plan["new_doc"]
        if label_path.exists() and labels_sha(doc) != plan["labels_sha"] and not a.force:
            raise SystemExit("标注文件在提议之后被改过：重新运行一次提议，或加 --force")
        counts = apply_plan(plan, doc, parse_ids(a.not_shuttle), parse_ids(a.unsure))
        schema.save(doc, label_path)
        print(f"已写入 {label_path}（reviewed）：采用 {counts['accepted']}，不是球 {counts['not_shuttle']}，"
              f"拿不准 {counts['unsure']}，矛盾去掉 {counts['ignored']}，新增看不见 {counts['invisible']}")
        print(f"预览：python scripts/auto_label.py {cid} --preview；训练：python scripts/shuttle_loop.py")
        return 0

    propose(cid, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
