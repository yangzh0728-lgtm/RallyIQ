"""Continuous shuttle-model improvement: label -> train -> compare -> keep only if better.

    python scripts/shuttle_loop.py              # run one round (train + compare + maybe promote)
    python scripts/shuttle_loop.py --status     # current model, history, what to label next
    python scripts/shuttle_loop.py --rollback   # go back to the previous model

Each round:

1. Collects every **reviewed** label in ``evaluation/labels/``. Each clip is split by time:
   the first 70% trains, the last 30% validates (``--val-frac``). ``--holdout ID`` keeps a
   whole clip for validation only, e.g. a camera the model has never been trained on.
   Clips marked ``holdout: true`` in a manifest (DJI_0009 in ``evaluation/own_clips.yaml``)
   are always held out. Own footage from ``own_clips.yaml`` is cut by
   ``python -m evaluation.tools.extract_clips`` into the same clips directory.
2. Fine-tunes a candidate starting from the original pretrained weights (``--from champion``
   to continue from the current model instead), picking the best epoch by F1 at the
   detector's operating peak threshold (0.2).
3. Evaluates the model currently in use (the "champion") on the **same** validation frames,
   so the comparison is fair even though the validation set grows as labels are added.
4. Promotes the candidate only if F1@0.2 improves by at least ``--min-gain`` (0.01) and
   precision@0.2 drops by no more than ``--max-precision-drop`` (0.03). The margin is there
   because the candidate's epoch was itself picked on these frames, which flatters it a little. A promoted model is
   copied to ``weights/shuttle/heatmap.pt``, which the pipeline loads by default (unless
   ``RALLYIQ_SHUTTLE_WEIGHTS`` points elsewhere). Every model is kept in ``weights/shuttle/loop/``.
5. Prints which clips to label next, with the exact commands.

A round is skipped when no labels changed since the last one (``--force`` to run anyway).
``--holdout`` is remembered for later rounds (``--holdout`` with no ids clears it).
State lives in ``weights/shuttle/loop/state.json``; a log of every round is appended to
``weights/shuttle/loop/history.md``.

The validation numbers are frame-level detection scores on your own labels. Before trusting
a promoted model for match statistics, also rerun ``scripts/real_run.py`` and
``scripts/hits_ablation.py`` on a clip that was not trained on (the script prints the command).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import schema
from scripts.finetune_shuttle import clip_labels, split_by_time

LABELS = ROOT / "evaluation" / "labels"
MANIFEST = ROOT / "evaluation" / "clips.yaml"
OWN_MANIFEST = schema.OWN_MANIFEST
PRIORITY_CAMERAS = {"side_court_fixed", "amateur_baseline", "amateur_side"}
WEIGHTS = ROOT / "weights" / "shuttle"
DEFAULT_CLIPS = Path.home() / "rallyiq-data" / "bwf" / "clips"
RUNS = Path.home() / "rallyiq-data" / "runs"
SELECT_AT = 0.2


# ------------------------------------------------------------------ state


def loop_dir(weights_dir: Path) -> Path:
    return weights_dir / "loop"


def load_state(weights_dir: Path) -> dict:
    p = loop_dir(weights_dir) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(weights_dir: Path, state: dict) -> None:
    d = loop_dir(weights_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False))
    tmp.replace(d / "state.json")


def init_state(weights_dir: Path, base: Path) -> dict:
    """First run: keep a copy of the pretrained checkpoint; it is the first champion."""
    d = loop_dir(weights_dir)
    d.mkdir(parents=True, exist_ok=True)
    kept = d / "base.pt"
    if not kept.exists():
        shutil.copyfile(base, kept)
    return {"base": str(kept), "base_source": str(base), "champion": str(kept), "previous": [], "rounds": []}


def install(weights_dir: Path, model: Path) -> Path:
    """Make ``model`` the pipeline default (weights/shuttle/heatmap.pt)."""
    dst = weights_dir / "heatmap.pt"
    tmp = weights_dir / "heatmap.pt.tmp"
    shutil.copyfile(model, tmp)
    tmp.replace(dst)
    return dst


# ------------------------------------------------------------------- data


def reviewed_docs(labels_dir: Path) -> dict[str, dict]:
    return {cid: d for cid, d in schema.load_dir(labels_dir).items()
            if d.get("status") == "reviewed" and d.get("shuttle")}


def labels_fingerprint(docs: dict[str, dict]) -> str:
    h = hashlib.sha256()
    for cid in sorted(docs):
        h.update(cid.encode())
        h.update(json.dumps(docs[cid]["shuttle"], sort_keys=True).encode())
    return h.hexdigest()[:16]


def labeled_clips(docs: dict[str, dict], clips_dir: Path):
    """One TrainClip per reviewed clip with all its labeled frames, plus clips with no local video."""
    from rallyiq.perception.shuttle.training import TrainClip

    clips, missing = [], []
    for cid, doc in sorted(docs.items()):
        video, meta_p = clips_dir / f"{cid}.mp4", clips_dir / f"{cid}.meta.json"
        if not video.exists() or not meta_p.exists():
            missing.append(cid)
            continue
        start_ms = int(json.loads(meta_p.read_text(encoding="utf-8"))["start_ms"])
        labels = clip_labels(doc, start_ms)
        if labels:
            clips.append(TrainClip(cid, video, float(doc["video"]["fps"]), labels))
    return clips, missing


def build_split(docs: dict[str, dict], clips_dir: Path, val_frac: float, holdout: list[str]):
    """TrainClips for training and validation, plus clips that could not be used."""
    from rallyiq.perception.shuttle.training import TrainClip

    clips, missing = labeled_clips(docs, clips_dir)
    train, val = [], []
    for c in clips:
        if c.clip_id in holdout:
            val.append(c)
            continue
        a, b = split_by_time(c.labels, 1.0 - val_frac)
        if a:
            train.append(TrainClip(f"{c.clip_id}[train]", c.video, c.fps, a))
        if b:
            val.append(TrainClip(f"{c.clip_id}[val]", c.video, c.fps, b))
    return train, val, missing


# --------------------------------------------------------------- decision


def decide(candidate: dict, champion: dict, at: float = SELECT_AT, min_gain: float = 0.01,
           max_precision_drop: float = 0.03) -> tuple[bool, str]:
    """``candidate`` / ``champion``: per-threshold metrics as returned by training.evaluate."""
    c, k = candidate.get(str(at), {}), champion.get(str(at), {})
    cf, kf = c.get("f1", 0.0), k.get("f1", 0.0)
    cp, kp = c.get("precision", 0.0), k.get("precision", 0.0)
    cr, kr = c.get("recall", 0.0), k.get("recall", 0.0)
    nums = f"F1@{at} {kf:.3f} -> {cf:.3f}（精确率 {kp:.3f} -> {cp:.3f}，召回率 {kr:.3f} -> {cr:.3f}）"
    if cf < kf + min_gain:
        return False, f"{nums}，F1 提升不足 {min_gain}"
    if cp < kp - max_precision_drop:
        return False, f"{nums}，精确率下降超过 {max_precision_drop}"
    return True, nums


# ------------------------------------------------------------ label check

KINDS = {
    "fp": "误报（标注说看不见，模型说有球）",
    "wrong": "位置不同（模型的点离标注的点太远）",
    "fn": "漏检（标注说有球，模型没找到）",
}


def check_order(mistakes: list[dict]) -> list[dict]:
    """Most telling first: confident detections the labels disagree with, then misses."""
    return sorted(mistakes, key=lambda m: (m["kind"] == "fn", -(m["score"] or 0.0), m["clip"], m["frame"]))


def contact_sheet(mistakes: list[dict], clips_dir: Path, out: Path, n: int = 48, cols: int = 6,
                  tile: tuple[int, int] = (320, 240)) -> Path | None:
    """A grid of crops around the first ``n`` frames of :func:`check_order`.

    Green circle: the label. Magenta circle: the model. Caption: frame, kind, model score."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    picked = check_order(mistakes)[:n]
    if not picked:
        return None
    tw, th = tile
    tiles: dict[int, np.ndarray] = {}
    caps: dict[str, object] = {}
    for i in sorted(range(len(picked)), key=lambda i: (picked[i]["clip"], picked[i]["frame"])):
        m = picked[i]
        cap = caps.get(m["clip"])
        if cap is None:
            cap = caps[m["clip"]] = cv2.VideoCapture(str(clips_dir / f"{m['clip']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame"])
        ok, fr = cap.read()
        img = np.zeros((th, tw, 3), np.uint8)
        if ok:
            cx, cy = m["xy"] or m["label"] or (fr.shape[1] / 2, fr.shape[0] / 2)
            x0 = int(min(max(cx - tw / 2, 0), max(fr.shape[1] - tw, 0)))
            y0 = int(min(max(cy - th / 2, 0), max(fr.shape[0] - th, 0)))
            crop = fr[y0:y0 + th, x0:x0 + tw]
            img[:crop.shape[0], :crop.shape[1]] = crop
            for pt, color in ((m["label"], (0, 255, 0)), (m["xy"], (255, 0, 255))):
                if pt:
                    cv2.circle(img, (int(pt[0] - x0), int(pt[1] - y0)), 12, color, 1, cv2.LINE_AA)
        score = "" if m["score"] is None else f" {m['score']:.2f}"
        cv2.rectangle(img, (0, th - 18), (tw, th), (0, 0, 0), -1)
        cv2.putText(img, f"{m['clip'][-6:]} #{m['frame']} {m['kind']}{score}", (4, th - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles[i] = img
    for cap in caps.values():
        cap.release()
    rows = []
    for r in range(0, len(picked), cols):
        row = [tiles[i] for i in range(r, min(r + cols, len(picked)))]
        row += [np.zeros((th, tw, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    head = np.zeros((26, tw * cols, 3), np.uint8)
    cv2.putText(head, "green = label   magenta = model   fp: label says no shuttle, model fires   "
                "wrong: model elsewhere   fn: model misses", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, *rows]))
    return out


def run_check(weights: Path, clips: list, trained: dict[str, set[int]], out_base: Path, clips_dir: Path,
              cfg) -> dict:
    """Find labeled frames ``weights`` disagrees with; write ``<out_base>.json`` and ``.jpg``."""
    from rallyiq.perception.shuttle.training import disagreements

    res, mistakes = disagreements(weights, clips, cfg)
    for m in mistakes:
        m["part"] = "train" if m["frame"] in trained.get(m["clip"], ()) else "val"
    doc = {"weights": str(weights), "threshold": cfg.select_at, "created": time.strftime("%Y-%m-%d %H:%M"),
           "scores": res, "frames": sorted(mistakes, key=lambda m: (m["clip"], m["frame"]))}
    out_base.parent.mkdir(parents=True, exist_ok=True)
    path = out_base.with_name(out_base.name + ".json")
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    doc["path"] = str(path)
    sheet = contact_sheet(mistakes, clips_dir, out_base.with_name(out_base.name + ".jpg"))
    doc["sheet"] = str(sheet) if sheet else None
    return doc


def check_summary(doc: dict) -> str:
    fr = doc["frames"]
    by = {k: sum(1 for m in fr if m["kind"] == k) for k in KINDS}
    parts = {p: sum(1 for m in fr if m.get("part") == p) for p in ("train", "val")}
    L = [(f"模型（{Path(doc['weights']).name}）和标注对不上的帧：共 {len(fr)} 帧"
          f"（训练部分 {parts['train']}，验证部分 {parts['val']}）")]
    L += [f"  {KINDS[k]}：{v}" for k, v in by.items() if v]
    if doc.get("sheet"):
        L.append(f"拼图（最有把握的前 48 帧）：open {doc['sheet']}")
    for cid in sorted({m["clip"] for m in fr}):
        L.append(f"逐帧检查：python -m evaluation.tools.label_tool {cid} --check {doc['path']}")
    return "\n".join(L)


# ------------------------------------------------------------ suggestions


def latest_pred(clip_id: str, runs: Path | None = None) -> Path | None:
    runs = runs or RUNS
    preds = sorted(runs.glob(f"*/preds/{clip_id}.json"), key=lambda p: p.stat().st_mtime) if runs.exists() else []
    return preds[-1] if preds else None


def camera(manifest: dict, cid: str) -> str:
    return (manifest.get(cid, {}).get("conditions") or {}).get("camera", "") or ""


def manifest_holdout(manifest: dict) -> list[str]:
    """Clips marked ``holdout: true`` in a manifest (e.g. DJI_0009) never train."""
    return sorted(cid for cid, c in manifest.items() if c.get("holdout"))


def suggest_next(manifest: dict, docs_all: dict[str, dict], clips_dir: Path, runs: Path | None = None,
                 n: int = 3) -> list[str]:
    """Clips to label next: side-court and own footage first, then the rest; skip what is reviewed.

    Whole-clip holdouts (``holdout: true``) come last: labelling them only adds test frames."""
    lines = []
    reviewed = {cid for cid, d in docs_all.items() if d.get("status") == "reviewed"}
    started = {cid for cid, d in docs_all.items() if d.get("status") != "reviewed"}

    def rank(cid: str) -> tuple:
        return (cid not in started, bool(manifest[cid].get("holdout")),
                camera(manifest, cid) not in PRIORITY_CAMERAS, manifest[cid].get("split") != "train", cid)

    todo = sorted((c for c in manifest if c not in reviewed and (clips_dir / f"{c}.mp4").exists()), key=rank)
    for cid in todo[:n]:
        cam = camera(manifest, cid) or "?"
        note = "（已开始，接着标）" if cid in started else ""
        pred = latest_pred(cid, runs)
        if pred is None:
            lines.append(f"- {cid}（{cam}）{note}：先生成预标注 `python scripts/real_run.py --only {cid}`，"
                         f"再 `python -m evaluation.tools.label_tool {cid} --prefill \"$(ls -t ~/rallyiq-data/runs/*/preds/{cid}.json | head -1)\"`")
        else:
            lines.append(f"- {cid}（{cam}）{note}：`python -m evaluation.tools.label_tool {cid} --prefill {pred}`")
    missing = [c for c in manifest if c not in reviewed and not (clips_dir / f"{c}.mp4").exists()]
    if not todo and missing:
        own = [c for c in missing if schema.is_own(manifest[c])]
        if own:
            lines.append(f"- 自有素材还没切片：{', '.join(own[:5])}。接上硬盘后运行 "
                         f"`python -m evaluation.tools.extract_clips --only {' '.join(own[:3])}`")
        yt = [c for c in missing if not schema.is_own(manifest[c])]
        if yt:
            lines.append(f"- 本地还没有视频的片段：{', '.join(yt[:5])}… 先运行 "
                         f"`python -m evaluation.tools.extract_clips --download --only {yt[0]}`")
    return lines


# ------------------------------------------------------------------- main


def status(state: dict, manifest: dict, labels_dir: Path, clips_dir: Path, weights_dir: Path,
           runs: Path | None = None) -> str:
    docs_all = schema.load_dir(labels_dir) if labels_dir.exists() else {}
    rev = {c: d for c, d in docs_all.items() if d.get("status") == "reviewed" and d.get("shuttle")}
    L = ["== 羽毛球模型持续训练 =="]
    if state:
        L.append(f"现役模型：{state['champion']}")
        L.append(f"流水线默认加载：{weights_dir / 'heatmap.pt'}"
                 f"{'' if (weights_dir / 'heatmap.pt').exists() else '（还没有，仍用原始预训练权重）'}")
        for r in state.get("rounds", [])[-5:]:
            L.append(f"  第 {r['round']} 轮 {r['time']}：{'采用' if r['promoted'] else '未采用'}，{r['reason']}")
    else:
        L.append("还没有跑过任何一轮。")
    ld = loop_dir(weights_dir)
    checks = sorted([*ld.glob("round-*-check.json"), *ld.glob("check-*.json")], key=lambda p: p.stat().st_mtime) \
        if ld.exists() else []
    if checks:
        L.append(f"最近一次标注检查：{checks[-1]}（标注工具加 --check 这个文件，逐帧看）")
    done = [f"{c}（{len(d['shuttle'])} 帧）" for c, d in sorted(rev.items())]
    L.append("已复核的标注：" + (", ".join(done) or "无"))
    held = sorted(set(state.get("holdout", [])) | set(manifest_holdout(manifest)))
    if held:
        L.append("只用于验证的片段：" + ", ".join(held))
    L.append("下一步建议标注：")
    L += suggest_next(manifest, docs_all, clips_dir, runs) or ["- （没有可标的片段）"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="only show the current state and what to label next")
    ap.add_argument("--rollback", action="store_true", help="reinstall the previous champion")
    ap.add_argument("--check", metavar="WEIGHTS",
                    help="only list labeled frames this checkpoint disagrees with (no training)")
    ap.add_argument("--no-check", action="store_true", help="skip the label check after a round")
    ap.add_argument("--force", action="store_true", help="run even if no labels changed since the last round")
    ap.add_argument("--from", dest="start", choices=["base", "champion"], default="base",
                    help="start each round from the original pretrained weights (default) or the current model")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="clip ids used only for validation (remembered; no ids clears it)")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--max-precision-drop", type=float, default=0.03)
    ap.add_argument("--device", help="cuda | mps | cpu")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--weights-dir", type=Path, default=WEIGHTS)
    ap.add_argument("--base", help="pretrained checkpoint for the first round (default: the installed one)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    ap.add_argument("--own-manifest", type=Path, default=OWN_MANIFEST, help=argparse.SUPPRESS)
    ap.add_argument("--runs", type=Path, default=RUNS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    clips_dir, wdir = a.clips.expanduser(), a.weights_dir.expanduser()
    manifest = schema.load_manifests(a.manifest, a.own_manifest)
    runs = a.runs.expanduser()
    state = load_state(wdir)

    if a.status:
        print(status(state, manifest, a.labels, clips_dir, wdir, runs))
        return 0

    if a.rollback:
        if not state or not state.get("previous"):
            print("没有可以回退的上一个模型。")
            return 1
        state["champion"] = state["previous"].pop()
        install(wdir, Path(state["champion"]))
        save_state(wdir, state)
        print(f"已回退到 {state['champion']}（已复制为 {wdir / 'heatmap.pt'}）")
        return 0

    from rallyiq.perception.shuttle.detector import find_weights
    from rallyiq.perception.shuttle.training import TrainConfig, train

    if a.check:
        docs = reviewed_docs(a.labels)
        tr, _va, _missing = build_split(docs, clips_dir, a.val_frac, state.get("holdout", []))
        clips, missing = labeled_clips(docs, clips_dir)
        if not clips:
            raise SystemExit("没有已复核且本地有视频的片段")
        cfg = TrainConfig(device=a.device, select_at=SELECT_AT)
        doc = run_check(Path(a.check).expanduser(), clips, {c.video.stem: set(c.labels) for c in tr},
                        loop_dir(wdir) / f"check-{Path(a.check).stem}", clips_dir, cfg)
        print(check_summary(doc))
        return 0

    if not state:
        base = find_weights(a.base)
        if base is None:
            raise SystemExit("找不到预训练权重（见 rallyiq/perception/shuttle/README.md）")
        state = init_state(wdir, base)
        save_state(wdir, state)
        print(f"第一次运行：已把预训练权重 {base} 备份为 {state['base']}")

    docs = reviewed_docs(a.labels)
    if not docs:
        print("还没有已复核（reviewed）的标注，先标一段：")
        print("\n".join(suggest_next(manifest, schema.load_dir(a.labels), clips_dir, runs)))
        return 1
    fp = labels_fingerprint(docs)
    last = state["rounds"][-1] if state["rounds"] else None
    if last and last.get("labels") == fp and not a.force:
        print("自上一轮以来标注没有变化，先多标一些（或加 --force 重跑）。下一步建议标注：")
        print("\n".join(suggest_next(manifest, schema.load_dir(a.labels), clips_dir, runs)))
        return 0

    if a.holdout is not None:
        state["holdout"] = list(a.holdout)
    holdout = sorted(set(state.get("holdout", [])) | set(manifest_holdout(manifest)))
    tr, va, missing = build_split(docs, clips_dir, a.val_frac, holdout)
    for cid in missing:
        print(f"跳过 {cid}：{clips_dir} 里没有视频")
    if not tr or not va:
        raise SystemExit("训练或验证数据为空：至少需要一段已复核且本地有视频的片段")
    print("训练：" + ", ".join(f"{c.clip_id} {len(c.labels)} 帧" for c in tr))
    print("验证：" + ", ".join(f"{c.clip_id} {len(c.labels)} 帧" for c in va))

    n = len(state["rounds"]) + 1
    out = loop_dir(wdir) / f"round-{n:03d}.pt"
    last = loop_dir(wdir) / f"round-{n:03d}-last.pt"
    start = Path(state["base"] if a.start == "base" else state["champion"])
    cfg = TrainConfig(epochs=a.epochs, steps_per_epoch=a.steps, batch_size=a.batch_size, lr=a.lr,
                      device=a.device, select_at=SELECT_AT)
    compare = {} if Path(state["champion"]) == start else {"champion": Path(state["champion"])}
    t0 = time.monotonic()
    report = train(start, tr, va, out, cfg, compare=compare, last_out=last)
    champ_val = (report["compare"].get("champion", {}).get("val") if compare else None) or report["history"][0]["val"]
    ok, reason = decide(report["best_val"], champ_val, SELECT_AT, a.min_gain, a.max_precision_drop)
    if report["best_epoch"] == 0 and a.start == "champion":
        ok, reason = False, "训练没有带来改进（最好的是第 0 轮）"

    rec = {
        "round": n, "time": time.strftime("%Y-%m-%d %H:%M"), "labels": fp, "candidate": str(out),
        "start": str(start), "best_epoch": report["best_epoch"], "promoted": ok, "reason": reason,
        "train": report["train_clips"], "val": report["val_clips"],
        "candidate_val": report["best_val"], "champion_val": champ_val,
        "minutes": round((time.monotonic() - t0) / 60, 1),
    }
    check = None
    if not a.no_check:
        print("\n正在用最后一轮的模型找标注可能有错的帧……")
        clips, _ = labeled_clips(docs, clips_dir)
        check = run_check(last, clips, {c.video.stem: set(c.labels) for c in tr},
                          loop_dir(wdir) / f"round-{n:03d}-check", clips_dir, cfg)
        rec["check"] = {"path": check["path"], "frames": len(check["frames"])}
    if ok:
        state["previous"].append(state["champion"])
        state["champion"] = str(out)
        install(wdir, out)
    state["rounds"].append(rec)
    save_state(wdir, state)
    with open(loop_dir(wdir) / "history.md", "a", encoding="utf-8") as f:
        f.write(f"- 第 {n} 轮 {rec['time']}：{'采用' if ok else '未采用'}，{reason}；"
                f"训练 {sum(rec['train'].values())} 帧，验证 {sum(rec['val'].values())} 帧\n")

    print()
    print(f"== 第 {n} 轮结果：{'采用新模型' if ok else '保留现役模型'} ==")
    print(reason)
    if ok:
        print(f"新模型已复制为 {wdir / 'heatmap.pt'}，之后的 real_run 会默认使用它。不满意可运行 --rollback 回退。")
        untrained = sorted((c for c in manifest if c not in docs and (clips_dir / f"{c}.mp4").exists()),
                           key=lambda c: (camera(manifest, c) not in PRIORITY_CAMERAS, c))[:2]
        if untrained:
            print("建议在没参与训练的片段上核对击球结果：")
            print(f"  python scripts/real_run.py --only {' '.join(untrained)} --out ~/rallyiq-data/runs/round-{n:03d}")
            print(f"  python scripts/hits_ablation.py --run ~/rallyiq-data/runs/round-{n:03d} "
                  + " ".join(f"--clip {c}" for c in untrained))
    if check is not None:
        print()
        print(check_summary(check))
        print("先看拼图：如果多数“误报”其实是看得见的球（比如发球前在手里），就用上面的命令把这些帧改对，再跑一次本脚本。")
    print("下一步建议标注：")
    print("\n".join(suggest_next(manifest, schema.load_dir(a.labels), clips_dir, runs)) or "- （没有可标的片段）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
