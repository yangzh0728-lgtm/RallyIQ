"""RallyIQ 识别基准：按模块、按条件分组报告精度。

用法（在仓库根目录）：
    python -m evaluation.benchmark --pred path/to/predictions
    python -m evaluation.benchmark --pred preds --split eval --include-draft \
        --json out/bench.json --markdown out/bench.md

predictions 目录里每个片段一个 <clip_id>.json，格式同 labels（见 schema.py）。
没有预测文件的片段按"全部漏检"计分，避免挑着片段跑出好看的数字。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import metrics as M
from .schema import load_dir, load_manifest

HERE = Path(__file__).resolve().parent
GROUP_KEYS = ("discipline", "camera", "source", "venue")

# README 验收门槛：达到后才可以去掉"待复核"
TARGETS = {
    ("hits", "f1"): 0.90,
    ("shuttle", "recall"): 0.90,
}


def _new_bucket() -> dict[str, Any]:
    return {
        "clips": 0,
        "court": M.CourtStats(),
        "shuttle": {k: M.ShuttleStats() for k in ("all", "far", "near", "unknown")},
        "hits": M.EventStats(),
        "rallies": M.RallyStats(),
    }


def _merge(bucket: dict[str, Any], res: dict[str, Any]) -> None:
    bucket["clips"] += 1
    bucket["court"].merge(res["court"])
    for k, v in res["shuttle"].items():
        bucket["shuttle"][k].merge(v)
    bucket["hits"].merge(res["hits"])
    bucket["rallies"].merge(res["rallies"])


def _summ(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "clips": bucket["clips"],
        "court": bucket["court"].summary(),
        "shuttle": {k: v.summary() for k, v in bucket["shuttle"].items()},
        "hits": bucket["hits"].summary(),
        "rallies": bucket["rallies"].summary(),
    }


def evaluate_clip(gt: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "court": M.eval_court(gt, pred),
        "shuttle": M.eval_shuttle(gt, pred),
        "hits": M.eval_hits(gt, pred),
        "rallies": M.eval_rallies(gt, pred),
    }


def run(labels_dir: Path, pred_dir: Path | None, manifest: dict[str, dict], split: str | None,
        include_draft: bool) -> dict[str, Any]:
    labels = load_dir(labels_dir)
    preds = load_dir(pred_dir) if pred_dir else {}
    overall = _new_bucket()
    groups: dict[str, dict[str, dict]] = {k: defaultdict(_new_bucket) for k in GROUP_KEYS}
    per_clip: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for cid, gt in labels.items():
        meta = manifest.get(cid)
        if meta is None:
            skipped[cid] = "不在 clips.yaml 中"
            continue
        if split and meta.get("split") != split:
            continue
        if gt.get("status") != "reviewed" and not include_draft:
            skipped[cid] = "标注未复核（status != reviewed），用 --include-draft 计入"
            continue
        pred = preds.get(cid)
        if pred is not None:
            if pred["video"]["width"] != gt["video"]["width"] or pred["video"]["height"] != gt["video"]["height"]:
                raise SystemExit(f"{cid}: 预测与标注的分辨率不一致，请在同一分辨率的抽片上推理")
        res = evaluate_clip(gt, pred)
        _merge(overall, res)
        cond = meta.get("conditions", {})
        for k in GROUP_KEYS:
            _merge(groups[k][str(cond.get(k, "unknown"))], res)
        per_clip[cid] = {"has_prediction": pred is not None, "status": gt.get("status"), **_summ(
            {"clips": 1, "court": res["court"], "shuttle": res["shuttle"], "hits": res["hits"], "rallies": res["rallies"]})}

    report = {
        "settings": {
            "split": split, "include_draft": include_draft,
            "shuttle_hit_px@1080p": M.SHUTTLE_HIT_PX, "hit_tolerance_frames": M.HIT_TOL_FRAMES,
            "court_ok_px@1080p": M.COURT_OK_PX, "rally_min_iou": M.RALLY_MIN_IOU,
        },
        "overall": _summ(overall),
        "groups": {k: {g: _summ(b) for g, b in sorted(v.items())} for k, v in groups.items()},
        "clips": per_clip,
        "skipped": skipped,
    }
    return report


# ----------------------------------------------------------------- 输出
def _f(x: Any, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.1f}"


def _row(name: str, s: dict[str, Any]) -> str:
    c, sh, far, h, r = s["court"], s["shuttle"]["all"], s["shuttle"]["far"], s["hits"], s["rallies"]
    return (f"| {name} | {s['clips']} | {_f(c['success_rate'])} | {_f(c['mean_corner_err_px'], False)} | "
            f"{_f(sh['precision'])} | {_f(sh['recall'])} | {_f(far['recall'])} | "
            f"{_f(h['f1'])} | {_f(h['f1_ignore_side'])} | {_f(r['f1'])} |")


HEADER = ("| 范围 | 片段 | 标定成功率 | 角点误差px | 球精确率 | 球召回 | 远场球召回 | 击球F1 | 击球F1(不分方) | 回合F1 |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def to_markdown(rep: dict[str, Any]) -> str:
    lines = ["# RallyIQ 识别基准", ""]
    st = rep["settings"]
    lines.append(f"划分：{st['split'] or '全部'}；草稿标注{'计入' if st['include_draft'] else '不计入'}；"
                 f"球命中 ≤{st['shuttle_hit_px@1080p']:.0f}px@1080p；击球 ±{st['hit_tolerance_frames']} 帧；"
                 f"标定成功 ≤{st['court_ok_px@1080p']:.0f}px；回合 IoU ≥{st['rally_min_iou']}")
    lines += ["", "## 总体", "", HEADER, _row("全部", rep["overall"])]
    o = rep["overall"]
    lines.append("")
    for (mod, key), thr in TARGETS.items():
        v = o[mod]["all"][key] if mod == "shuttle" else o[mod][key]
        mark = "达标" if v is not None and v >= thr else "未达标"
        lines.append(f"- {mod}.{key} = {_f(v)}（门槛 {thr:.0%}，{mark}）")
    for k, groups in rep["groups"].items():
        lines += ["", f"## 按 {k} 分组", "", HEADER]
        lines += [_row(g, s) for g, s in groups.items()]
    lines += ["", "## 逐片段", "", HEADER]
    lines += [_row(cid + ("" if s["has_prediction"] else " (无预测)"), s) for cid, s in rep["clips"].items()]
    if rep["skipped"]:
        lines += ["", "## 跳过", ""] + [f"- {k}: {v}" for k, v in rep["skipped"].items()]
    worst = _weakest(rep)
    if worst:
        lines += ["", "## 最弱条件（优先改进）", ""] + [f"- {w}" for w in worst]
    return "\n".join(lines) + "\n"


def _weakest(rep: dict[str, Any], n: int = 3) -> list[str]:
    """README：每周挑出最弱的场景优先改进。"""
    items = []
    for k, groups in rep["groups"].items():
        for g, s in groups.items():
            for label, v in (("球召回", s["shuttle"]["all"]["recall"]), ("远场球召回", s["shuttle"]["far"]["recall"]),
                             ("击球F1", s["hits"]["f1"]), ("标定成功率", s["court"]["success_rate"])):
                if v is not None:
                    items.append((v, f"{k}={g}: {label} {_f(v)}"))
    return [t for _, t in sorted(items)[:n]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=HERE / "labels")
    ap.add_argument("--manifest", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--pred", type=Path, required=True, help="预测目录，每个片段一个 <clip_id>.json")
    ap.add_argument("--split", choices=["eval", "train"], default="eval",
                    help="默认只在 eval 划分上评测；--split train 用于检查微调数据")
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--include-draft", action="store_true", help="把未复核的草稿标注也计入")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)

    rep = run(a.labels, a.pred, load_manifest(a.manifest), None if a.all_splits else a.split, a.include_draft)
    md = to_markdown(rep)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
