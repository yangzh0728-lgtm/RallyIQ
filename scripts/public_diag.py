"""Why are the public-dataset scores low? Per-clip diagnostics on labels/ vs preds/.

    python scripts/public_diag.py                       # ~/rallyiq-data/public/shuttleset
    python scripts/public_diag.py --dir <out> --frames  # also save label-vs-pred court overlays

Checks, per clip:
* time sync: hit F1 when predictions are shifted by -2..+2 s. A peak away from 0 means the
  downloaded section does not start where the label assumes (a cut/download offset), not a
  detector problem.
* court: corner error as scored, and the smallest error over the court's symmetries
  (180 deg turn, left/right and near/far mirror). A big drop means our corners are the right
  court with the wrong labelling, not a wrong court.
* hits: of the missed label hits, how many fall outside every predicted rally, and how many
  have no detected shuttle within +-5 frames; same for false hits against label rallies.
Writes diag.md next to report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import metrics as M  # noqa: E402
from evaluation import schema  # noqa: E402

# corner index permutations for [far-left, far-right, near-right, near-left]
SYMS = {"as is": [0, 1, 2, 3], "180°": [2, 3, 0, 1], "mirror L/R": [1, 0, 3, 2], "mirror N/F": [3, 2, 1, 0]}


def _shift_f1(gt: dict[str, Any], pred: dict[str, Any], shift_ms: float) -> float:
    fm = M.frame_ms(gt)
    g = [h for h in gt["hits"] if M._in_range(h["t_ms"], gt)]
    p = [{**h, "t_ms": h["t_ms"] + shift_ms} for h in pred.get("hits", [])]
    p = [h for h in p if M._in_range(h["t_ms"], gt)]
    tp = len(M._greedy_match(g, p, M.HIT_TOL_FRAMES * fm + 1e-6, same_side=False))
    return 2 * tp / (len(g) + len(p)) if g or p else 0.0


def time_sync(gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    fm = M.frame_ms(gt)
    shifts = np.arange(-2000, 2000 + 1e-6, fm)
    f1 = [_shift_f1(gt, pred, s) for s in shifts]
    best = np.flatnonzero(np.isclose(f1, max(f1)))
    k = int(best[len(best) // 2])  # middle of the plateau: hits match within +-tol of the true shift
    return {"f1_at_0": round(_shift_f1(gt, pred, 0.0), 3), "best_shift_ms": round(float(shifts[k])),
            "best_shift_frames": round(float(shifts[k] / fm), 1), "f1_at_best": round(f1[k], 3)}


def court(gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    idx = M._TimeIndex(pred.get("court", []))
    tol = M.frame_ms(gt) / 2 + 1
    s = M.px_scale(gt)
    errs: dict[str, list[float]] = {k: [] for k in SYMS}
    seen = 0
    for g in gt["court"]:
        if not g["visible"]:
            continue
        seen += 1
        p = idx.nearest(g["t_ms"], tol)
        if not (p and p.get("visible") and p.get("corners")):
            continue
        gc, pc = np.asarray(g["corners"], float), np.asarray(p["corners"], float)
        for k, perm in SYMS.items():
            errs[k].append(float(np.mean(np.linalg.norm(gc - pc[perm], axis=1))) * s)
    out: dict[str, Any] = {"label_times": seen, "with_pred": len(errs["as is"])}
    for k, e in errs.items():
        out[k] = round(float(np.median(e)), 1) if e else None
    if errs["as is"]:
        e = np.asarray(errs["as is"])
        best = np.min(np.stack([errs[k] for k in SYMS]), axis=0)
        out["share_ok_as_is"] = round(float(np.mean(e <= M.COURT_OK_PX)), 3)
        out["share_ok_any_sym"] = round(float(np.mean(best <= M.COURT_OK_PX)), 3)
        out["share_le_30px_any_sym"] = round(float(np.mean(best <= 30)), 3)
        out["p90_any_sym"] = round(float(np.percentile(best, 90)), 1)
    return out


def _inside(t: float, spans: list[dict[str, Any]]) -> bool:
    return any(r["start_ms"] <= t <= r["end_ms"] for r in spans)


def hits(gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    fm = M.frame_ms(gt)
    g = [h for h in gt["hits"] if M._in_range(h["t_ms"], gt)]
    p = [h for h in pred.get("hits", []) if M._in_range(h["t_ms"], gt)]
    m = M._greedy_match(g, p, M.HIT_TOL_FRAMES * fm + 1e-6, same_side=False)
    mg, mp = {i for i, _, _ in m}, {j for _, j, _ in m}
    shuttle_t = np.array(sorted(s["t_ms"] for s in pred.get("shuttle", []) if s.get("visible")))

    def near_shuttle(t: float) -> bool:
        if not shuttle_t.size:
            return False
        k = np.searchsorted(shuttle_t, t)
        return any(0 <= i < shuttle_t.size and abs(shuttle_t[i] - t) <= 5 * fm for i in (k - 1, k))

    miss = [g[i] for i in range(len(g)) if i not in mg]
    fp = [p[j] for j in range(len(p)) if j not in mp]
    near_miss = []
    for h in miss:  # a pred hit within 4..15 frames: right event, timed off
        d = [abs(q["t_ms"] - h["t_ms"]) / fm for q in p]
        if d and M.HIT_TOL_FRAMES < min(d) <= 15:
            near_miss.append(min(d))
    pr = pred.get("rallies", [])
    return {
        "gt": len(g), "pred": len(p), "tp_any_side": len(m),
        "missed": len(miss),
        "missed_outside_pred_rally": sum(not _inside(h["t_ms"], pr) for h in miss),
        "missed_no_shuttle_±5f": sum(not near_shuttle(h["t_ms"]) for h in miss),
        "missed_pred_within_4_15f": len(near_miss),
        "false": len(fp),
        "false_outside_gt_rally": sum(not _inside(h["t_ms"], gt["rallies"]) for h in fp),
        "shuttle_frames_in_gt_rallies": round(float(np.mean([_inside(t, gt["rallies"]) for t in shuttle_t])), 3)
        if shuttle_t.size else None,
    }


def overlay(clip: Path, gt: dict[str, Any], pred: dict[str, Any], out: Path) -> Path | None:
    import cv2

    ts = [c["t_ms"] for c in gt["court"] if c["visible"]]
    if not ts or not clip.exists():
        return None
    t = ts[len(ts) // 2]
    meta = json.loads(clip.with_suffix(".meta.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, t - meta.get("start_ms", 0))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    g = next(c for c in gt["court"] if c["t_ms"] == t)
    p = M._TimeIndex(pred.get("court", [])).nearest(t, M.frame_ms(gt))
    for doc, col in ((g, (0, 255, 0)), (p, (0, 0, 255))):
        if doc and doc.get("corners"):
            pts = np.asarray(doc["corners"], np.int32)
            cv2.polylines(img, [pts], True, col, 2)
            for k, (x, y) in enumerate(pts):
                cv2.putText(img, "FL FR NR NL".split()[k], (int(x) + 4, int(y) - 4), 0, 0.6, col, 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / "rallyiq-data" / "public" / "shuttleset")
    ap.add_argument("--frames", action="store_true", help="save label (green) vs pred (red) court overlays")
    a = ap.parse_args(argv)
    d = a.dir.expanduser()
    rows = []
    for lp in sorted((d / "labels").glob("*.json")):
        if lp.name.startswith("._"):  # macOS metadata files on exFAT
            continue
        cid = lp.stem
        pp = d / "preds" / f"{cid}.json"
        if not pp.exists():
            continue
        gt, pred = schema.load(lp), json.loads(pp.read_text(encoding="utf-8"))
        r = {"clip": cid, "sync": time_sync(gt, pred), "court": court(gt, pred), "hits": hits(gt, pred)}
        if a.frames:
            img = overlay(d / "clips" / f"{cid}.mp4", gt, pred, d / "diag" / f"{cid}_court.jpg")
            r["overlay"] = str(img) if img else None
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    md = ["# public eval diagnostics", ""]
    for r in rows:
        md += [f"## {r['clip']}", "", "```json", json.dumps({k: v for k, v in r.items() if k != "clip"},
                                                            ensure_ascii=False, indent=1), "```", ""]
    (d / "diag.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{d / 'diag.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
