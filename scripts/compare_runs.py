"""Compare two ``real_run.py`` outputs: speed per stage and how much the results moved.

    # baseline from main, then this branch, on the same clips
    git worktree add ../rallyiq-base origin/main
    python ../rallyiq-base/scripts/real_run.py --only ae26_ms_f --out ~/rallyiq-data/runs/speed-before
    python scripts/real_run.py --only ae26_ms_f --out ~/rallyiq-data/runs/speed-after
    python scripts/compare_runs.py ~/rallyiq-data/runs/speed-before ~/rallyiq-data/runs/speed-after

Without labels this cannot say which run is more accurate; it says whether the
outputs stayed the same: shuttle points matched within ``--px`` pixels, player
keypoints moved, hits matched within ``--hit-frames`` frames.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

STAGE_COLS = ("calibration", "players", "shuttle", "events")


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def shuttle_agreement(a: dict, b: dict, px: float) -> dict[str, Any]:
    def det(s):
        return {p["frame"]: (p["x"], p["y"]) for p in s["points"] if p.get("source", "detected") == "detected"}

    da, db = det(a), det(b)
    both = set(da) & set(db)
    close = sum(1 for f in both if abs(da[f][0] - db[f][0]) <= px and abs(da[f][1] - db[f][1]) <= px)
    union = set(da) | set(db)
    return {"shuttle_a": len(da), "shuttle_b": len(db),
            "shuttle_same": round(close / max(len(union), 1), 3)}


def players_agreement(a: dict, b: dict) -> dict[str, Any]:
    fa = {f["frame"]: f for f in a["frames"]}
    moved = []
    count_same = total = 0
    for fr in b["frames"]:
        other = fa.get(fr["frame"])
        if other is None:
            continue
        total += 1
        pa = {p["player_id"]: p for p in other["players"] if not p.get("interpolated")}
        pb = {p["player_id"]: p for p in fr["players"] if not p.get("interpolated")}
        count_same += len(pa) == len(pb)
        for pid in set(pa) & set(pb):
            ka, kb = pa[pid]["keypoints"], pb[pid]["keypoints"]
            d = [((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 for (x1, y1, c1), (x2, y2, c2) in zip(ka, kb)
                 if c1 >= 0.3 and c2 >= 0.3]
            if d:
                moved.append(statistics.median(d))
    return {"players_count_same": round(count_same / max(total, 1), 3),
            "kp_shift_px_median": round(statistics.median(moved), 2) if moved else None,
            "kp_shift_px_p95": round(sorted(moved)[int(0.95 * (len(moved) - 1))], 2) if moved else None}


def hits_agreement(a: dict, b: dict, tol: int) -> dict[str, Any]:
    ha = sorted(s["frame"] for s in a.get("shots", []))
    hb = sorted(s["frame"] for s in b.get("shots", []))
    used: set[int] = set()
    matched = 0
    for f in ha:
        best = min((j for j, g in enumerate(hb) if j not in used and abs(g - f) <= tol),
                   key=lambda j: abs(hb[j] - f), default=None)
        if best is not None:
            used.add(best)
            matched += 1
    return {"hits_a": len(ha), "hits_b": len(hb), "hits_matched": matched}


def compare(before: Path, after: Path, px: float, tol: int) -> list[dict[str, Any]]:
    ra = {r["clip_id"]: r for r in _load(before / "results.json") or []}
    rb = {r["clip_id"]: r for r in _load(after / "results.json") or []}
    rows = []
    for clip in sorted(set(ra) & set(rb)):
        a, b = ra[clip], rb[clip]
        row: dict[str, Any] = {"clip": clip, "wall_a": a.get("wall_s"), "wall_b": b.get("wall_s")}
        for s in STAGE_COLS:
            row[f"{s}_a"] = (a["stages"].get(s) or {}).get("s")
            row[f"{s}_b"] = (b["stages"].get(s) or {}).get("s")
        art_a, art_b = Path(a["artifacts"]), Path(b["artifacts"])
        sa, sb = _load(art_a / "shuttle.json"), _load(art_b / "shuttle.json")
        if sa and sb:
            row.update(shuttle_agreement(sa, sb, px))
        pa, pb = _load(art_a / "players.json"), _load(art_b / "players.json")
        if pa and pb:
            row.update(players_agreement(pa, pb))
        ea, eb = _load(art_a / "events.json"), _load(art_b / "events.json")
        if ea and eb:
            row.update(hits_agreement(ea, eb, tol))
        for k in ("hits_ok", "rallies", "shuttle_detected_frac", "players_both_frac"):
            row[f"{k}_a"], row[f"{k}_b"] = a.get(k), b.get(k)
        rows.append(row)
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    def f(v):
        return "–" if v is None else str(v)

    L = ["## 耗时（秒，前 → 后）", "", "| 片段 | 总计 | " + " | ".join(STAGE_COLS) + " | 加速 |", "|" + "---|" * (len(STAGE_COLS) + 3)]
    for r in rows:
        speed = (f"{r['wall_a'] / r['wall_b']:.1f}×" if r.get("wall_a") and r.get("wall_b") else "–")
        cells = [f"{f(r[f'{s}_a'])} → {f(r[f'{s}_b'])}" for s in STAGE_COLS]
        L.append(f"| {r['clip']} | {f(r['wall_a'])} → {f(r['wall_b'])} | " + " | ".join(cells) + f" | {speed} |")
    L += ["", "## 结果是否变化（无标注，只比较前后是否一致）", "",
          "| 片段 | 球点一致 | 球点数 | 球员人数一致帧 | 关键点位移 px 中位 / p95 | 击球匹配 | 击球 ok | 回合 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(
            f"| {r['clip']} | {f(r.get('shuttle_same'))} | {f(r.get('shuttle_a'))} → {f(r.get('shuttle_b'))} | "
            f"{f(r.get('players_count_same'))} | {f(r.get('kp_shift_px_median'))} / {f(r.get('kp_shift_px_p95'))} | "
            f"{f(r.get('hits_matched'))}/{f(r.get('hits_a'))} → {f(r.get('hits_b'))} | "
            f"{f(r.get('hits_ok_a'))} → {f(r.get('hits_ok_b'))} | {f(r.get('rallies_a'))} → {f(r.get('rallies_b'))} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--px", type=float, default=3.0, help="shuttle points closer than this count as the same")
    ap.add_argument("--hit-frames", type=int, default=2, help="hits closer than this many frames count as the same")
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    rows = compare(a.before.expanduser(), a.after.expanduser(), a.px, a.hit_frames)
    if not rows:
        print("两个目录里没有共同的片段（results.json）", file=sys.stderr)
        return 1
    md = render(rows)
    print(md)
    if a.markdown:
        a.markdown.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
