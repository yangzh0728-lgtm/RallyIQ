"""把流水线产物（rallyiq/contracts 的 JSON）转换成评测预测格式（evaluation/schema.py）。

    python -m evaluation.from_contracts data/videos/<video_id>/artifacts \
        --clip-meta ~/rallyiq-data/bwf/clips/ae26_ms_f.meta.json --out preds/

读取 artifacts 目录下的 probe（VideoMeta）/ calibration / shuttle / events / players（缺的跳过），
写出 preds/<clip_id>.json。

时间换算：流水线处理的是抽出来的片段文件，它的 t_ms 从片段开头算起；
评测标注用原 YouTube 视频时间，所以这里加上片段 meta 里的 start_ms。

转换规则：
- 球场：每帧输出一条。帧落在状态不是 undeterminable、且有角点或单应的标定段里时，
  记为可见；角点顺序从合约的「近左、近右、远右、远左」转成评测的「远左、远右、近右、近左」。
  其他帧记为不可见（这样特写/回放时"没输出"会被算作正确）。
- 羽毛球：每帧输出一条。默认只把 source=detected 的点算作可见，
  插值点不计入（README：原始检测与补点分开）；--include-interpolated 可改。
- 击球：hitter → near/far，优先用同帧 players 里该球员的 side，其次用 hitter_court 的 y
  （y < 6.7 m 为近端）；都没有时为 null（只计入"不分击球方"的 F1）。
- 回合：直接用 start_ms / end_ms。
- 合约里没有"已拒绝"状态；修正后的结果在 events_effective.json，评测模型时用原始的 events.json。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import schema
from .metrics import NET_Y

COURT_CORNERS_M = [[0.0, 0.0], [6.10, 0.0], [6.10, 13.40], [0.0, 13.40]]  # 近左 近右 远右 远左


def _read(d: Path, name: str) -> dict[str, Any] | None:
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _corners(seg: dict[str, Any]) -> list[list[float]] | None:
    if seg.get("status") == "undeterminable":
        return None
    c = seg.get("image_corners")
    if c is None and seg.get("homography") is not None:
        from rallyiq.contracts.geometry import court_to_image
        c = [court_to_image(seg["homography"], p) for p in COURT_CORNERS_M]
    if c is None:
        return None
    nl, nr, fr, fl = c
    return [list(fl), list(fr), list(nr), list(nl)]


def _side_lookup(players: dict[str, Any] | None) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for fr in (players or {}).get("frames", []):
        for p in fr.get("players", []):
            if p.get("player_id") and p.get("side"):
                out[(fr["frame"], p["player_id"])] = p["side"]
    return out


def convert(art: Path, clip_id: str, offset_ms: int, include_interpolated: bool = False) -> dict[str, Any]:
    # 流水线把 VideoMeta 存为 probe.json（产物以阶段命名）；video_meta.json 兼容旧写法
    meta = _read(art, "probe") or _read(art, "video_meta")
    if meta is None:
        raise SystemExit(f"{art} 里没有 probe.json")
    fps, n = float(meta["fps"]), int(meta["frame_count"])
    t = lambda f: int(round(offset_ms + f * 1000.0 / fps))  # noqa: E731
    doc = schema.empty_doc(clip_id, "prediction", meta["width"], meta["height"], fps,
                           (offset_ms, offset_ms + int(meta["duration_ms"])))
    doc["source_video_id"] = meta["video_id"]

    cal = _read(art, "calibration")
    if cal is not None:
        per_frame: list[list[list[float]] | None] = [None] * n
        for seg in cal.get("segments", []):
            c = _corners(seg)
            for f in range(max(0, seg["start_frame"]), min(n - 1, seg["end_frame"]) + 1):
                per_frame[f] = c
        doc["court"] = [{"t_ms": t(f), "visible": c is not None, "corners": c} for f, c in enumerate(per_frame)]

    sh = _read(art, "shuttle")
    if sh is not None:
        pts = {}
        for p in sh.get("points", []):
            if p.get("source", "detected") == "detected" or include_interpolated:
                pts[p["frame"]] = p
        doc["shuttle"] = [
            {"t_ms": t(f), "visible": True, "xy": [pts[f]["x"], pts[f]["y"]], "score": pts[f].get("confidence")}
            if f in pts else {"t_ms": t(f), "visible": False, "xy": None}
            for f in range(n)
        ]

    ev = _read(art, "events")
    if ev is not None:
        sides = _side_lookup(_read(art, "players"))
        for s in ev.get("shots", []):
            side = sides.get((s["frame"], s.get("hitter"))) if s.get("hitter") else None
            if side is None and s.get("hitter_court"):
                side = "near" if s["hitter_court"][1] < NET_Y else "far"
            doc["hits"].append({"t_ms": t(s["frame"]), "side": side, "player": s.get("hitter"),
                                "score": s.get("confidence")})
        doc["rallies"] = [{"start_ms": t(r["start_frame"]), "end_ms": t(r["end_frame"])}
                          for r in ev.get("rallies", []) if r["end_frame"] > r["start_frame"]]
    return schema.validate(doc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifacts", type=Path, help="data/videos/<video_id>/artifacts")
    ap.add_argument("--clip-meta", type=Path, help="片段的 .meta.json（提供 clip_id 与 start_ms）")
    ap.add_argument("--clip-id")
    ap.add_argument("--offset-ms", type=int, help="片段在原视频中的起点；默认取 --clip-meta")
    ap.add_argument("--include-interpolated", action="store_true")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    a = ap.parse_args(argv)
    cm = json.loads(a.clip_meta.expanduser().read_text(encoding="utf-8")) if a.clip_meta else {}
    clip_id = a.clip_id or cm.get("clip_id")
    offset = a.offset_ms if a.offset_ms is not None else cm.get("start_ms")
    if not clip_id or offset is None:
        raise SystemExit("需要 --clip-meta，或同时给 --clip-id 和 --offset-ms")
    doc = convert(a.artifacts, clip_id, int(offset), a.include_interpolated)
    out = a.out / f"{clip_id}.json"
    schema.save(doc, out)
    print(f"写出 {out}：court {len(doc['court'])} 帧，shuttle {len(doc['shuttle'])} 帧，"
          f"hits {len(doc['hits'])}，rallies {len(doc['rallies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
