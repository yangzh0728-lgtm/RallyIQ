"""标注 / 预测文件格式（labels/<clip_id>.json 与 predictions/<clip_id>.json 共用）。

所有时间都是原 YouTube 视频的毫秒时间（README：统一使用原视频毫秒时间），
所有像素坐标都是源视频分辨率下的像素（video.width × video.height）。

{
  "schema_version": 1,
  "clip_id": "ae26_ms_f",
  "kind": "label" | "prediction",
  "status": "draft" | "reviewed",        # 只有 reviewed 的标注计入正式基准
  "video": {"width": 1920, "height": 1080, "fps": 25.0},
  "range_ms": [900000, 990000],          # 已完整标注事件（击球/回合）的时间范围
  "court": [                              # 球场外边界四角（双打边线 × 底线）
    {"t_ms": 900000, "visible": true,
     "corners": [[x,y], [x,y], [x,y], [x,y]]}   # 顺序：远左、远右、近右、近左（画面中）
  ],
  "shuttle": [{"t_ms": 900040, "visible": true, "xy": [x, y]}],   # 不可见时 xy 为 null
  "hits":    [{"t_ms": 901200, "side": "near" | "far" | null, "player": null}],
  "rallies": [{"start_ms": 900500, "end_ms": 912300}]
}

预测文件可以给 shuttle / hits 加 "score"，给 court 加 "confidence"，基准脚本会忽略未知字段。
字段允许为空，但不要填伪造的默认值（README：不填伪造的默认值）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SIDES = {"near", "far", None}


class SchemaError(ValueError):
    pass


def empty_doc(clip_id: str, kind: str, width: int, height: int, fps: float,
              range_ms: tuple[int, int]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "kind": kind,
        "status": "draft",
        "video": {"width": width, "height": height, "fps": fps},
        "range_ms": [int(range_ms[0]), int(range_ms[1])],
        "court": [],
        "shuttle": [],
        "hits": [],
        "rallies": [],
    }


def _xy(v: Any, where: str) -> None:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v)):
        raise SchemaError(f"{where}: 需要 [x, y]，得到 {v!r}")


def validate(doc: dict[str, Any]) -> dict[str, Any]:
    """检查格式并按时间排序；返回同一个 dict。"""
    cid = doc.get("clip_id", "?")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"{cid}: schema_version 应为 {SCHEMA_VERSION}")
    for k in ("clip_id", "video", "range_ms"):
        if k not in doc:
            raise SchemaError(f"{cid}: 缺少字段 {k}")
    v = doc["video"]
    if not (v.get("width", 0) > 0 and v.get("height", 0) > 0 and v.get("fps", 0) > 0):
        raise SchemaError(f"{cid}: video.width/height/fps 必须为正数")
    r0, r1 = doc["range_ms"]
    if r1 <= r0:
        raise SchemaError(f"{cid}: range_ms 结束必须晚于开始")

    for i, c in enumerate(doc.setdefault("court", [])):
        w = f"{cid}.court[{i}]"
        if "t_ms" not in c or "visible" not in c:
            raise SchemaError(f"{w}: 需要 t_ms 和 visible")
        if c["visible"]:
            cs = c.get("corners")
            if not (isinstance(cs, list) and len(cs) == 4):
                raise SchemaError(f"{w}: visible 时需要 4 个角点")
            for j, p in enumerate(cs):
                _xy(p, f"{w}.corners[{j}]")
    for i, s in enumerate(doc.setdefault("shuttle", [])):
        w = f"{cid}.shuttle[{i}]"
        if "t_ms" not in s or "visible" not in s:
            raise SchemaError(f"{w}: 需要 t_ms 和 visible")
        if s["visible"]:
            _xy(s.get("xy"), w)
    for i, h in enumerate(doc.setdefault("hits", [])):
        if "t_ms" not in h:
            raise SchemaError(f"{cid}.hits[{i}]: 需要 t_ms")
        if h.get("side") not in SIDES:
            raise SchemaError(f"{cid}.hits[{i}]: side 只能是 near/far/null")
    for i, r in enumerate(doc.setdefault("rallies", [])):
        if not ("start_ms" in r and "end_ms" in r and r["end_ms"] > r["start_ms"]):
            raise SchemaError(f"{cid}.rallies[{i}]: 需要 start_ms < end_ms")

    doc["court"].sort(key=lambda c: c["t_ms"])
    doc["shuttle"].sort(key=lambda s: s["t_ms"])
    doc["hits"].sort(key=lambda h: h["t_ms"])
    doc["rallies"].sort(key=lambda r: r["start_ms"])
    return doc


def load(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return validate(json.load(f))


def save(doc: dict[str, Any], path: str | Path) -> None:
    validate(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    tmp.replace(path)


def load_dir(directory: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(directory).glob("*.json")):
        doc = load(p)
        out[doc["clip_id"]] = doc
    return out


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    import yaml  # PyYAML

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    clips = {}
    for c in data["clips"]:
        if c["id"] in clips:
            raise SchemaError(f"clips.yaml 中片段编号重复：{c['id']}")
        clips[c["id"]] = c
    return clips
