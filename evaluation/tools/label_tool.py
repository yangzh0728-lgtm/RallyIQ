"""逐帧标注工具（OpenCV 窗口，在本机运行）。

    pip install opencv-python numpy pyyaml
    python -m evaluation.tools.label_tool ae26_ms_f
    python -m evaluation.tools.label_tool ae26_ms_f --prefill preds/ae26_ms_f.json   # 用模型预测打底，人工修正

读取 ~/rallyiq-data/bwf/clips/<clip_id>.mp4，写入 evaluation/labels/<clip_id>.json。
标注文件里所有时间都是原视频毫秒时间，坐标是源视频像素。

按键（窗口需获得焦点）：
  通用   a / d 上一帧 / 下一帧     A / D 后退 / 前进 10 帧     空格 播放 / 暂停
         1 球场模式   2 羽毛球模式   3 事件模式   s 保存   q 保存并退出   Esc 不保存退出
         r 切换 草稿 / 已复核（只有已复核的标注计入正式基准）
  球场   依次点击 远左、远右、近右、近左 四个外角（双打边线与底线交点）
         c 把最近一次的球场标注复制到当前帧（固定机位时只需偶尔核对）
         x 当前帧球场不可见（特写、回放、观众镜头）   Backspace 撤销上一个角点
  羽毛球 左键点击球的位置 → 记为可见并跳到下一标注帧
         n 当前帧球不可见并跳到下一帧   x 删除当前帧的球标注
  事件   h 近端球员击球   j 远端球员击球   [ 回合开始   ] 回合结束
         u 删除离当前帧最近（±5 帧内）的击球
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .. import schema

HERE = Path(__file__).resolve().parents[1]
DEFAULT_CLIPS = Path.home() / "rallyiq-data" / "bwf" / "clips"
MODES = {1: "court", 2: "shuttle", 3: "events"}


class Session:
    """与界面无关的标注状态，便于测试。"""

    def __init__(self, doc: dict[str, Any], start_ms: int, n_frames: int, shuttle_stride: int = 1):
        self.doc = doc
        self.start_ms = start_ms
        self.fps = float(doc["video"]["fps"])
        self.n = n_frames
        self.stride = max(1, shuttle_stride)
        self.frame = 0
        self.mode = "court"
        self.pending: list[list[float]] = []
        self.open_rally: int | None = None
        self.dirty = False

    # 帧号 ↔ 原视频毫秒
    def t(self, frame: int | None = None) -> int:
        f = self.frame if frame is None else frame
        return int(round(self.start_ms + f * 1000.0 / self.fps))

    def seek(self, f: int) -> None:
        self.frame = max(0, min(self.n - 1, f))

    def _at(self, key: str) -> dict | None:
        tol = 500.0 / self.fps
        for item in self.doc[key]:
            if abs(item["t_ms"] - self.t()) <= tol:
                return item
        return None

    def _set(self, key: str, item: dict) -> None:
        old = self._at(key)
        if old is not None:
            self.doc[key].remove(old)
        self.doc[key].append(item)
        self.doc[key].sort(key=lambda d: d["t_ms"])
        self.dirty = True

    # ---------------- 球场
    def court_click(self, x: float, y: float) -> None:
        self.pending.append([round(x, 1), round(y, 1)])
        if len(self.pending) == 4:
            self._set("court", {"t_ms": self.t(), "visible": True, "corners": self.pending})
            self.pending = []

    def court_undo(self) -> None:
        if self.pending:
            self.pending.pop()

    def court_copy_prev(self) -> bool:
        prev = [c for c in self.doc["court"] if c["visible"] and c["t_ms"] <= self.t()]
        if not prev:
            return False
        self._set("court", {"t_ms": self.t(), "visible": True, "corners": [list(p) for p in prev[-1]["corners"]]})
        return True

    def court_hidden(self) -> None:
        self.pending = []
        self._set("court", {"t_ms": self.t(), "visible": False, "corners": None})

    def court_for_display(self) -> dict | None:
        prev = [c for c in self.doc["court"] if c["t_ms"] <= self.t() + 500.0 / self.fps]
        return prev[-1] if prev else None

    # ---------------- 羽毛球
    def shuttle_click(self, x: float, y: float) -> None:
        self._set("shuttle", {"t_ms": self.t(), "visible": True, "xy": [round(x, 1), round(y, 1)]})
        self.seek(self.frame + self.stride)

    def shuttle_hidden(self) -> None:
        self._set("shuttle", {"t_ms": self.t(), "visible": False, "xy": None})
        self.seek(self.frame + self.stride)

    def shuttle_delete(self) -> None:
        old = self._at("shuttle")
        if old is not None:
            self.doc["shuttle"].remove(old)
            self.dirty = True

    def shuttle_here(self) -> dict | None:
        return self._at("shuttle")

    # ---------------- 事件
    def hit(self, side: str) -> None:
        self._set("hits", {"t_ms": self.t(), "side": side, "player": None})

    def hit_delete_nearest(self, max_frames: int = 5) -> bool:
        tol = max_frames * 1000.0 / self.fps
        cands = [h for h in self.doc["hits"] if abs(h["t_ms"] - self.t()) <= tol]
        if not cands:
            return False
        self.doc["hits"].remove(min(cands, key=lambda h: abs(h["t_ms"] - self.t())))
        self.dirty = True
        return True

    def rally_start(self) -> None:
        self.open_rally = self.t()

    def rally_end(self) -> bool:
        if self.open_rally is None or self.t() <= self.open_rally:
            return False
        self.doc["rallies"].append({"start_ms": self.open_rally, "end_ms": self.t()})
        self.doc["rallies"].sort(key=lambda r: r["start_ms"])
        self.open_rally = None
        self.dirty = True
        return True

    def toggle_status(self) -> str:
        self.doc["status"] = "reviewed" if self.doc.get("status") != "reviewed" else "draft"
        self.dirty = True
        return self.doc["status"]


def load_or_create(clip_id: str, meta: dict, labels_dir: Path, prefill: Path | None) -> dict:
    path = labels_dir / f"{clip_id}.json"
    if path.exists():
        return schema.load(path)
    rng = (meta["start_ms"], meta["start_ms"] + meta["duration_ms"])
    doc = schema.empty_doc(clip_id, "label", meta["width"], meta["height"], meta["fps"], rng)
    if prefill:
        p = schema.load(prefill)
        for k in ("court", "shuttle", "hits", "rallies"):
            doc[k] = [{kk: vv for kk, vv in it.items() if kk not in ("score", "confidence")} for it in p.get(k, [])]
        doc["prefilled_from"] = str(prefill.name)
    return doc


# ------------------------------------------------------------------ 界面
class FrameReader:
    """按需解码，保留最近 120 帧缓存；不把整段 1080p 视频读进内存。"""

    def __init__(self, cv2: Any, path: Path, cache: int = 120):
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        self.cache: dict[int, Any] = {}
        self.order: list[int] = []
        self.max = cache

    def get(self, i: int) -> Any:
        if i in self.cache:
            return self.cache[i]
        if i != self.pos:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:
            raise RuntimeError(f"无法读取第 {i} 帧")
        self.pos = i + 1
        self.cache[i] = fr
        self.order.append(i)
        if len(self.order) > self.max:
            self.cache.pop(self.order.pop(0), None)
        return fr


def run_gui(clip_id: str, clips_dir: Path, labels_dir: Path, prefill: Path | None, stride: int) -> int:
    import cv2  # 只有界面需要 OpenCV

    video = clips_dir / f"{clip_id}.mp4"
    meta_p = clips_dir / f"{clip_id}.meta.json"
    if not video.exists() or not meta_p.exists():
        sys.exit(f"找不到 {video}，请先运行 python -m evaluation.tools.extract_clips --only {clip_id}")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    reader = FrameReader(cv2, video)
    if reader.n == 0:
        sys.exit("视频读取失败")
    doc = load_or_create(clip_id, meta, labels_dir, prefill)
    s = Session(doc, meta["start_ms"], reader.n, stride)
    out_path = labels_dir / f"{clip_id}.json"
    win = f"RallyIQ label - {clip_id}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    playing = False
    msg = ""

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if s.mode == "court":
            s.court_click(x, y)
        elif s.mode == "shuttle":
            s.shuttle_click(x, y)

    cv2.setMouseCallback(win, on_mouse)
    colors = {"court": (0, 255, 0), "shuttle": (0, 255, 255), "events": (255, 128, 0)}
    while True:
        img = reader.get(s.frame).copy()
        c = s.court_for_display()
        if c and c["visible"]:
            pts = [tuple(int(v) for v in p) for p in c["corners"]]
            for i in range(4):
                cv2.line(img, pts[i], pts[(i + 1) % 4], (0, 255, 0), 1, cv2.LINE_AA)
        for p in s.pending:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        sh = s.shuttle_here()
        if sh and sh["visible"]:
            cv2.circle(img, (int(sh["xy"][0]), int(sh["xy"][1])), 8, (0, 255, 255), 1, cv2.LINE_AA)
        near_hits = [h for h in doc["hits"] if abs(h["t_ms"] - s.t()) <= 3000 / s.fps]
        in_rally = any(r["start_ms"] <= s.t() <= r["end_ms"] for r in doc["rallies"]) or s.open_rally is not None
        bar = (f"[{s.mode}] frame {s.frame}/{s.n - 1}  t={s.t()}ms  {doc['status']}  "
               f"hits={len(doc['hits'])} rallies={len(doc['rallies'])} shuttle={len(doc['shuttle'])}"
               f"{'  IN-RALLY' if in_rally else ''}{'  HIT ' + near_hits[0]['side'] if near_hits else ''}  {msg}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[s.mode], 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(int(1000 / s.fps) if playing else 30) & 0xFF
        if playing:
            s.seek(s.frame + 1)
        if k == 255:
            continue
        ch = chr(k)
        msg = ""
        if ch == " ":
            playing = not playing
        elif ch in "dD":
            s.seek(s.frame + (10 if ch == "D" else 1))
        elif ch in "aA":
            s.seek(s.frame - (10 if ch == "A" else 1))
        elif ch in "123":
            s.mode = MODES[int(ch)]; s.pending = []
        elif ch == "s":
            schema.save(doc, out_path); s.dirty = False; msg = "saved"
        elif ch == "q":
            schema.save(doc, out_path); break
        elif k == 27:
            break
        elif ch == "r":
            msg = s.toggle_status()
        elif s.mode == "court":
            if ch == "c":
                msg = "copied" if s.court_copy_prev() else "no previous court"
            elif ch == "x":
                s.court_hidden()
            elif k in (8, 127):
                s.court_undo()
        elif s.mode == "shuttle":
            if ch == "n":
                s.shuttle_hidden()
            elif ch == "x":
                s.shuttle_delete()
        elif s.mode == "events":
            if ch == "h":
                s.hit("near")
            elif ch == "j":
                s.hit("far")
            elif ch == "u":
                msg = "deleted" if s.hit_delete_nearest() else "no hit nearby"
            elif ch == "[":
                s.rally_start(); msg = "rally start"
            elif ch == "]":
                msg = "rally saved" if s.rally_end() else "press [ first"
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_id")
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--labels", type=Path, default=HERE / "labels")
    ap.add_argument("--prefill", type=Path, help="用模型预测文件打底")
    ap.add_argument("--shuttle-stride", type=int, default=1, help="羽毛球模式下每次前进的帧数")
    a = ap.parse_args(argv)
    return run_gui(a.clip_id, a.clips.expanduser(), a.labels, a.prefill, a.shuttle_stride)


if __name__ == "__main__":
    raise SystemExit(main())
