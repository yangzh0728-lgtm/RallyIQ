"""各模块的评测指标。

阈值来自 README 的验收标准：
  - 击球：同一侧球员 ±3 帧内匹配，F1 ≥ 0.90 才去掉"待复核"
  - 羽毛球：1080p 下 ≤ 12 像素算命中，可见球召回 ≥ 0.90
像素误差统一换算到 1080p（乘以 1080 / 视频高度），不同分辨率的片段可以直接比较。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np

REF_HEIGHT = 1080
SHUTTLE_HIT_PX = 12.0      # 1080p 像素
HIT_TOL_FRAMES = 3
COURT_OK_PX = 10.0         # 角点平均误差 ≤ 10px@1080p 视为标定成功
RALLY_MIN_IOU = 0.5

# 球场外边界（双打）世界坐标，单位米；与 schema 中角点顺序对应：远左、远右、近右、近左
COURT_W, COURT_L = 6.10, 13.40
COURT_WORLD = np.array([[0, COURT_L], [COURT_W, COURT_L], [COURT_W, 0], [0, 0]], dtype=float)
NET_Y = COURT_L / 2


def px_scale(doc: dict[str, Any]) -> float:
    return REF_HEIGHT / float(doc["video"]["height"])


def frame_ms(doc: dict[str, Any]) -> float:
    return 1000.0 / float(doc["video"]["fps"])


# ----------------------------------------------------------------- 工具函数
def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4 点 DLT，求 src→dst 的单应矩阵（不依赖 OpenCV）。"""
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, pt: Iterable[float]) -> np.ndarray:
    x, y = pt
    p = h @ np.array([x, y, 1.0])
    return p[:2] / p[2]


class _TimeIndex:
    """按 t_ms 查找最近的条目。"""

    def __init__(self, items: list[dict[str, Any]], key: str = "t_ms"):
        self.items = sorted(items, key=lambda d: d[key])
        self.times = [d[key] for d in self.items]

    def nearest(self, t: float, tol: float | None = None) -> dict[str, Any] | None:
        if not self.items:
            return None
        i = bisect.bisect_left(self.times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.items):
                if best is None or abs(self.times[j] - t) < abs(best[0] - t):
                    best = (self.times[j], self.items[j])
        if tol is not None and abs(best[0] - t) > tol:
            return None
        return best[1]


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


# ----------------------------------------------------------------- 球场标定
@dataclass
class CourtStats:
    frames: int = 0               # 标注里球场可见的帧数
    detected: int = 0             # 预测给出了角点
    ok: int = 0                   # 平均角点误差 ≤ COURT_OK_PX
    errors: list[float] = field(default_factory=list)
    false_calib: int = 0          # 球场不可见（特写/回放）时仍输出了标定
    hidden_frames: int = 0

    def merge(self, o: "CourtStats") -> None:
        self.frames += o.frames; self.detected += o.detected; self.ok += o.ok
        self.errors += o.errors; self.false_calib += o.false_calib; self.hidden_frames += o.hidden_frames

    def summary(self) -> dict[str, Any]:
        e = np.asarray(self.errors)
        return {
            "frames": self.frames,
            "detection_rate": self.detected / self.frames if self.frames else None,
            "success_rate": self.ok / self.frames if self.frames else None,
            "mean_corner_err_px": float(e.mean()) if e.size else None,
            "median_corner_err_px": float(np.median(e)) if e.size else None,
            "false_calibration_rate": self.false_calib / self.hidden_frames if self.hidden_frames else None,
        }


def eval_court(gt: dict[str, Any], pred: dict[str, Any] | None) -> CourtStats:
    st = CourtStats()
    tol = frame_ms(gt) / 2 + 1
    idx = _TimeIndex(pred.get("court", []) if pred else [])
    s = px_scale(gt)
    for g in gt["court"]:
        p = idx.nearest(g["t_ms"], tol)
        p_vis = bool(p and p.get("visible") and p.get("corners"))
        if not g["visible"]:
            st.hidden_frames += 1
            st.false_calib += int(p_vis)
            continue
        st.frames += 1
        if not p_vis:
            continue
        st.detected += 1
        err = float(np.mean(np.linalg.norm(np.asarray(g["corners"]) - np.asarray(p["corners"]), axis=1))) * s
        st.errors.append(err)
        st.ok += int(err <= COURT_OK_PX)
    return st


# ----------------------------------------------------------------- 羽毛球
@dataclass
class ShuttleStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    n: int = 0                    # 标注帧数
    errors: list[float] = field(default_factory=list)

    def merge(self, o: "ShuttleStats") -> None:
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn; self.tn += o.tn; self.n += o.n
        self.errors += o.errors

    def summary(self) -> dict[str, Any]:
        p, r, f = _prf(self.tp, self.fp, self.fn)
        n = self.n
        e = np.asarray(self.errors)
        return {
            "frames": n, "visible_frames": self.tp + self.fn,
            "precision": p, "recall": r, "f1": f,
            "accuracy": (self.tp + self.tn) / n if n else None,  # 逐帧判断正确的比例
            "mean_err_px": float(e.mean()) if e.size else None,
        }


def eval_shuttle(gt: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, ShuttleStats]:
    """返回 {"all", "far", "near", "unknown"} 四组统计，按标注球的位置分组。

    分组方法：用时间上最近的球场标注求地面单应，把球的画面位置投到地面，
    投影在网后（y > 6.7 m）算 far。球在空中，这不是真实位置，只用来近似区分
    "画面远端的小球"这一难点；没有球场标注时归入 unknown。
    """
    groups = {k: ShuttleStats() for k in ("all", "far", "near", "unknown")}
    tol = frame_ms(gt) / 2 + 1
    idx = _TimeIndex(pred.get("shuttle", []) if pred else [])
    court_idx = _TimeIndex([c for c in gt["court"] if c["visible"]])
    s = px_scale(gt)
    for g in gt["shuttle"]:
        p = idx.nearest(g["t_ms"], tol)
        p_vis = bool(p and p.get("visible") and p.get("xy") is not None)
        region = "unknown"
        if g["visible"]:
            c = court_idx.nearest(g["t_ms"])
            if c is not None:
                h = homography(np.asarray(c["corners"], float), COURT_WORLD)
                region = "far" if apply_h(h, g["xy"])[1] > NET_Y else "near"
        st = ShuttleStats(n=1)
        if g["visible"]:
            if p_vis:
                d = float(np.linalg.norm(np.asarray(g["xy"]) - np.asarray(p["xy"]))) * s
                if d <= SHUTTLE_HIT_PX:
                    st.tp = 1; st.errors.append(d)
                else:
                    st.fp = 1; st.fn = 1   # 位置错：既是误检也是漏检（TrackNet 惯例）
            else:
                st.fn = 1
        else:
            if p_vis:
                st.fp = 1
            else:
                st.tn = 1
        groups["all"].merge(st)
        groups[region].merge(st)
    return groups


# ----------------------------------------------------------------- 击球
@dataclass
class EventStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_any_side: int = 0          # 只看时间、不管击球方的匹配数
    offsets: list[float] = field(default_factory=list)   # 匹配上的时间误差（帧）

    def merge(self, o: "EventStats") -> None:
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn
        self.tp_any_side += o.tp_any_side; self.offsets += o.offsets

    def summary(self) -> dict[str, Any]:
        p, r, f = _prf(self.tp, self.fp, self.fn)
        n_gt, n_pred = self.tp + self.fn, self.tp + self.fp
        _, _, f_any = _prf(self.tp_any_side, n_pred - self.tp_any_side, n_gt - self.tp_any_side)
        o = np.abs(np.asarray(self.offsets))
        return {
            "gt": n_gt, "pred": n_pred, "precision": p, "recall": r, "f1": f,
            "f1_ignore_side": f_any,
            "mean_abs_offset_frames": float(o.mean()) if o.size else None,
        }


def _in_range(t: float, doc: dict[str, Any]) -> bool:
    return doc["range_ms"][0] <= t <= doc["range_ms"][1]


def _greedy_match(gt: list[dict], pred: list[dict], tol: float, same_side: bool) -> list[tuple[int, int, float]]:
    cands = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            dt = p["t_ms"] - g["t_ms"]
            if abs(dt) > tol:
                continue
            if same_side and g.get("side") and p.get("side") != g.get("side"):
                continue
            cands.append((abs(dt), i, j, dt))
    cands.sort()
    used_g, used_p, out = set(), set(), []
    for _, i, j, dt in cands:
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); out.append((i, j, dt))
    return out


def eval_hits(gt: dict[str, Any], pred: dict[str, Any] | None) -> EventStats:
    fm = frame_ms(gt)
    tol = HIT_TOL_FRAMES * fm + 1e-6
    g = [h for h in gt["hits"] if _in_range(h["t_ms"], gt)]
    p = [h for h in (pred.get("hits", []) if pred else []) if _in_range(h["t_ms"], gt)]
    m = _greedy_match(g, p, tol, same_side=True)
    m_any = _greedy_match(g, p, tol, same_side=False)
    return EventStats(tp=len(m), fp=len(p) - len(m), fn=len(g) - len(m),
                      tp_any_side=len(m_any), offsets=[dt / fm for _, _, dt in m])


# ----------------------------------------------------------------- 回合
@dataclass
class RallyStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    start_err_ms: list[float] = field(default_factory=list)
    end_err_ms: list[float] = field(default_factory=list)

    def merge(self, o: "RallyStats") -> None:
        self.tp += o.tp; self.fp += o.fp; self.fn += o.fn
        self.start_err_ms += o.start_err_ms; self.end_err_ms += o.end_err_ms

    def summary(self) -> dict[str, Any]:
        p, r, f = _prf(self.tp, self.fp, self.fn)
        se, ee = np.abs(self.start_err_ms), np.abs(self.end_err_ms)
        return {
            "gt": self.tp + self.fn, "pred": self.tp + self.fp, "precision": p, "recall": r, "f1": f,
            "mean_abs_start_err_ms": float(se.mean()) if se.size else None,
            "mean_abs_end_err_ms": float(ee.mean()) if ee.size else None,
        }


def _iou(a: dict, b: dict) -> float:
    inter = min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"])
    union = max(a["end_ms"], b["end_ms"]) - min(a["start_ms"], b["start_ms"])
    return max(0.0, inter) / union if union > 0 else 0.0


def eval_rallies(gt: dict[str, Any], pred: dict[str, Any] | None) -> RallyStats:
    r0, r1 = gt["range_ms"]
    overl = lambda r: r["end_ms"] > r0 and r["start_ms"] < r1  # noqa: E731
    g = [r for r in gt["rallies"] if overl(r)]
    p = [r for r in (pred.get("rallies", []) if pred else []) if overl(r)]
    cands = sorted(((_iou(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
    used_g, used_p = set(), set()
    st = RallyStats()
    for iou, i, j in cands:
        if iou < RALLY_MIN_IOU:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j)
        st.tp += 1
        st.start_err_ms.append(p[j]["start_ms"] - g[i]["start_ms"])
        st.end_err_ms.append(p[j]["end_ms"] - g[i]["end_ms"])
    st.fp = len(p) - st.tp
    st.fn = len(g) - st.tp
    return st


def to_dict(x: Any) -> Any:
    return asdict(x)
