"""Rally sequence hit detector (train_v10): a dilated temporal CNN over per-frame shuttle + pose inputs.

Inputs per ~30 Hz frame (27): the shuttle (visible, x/W, y/H, velocity, confidence, frames since seen) and, for each side, the
player nearest the shuttle (wrists and hip centre relative to the shuttle in body heights, body height / H, wrist speed, wrist
above shoulder, present). Model: single-stage MS-TCN style stack (Farha & Gall, CVPR 2019): 1x1 input conv to 64 channels,
8 residual layers of dilated 3-tap convs (dilation 1..128), heads for P(hit) and P(far side). Weights trained on the
ShuttleSet matches (all 44, 3 seeds; ``models/seq_hits.npz``; docs/train_v10). Inference is numpy only.

Use: ``EventsConfig.hit_model = "seq"`` replaces the hits inside each rally by the model's peaks. It wins on broadcast
footage (ShuttleSet test matches) and loses on sam's own footage, so only the broadcast profile turns it on.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .types import HitEvent, PoseFrame, RallySegment, ReviewStatus, ShuttleObs

BUNDLED = Path(__file__).with_name("models") / "seq_hits.npz"
NF = 27
SOURCE = "seq_v1"
HITTER_WINDOW_S = 1.0


def frame_features(shuttle: Sequence[ShuttleObs], poses: Sequence[PoseFrame], fps: float, W: float, H: float,
                   nframes: int) -> tuple[np.ndarray, np.ndarray]:
    st = max(1, round(fps / 30))
    grid = np.arange(0, nframes, st)
    sh = {o.frame: o for o in shuttle if o.visible and o.x is not None}
    pf = {p.frame: p for p in poses}
    pk = sorted(pf)
    X = np.zeros((len(grid), NF), np.float32)
    last = None
    since = 10
    prev_w: dict = {}
    for i, g in enumerate(grid):
        o = sh.get(g) or sh.get(g + 1)
        if o is not None:
            if last is not None:
                X[i, 3] = (o.x - last[0]) / W * 30 / (fps / st) * st
                X[i, 4] = (o.y - last[1]) / H * 30 / (fps / st) * st
            X[i, 0], X[i, 1], X[i, 2], X[i, 5] = 1, o.x / W, o.y / H, o.confidence
            last = (o.x, o.y)
            since = 0
        else:
            since = min(since + 1, 10)
            if last is not None:
                X[i, 1], X[i, 2] = last[0] / W, last[1] / H
        X[i, 6] = since / 10
        j = int(np.searchsorted(pk, g))
        cand = [pk[q] for q in (j - 1, j) if 0 <= q < len(pk) and abs(pk[q] - g) <= 2]
        if not cand:
            continue
        P = pf[min(cand, key=lambda q: abs(q - g))]
        sx, sy = last if last is not None else (W / 2, H / 2)
        for si, side in enumerate(("near", "far")):
            best = None
            for p in P.players:
                if p.side != side:
                    continue
                kp = [(k.x, k.y) if k.confidence >= 0.3 else None for k in p.keypoints]
                ys = [k[1] for k in kp if k]
                if len(ys) < 4:
                    continue
                h = max(max(ys) - min(ys), 20.0)
                wr = [kp[q] if len(kp) > q else None for q in (9, 10)]
                d = min([math.hypot(w[0] - sx, w[1] - sy) for w in wr if w] or [1e9]) / h
                if best is None or d < best[0]:
                    best = (d, p, kp, h, wr)
            if best is None:
                continue
            _, p, kp, h, wr = best
            b = 7 + si * 10
            X[i, b] = 1
            for q, w in enumerate(wr):
                if w:
                    X[i, b + 1 + 2 * q] = np.clip((w[0] - sx) / h, -5, 5)
                    X[i, b + 2 + 2 * q] = np.clip((w[1] - sy) / h, -5, 5)
            hp = [kp[q] for q in (11, 12) if len(kp) > q and kp[q]]
            if hp:
                X[i, b + 5] = np.clip((np.mean([q[0] for q in hp]) - sx) / h, -5, 5)
                X[i, b + 6] = np.clip((np.mean([q[1] for q in hp]) - sy) / h, -5, 5)
            X[i, b + 7] = h / H
            wv = [w for w in wr if w]
            key = (side, p.player_id)
            if wv and key in prev_w:
                X[i, b + 8] = np.clip(max(math.hypot(w[0] - prev_w[key][0], w[1] - prev_w[key][1]) for w in wv)
                                      / h * (fps / st) / 10, 0, 3)
            if wv:
                prev_w[key] = wv[0]
            shs = [kp[q] for q in (5, 6) if len(kp) > q and kp[q]]
            if shs and wv:
                X[i, b + 9] = np.clip((min(q[1] for q in shs) - min(w[1] for w in wv)) / h, -2, 2)
    return grid, X


class SeqHitModel:
    def __init__(self, spec: dict):
        self.mu, self.sd = spec["mu"], spec["sd"]
        self.members = spec["members"]  # list of dicts of numpy weights
        self.thr, self.gap_s = float(spec["thr"]), float(spec["gap_s"])

    @staticmethod
    def _conv(h: np.ndarray, w: np.ndarray, b: np.ndarray, dil: int = 1) -> np.ndarray:
        """h (C, T), w (O, C, k): same-length dilated conv."""
        k = w.shape[2]
        if k == 1:
            return w[:, :, 0] @ h + b[:, None]
        pad = dil * (k // 2)
        hp = np.pad(h, ((0, 0), (pad, pad)))
        T = h.shape[1]
        out = b[:, None] + sum(w[:, :, j] @ hp[:, j * dil: j * dil + T] for j in range(k))
        return out

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = ((X - self.mu) / self.sd).T.astype(np.float32)
        hits, fars = [], []
        for m in self.members:
            h = self._conv(x, m["inp.weight"], m["inp.bias"])
            for i in range(8):
                c = self._conv(h, m[f"layers.{i}.weight"], m[f"layers.{i}.bias"], 2 ** i)
                h = h + self._conv(np.maximum(c, 0), m[f"mix.{i}.weight"], m[f"mix.{i}.bias"])
            hits.append(1 / (1 + np.exp(-self._conv(h, m["hit.weight"], m["hit.bias"])[0])))
            fars.append(1 / (1 + np.exp(-self._conv(h, m["side.weight"], m["side.bias"])[0])))
        return np.mean(hits, 0), np.mean(fars, 0)


@lru_cache(maxsize=2)
def _load(path: str) -> Optional[SeqHitModel]:
    p = Path(path)
    if not p.exists():
        return None
    z = np.load(p)
    n = int(z["n_members"])
    members = [{k.split("/", 1)[1]: z[k].astype(np.float32) for k in z.files if k.startswith(f"m{i}/")} for i in range(n)]
    return SeqHitModel({"mu": z["mu"], "sd": z["sd"], "members": members, "thr": z["thr"], "gap_s": z["gap_s"]})


def get_model(setting: str) -> Optional[SeqHitModel]:
    if not setting or setting in ("rules", "off"):
        return None
    return _load(str(BUNDLED if setting in ("seq", "auto") else Path(setting).expanduser()))


def peaks(grid: np.ndarray, p: np.ndarray, fps: float, thr: float, gap_s: float) -> list[int]:
    cand = [i for i in range(1, len(p) - 1) if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1]]
    keep: list[int] = []
    for i in sorted(cand, key=lambda i: -p[i]):
        if all(abs(grid[i] - grid[j]) / fps >= gap_s for j in keep):
            keep.append(i)
    return sorted(keep)


def apply(rallies: list[RallySegment], shuttle: Sequence[ShuttleObs], poses: Sequence[PoseFrame], model: SeqHitModel,
          fps: float, frame_size: tuple[int, int], margin_ms: float = 300.0) -> None:
    """Replace each rally's hits by the model's peaks inside the rally (+-margin)."""
    if not rallies:
        return
    nframes = max([o.frame for o in shuttle] + [f.frame for f in poses] + [0]) + 1
    # frame -> time from the observations themselves (a rounded fps drifts by hundreds of ms over a minute)
    ft = sorted({(o.frame, o.t_ms) for o in shuttle} | {(f.frame, f.t_ms) for f in poses})
    fr_k = np.array([a for a, _ in ft], float)
    t_k = np.array([b for _, b in ft], float)
    if len(ft) >= 2 and fr_k[-1] > fr_k[0]:
        fps = (fr_k[-1] - fr_k[0]) * 1000.0 / (t_k[-1] - t_k[0])
    grid, X = frame_features(shuttle, poses, fps, frame_size[0], frame_size[1], nframes)
    p_hit, p_far = model.predict(X)
    pk = peaks(grid, p_hit, fps, model.thr, model.gap_s)
    pf = {f.frame: f for f in poses}
    sh = {o.frame: o for o in shuttle if o.visible and o.x is not None}
    pfk = sorted(pf)

    def hitter(frame: int, side: str) -> tuple[Optional[str], Optional[tuple[float, float]]]:
        """The player on the model's side (side head) nearest the shuttle, from the closest pose frame (within
        ``HITTER_WINDOW_S``) that has a player on that side; None when there is none."""
        j = int(np.searchsorted(pfk, frame))
        lo, hi = j - 1, j
        while lo >= 0 or hi < len(pfk):
            # take the closer of the two frontier frames first
            if hi >= len(pfk) or (lo >= 0 and frame - pfk[lo] <= pfk[hi] - frame):
                q, lo = pfk[lo], lo - 1
            else:
                q, hi = pfk[hi], hi + 1
            if abs(q - frame) > HITTER_WINDOW_S * fps:
                if (lo < 0 or frame - pfk[lo] > HITTER_WINDOW_S * fps) and (hi >= len(pfk) or pfk[hi] - frame > HITTER_WINDOW_S * fps):
                    break
                continue
            cands = [p for p in pf[q].players if p.side == side]
            if not cands:
                continue
            o = sh.get(frame)
            if o is None:
                b = cands[0]
            else:
                def dist(p):
                    pts = [(k.x, k.y) for k in p.keypoints if k.confidence >= 0.3]
                    return min((math.hypot(x - o.x, y - o.y) for x, y in pts), default=1e9)
                b = min(cands, key=dist)
            return b.player_id, b.foot_court_xy
        return None, None

    for r in rallies:
        new = []
        for i in pk:
            f = int(grid[i])
            t = float(np.interp(f, fr_k, t_k)) if len(ft) >= 2 else f * 1000.0 / fps
            if not (r.start_ms - margin_ms <= t <= r.end_ms + margin_ms):
                continue
            side = "far" if p_far[i] > 0.5 else "near"
            pid, xy = hitter(f, side)
            o = sh.get(f)
            new.append(HitEvent(hit_id="", rally_id=r.rally_id, index_in_rally=0, frame=f, t_ms=t, hitter_id=pid,
                                shuttle_xy_px=(o.x, o.y) if o else None, shuttle_court_xy=None,
                                confidence=round(float(p_hit[i]), 3),
                                status=ReviewStatus.OK if p_hit[i] >= 0.7 and pid else ReviewStatus.NEEDS_REVIEW,
                                scores={"p_seq": round(float(p_hit[i]), 3)}, hitter_court_xy=xy, source=SOURCE))
        r.hits = new
        for k, h in enumerate(r.hits, start=1):
            h.index_in_rally = k
            h.hit_id = f"{r.rally_id}_S{k:02d}"
