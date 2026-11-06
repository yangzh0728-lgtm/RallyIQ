"""Fine-tune the heatmap shuttle network on labeled clips (needs torch + OpenCV).

Input is plain per-frame labels, so this module does not depend on the evaluation
package: ``TrainClip.labels`` maps a frame index of the clip video to ``(x, y)`` in
source pixels (shuttle visible) or ``None`` (labeled as not visible). Frames without
an entry are unlabeled and do not contribute to the loss.

What it does:

* Decodes only the labeled frame range of each clip once, resized to the network
  input (512x288) exactly like the detector (``HeatmapShuttleDetector._prep``), kept
  as uint8 in memory. A background median per clip is fed when the checkpoint uses it.
* Samples windows of ``seq_len`` frames (frame step ``round(fps / 30)`` like the
  detector's stream split) that contain at least one labeled frame, with horizontal
  flip and brightness/contrast jitter.
* Target per frame: a Gaussian blob (sigma 2.5 input px) at the shuttle, zeros when
  not visible. Loss: the weighted BCE used by TrackNet-style models, masked to labeled
  frames. BatchNorm stays frozen by default (batches are tiny).
* Validation on held-out clips: one centre-weighted pass like the detector, then a peak
  per frame; a hit is a peak within ``tol_px`` source pixels (scaled for 1080p). Reports
  precision / recall / F1 at a few peak thresholds; the best epoch by F1 at 0.5 is kept.

The checkpoint is written in the format the detector loads (``model`` + ``param_dict``)
plus a ``finetune`` block describing the data and the validation result.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("rallyiq.shuttle.train")

SIGMA = 2.5  # input px


@dataclass
class TrainClip:
    clip_id: str
    video: Path
    fps: float
    labels: dict[int, tuple[float, float] | None]  # frame -> source px, None = not visible
    camera: str | None = None


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    steps_per_epoch: int = 300
    freeze_bn: bool = True
    hflip: float = 0.5
    color_jitter: float = 0.2
    device: str | None = None
    seed: int = 0
    tol_px: float = 10.0  # at 1080p; scaled with the frame height
    thresholds: tuple[float, ...] = (0.5, 0.3, 0.2)


@dataclass
class ValResult:
    epoch: int
    loss: float
    per_threshold: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        return self.per_threshold.get("0.5", {}).get("f1", 0.0)


# --------------------------------------------------------------------- data


class ClipFrames:
    """The labeled range of one clip, prepped to network input size (uint8)."""

    def __init__(self, clip: TrainClip, seq_len: int, use_bg: bool, target_fps: float = 30.0):
        import cv2

        from .model import INPUT_HEIGHT, INPUT_WIDTH

        if not clip.labels:
            raise ValueError(f"{clip.clip_id}: no labeled frames")
        self.clip = clip
        self.step = max(1, int(round(clip.fps / target_fps))) if target_fps else 1
        span = (seq_len - 1) * self.step
        self.first = max(0, min(clip.labels) - span)
        last = max(clip.labels) + span
        cap = cv2.VideoCapture(str(clip.video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {clip.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
        frames = []
        size = None
        for _ in range(self.first, last + 1):
            ok, fr = cap.read()
            if not ok:
                break
            size = size or (fr.shape[1], fr.shape[0])
            img = cv2.resize(fr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            frames.append(img[:, :, ::-1])  # RGB
        cap.release()
        if not frames:
            raise RuntimeError(f"{clip.clip_id}: no frames decoded from {clip.video}")
        self.frames = np.ascontiguousarray(np.stack(frames))  # (N, H, W, 3) uint8
        self.width, self.height = size
        self.sx, self.sy = INPUT_WIDTH / self.width, INPUT_HEIGHT / self.height
        self.seq_len = seq_len
        self.median = np.median(self.frames[:: max(1, len(self.frames) // 120)], axis=0).astype(np.uint8) \
            if use_bg else None
        # Window starts (absolute frame index) whose frames are all decoded and that hold a label.
        n_abs = self.first + len(self.frames)
        labeled = set(clip.labels)
        self.starts = [
            s for s in range(self.first, n_abs - span)
            if any(s + k * self.step in labeled for k in range(seq_len))
        ]

    def window(self, start: int) -> tuple[list[int], np.ndarray]:
        idx = [start + k * self.step for k in range(self.seq_len)]
        return idx, self.frames[[i - self.first for i in idx]]

    def target(self, idx: Sequence[int], flip: bool) -> tuple[np.ndarray, np.ndarray]:
        from .model import INPUT_HEIGHT, INPUT_WIDTH

        L = len(idx)
        hm = np.zeros((L, INPUT_HEIGHT, INPUT_WIDTH), np.float32)
        mask = np.zeros(L, np.float32)
        yy, xx = np.mgrid[0:INPUT_HEIGHT, 0:INPUT_WIDTH]
        for k, f in enumerate(idx):
            if f not in self.clip.labels:
                continue
            mask[k] = 1.0
            xy = self.clip.labels[f]
            if xy is None:
                continue
            x, y = (xy[0] + 0.5) * self.sx - 0.5, (xy[1] + 0.5) * self.sy - 0.5
            if flip:
                x = INPUT_WIDTH - 1 - x
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA**2))
            g[g < 0.05] = 0.0
            hm[k] = g
        return hm, mask


def make_input(frames: np.ndarray, median: np.ndarray | None, flip: bool, gain: float, bias: float) -> np.ndarray:
    """(L, H, W, 3) uint8 [+ median] -> (C, H, W) float32, channel order as the detector."""
    parts = ([median] if median is not None else []) + list(frames)
    x = np.stack(parts).astype(np.float32) / 255.0
    if flip:
        x = x[:, :, ::-1]
    if gain != 1.0 or bias != 0.0:
        x = np.clip(x * gain + bias, 0.0, 1.0)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(-1, x.shape[1], x.shape[2]))


def wbce(pred, target, mask):
    """Weighted BCE (TrackNetV3), averaged over labeled frames. pred/target: (B, L, H, W), mask: (B, L)."""
    import torch

    eps = 1e-6
    p = pred.clamp(eps, 1 - eps)
    loss = -((1 - p) ** 2 * target * torch.log(p) + p**2 * (1 - target) * torch.log(1 - p))
    per_frame = loss.mean(dim=(2, 3))
    denom = mask.sum().clamp(min=1.0)
    return (per_frame * mask).sum() / denom


# --------------------------------------------------------------------- model


def load_model(weights: Path, device):
    import torch

    from .model import build_heatmap_net

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    params = dict(ckpt.get("param_dict", {})) if isinstance(ckpt, dict) else {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    seq_len = int(params.get("seq_len", 8))
    bg_mode = str(params.get("bg_mode", "concat") or "")
    net = build_heatmap_net(seq_len, bg_mode)
    net.load_state_dict(state)
    params.update(seq_len=seq_len, bg_mode=bg_mode)
    return net.to(device), params


def _set_bn_eval(net) -> None:
    from torch import nn

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# ---------------------------------------------------------------- validate


def evaluate(net, data: Sequence[ClipFrames], cfg: TrainConfig, device) -> tuple[float, dict[str, dict[str, float]]]:
    """Centre-weighted sliding pass over every labeled frame; detection P/R/F1 per threshold."""
    import torch

    from .detector import ensemble_weights
    from .heatmap import extract_peaks
    from .model import INPUT_HEIGHT, INPUT_WIDTH

    net.eval()
    counts = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in cfg.thresholds}
    losses = []
    for cf in data:
        L = cf.seq_len
        w = ensemble_weights(L, "weight")
        acc: dict[int, np.ndarray] = {}
        acc_w: dict[int, float] = {}
        tol = cfg.tol_px * cf.height / 1080.0
        with torch.no_grad():
            for b0 in range(0, len(cf.starts), cfg.batch_size):
                starts = cf.starts[b0:b0 + cfg.batch_size]
                wins = [cf.window(s) for s in starts]
                x = torch.from_numpy(np.stack([make_input(fr, cf.median, False, 1.0, 0.0) for _, fr in wins]))
                y = net(x.to(device)).float()
                tg = [cf.target(idx, False) for idx, _ in wins]
                losses.append(float(wbce(y, torch.from_numpy(np.stack([t[0] for t in tg])).to(device),
                                         torch.from_numpy(np.stack([t[1] for t in tg])).to(device))))
                y = y.cpu().numpy()
                for (idx, _), hms in zip(wins, y):
                    for k, f in enumerate(idx):
                        if f in cf.clip.labels:
                            acc[f] = acc.get(f, 0) + w[k] * hms[k]
                            acc_w[f] = acc_w.get(f, 0.0) + float(w[k])
        for f, hm in acc.items():
            hm = hm / max(acc_w[f], 1e-6)
            truth = cf.clip.labels[f]
            peaks = extract_peaks(hm, min(cfg.thresholds), 1)
            for t in cfg.thresholds:
                c = counts[t]
                p = [(x, y) for x, y, s in peaks if s >= t]
                if truth is None:
                    c["fp" if p else "tn"] += 1
                    continue
                if not p:
                    c["fn"] += 1
                    continue
                px = (p[0][0] + 0.5) * cf.width / INPUT_WIDTH - 0.5
                py = (p[0][1] + 0.5) * cf.height / INPUT_HEIGHT - 0.5
                if math.hypot(px - truth[0], py - truth[1]) <= tol:
                    c["tp"] += 1
                else:  # wrong place: a false detection and a miss
                    c["fp"] += 1
                    c["fn"] += 1
    out = {}
    for t, c in counts.items():
        prec = c["tp"] / max(c["tp"] + c["fp"], 1)
        rec = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[str(t)] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), **c}
    return (float(np.mean(losses)) if losses else 0.0), out


# ------------------------------------------------------------------- train


def train(
    weights: Path,
    train_clips: Sequence[TrainClip],
    val_clips: Sequence[TrainClip],
    out: Path,
    cfg: TrainConfig | None = None,
    progress: Callable[[str], None] = print,
) -> dict:
    """Fine-tune ``weights`` and write the best checkpoint to ``out``. Returns the report."""
    import torch

    from .detector import pick_device

    cfg = cfg or TrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    net, params = load_model(weights, device)
    L, use_bg = params["seq_len"], params["bg_mode"] == "concat"

    t0 = time.monotonic()
    train_data = [ClipFrames(c, L, use_bg) for c in train_clips]
    val_data = [ClipFrames(c, L, use_bg) for c in val_clips]
    progress(f"decoded {sum(len(d.frames) for d in train_data)} train / "
             f"{sum(len(d.frames) for d in val_data)} val frames in {time.monotonic() - t0:.0f}s on {device}")
    pool = [(d, s) for d in train_data for s in d.starts]
    if not pool:
        raise ValueError("no training windows: check that the train clips have labeled frames")

    history: list[ValResult] = []
    if val_data:
        loss, res = evaluate(net, val_data, cfg, device)
        history.append(ValResult(0, loss, res))
        progress(f"epoch 0 (pretrained): val loss {loss:.4f} " + _fmt(res))
    best = history[0].f1 if history else -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best_epoch = 0

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        if cfg.freeze_bn:
            _set_bn_eval(net)
        tl = []
        for _ in range(cfg.steps_per_epoch):
            xs, hms, masks = [], [], []
            for d, s in random.choices(pool, k=cfg.batch_size):
                flip = random.random() < cfg.hflip
                j = cfg.color_jitter
                gain, bias = 1.0 + random.uniform(-j, j), random.uniform(-j / 2, j / 2)
                idx, fr = d.window(s)
                xs.append(make_input(fr, d.median, flip, gain, bias))
                hm, m = d.target(idx, flip)
                hms.append(hm)
                masks.append(m)
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = net(x)
            loss = wbce(y, torch.from_numpy(np.stack(hms)).to(device), torch.from_numpy(np.stack(masks)).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(float(loss.detach()))
        msg = f"epoch {epoch}: train loss {np.mean(tl):.4f}"
        if val_data:
            vloss, res = evaluate(net, val_data, cfg, device)
            history.append(ValResult(epoch, vloss, res))
            msg += f", val loss {vloss:.4f} " + _fmt(res)
            if history[-1].f1 > best:
                best, best_epoch = history[-1].f1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
        progress(msg)

    report = {
        "base_weights": str(weights),
        "best_epoch": best_epoch,
        "train_clips": {c.clip_id: len(c.labels) for c in train_clips},
        "val_clips": {c.clip_id: len(c.labels) for c in val_clips},
        "config": {k: v for k, v in asdict(cfg).items() if k != "device"},
        "history": [{"epoch": h.epoch, "val_loss": round(h.loss, 5), "val": h.per_threshold} for h in history],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "param_dict": params, "finetune": report}, out)
    progress(f"saved epoch {best_epoch} -> {out}")
    return report


def _fmt(res: dict[str, dict[str, float]]) -> str:
    return " ".join(f"@{t}: P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f}" for t, r in res.items())
