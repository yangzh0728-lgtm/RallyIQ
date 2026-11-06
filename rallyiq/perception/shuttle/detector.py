"""Per-frame shuttle candidates from a multi-frame heatmap network (needs torch + OpenCV).

How a video is processed:

* Frames are streamed (never all held in memory). Every frame is resized to the
  network input (512x288), optionally after cropping to a region of interest.
* A background median image per camera segment is computed from sampled frames
  and fed as an extra input, which suppresses static bright objects.
* The network sees ``seq_len`` consecutive frames at a time. Windows slide by one
  frame and each frame's heatmap is a weighted average over every window that
  contained it (temporal ensemble), which stabilises faint detections.
* High frame rate video (e.g. 60/120 fps) is split into interleaved streams at
  roughly ``target_fps`` (frames 0, 4, 8, ... / 1, 5, 9, ... at 120 fps), so the
  motion between the frames the network sees matches what it was trained on.
  Every frame is still processed exactly once.
* Each final heatmap yields up to ``top_k`` peaks down to ``peak_threshold``; the
  tracker picks the consistent ones.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .heatmap import extract_peaks

log = logging.getLogger("rallyiq.shuttle")

Candidate = tuple[float, float, float]

HEATMAP_FILENAMES = ("heatmap.pt", "TrackNet_best.pt")


@dataclass
class DetectorConfig:
    batch_size: int = 8
    device: str | None = None  # "cuda" | "mps" | "cpu"; auto when None
    ensemble: str = "weight"  # "weight" | "average" | "none"
    peak_threshold: float = 0.2  # 0.1 raises coverage but costs ok hits (2026-11-06), see shuttle/README.md
    top_k: int = 3
    target_fps: float = 30.0
    median_samples: int = 120


@dataclass
class Segment:
    """Frames ``start..end`` (inclusive) share one camera pose / background / ROI."""

    start: int
    end: int
    roi: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 in original px


def find_weights(explicit: str | os.PathLike | None = None) -> Path | None:
    """Locate the heatmap checkpoint file. Returns ``None`` when nothing is installed."""
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    if env := os.environ.get("RALLYIQ_SHUTTLE_WEIGHTS"):
        roots.append(Path(env).expanduser())
    roots.append(Path(__file__).resolve().parents[3] / "weights" / "shuttle")
    roots.append(Path("~/.cache/rallyiq/weights/shuttle").expanduser())
    for r in roots:
        if r.is_file():
            return r
        for name in HEATMAP_FILENAMES:
            if (r / name).is_file():
                return r / name
    return None


def pick_device(requested: str | None = None):
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensemble_weights(seq_len: int, mode: str) -> np.ndarray:
    """Weight of a frame's heatmap by its position in the window (centre counts most)."""
    if mode == "average" or mode == "none":
        w = np.ones(seq_len, dtype=np.float32)
    elif mode == "weight":
        w = np.array([min(i + 1, seq_len - i) for i in range(seq_len)], dtype=np.float32)
    else:
        raise ValueError(f"unknown ensemble mode {mode!r}")
    return w / w.sum()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HeatmapShuttleDetector:
    def __init__(self, weights: Path, config: DetectorConfig | None = None):
        import torch

        from .model import build_heatmap_net

        self.cfg = config or DetectorConfig()
        self.device = pick_device(self.cfg.device)
        self.weights = Path(weights)
        self.sha = _sha256(self.weights)
        try:
            ckpt = torch.load(self.weights, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - older checkpoints pickle extra metadata
            log.warning("checkpoint %s needs full unpickling; loading it as a trusted local file", self.weights)
            ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
        params = ckpt.get("param_dict", {}) if isinstance(ckpt, dict) else {}
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.seq_len = int(params.get("seq_len", 8))
        self.bg_mode = str(params.get("bg_mode", "concat") or "")
        self.net = build_heatmap_net(self.seq_len, self.bg_mode)
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()
        self.weights_vec = ensemble_weights(self.seq_len, self.cfg.ensemble)
        log.info("shuttle detector: seq_len=%d bg=%r device=%s", self.seq_len, self.bg_mode, self.device)

    @property
    def name(self) -> str:
        return f"shuttle-heatmap-seq{self.seq_len}@{self.sha[:12]}"

    # ---------------------------------------------------------------- frames

    @staticmethod
    def _prep_u8(frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
        """BGR uint8 frame -> (3, 288, 512) uint8 RGB."""
        import cv2

        from .model import INPUT_HEIGHT, INPUT_WIDTH

        if roi is not None:
            x0, y0, x1, y1 = roi
            frame_bgr = frame_bgr[y0:y1, x0:x1]
        img = cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))

    @classmethod
    def _prep(cls, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
        """BGR uint8 frame -> (3, 288, 512) float32 RGB in 0..1."""
        return cls._prep_u8(frame_bgr, roi).astype(np.float32) / 255.0

    def _medians(self, video: Path, segments: list[Segment]) -> list[np.ndarray | None]:
        import cv2

        if self.bg_mode != "concat":
            return [None] * len(segments)
        # One sequential decode instead of a seek per sample: broadcast footage has
        # dozens of short camera segments, so nearly every frame is a sample, and each
        # seek decodes from the previous keyframe (~100 ms per seek at 1080p).
        wanted: dict[int, list[int]] = {}  # frame -> segments that sample it
        for si, seg in enumerate(segments):
            n = seg.end - seg.start + 1
            k = min(self.cfg.median_samples, n)
            for i in np.unique(np.linspace(seg.start, seg.end, k).astype(int)):
                wanted.setdefault(int(i), []).append(si)
        samples: list[list[np.ndarray]] = [[] for _ in segments]
        out: list[np.ndarray | None] = [None] * len(segments)
        remaining = [sum(1 for v in wanted.values() if si in v) for si in range(len(segments))]

        def close(si: int) -> None:
            fr = samples[si]
            # Stored as uint8 at network size to keep memory low; the median of
            # the quantised values equals the old float median up to rounding.
            out[si] = (np.median(np.stack(fr), axis=0).astype(np.float32) / 255.0) if fr else None
            samples[si] = []

        cap = cv2.VideoCapture(str(video))
        last = max(wanted) if wanted else -1
        i = 0
        try:
            while i <= last:
                if i not in wanted:
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    break
                for si in wanted[i]:
                    samples[si].append(self._prep_u8(fr, segments[si].roi))
                    remaining[si] -= 1
                    if remaining[si] == 0:
                        close(si)
                i += 1
        finally:
            cap.release()
        for si in range(len(segments)):
            if samples[si]:
                close(si)
        return out

    # ----------------------------------------------------------------- detect

    def detect(
        self,
        video: Path,
        fps: float,
        frame_count: int,
        segments: list[Segment] | None = None,
        progress: Callable[[float], None] = lambda p: None,
    ) -> tuple[dict[int, list[Candidate]], tuple[int, int]]:
        """Return ``({frame: [(x, y, score), ...]}, (width, height))`` in original px."""
        import cv2
        import torch

        from .model import INPUT_HEIGHT, INPUT_WIDTH

        cap = cv2.VideoCapture(str(video))
        ok, first = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read video {video}")
        size = [first.shape[1], first.shape[0]]
        segments = [
            Segment(s.start, s.end, _clamp_roi(s.roi, size[0], size[1]))
            for s in (segments or [Segment(0, frame_count - 1 if frame_count > 0 else 1 << 40)])
        ]
        medians = self._medians(video, segments)
        stride = max(1, int(round(fps / self.cfg.target_fps))) if self.cfg.target_fps else 1
        L = self.seq_len

        def seg_index(f: int) -> int:
            for i, s in enumerate(segments):
                if s.start <= f <= s.end:
                    return i
            return len(segments) - 1

        # Per-stream sliding buffers: deque of (frame_idx, prepped) and ensemble accumulators.
        streams: dict[tuple[int, int], deque] = {}
        acc: dict[int, np.ndarray] = {}
        acc_w: dict[int, float] = {}
        pending: list[tuple[list[int], np.ndarray, int, int]] = []  # (frame idxs, uint8 frames, n_real, segment)
        results: dict[int, list[Candidate]] = {}

        def finalize(f: int) -> None:
            hm = acc.pop(f) / max(acc_w.pop(f), 1e-6)
            seg = segments[seg_index(f)]
            x0, y0, x1, y1 = seg.roi or (0, 0, size[0], size[1])
            sx, sy = (x1 - x0) / INPUT_WIDTH, (y1 - y0) / INPUT_HEIGHT
            results[f] = [
                (x0 + (x + 0.5) * sx - 0.5, y0 + (y + 0.5) * sy - 0.5, s)
                for x, y, s in extract_peaks(hm, self.cfg.peak_threshold, self.cfg.top_k)
            ]

        med_dev: dict = {}

        def background(seg_i: int, first_frame):
            if medians[seg_i] is None:
                return first_frame
            if seg_i not in med_dev:
                med_dev[seg_i] = torch.from_numpy(medians[seg_i]).to(self.device)
            return med_dev[seg_i]

        def run_batch() -> None:
            if not pending:
                return
            # Windows travel as uint8 and become float on the device: 4x less copying
            # than float32 windows built on the CPU (every frame sits in L windows).
            with torch.no_grad():
                x = torch.from_numpy(np.stack([p[1] for p in pending])).to(self.device)
                x = x.float().div_(255.0)
                if self.bg_mode == "concat":
                    bg = torch.stack([background(p[3], x[i, :3]) for i, p in enumerate(pending)])
                    x = torch.cat([bg, x], dim=1)
                y = self.net(x).float().cpu().numpy()
            for (idxs, _inp, n_real, _seg), hms in zip(pending, y):
                for pos, f in enumerate(idxs[:n_real]):
                    w = float(self.weights_vec[pos])
                    if f in acc:
                        acc[f] += w * hms[pos]
                        acc_w[f] += w
                    else:
                        acc[f] = w * hms[pos]
                        acc_w[f] = w
                # The first frame of a window gets no more contributions after it.
                if self.cfg.ensemble == "none":
                    for f in idxs[:n_real]:
                        if f in acc:
                            finalize(f)
                else:
                    finalize(idxs[0])
            pending.clear()

        def emit(key, buf: deque) -> None:
            seg_i = key[1]
            items = list(buf)
            n_real = len(items)
            if n_real < L:
                items = items + [items[-1]] * (L - n_real)
            idxs = [f for f, _ in items]
            pending.append((idxs, np.concatenate([fr for _, fr in items], axis=0), n_real, seg_i))
            if len(pending) >= self.cfg.batch_size:
                run_batch()

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {video}")
        f = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                seg_i = seg_index(f)
                key = (f % stride, seg_i)
                buf = streams.setdefault(key, deque(maxlen=L))
                buf.append((f, self._prep_u8(frame, segments[seg_i].roi)))
                if len(buf) == L:
                    emit(key, buf)
                    if self.cfg.ensemble == "none":
                        buf.clear()
                # A stream whose segment just ended flushes its tail.
                if f == segments[seg_i].end:
                    for k in [k for k in streams if k[1] == seg_i]:
                        self._flush(streams.pop(k), emit, k, L)
                f += 1
                if f % 64 == 0:
                    progress(min(0.99, f / max(frame_count, 1)))
            for k in list(streams):
                self._flush(streams.pop(k), emit, k, L)
            run_batch()
            for rest in sorted(acc):
                finalize(rest)
        finally:
            cap.release()
        progress(1.0)
        if frame_count and f != frame_count:
            log.info("decoded %d frames, metadata said %d", f, frame_count)
        return results, (size[0], size[1])

    @staticmethod
    def _flush(buf: deque, emit, key, L: int) -> None:
        """Process the last frames of a stream that never started a full window of their own."""
        items = list(buf)
        if not items:
            return
        if len(items) < L:
            # Short stream (or ensemble 'none' leftovers): one padded window.
            emit(key, deque(items))
            return
        # Full buffer: windows starting at each of the remaining L-1 frames, padded.
        for i in range(1, L):
            emit(key, deque(items[i:]))


def _clamp_roi(roi, w: int, h: int):
    if roi is None:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in roi)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def court_roi(image_corners: list[list[float]], width: int, height: int, min_zoom: float = 1.25):
    """A 16:9 crop around the court that keeps the air space above it.

    Returns ``None`` when the crop would not enlarge the shuttle by at least
    ``min_zoom`` (then the full frame is just as good).
    """
    pts = np.asarray(image_corners, dtype=float)
    cx0, cx1 = pts[:, 0].min(), pts[:, 0].max()
    cy0, cy1 = pts[:, 1].min(), pts[:, 1].max()
    cw, ch = cx1 - cx0, cy1 - cy0
    x0, x1 = cx0 - 0.25 * cw, cx1 + 0.25 * cw
    y0, y1 = cy0 - 1.2 * ch, cy1 + 0.1 * ch  # high clears go well above the far baseline
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
    # Grow to 16:9 around the centre, staying inside the frame.
    target = 16 / 9
    w, h = x1 - x0, y1 - y0
    if w / h < target:
        grow = h * target - w
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    else:
        grow = w / target - h
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    if x0 < 0:
        x1, x0 = x1 - x0, 0.0
    if x1 > width:
        x0, x1 = x0 - (x1 - width), float(width)
    if y0 < 0:
        y1, y0 = y1 - y0, 0.0
    if y1 > height:
        y0, y1 = y0 - (y1 - height), float(height)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    zoom = min(width / max(x1 - x0, 1), height / max(y1 - y0, 1))
    if zoom < min_zoom:
        return None
    return int(x0), int(y0), int(round(x1)), int(round(y1))
