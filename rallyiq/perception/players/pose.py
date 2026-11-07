"""Multi-person pose estimation backends.

The tracker only depends on the ``PoseEstimator`` protocol, so models can be
swapped without touching the rest of the module.

Default backend: RTMO (one-stage multi-person pose, COCO-17 keypoints) exported
to ONNX and run with onnxruntime. RTMO and its body7 weights are Apache-2.0,
which fits the dependency rule in the README. Download the ONNX file once into
``weights/`` (git-ignored) and point ``RALLYIQ_POSE_MODEL`` at it, see
``README.md`` in this folder.

Notes for phone footage: far-side players are often < 80 px tall in 1080p.
``tile_far_half`` runs a second pass on an upscaled crop of the upper part of
the frame and merges the results, which recovers many far players at roughly
twice the cost.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from .models import PersonDetection


class PoseEstimator(Protocol):
    def __call__(self, image_bgr: np.ndarray) -> list[PersonDetection]: ...


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _overlap_small(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box: 1 when one box sits inside the other."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / small) if small > 0 else 0.0


def same_person(a: PersonDetection, b: PersonDetection) -> bool:
    """Two boxes of one person from different passes. A box that includes the
    racket arm can be twice as wide as one that does not, so plain IoU misses
    them: compare overlap of the smaller box, and the feet."""
    if _overlap_small(a.bbox, b.bbox) > 0.6:
        return True
    ha, hb = a.height, b.height
    if min(ha, hb) <= 0 or not 0.7 <= ha / hb <= 1.4:
        return False
    fa = np.array([(a.bbox[0] + a.bbox[2]) / 2, a.bbox[3]])
    fb = np.array([(b.bbox[0] + b.bbox[2]) / 2, b.bbox[3]])
    return float(np.linalg.norm(fa - fb)) < 0.25 * max(ha, hb)


def merge_detections(dets: list[PersonDetection], iou_thr: float = 0.5) -> list[PersonDetection]:
    """Greedy NMS across passes, keeping the higher-scoring person."""
    keep: list[PersonDetection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(_box_iou(d.bbox, k.bbox) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _coreml_options() -> dict[str, str]:
    """CoreML as an ML Program: the old NeuralNetwork format only took 195 of RTMO's
    656 nodes (2026-11-06) and ran no faster than the CPU. The compiled model is
    cached so later runs skip the conversion."""
    cache = os.path.expanduser(os.environ.get("RALLYIQ_COREML_CACHE", "~/.cache/rallyiq/coreml"))
    opts = {"ModelFormat": "MLProgram", "MLComputeUnits": os.environ.get("RALLYIQ_COREML_UNITS", "ALL")}
    try:
        os.makedirs(cache, exist_ok=True)
        opts["ModelCacheDirectory"] = cache
    except OSError:
        pass
    return opts


def _session(ort, model_path: str, kind: str):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if kind == "coreml":
        return ort.InferenceSession(model_path, so, providers=[("CoreMLExecutionProvider", _coreml_options()),
                                                                "CPUExecutionProvider"])
    if kind == "cuda":
        return ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


class CpuFallbackSession:
    """CoreML session that hands a frame to the CPU when CoreML rejects it.

    RTMO has a dynamic-length tensor of person candidates; CoreML (ML Program)
    refuses it when the length is 0, i.e. on frames with nobody in them (blank
    frames, some crops of the far-court pass). Those frames run on the CPU, which
    gives the same result (keypoints agree to < 0.01 px on real frames).
    """

    def __init__(self, primary, make_cpu):
        self.primary = primary
        self._make_cpu = make_cpu
        self._cpu = None
        self.fallbacks = 0

    def get_inputs(self):
        return self.primary.get_inputs()

    def get_providers(self):
        return self.primary.get_providers()

    def run(self, output_names, feed):
        try:
            return self.primary.run(output_names, feed)
        except Exception:  # noqa: BLE001 - onnxruntime raises a generic RuntimeException
            if self._cpu is None:
                self._cpu = self._make_cpu()
            self.fallbacks += 1
            return self._cpu.run(output_names, feed)


def pick_session(model_path: str, input_size: tuple[int, int] = (640, 640)):
    """onnxruntime session on the best backend for this machine.

    ``RALLYIQ_POSE_PROVIDER``: ``auto`` (default), ``coreml``, ``cuda`` or ``cpu``.
    ``auto`` prefers CUDA, then CoreML as an ML Program (with a per-frame CPU
    fallback, see ``CpuFallbackSession``), then the CPU.
    """
    import logging

    import onnxruntime as ort

    log = logging.getLogger(__name__)
    want = os.environ.get("RALLYIQ_POSE_PROVIDER", "auto").lower()
    avail = ort.get_available_providers()
    if want == "auto":
        want = ("cuda" if "CUDAExecutionProvider" in avail
                else "coreml" if "CoreMLExecutionProvider" in avail else "cpu")
    needs = {"coreml": "CoreMLExecutionProvider", "cuda": "CUDAExecutionProvider"}
    if want not in ("cpu", "coreml", "cuda") or (want != "cpu" and needs[want] not in avail):
        want = "cpu"
    if want == "coreml":
        try:
            sess = CpuFallbackSession(_session(ort, model_path, "coreml"),
                                      lambda: _session(ort, model_path, "cpu"))
            return sess, "CoreMLExecutionProvider"
        except Exception as e:  # noqa: BLE001 - CoreML conversion can fail on some graphs
            log.warning("pose: CoreML unavailable (%s); using the CPU", e)
            want = "cpu"
    sess = _session(ort, model_path, want)
    return sess, sess.get_providers()[0]


def find_pose_model(explicit: str | None = None) -> str | None:
    """RTMO ONNX path: ``explicit`` -> ``$RALLYIQ_POSE_MODEL`` -> ``weights/pose.onnx`` or
    ``weights/rtmo-*.onnx`` in the repo -> the same names under ``~/.cache/rallyiq/weights/``.
    ``scripts/fetch_weights.sh`` downloads it to ``weights/pose.onnx``."""
    from pathlib import Path

    for cand in (explicit, os.environ.get("RALLYIQ_POSE_MODEL")):
        if cand:
            return cand if os.path.exists(cand) else None
    roots = [Path(__file__).resolve().parents[3] / "weights", Path("~/.cache/rallyiq/weights").expanduser()]
    for root in roots:
        if (root / "pose.onnx").is_file():
            return str(root / "pose.onnx")
        found = sorted(root.glob("rtmo-*.onnx")) if root.is_dir() else []
        if found:
            return str(found[0])
    return None


class RTMOOnnx:
    """RTMO ONNX (end-to-end export: outputs ``dets`` [1,N,5] and ``keypoints`` [1,N,17,3]).

    Preprocessing follows the reference deployment: letterbox to the model input
    size with the image at the top-left, pad value 114, BGR, no normalisation
    (it is baked into the graph).
    """

    def __init__(
        self,
        model_path: str | None = None,
        input_size: tuple[int, int] = (640, 640),
        score_thr: float = 0.3,
        providers: list[str] | None = None,
        tile_far_half: bool = False,
    ):
        import onnxruntime as ort

        model_path = find_pose_model(model_path)
        if not model_path:
            raise FileNotFoundError(
                "RTMO ONNX model not found. Run scripts/fetch_weights.sh, set RALLYIQ_POSE_MODEL or pass "
                "model_path (see rallyiq/perception/players/README.md)."
            )
        self.input_size = input_size  # (w, h)
        if providers is None:
            self.session, self.provider = pick_session(model_path, input_size)
        else:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.score_thr = score_thr
        self.tile_far_half = tile_far_half

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2

        w_in, h_in = self.input_size
        h, w = img.shape[:2]
        ratio = min(h_in / h, w_in / w)
        resized = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        return blob, ratio

    def _infer(self, img: np.ndarray) -> list[PersonDetection]:
        blob, ratio = self._preprocess(img)
        dets, kps = self.session.run(None, {self.input_name: blob})[:2]
        dets, kps = dets[0], kps[0]
        out = []
        for d, k in zip(dets, kps):
            score = float(d[4])
            if score < self.score_thr:
                continue
            bbox = d[:4] / ratio
            k = k.copy()
            k[:, :2] /= ratio
            out.append(PersonDetection(bbox, k, score))
        return out

    supports_far_roi = True

    def _scale(self, w: int, h: int) -> float:
        w_in, h_in = self.input_size
        return min(h_in / h, w_in / w)

    def far_player_px(self, image_shape: tuple[int, ...], roi: tuple) -> float | None:
        """Height a far-baseline player has in the model input on the full frame."""
        h, w = image_shape[:2]
        return float(roi[4]) * self._scale(w, h)

    def _infer_roi(self, img: np.ndarray, roi: tuple, full: list[PersonDetection]) -> list[PersonDetection]:
        x1, y1, x2, y2 = (int(v) for v in roi[:4])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        out = []
        for d in self._infer(crop):
            # Someone cut off by the crop's lower edge (a near player's upper
            # body) would get a fake foot point on the far half.
            if y2 < img.shape[0] - 2 and d.bbox[3] >= crop.shape[0] - 3:
                continue
            d.bbox[[0, 2]] += x1
            d.bbox[[1, 3]] += y1
            d.keypoints[:, 0] += x1
            d.keypoints[:, 1] += y1
            if any(same_person(d, f) for f in full):
                continue  # same person the full-frame pass already has, keep that one
            out.append(d)
        return out

    def detect_roi(self, image_bgr: np.ndarray, roi: tuple, dets: list[PersonDetection]) -> list[PersonDetection]:
        """``dets`` (a full-frame pass) plus the people a second pass on the upscaled ``roi`` finds."""
        return merge_detections(dets + self._infer_roi(image_bgr, roi, dets))

    def __call__(self, image_bgr: np.ndarray, far_roi: tuple | None = None) -> list[PersonDetection]:
        dets = self._infer(image_bgr)
        if far_roi is not None:
            return merge_detections(dets + self._infer_roi(image_bgr, far_roi, dets))
        if not self.tile_far_half:
            return dets
        # Upper 55% of the frame usually holds the far half of the court.
        h = image_bgr.shape[0]
        crop = image_bgr[: int(0.55 * h)]
        far = self._infer(crop)  # letterbox upscales the crop relative to full frame
        return merge_detections(dets + far)
