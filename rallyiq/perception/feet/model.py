"""Foot landmark model: BlazePose GHUM landmarks (MediaPipe, Apache-2.0) run on one player box.

The model takes a 256x256 RGB crop around one person and returns 33 body landmarks; RallyIQ keeps the
ankles (27, 28), heels (29, 30) and foot tips (31, 32). It is a TFLite file run with LiteRT
(``pip install ai-edge-litert``, Apache-2.0).

``scripts/fetch_weights.sh`` puts the "heavy" variant in ``weights/feet.tflite`` (about 20 ms per player
on a laptop CPU). The "full" variant (about 8 ms) has the same inputs and outputs and still works, but on
amateur video its heel and toe points wander a third more. ``$RALLYIQ_FOOT_MODEL`` overrides the path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_NAME = "blazepose"
INPUT = 256
MARGIN = 0.25  # box padding on each side, as a share of the longer box side
ANKLES, HEELS, TOES = (27, 28), (29, 30), (31, 32)


def model_name(path: str) -> str:
    """``blazepose-heavy`` / ``blazepose-full`` (told apart by size: about 27 MB vs 6 MB), for the artifact."""
    return f"{MODEL_NAME}-{'heavy' if os.path.getsize(path) > 15_000_000 else 'full'}"


def find_foot_model(explicit: str | None = None) -> str | None:
    """``explicit`` -> ``$RALLYIQ_FOOT_MODEL`` -> ``weights/feet.tflite`` in the repo -> ``~/.cache/rallyiq/weights``."""
    for cand in (explicit, os.environ.get("RALLYIQ_FOOT_MODEL")):
        if cand:
            return cand if os.path.exists(cand) else None
    for root in (Path(__file__).resolve().parents[3] / "weights", Path("~/.cache/rallyiq/weights").expanduser()):
        if (root / "feet.tflite").is_file():
            return str(root / "feet.tflite")
    return None


def _interpreter(path: str, threads: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older installs
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
        except ImportError:
            from tensorflow.lite import Interpreter  # type: ignore[no-redef]
    return Interpreter(model_path=path, num_threads=threads)


def runtime_available() -> bool:
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


class FootLandmarker:
    """``(image_bgr, bbox) -> (3, 2, 3)``: rows ankle / heel / toe, columns left / right, values [x, y, conf]."""

    def __init__(self, model_path: str | None = None, threads: int | None = None):
        path = find_foot_model(model_path)
        if not path:
            raise FileNotFoundError("foot model not found: run scripts/fetch_weights.sh or set RALLYIQ_FOOT_MODEL")
        self.path = path
        self.it = _interpreter(path, threads or min(4, os.cpu_count() or 1))
        self.it.allocate_tensors()
        self._in = self.it.get_input_details()[0]["index"]
        self._out = self.it.get_output_details()[0]["index"]

    def __call__(self, image_bgr: np.ndarray, bbox) -> np.ndarray:
        import cv2

        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1, 1.0) * (1 + 2 * MARGIN)
        k = INPUT / side
        M = np.array([[k, 0, INPUT / 2 - cx * k], [0, k, INPUT / 2 - cy * k]], dtype=np.float32)
        crop = cv2.warpAffine(image_bgr, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR)
        x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self.it.set_tensor(self._in, x)
        self.it.invoke()
        lm = self.it.get_tensor(self._out).reshape(-1, 5)[:33]
        xy = lm[:, :2] / k + [cx - side / 2, cy - side / 2]
        vis = 1.0 / (1.0 + np.exp(-lm[:, 3]))
        out = np.zeros((3, 2, 3))
        for r, idx in enumerate((ANKLES, HEELS, TOES)):
            for c, j in enumerate(idx):
                out[r, c] = (xy[j, 0], xy[j, 1], vis[j])
        return out
