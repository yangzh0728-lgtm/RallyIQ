"""Stage ``feet``: heel and toe keypoints of each tracked player, during the rallies.

Runs the foot landmark model (``model.py``) on every player box from the ``players`` stage, on the
frames of each rally (plus ``MARGIN_S`` either side; the whole video when there are no rallies yet).
A player whose box is largely covered by another player's box (the near player standing in front of
the far one) is skipped for that frame: the model would find the front player's feet.

Without the model or its runtime the stage returns an empty result and footwork uses the ankles alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ...contracts import FeetFrame, FootLandmarks, PlayerFeet, PlayerTracks
from ...pipeline.context import StageContext, demo_mode

log = logging.getLogger("rallyiq.feet")

MARGIN_S = 1.0
MAX_COVER = 0.4  # skip a player when another box covers this share of its lower half


def _cover(a, b) -> float:
    """Share of the lower half of box ``a`` (where the feet are) covered by box ``b``."""
    x1, y1, x2, y2 = a
    y1 = (y1 + y2) / 2
    ix = max(0.0, min(x2, b[2]) - max(x1, b[0]))
    iy = max(0.0, min(y2, b[3]) - max(y1, b[1]))
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    return ix * iy / area


def wanted_frames(spans: list[tuple[int, int]], fps: float, n_frames: int) -> Callable[[int], bool]:
    if not spans:
        return lambda f: True
    m = int(round(MARGIN_S * fps))
    keep = np.zeros(max(n_frames, max(b for _, b in spans) + m + 1), dtype=bool)
    for a, b in spans:
        keep[max(0, a - m):b + m + 1] = True
    return lambda f: 0 <= f < len(keep) and bool(keep[f])


def detect_feet(frames, players: PlayerTracks, landmarker, wanted: Callable[[int], bool],
                report: Callable[[float], None] = lambda p: None, total: int = 1) -> list[FeetFrame]:
    """``frames``: iterable of (index, t_ms, image) or (index, t_ms, None) for frames to skip."""
    by_frame = {pf.frame: pf for pf in players.frames}
    out: list[FeetFrame] = []
    for i, t_ms, img in frames:
        if i % 60 == 0:
            report(min(i / max(total, 1), 0.99))
        pf = by_frame.get(i)
        if img is None or pf is None or not wanted(i):
            continue
        people = [p for p in pf.players if not p.interpolated]
        found = []
        for p in people:
            if any(_cover(p.bbox, q.bbox) > MAX_COVER and q.bbox[3] > p.bbox[3] for q in people if q is not p):
                continue  # someone nearer the camera stands in front of this player's feet
            r = landmarker(img, p.bbox)
            rnd = lambda a: [round(float(a[0]), 1), round(float(a[1]), 1), round(float(a[2]), 3)]  # noqa: E731
            found.append(PlayerFeet(player_id=p.player_id, track_id=p.track_id, ankle=[rnd(r[0, 0]), rnd(r[0, 1])],
                                    heel=[rnd(r[1, 0]), rnd(r[1, 1])], toe=[rnd(r[2, 0]), rnd(r[2, 1])]))
        if found:
            out.append(FeetFrame(frame=i, t_ms=int(round(t_ms)), players=found))
    report(1.0)
    return out


def iter_selected(path: str, wanted: Callable[[int], bool]):
    """Decode only the frames ``wanted`` asks for (others are grabbed and skipped)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    try:
        while cap.grab():
            if wanted(i):
                ok, img = cap.retrieve()
                yield i, i * 1000.0 / fps, img if ok else None
            i += 1
    finally:
        cap.release()


def run(ctx: StageContext) -> FootLandmarks:
    meta = ctx.require_meta()
    empty = FootLandmarks(video_id=meta.video_id, model=None, frames=[])
    players = ctx.players
    if players is None or not players.frames or demo_mode(ctx) or str(ctx.config.get("feet", "auto")).lower() == "off":
        return empty
    from .model import FootLandmarker, find_foot_model, model_name, runtime_available

    path = find_foot_model(ctx.config.get("foot_model"))
    if not path or not runtime_available():
        log.warning("feet: %s; footwork will use the ankles only",
                    "no foot model (scripts/fetch_weights.sh or RALLYIQ_FOOT_MODEL)" if not path
                    else "no TFLite runtime (pip install ai-edge-litert)")
        return empty
    lm = FootLandmarker(path)
    events = ctx.events
    spans = [(r.start_frame, r.end_frame) for r in events.rallies] if events is not None else []
    wanted = wanted_frames(spans, meta.fps, meta.frame_count)
    frames = ((i, float(meta.t_ms(i)), img) for i, _t, img in iter_selected(str(ctx.video_path), wanted))
    found = detect_feet(frames, players, lm, wanted, ctx.report, meta.frame_count)
    import os

    return FootLandmarks(video_id=meta.video_id, model=f"{model_name(path)}:{os.path.basename(path)}", frames=found)
