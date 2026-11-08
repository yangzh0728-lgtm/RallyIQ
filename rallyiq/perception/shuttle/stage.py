"""Stage ``shuttle``: per-frame shuttle detection + trajectory tracking.

``run(ctx) -> ShuttleTrack``. Settings come from ``ctx.config["shuttle"]`` (all optional):

* ``weights``: checkpoint file or directory (else ``RALLYIQ_SHUTTLE_WEIGHTS``,
  ``weights/shuttle/`` in the repo, ``~/.cache/rallyiq/weights/shuttle/``).
* ``require_weights``: raise instead of returning an empty track when no weights are found.
* ``device``, ``batch_size``, ``peak_threshold``, ``top_k``, ``target_fps``, ``ensemble``.
* ``roi``: ``"off"`` (default) or ``"auto"``: crop to the calibrated court plus the air
  above it, which enlarges small far-court shuttles. Experimental until measured.
* ``tracker``: overrides for ``TrackerConfig`` fields.
* ``reuse_candidates``: re-track the raw peaks saved by an earlier run of this video
  (``<work_dir>/shuttle_candidates.json``) instead of running the network again. For
  re-running the court lock / tracker after a calibration fix; ignored when no file.
* ``court_lock``: ``"auto"`` (default) or ``"off"``. When several courts are in view,
  keep only trajectories over the target court: the air box above the calibrated
  court (or, without calibration, the band spanned by the tracked players), with a
  bonus for trajectories that pass within reach of our players. Shuttles over the
  other courts listed by calibration are rejected. See ``target_court.py``.

Besides the contract output, every raw candidate (including rejected ones) is written
to ``<work_dir>/shuttle_candidates.json`` so nothing the model saw is lost.

When no weights are installed the stage returns a track with ``detector=None`` and no
points: downstream stages must read that as "shuttle not analysed", not "never visible".
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path

from ...contracts import CourtCalibration, FrameGap, PlayerTracks, ShuttlePoint, ShuttleTrack, VideoMeta
from ...pipeline.context import StageContext, demo_mode
from ..target_court import PlayerProximity, PlayerSpanRegion, SegmentRegions
from .tracker import Candidate, RegionFn, RestFn, ShuttleTracker, SupportFn, TrackerConfig

log = logging.getLogger("rallyiq.shuttle")

CANDIDATES_FILE = "shuttle_candidates.json"


def tracker_config(overrides: dict | None, peak_threshold: float | None = None) -> TrackerConfig:
    """Tracker settings. ``peak_threshold`` is the detector's: the tracker keeps weak
    candidates down to it (unless ``weak_score`` is overridden), otherwise lowering the
    detector threshold would only add candidates the tracker throws away."""
    cfg = TrackerConfig()
    if peak_threshold is not None:
        cfg.weak_score = float(peak_threshold)
    names = {f.name for f in fields(TrackerConfig)}
    for k, v in (overrides or {}).items():
        if k not in names:
            raise ValueError(f"unknown shuttle tracker setting {k!r}")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    return cfg


def court_lock(
    calibration: CourtCalibration | None, players: PlayerTracks | None
) -> tuple[RegionFn | None, SupportFn | None, str]:
    """Region and player-support functions that keep the tracker on the target court.

    Returns ``(region, support, mode)``; ``mode`` says which evidence was used.
    """
    regions = SegmentRegions(calibration)
    span = PlayerSpanRegion.from_players(players)
    region: RegionFn | None = None
    if regions:
        def region(frame: int, x: float, y: float) -> float:
            r = regions.for_frame(frame)
            if r is not None:
                return r.air_weight(x, y)
            return span.air_weight(x, y) if span is not None else 1.0
        mode = "calibration"
    elif span is not None:
        def region(frame: int, x: float, y: float) -> float:
            return span.air_weight(x, y)
        mode = "player_span"
    else:
        mode = "none"
    prox = PlayerProximity(players)
    support = prox.near if prox else None
    return region, support, mode


# Racket contact, in body heights of the player's box: a wrist this close to the shuttle,
# while the wrist moves at least this fast (a swing, not a hand hanging at the side).
CONTACT_REACH_BODIES = 0.45
SWING_BODIES_PER_S = 1.0
SWING_WINDOW_S = 0.2
HELD_REACH_BODIES = 0.15
# A far player is small: the racket meets the shuttle within this share of the frame
# height however small the box (detections at a far racket sit a few px off).
CONTACT_MIN_FRAME_FRAC = 0.035
_WRISTS = (9, 10)  # COCO-17 left / right wrist
_MIN_KP_CONF = 0.3


class RacketContact:
    """Where our players' rackets can meet the shuttle, from the wrist keypoints.

    ``contact`` is true near a wrist of P1/P2 that is swinging around that frame (the
    box-based reach covers half the image for a big near player, so a neighbour court's
    shuttle passing behind them looked ours). ``held`` is true right at a wrist (a
    shuttle in hand before a serve sits still without being a scene feature). Players
    without wrist keypoints fall back to the box reach for ``contact``.
    """

    def __init__(self, players: PlayerTracks | None, fps: float, frame_height: float = 1080.0):
        self.fps = float(fps)
        self.min_reach = CONTACT_MIN_FRAME_FRAC * float(frame_height)
        self.win = max(1, round(SWING_WINDOW_S * self.fps))
        # frame -> [(pid, h, box, wrists)]
        self.by_frame: dict[int, list] = {}
        self.wrists: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}  # pid -> frame -> wrist
        for fr in (players.frames if players else []):
            rows = []
            for p in fr.players:
                if p.player_id not in ("P1", "P2"):
                    continue
                x1, y1, x2, y2 = p.bbox
                h = max(y2 - y1, 1.0)
                wd = {i: (k[0], k[1]) for i, k in enumerate(p.keypoints)
                      if i in _WRISTS and len(k) > 2 and k[2] >= _MIN_KP_CONF}
                ws = list(wd.values())
                rows.append((p.player_id, h, (x1, y1, x2, y2), ws))
                self.wrists.setdefault(p.player_id, {})[fr.frame] = wd
            if rows:
                self.by_frame[fr.frame] = rows
        self._swing: dict[tuple[str, int], float] = {}

    def __bool__(self) -> bool:
        return bool(self.by_frame)

    def swing(self, pid: str, frame: int, h: float) -> float:
        """Fastest wrist speed (body heights per second) within ``win`` frames."""
        key = (pid, frame)
        if key not in self._swing:
            track = self.wrists.get(pid, {})
            best = 0.0
            for f in range(frame - self.win, frame + self.win + 1):
                a, b = track.get(f - 1), track.get(f)
                for i in (a.keys() & b.keys()) if a and b else ():
                    best = max(best, math.dist(a[i], b[i]))
            self._swing[key] = best * self.fps / h
        return self._swing[key]

    def contact(self, frame: int, x: float, y: float) -> bool:
        for pid, h, (x1, y1, x2, y2), ws in self.by_frame.get(frame, ()):
            if ws:
                reach = max(CONTACT_REACH_BODIES * h, self.min_reach)
                if (any(math.dist(w, (x, y)) <= reach for w in ws)
                        and self.swing(pid, frame, h) >= SWING_BODIES_PER_S):
                    return True
            elif (x1 - CONTACT_REACH_BODIES * h <= x <= x2 + CONTACT_REACH_BODIES * h
                  and y1 - CONTACT_REACH_BODIES * h <= y <= y1 + 0.6 * h):
                return True
        return False

    def held(self, frame: int, x: float, y: float) -> bool:
        return any(math.dist(w, (x, y)) <= HELD_REACH_BODIES * h
                   for _, h, _, ws in self.by_frame.get(frame, ()) for w in ws)


def strict_support(players: PlayerTracks | None, fps: float = 30.0) -> SupportFn | None:
    rc = RacketContact(players, fps)
    return rc.contact if rc else None


def rest_area(calibration: CourtCalibration | None, players: PlayerTracks | None) -> RestFn | None:
    """Where a still shuttle of ours can be: on the target court's floor (with run-off)
    or in a target player's box above the knees (held; a pile on the floor next to a
    player's feet is not in hand). ``None`` without a calibration."""
    regions = SegmentRegions(calibration)
    if not regions:
        return None
    boxes: dict[int, list] = {}
    for fr in (players.frames if players else []):
        boxes[fr.frame] = [p.bbox for p in fr.players if p.player_id in ("P1", "P2")]

    def rest(frame: int, x: float, y: float) -> bool:
        for x1, y1, x2, y2 in boxes.get(frame, ()):
            m = 0.1 * (y2 - y1)
            if x1 - m <= x <= x2 + m and y1 - m <= y <= y1 + 0.7 * (y2 - y1):
                return True
        r = regions.for_frame(frame)
        return r is not None and r.floor_outside_m([x, y]) == 0.0

    return rest


def build_track(
    meta: VideoMeta,
    candidates: dict[int, Sequence[Candidate]],
    width: int,
    height: int,
    detector: str | None,
    config: TrackerConfig | None = None,
    region: RegionFn | None = None,
    support: SupportFn | None = None,
    rest: RestFn | None = None,
    strict: SupportFn | None = None,
    held: SupportFn | None = None,
) -> tuple[ShuttleTrack, dict[int, int]]:
    """Run the tracker and convert to the contract. Also returns the selected
    candidate index per frame (for the raw-candidates file)."""
    frame_count = max(meta.frame_count, (max(candidates) + 1) if candidates else 0)
    if abs(width / height - meta.width / meta.height) > 0.02:
        raise ValueError(
            f"decoded frames are {width}x{height} but the probe says {meta.width}x{meta.height}; "
            "the decoder did not apply the video rotation"
        )
    # Coordinates are in decoded-frame px; scale if the decoder's size differs from the probe.
    sx, sy = meta.width / width, meta.height / height
    tracker = ShuttleTracker(meta.fps, meta.width, meta.height, config, region=region, support=support, rest=rest,
                             strict_support=strict, held=held)
    scaled = {f: [(x * sx, y * sy, s) for x, y, s in cs] for f, cs in candidates.items()}
    res = tracker.track(scaled, frame_count)
    if tracker.rejected_foreign:
        log.info("shuttle: dropped %d trajectories outside the target court", tracker.rejected_foreign)
    points = [
        ShuttlePoint(
            frame=p.frame, t_ms=meta.t_ms(p.frame), x=round(p.x, 2), y=round(p.y, 2),
            confidence=round(p.confidence, 4), source=p.source,
        )
        for p in res.points
    ]
    gaps = [FrameGap(start_frame=g.start_frame, end_frame=g.end_frame, reason=g.reason) for g in res.gaps]
    return ShuttleTrack(video_id=meta.video_id, detector=detector, points=points, gaps=gaps), res.selected


def segments_for(meta: VideoMeta, calibration: CourtCalibration | None, roi_mode: str):
    """Camera segments (background median per segment) covering every frame, with optional ROI."""
    from .detector import Segment, court_roi

    last = max(meta.frame_count - 1, 0)
    segs = sorted(calibration.segments, key=lambda s: s.start_frame) if calibration else []
    if not segs:
        return [Segment(0, last)]
    out = []
    for i, s in enumerate(segs):
        start = 0 if i == 0 else s.start_frame
        end = last if i == len(segs) - 1 else segs[i + 1].start_frame - 1
        roi = None
        if roi_mode == "auto" and s.image_corners and s.status != "undeterminable":
            roi = court_roi(s.image_corners, meta.width, meta.height)
        out.append(Segment(start, max(start, end), roi))
    return out


def save_candidates(path: Path, candidates: dict[int, Sequence[Candidate]], selected: dict[int, int],
                    detector: str, size: tuple[int, int]) -> None:
    doc = {
        "detector": detector,
        "frame_size": list(size),
        "note": "raw heatmap peaks in decoded-frame px; 'selected' = index kept by the tracker or -1",
        "frames": {
            str(f): {"candidates": [[round(x, 2), round(y, 2), round(s, 4)] for x, y, s in cs],
                     "selected": selected.get(f, -1)}
            for f, cs in sorted(candidates.items()) if cs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))


def load_candidates(path: Path) -> tuple[dict[int, list[Candidate]], tuple[int, int], str | None]:
    """Inverse of ``save_candidates``: ``(candidates, frame_size, detector)``."""
    doc = json.loads(Path(path).read_text())
    cands = {int(f): [tuple(c) for c in v["candidates"]] for f, v in doc.get("frames", {}).items()}
    return cands, tuple(doc["frame_size"]), doc.get("detector")


def _reuse_path(ctx: StageContext) -> Path | None:
    cfg = dict(ctx.config.get("shuttle") or {})
    p = Path(ctx.work_dir) / CANDIDATES_FILE
    return p if cfg.get("reuse_candidates") and p.exists() else None


class Cancelled(Exception):
    """Raised inside a background detection when the job no longer needs it."""


def _detector(ctx: StageContext):
    """``(detector, DetectorConfig)`` or ``None`` when no weights are installed."""
    cfg = dict(ctx.config.get("shuttle") or {})
    from .detector import DetectorConfig, HeatmapShuttleDetector, find_weights

    weights = find_weights(cfg.get("weights"))
    if weights is None:
        msg = ("no shuttle model weights found; put the checkpoint in weights/shuttle/ "
               "or set RALLYIQ_SHUTTLE_WEIGHTS (see rallyiq/perception/shuttle/README.md)")
        if cfg.get("require_weights"):
            raise RuntimeError(msg)
        log.warning(msg)
        return None
    det_cfg = DetectorConfig()
    for k in ("device", "batch_size", "peak_threshold", "top_k", "target_fps", "ensemble"):
        if k in cfg:
            setattr(det_cfg, k, cfg[k])
    det = HeatmapShuttleDetector(weights, det_cfg)
    if "peak_threshold" not in cfg and det.recommended_threshold is not None:
        det_cfg.peak_threshold = det.recommended_threshold  # det.cfg is det_cfg
        log.info("shuttle: peak threshold %.2f from %s", det_cfg.peak_threshold, weights.name)
    return det, det_cfg


def _detect(ctx: StageContext, detector, progress):
    meta = ctx.require_meta()
    cfg = dict(ctx.config.get("shuttle") or {})
    segments = segments_for(meta, ctx.calibration, cfg.get("roi", "off"))
    return detector.detect(ctx.video_path, meta.fps, meta.frame_count, segments, progress=progress)


def prefetch(ctx: StageContext, report, cancel) -> "Callable[[], tuple] | None":
    """Start the detector in a background thread while the players stage runs.

    Detection only needs the calibration; the players are used afterwards (court
    lock), so the two heaviest stages can overlap: pose runs on the CPU
    (onnxruntime) and the shuttle network on the GPU (MPS / CUDA). Returns a
    function that waits for and returns ``(detector, det_cfg, candidates, size)``,
    or ``None`` when there is nothing worth starting (demo mode, no weights, or no
    accelerator, where both stages would fight for the same CPU cores).
    """
    if demo_mode(ctx) or _reuse_path(ctx) is not None:
        return None
    got = _detector(ctx)
    if got is None:
        return None
    detector, det_cfg = got
    if detector.device.type == "cpu" and str(ctx.config.get("parallel_stages", "auto")).lower() != "on":
        return None
    import threading

    box: dict = {}

    def progress(p: float) -> None:
        if cancel.is_set():
            raise Cancelled()
        report(0.95 * p)

    def work() -> None:
        try:
            box["result"] = _detect(ctx, detector, progress)
        except BaseException as e:  # noqa: BLE001 - re-raised in the shuttle stage
            box["error"] = e

    t = threading.Thread(target=work, name="rallyiq-shuttle-detect", daemon=True)
    t.start()

    def wait():
        t.join()
        if "error" in box:
            raise box["error"]
        return detector, det_cfg, *box["result"]

    return wait


def run(ctx: StageContext) -> ShuttleTrack:
    meta = ctx.require_meta()
    if demo_mode(ctx) and (players := ctx.players) is not None and players.frames:
        from ...pipeline.demo import demo_shuttle

        return demo_shuttle(meta, players)

    cfg = dict(ctx.config.get("shuttle") or {})
    started = ctx.prefetched.pop("shuttle", None)
    reuse = _reuse_path(ctx)
    if started is not None:
        detector, det_cfg, candidates, size = started()
    elif reuse is not None:
        got = _detector(ctx)
        if got is None:
            return ShuttleTrack(video_id=meta.video_id, detector=None, points=[])
        detector, det_cfg = got
        candidates, size, _ = load_candidates(reuse)
        log.info("shuttle: re-tracking %d frames of saved candidates", len(candidates))
    else:
        got = _detector(ctx)
        if got is None:
            return ShuttleTrack(video_id=meta.video_id, detector=None, points=[])
        detector, det_cfg = got
        candidates, size = _detect(ctx, detector, lambda p: ctx.report(0.95 * p))
    region = support = rest = strict = held = None
    if cfg.get("court_lock", "auto") != "off":
        region, support, mode = court_lock(ctx.calibration, ctx.players)
        rest = rest_area(ctx.calibration, ctx.players)
        rc = RacketContact(ctx.players, meta.fps, meta.height)
        strict, held = (rc.contact, rc.held) if rc else (None, None)
        log.info("shuttle: court lock using %s", mode)
    track, selected = build_track(meta, candidates, size[0], size[1], detector.name,
                                  tracker_config(cfg.get("tracker"), det_cfg.peak_threshold), region, support, rest, strict, held)
    save_candidates(Path(ctx.work_dir) / CANDIDATES_FILE, candidates, selected, detector.name, size)
    ctx.report(1.0)
    return track
