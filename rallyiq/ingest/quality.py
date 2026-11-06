"""Input video assessment: what kind of footage is this, and how far can each output be trusted?

README: "自动评估输入 ... 据此选择处理方式，并说明每项结论在这段视频上的可信程度" and
"降级而不拒绝". This module never blocks analysis; it reads the probe result and the
stage outputs that already exist and turns them into:

* ``checks``: measured properties (frame rate, resolution, camera view and height,
  camera motion, court visibility, player and shuttle visibility, brightness, blur);
* ``capabilities``: per output (court metres, players, shuttle, rallies, hits) a
  README status level (``ok`` / ``needs_review`` / ``undeterminable``) with reasons.

Everything is a pure function of its inputs except ``sample_picture_stats``, which
reads a dozen frames with OpenCV when it is installed (skipped otherwise).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from ..contracts import (
    CapabilityReliability,
    CourtCalibration,
    EventsResult,
    PlayerTracks,
    QualityCheck,
    ShuttleTrack,
    Status,
    VideoMeta,
    VideoQuality,
)

OK, REVIEW, UNDET = Status.ok, Status.needs_review, Status.undeterminable

VIEW_ZH = {"end_center": "后场正中", "end_oblique": "后场偏斜", "side": "侧面", "unknown": "未知"}
HEIGHT_ZH = {"high": "高位", "medium": "中等高度", "low": "低位", "unknown": "未知"}

# Thresholds (rough, documented in rallyiq/ingest/README.md).
FPS_LOW, FPS_HIGH = 24.5, 49.0
SHORT_SIDE_LOW, SHORT_SIDE_GOOD = 540, 1080
OBLIQUE_OFFSET = 0.15  # court centre off the image centre, as a fraction of image width
SIDE_VIEW_DEG = 45.0  # baseline steeper than this in the image -> side view
DARK, BRIGHT = 55.0, 205.0  # mean grey level
BLURRY = 60.0  # variance of the Laplacian at 640 px width
WRONG_COURT_MAX_VIS = 0.15  # both sides' player visibility below this with a calibrated court


def _status_worst(*s: Status) -> Status:
    order = {OK: 0, REVIEW: 1, UNDET: 2}
    return max(s, key=lambda x: order[x]) if s else UNDET


# ----------------------------------------------------------------- camera geometry


def camera_view(corners: list[list[float]], width: int, height: int) -> tuple[str, str, float, float]:
    """Classify the camera from the outer court corners in px.

    ``corners``: near-left, near-right, far-right, far-left. Returns
    ``(view, height_class, centre_offset, baseline_angle_deg)``.
    """
    (nlx, nly), (nrx, nry), (frx, fry), (flx, fly) = corners
    angle = math.degrees(math.atan2(abs(nry - nly), abs(nrx - nlx) + 1e-9))
    cx = (nlx + nrx + frx + flx) / 4
    offset = (cx - width / 2) / width
    if angle > SIDE_VIEW_DEG:
        view = "side"
    else:
        view = "end_oblique" if abs(offset) > OBLIQUE_OFFSET else "end_center"
    # Height from foreshortening: how long the 13.4 m court looks relative to its
    # 6.1 m near baseline. Top-down would be ~2.2; a camera at head height
    # behind the baseline squeezes it to well under 0.5.
    near_w = math.hypot(nrx - nlx, nry - nly)
    depth = math.hypot((nlx + nrx) / 2 - (flx + frx) / 2, (nly + nry) / 2 - (fly + fry) / 2)
    if view == "side" or near_w < 1:
        h = "unknown"
    else:
        ratio = depth / near_w
        h = "low" if ratio < 0.45 else "high" if ratio > 0.95 else "medium"
    return view, h, offset, angle


def _corners_in_frame(corners: list[list[float]], width: int, height: int, margin: float = 0.02) -> int:
    mx, my = width * margin, height * margin
    return sum(-mx <= x <= width + mx and -my <= y <= height + my for x, y in corners)


# ---------------------------------------------------------------------- pictures


def sample_picture_stats(video_path: Path | str, frame_count: int, n: int = 12) -> dict | None:
    """Mean brightness and sharpness over ``n`` evenly spaced frames; ``None`` without OpenCV."""
    try:
        import cv2  # noqa: PLC0415 - optional dependency
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    bright, sharp = [], []
    try:
        total = frame_count or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = [int((i + 0.5) * total / n) for i in range(n)] if total else []
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if g.shape[1] > 640:
                g = cv2.resize(g, (640, int(round(g.shape[0] * 640 / g.shape[1]))), interpolation=cv2.INTER_AREA)
            bright.append(float(g.mean()))
            sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    finally:
        cap.release()
    if not bright:
        return None
    bright.sort()
    sharp.sort()
    return {"brightness": bright[len(bright) // 2], "sharpness": sharp[len(sharp) // 2], "frames": len(bright)}


# ------------------------------------------------------------------- visibility


def _rally_frames(events: EventsResult | None) -> list[tuple[int, int]]:
    return [(r.start_frame, r.end_frame) for r in events.rallies] if events and events.rallies else []


def _in_ranges(f: int, ranges: list[tuple[int, int]]) -> bool:
    return not ranges or any(a <= f <= b for a, b in ranges)


def _expected_frames(ranges: list[tuple[int, int]], frame_count: int, stride: int) -> int:
    span = sum(b - a + 1 for a, b in ranges) if ranges else frame_count
    return max(1, math.ceil(span / max(stride, 1)))


def player_visibility(players: PlayerTracks, ranges: list[tuple[int, int]], frame_count: int) -> dict[str, float]:
    """Share of sampled frames (inside rallies, if known) with a real detection on each side."""
    frames = [pf.frame for pf in players.frames]
    diffs = Counter(b - a for a, b in zip(frames, frames[1:]) if b > a)
    stride = diffs.most_common(1)[0][0] if diffs else 1
    expected = _expected_frames(ranges, frame_count, stride)
    seen = {"near": 0, "far": 0}
    for pf in players.frames:
        if not _in_ranges(pf.frame, ranges):
            continue
        sides = {p.side for p in pf.players if not p.interpolated and p.side}
        for s in sides:
            seen[s] += 1
    return {s: min(1.0, seen[s] / expected) for s in seen}


def shuttle_visibility(shuttle: ShuttleTrack, ranges: list[tuple[int, int]], frame_count: int) -> float:
    det = sum(1 for p in shuttle.points if p.source == "detected" and _in_ranges(p.frame, ranges))
    return min(1.0, det / _expected_frames(ranges, frame_count, 1))


# ------------------------------------------------------------------------ assess


def fingerprint(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _pct(x: float) -> str:
    return f"{x:.0%}"


def assess_quality(
    meta: VideoMeta,
    calibration: CourtCalibration | None = None,
    players: PlayerTracks | None = None,
    shuttle: ShuttleTrack | None = None,
    events: EventsResult | None = None,
    picture: dict | None = None,
    fingerprint_inputs: object = None,
) -> VideoQuality:
    checks: list[QualityCheck] = []
    by_key: dict[str, QualityCheck] = {}

    def add(key, label, value, status, score=None, note=None):
        c = QualityCheck(key=key, label=label, value=value, score=score, status=status, note=note)
        checks.append(c)
        by_key[key] = c

    # --- format
    fps = meta.fps
    add("fps", "帧率", f"{fps:.0f} fps", REVIEW if fps < FPS_LOW else OK, fps,
        "低于 25 fps：快球之间可能只有一两帧，击球时刻和球速误差更大" if fps < FPS_LOW
        else ("30 fps 左右：快速杀球会有运动模糊和丢帧，击球按区域级精度给出" if fps < FPS_HIGH else None))
    short = min(meta.width, meta.height)
    add("resolution", "分辨率", f"{meta.width}×{meta.height}",
        REVIEW if short < SHORT_SIDE_LOW else OK, float(short),
        "分辨率偏低：远端球员和远端小球容易漏检" if short < SHORT_SIDE_LOW
        else ("建议 1080p 以上，远端小球更容易识别" if short < SHORT_SIDE_GOOD else None))
    if meta.variable_frame_rate:
        add("vfr", "可变帧率", "是", REVIEW, note="手机可变帧率：时间戳按平均帧率换算，个别时刻可能偏差几帧")

    # --- calibration / camera
    view, height = "unknown", "unknown"
    segs = calibration.segments if calibration else []
    calibrated = [s for s in segs if s.homography is not None]
    total = max(meta.frame_count, 1)
    cal_frames = sum(s.end_frame - s.start_frame + 1 for s in calibrated)
    cal_ok_frames = sum(s.end_frame - s.start_frame + 1 for s in calibrated if s.status == OK)
    if calibration is None:
        add("calibration", "球场标定", None, UNDET, note="还没有标定结果")
    else:
        cov = min(1.0, cal_frames / total)
        ok_cov = min(1.0, cal_ok_frames / total)
        errs = [s.reprojection_error_px for s in calibrated if s.reprojection_error_px is not None]
        err = max(errs) if errs else None
        manual = any(s.method == "manual" for s in calibrated)
        st = OK if ok_cov >= 0.9 else REVIEW if cov >= 0.5 else UNDET
        note = ("已手动点选四角" if manual else None) or (
            None if st == OK else "部分时段没有可靠的球场标定：这些时段不显示米和 m/s，可在网页上点选四角")
        add("calibration", "球场标定", f"{_pct(cov)} 时长已标定" + (f"，最大重投影误差 {err:.1f}px" if err is not None else ""),
            st, cov, note)
        main = max(calibrated, key=lambda s: s.end_frame - s.start_frame, default=None)
        if main is not None and main.image_corners:
            view, height, offset, angle = camera_view(main.image_corners, meta.width, meta.height)
            view_note = {
                "end_center": None,
                "end_oblique": "机位偏向一侧：远端一侧的球员和落点精度较低",
                "side": "侧面机位：前后场深度看得清，左右方向压缩，远端球员容易被遮挡",
            }[view]
            add("camera_view", "机位", VIEW_ZH[view], REVIEW if view == "side" else OK, round(offset, 3), view_note)
            add("camera_height", "机位高度", HEIGHT_ZH[height], REVIEW if height == "low" else OK, None,
                "机位偏低：远端场地被压缩，远端落点和跑位只适合区域级判断" if height == "low" else None)
            inside = _corners_in_frame(main.image_corners, meta.width, meta.height)
            add("court_in_frame", "场地完整度", f"{inside}/4 个角在画面内", OK if inside == 4 else REVIEW, float(inside),
                None if inside == 4 else "球场有角落在画面外：出画的球和该区域的跑位无法判断")
        moves = max(0, len(segs) - 1)
        minutes = max(meta.duration_ms / 60000, 1 / 60)
        rate = moves / minutes
        add("camera_motion", "镜头移动", "固定" if moves == 0 else f"{moves} 次移动或切换",
            OK if rate <= 1 else REVIEW, float(moves),
            None if rate <= 1 else "镜头频繁移动：每次移动后重新标定，切换瞬间附近的数据较不可靠")

    # --- visibility
    ranges = _rally_frames(events)
    scope = "回合内" if ranges else "全片"
    if players is not None:
        vis = player_visibility(players, ranges, meta.frame_count) if players.frames else {"near": 0.0, "far": 0.0}
        for side, label in (("near", "近端球员可见"), ("far", "远端球员可见")):
            v = vis[side]
            add(f"{side}_player_rate", label, f"{scope} {_pct(v)}", OK if v >= 0.8 else REVIEW if v >= 0.4 else UNDET, round(v, 3),
                None if v >= 0.8 else ("远端球员经常丢失：远端跑位和击球者归属需要复核" if side == "far"
                                       else "近端球员经常丢失（可能出画或被遮挡）"))
        # Calibration says "ok" but (almost) nobody stands on that court: the
        # wrong court was probably picked (hall footage, amateur camera angles),
        # and the players were filtered out as off-court.
        cal = by_key.get("calibration")
        if cal is not None and cal.status != UNDET and max(vis.values()) < WRONG_COURT_MAX_VIS:
            note = ("球场标定显示可用，但几乎没有球员站在这块场地上：很可能识别成了别的场地，"
                    "球员因此被当作场外人员过滤。请在左栏点“手动点选四角”或“换一块场地”。")
            add("court_choice", "场地选择", "疑似选错场地", UNDET, round(max(vis.values()), 3), note)
            cal.status, cal.note = UNDET, note
    if shuttle is not None:
        v = shuttle_visibility(shuttle, ranges, meta.frame_count)
        add("shuttle_rate", "羽毛球检出", f"{scope} {_pct(v)}", OK if v >= 0.6 else REVIEW if v >= 0.25 else UNDET, round(v, 3),
            None if v >= 0.6 else "球的检出率低：回合切分和击球时刻会漏判，补点和缺口已单独标记")

    # --- picture
    if picture:
        b, s = picture["brightness"], picture["sharpness"]
        add("brightness", "亮度", "偏暗" if b < DARK else "过曝" if b > BRIGHT else "正常",
            REVIEW if (b < DARK or b > BRIGHT) else OK, round(b, 1),
            "画面偏暗：球和远端球员更难识别" if b < DARK else "画面过亮：白色羽毛球容易融入背景" if b > BRIGHT else None)
        add("sharpness", "清晰度", "模糊" if s < BLURRY else "清晰", REVIEW if s < BLURRY else OK, round(s, 1),
            "画面模糊或对焦不准：骨架和小球的精度会下降" if s < BLURRY else None)

    caps = _capabilities(by_key, calibration is not None, players is not None, shuttle is not None, events is not None)
    return VideoQuality(
        video_id=meta.video_id,
        fingerprint=fingerprint(fingerprint_inputs, meta.model_dump(mode="json")),
        camera_view=view, camera_height=height,
        checks=checks, capabilities=caps, summary=_summary(view, height, fps, checks, caps),
    )


def _capabilities(c: dict[str, QualityCheck], has_cal: bool, has_players: bool, has_shuttle: bool,
                  has_events: bool) -> list[CapabilityReliability]:
    def st(*keys: str) -> tuple[Status, list[str]]:
        found = [c[k] for k in keys if k in c]
        bad = [f"{x.label}：{x.value or '无'}" for x in found if x.status != OK]
        return _status_worst(*(x.status for x in found)) if found else UNDET, bad

    out = []
    s, r = st("calibration", "court_in_frame", "court_choice") if has_cal else (UNDET, ["还没有标定结果"])
    out.append(CapabilityReliability(capability="court_metrics", label="球场坐标（米、m/s、热力图）", status=s, reasons=r))
    if has_players:
        s, r = st("near_player_rate", "far_player_rate", "resolution", "court_choice")
        s = _status_worst(s, REVIEW) if c.get("camera_view") and c["camera_view"].status != OK and s == OK else s
    else:
        s, r = UNDET, ["还没有球员结果"]
    out.append(CapabilityReliability(capability="players", label="球员跑位与身份", status=s, reasons=r))
    s, r = st("shuttle_rate", "fps", "resolution", "sharpness", "brightness") if has_shuttle else (UNDET, ["还没有羽毛球结果"])
    out.append(CapabilityReliability(capability="shuttle", label="羽毛球轨迹", status=s, reasons=r))
    if has_events:
        s, r = st("shuttle_rate", "fps")
        out.append(CapabilityReliability(capability="rallies", label="回合切分", status=s, reasons=r))
        # Hits stay "needs review" until the README accuracy bar (F1 >= 0.90) is met on our eval set.
        s, r = st("shuttle_rate", "fps", "near_player_rate", "far_player_rate")
        out.append(CapabilityReliability(capability="hits", label="击球与击球者", status=_status_worst(s, REVIEW),
                                         reasons=r + ["击球检测尚未达到 F1 ≥ 0.90 的去复核门槛"]))
    else:
        for cap, label in (("rallies", "回合切分"), ("hits", "击球与击球者")):
            out.append(CapabilityReliability(capability=cap, label=label, status=UNDET, reasons=["还没有回合与击球结果"]))
    return out


def _summary(view: str, height: str, fps: float, checks: list[QualityCheck], caps: list[CapabilityReliability]) -> str:
    head = f"{VIEW_ZH[view]}机位" + (f"（{HEIGHT_ZH[height]}）" if height != "unknown" else "") + f"，{fps:.0f} fps"
    weak = [x.label for x in checks if x.status != OK]
    usable = [x.label for x in caps if x.status == OK]
    parts = [head]
    parts.append("需要注意：" + "、".join(weak) if weak else "画面条件良好")
    if usable:
        parts.append("可直接使用：" + "、".join(usable))
    return "；".join(parts) + "。"


# ------------------------------------------------------------------------ cached


QUALITY_ARTIFACT = "quality"
PICTURE_ARTIFACT = "picture_stats"
_INPUTS = ("calibration", "players", "shuttle", "events")


class PictureStats(BaseModel):
    brightness: float
    sharpness: float
    frames: int


def _stamp(path: Path) -> list[int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


def quality_for(store, video_id: str, sample_pictures: bool = True) -> VideoQuality | None:
    """Assessment for a stored video, cached as the ``quality`` artifact until an input changes.

    Returns ``None`` before the probe has run.
    """
    meta = store.get_video_meta(video_id)
    if meta is None:
        return None
    video = store.video_path(video_id)
    stamps = {n: _stamp(store.artifact_path(video_id, n)) for n in _INPUTS}
    stamps["video"] = _stamp(video) if video else None
    fp = fingerprint(stamps, meta.model_dump(mode="json"))
    cached = store.load_artifact(video_id, QUALITY_ARTIFACT, VideoQuality)
    if cached is not None and cached.fingerprint == fp:
        return cached
    pic = store.load_artifact(video_id, PICTURE_ARTIFACT, PictureStats)
    if pic is None and sample_pictures and video is not None and video.exists():
        raw = sample_picture_stats(video, meta.frame_count)
        if raw is not None:
            pic = PictureStats(**raw)
            store.save_artifact(video_id, PICTURE_ARTIFACT, pic)
    q = assess_quality(
        meta,
        store.load_artifact(video_id, "calibration", CourtCalibration),
        store.load_artifact(video_id, "players", PlayerTracks),
        store.load_artifact(video_id, "shuttle", ShuttleTrack),
        store.load_artifact(video_id, "events", EventsResult),
        pic.model_dump() if pic else None,
        fingerprint_inputs=stamps,
    )
    store.save_artifact(video_id, QUALITY_ARTIFACT, q)
    return q
