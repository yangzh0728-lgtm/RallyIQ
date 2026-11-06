"""Stage ``probe``: read fps / size / frame count with ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from ..contracts import VideoMeta
from .context import StageContext


def probe_video(path: Path, video_id: str, filename: str | None = None) -> VideoMeta:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found; install FFmpeg")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration"
         ":stream_side_data=rotation:stream_tags=rotate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError("no video stream found")
    s = data["streams"][0]
    avg = Fraction(s.get("avg_frame_rate") or "0/1") if s.get("avg_frame_rate", "0/0") != "0/0" else Fraction(0)
    real = Fraction(s.get("r_frame_rate") or "0/1") if s.get("r_frame_rate", "0/0") != "0/0" else Fraction(0)
    fps = float(avg or real)
    if fps <= 0:
        raise ValueError("could not determine frame rate")
    frames = int(s.get("nb_read_packets") or s.get("nb_frames") or 0)
    duration = float(s.get("duration") or data.get("format", {}).get("duration") or 0)
    if not frames and duration:
        frames = int(round(duration * fps))
    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    if not rotation and s.get("tags", {}).get("rotate"):
        rotation = int(s["tags"]["rotate"])
    w, h = int(s["width"]), int(s["height"])
    if rotation % 180:
        w, h = h, w
    return VideoMeta(
        video_id=video_id,
        filename=filename or path.name,
        fps=fps,
        frame_count=frames,
        width=w,
        height=h,
        duration_ms=int(round(duration * 1000)) if duration else int(round(frames * 1000 / fps)),
        codec=s.get("codec_name"),
        rotation=rotation % 360,
        variable_frame_rate=bool(avg and real and abs(float(avg) - float(real)) > 0.5),
    )


def run(ctx: StageContext) -> VideoMeta:
    v = ctx.store.get_video(ctx.video_id)
    meta = probe_video(ctx.video_path, ctx.video_id, v.filename if v else None)
    ctx.store.set_video_meta(meta)
    ctx.meta = meta
    return meta
