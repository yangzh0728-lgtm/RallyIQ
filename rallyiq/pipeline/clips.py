"""Cut a short MP4 clip from the source video (time line "导出片段").

Re-encodes (H.264 + AAC, faststart) so the cut is frame-accurate and plays in any
browser or phone; stream copy would snap to the previous keyframe. Clips are
cached under the video's work dir, so repeated downloads are instant.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

MAX_CLIP_MS = 10 * 60 * 1000


class ClipError(ValueError):
    pass


def cut_clip(src: Path, out_dir: Path, start_ms: int, end_ms: int, timeout_s: float = 600) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found; install FFmpeg")
    start_ms = max(0, int(start_ms))
    end_ms = int(end_ms)
    if end_ms <= start_ms:
        raise ClipError("end_ms must be after start_ms")
    if end_ms - start_ms > MAX_CLIP_MS:
        raise ClipError(f"clip longer than {MAX_CLIP_MS // 60000} minutes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clip_{start_ms}_{end_ms}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start_ms / 1000:.3f}", "-i", str(src),
        "-t", f"{(end_ms - start_ms) / 1000:.3f}", "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='replace')[-400:]}") from e
    tmp.replace(out)
    return out
