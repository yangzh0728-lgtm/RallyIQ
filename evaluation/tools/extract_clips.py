"""把 clips.yaml 里的片段从本地视频切出来，放到仓库外的数据目录。

视频不进仓库。默认数据目录：~/rallyiq-data/bwf
    raw/<video_id>.<ext>      原视频（自行获取）
    clips/<clip_id>.mp4       切好的片段（逐帧精确切割，重新编码）
    clips/<clip_id>.meta.json 片段元数据：起点毫秒、fps、分辨率

用法：
    python -m evaluation.tools.extract_clips                 # 处理 raw/ 里已有的视频
    python -m evaluation.tools.extract_clips --only ae26_ms_f
    python -m evaluation.tools.extract_clips --download      # 用 yt-dlp 只下载需要的时间段

关于 --download：BWF 视频受版权保护，YouTube 服务条款限制下载。是否下载由你自行决定，
仅限个人内部研究和评测，不要再分发。该选项只下载清单里的片段区间（每段 1–2 分钟），
需要本机已安装 yt-dlp。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from ..schema import load_manifest

HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path.home() / "rallyiq-data" / "bwf"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": round(int(num) / int(den), 3)}


def find_raw(raw_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = raw_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def download_section(video_id: str, start_s: float, dur_s: float, dest: Path) -> Path:
    if not shutil.which("yt-dlp"):
        sys.exit("未找到 yt-dlp。请先安装（brew install yt-dlp），或手动把视频放到 raw/ 目录。")
    dest.mkdir(parents=True, exist_ok=True)
    # 多留 2 秒余量，之后再精确切割；文件名带区间，避免和整段原视频混淆
    a, b = max(0, start_s - 2), start_s + dur_s + 2
    out = dest / f"{video_id}__{int(a)}-{int(b)}.mp4"
    if not out.exists():
        subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
                        "--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts",
                        "-o", str(out), f"https://www.youtube.com/watch?v={video_id}"], check=True)
    return out


def cut(src: Path, offset_s: float, dur_s: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset_s:.3f}", "-i", str(src),
                    "-t", f"{dur_s:.3f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "16",
                    "-preset", "medium", "-c:a", "aac", str(out)], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--only", nargs="*", help="只处理这些片段编号")
    ap.add_argument("--download", action="store_true", help="用 yt-dlp 下载片段区间（见上方版权说明）")
    a = ap.parse_args(argv)

    data = a.data.expanduser()
    if data.resolve().is_relative_to(HERE.parent.resolve()):
        sys.exit("数据目录不能放在仓库里（视频不进仓库）。")
    clips = load_manifest(a.manifest)
    todo = [c for cid, c in clips.items() if not a.only or cid in a.only]
    missing = []
    for c in todo:
        out = data / "clips" / f"{c['id']}.mp4"
        meta_p = out.with_suffix(".meta.json")
        if out.exists() and meta_p.exists():
            print(f"[跳过] {c['id']} 已存在")
            continue
        if a.download:
            src = download_section(c["video_id"], c["start_s"], c["duration_s"], data / "raw")
            offset = c["start_s"] - max(0, c["start_s"] - 2)
        else:
            src = find_raw(data / "raw", c["video_id"])
            offset = c["start_s"]
            if src is None:
                missing.append(c)
                continue
        cut(src, offset, c["duration_s"], out)
        info = probe(out)
        meta = {"clip_id": c["id"], "video_id": c["video_id"], "start_ms": int(round(c["start_s"] * 1000)),
                "duration_ms": int(round(c["duration_s"] * 1000)), **info}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[完成] {c['id']}  {info['width']}x{info['height']} @ {info['fps']}fps → {out}")
    if missing:
        print("\n以下片段缺少原视频，请放到", data / "raw", "（文件名 <video_id>.mp4）或加 --download：")
        for c in missing:
            print(f"  {c['id']:14s} https://www.youtube.com/watch?v={c['video_id']}&t={c['start_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
