"""把 clips.yaml 里的片段从本地视频切出来，放到仓库外的数据目录。

视频不进仓库。默认数据目录：~/rallyiq-data/bwf
    raw/<video_id>.<ext>      原视频（自行获取）
    clips/<clip_id>.mp4       切好的片段（逐帧精确切割，重新编码）
    clips/<clip_id>.meta.json 片段元数据：起点毫秒、fps、分辨率

用法：
    python -m evaluation.tools.extract_clips                 # 处理 raw/ 里已有的视频
    python -m evaluation.tools.extract_clips --only ae26_ms_f
    python -m evaluation.tools.extract_clips --download      # 用 yt-dlp 只下载需要的时间段
    python -m evaluation.tools.extract_clips --only own0529_0008   # 自有素材（own_clips.yaml）

自有素材：own_clips.yaml 里的片段从外接硬盘的原片切出（--footage，默认
/Volumes/T7/YZH/Badminton Film，或环境变量 RALLYIQ_FOOTAGE），按文件名在根目录下查找，
只切清单里的时间段并缩到 1080p 以内，不复制整段原片。

关于 --download：BWF 视频受版权保护，YouTube 服务条款限制下载。是否下载由你自行决定，
仅限个人内部研究和评测，不要再分发。该选项只下载清单里的片段区间（每段 1–2 分钟），
需要本机已安装 yt-dlp。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..schema import OWN_MANIFEST, is_own, load_manifests

HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path.home() / "rallyiq-data" / "bwf"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")
DEFAULT_FOOTAGE = Path(os.environ.get("RALLYIQ_FOOTAGE", "/Volumes/T7/YZH/Badminton Film"))
MAX_HEIGHT = 1080  # 自有素材多是 4K：缩到 1080p，和 BWF 片段一致，片段也小得多


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


def find_own(root: Path, filename: str) -> Path | None:
    """原片按文件名在根目录下递归查找；跳过 macOS 的 ._ 资源文件。"""
    direct = root / filename
    if direct.is_file():
        return direct
    hits = sorted(p for p in root.rglob(filename) if p.is_file() and not p.name.startswith("._"))
    return hits[0] if hits else None


def download_section(video_id: str, start_s: float, dur_s: float, dest: Path) -> Path:
    if not shutil.which("yt-dlp"):
        sys.exit("未找到 yt-dlp。请先安装（brew install yt-dlp），或手动把视频放到 raw/ 目录。")
    dest.mkdir(parents=True, exist_ok=True)
    # 多留 2 秒余量，之后再精确切割；文件名带区间，避免和整段原视频混淆
    a, b = max(0, start_s - 2), start_s + dur_s + 2
    out = dest / f"{video_id}__{int(a)}-{int(b)}.mp4"
    if not out.exists():
        subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]",
                        "--merge-output-format", "mp4", "--download-sections", f"*{a}-{b}", "--force-keyframes-at-cuts",
                        "-o", str(out), f"https://www.youtube.com/watch?v={video_id}"], check=True)
    return out


def cut_cmd(src: Path, offset_s: float, dur_s: float, out: Path, max_height: int | None = None,
            audio: bool = True) -> list[str]:
    vf = ["-vf", f"scale=-2:'min({max_height},ih)'"] if max_height else []
    # 自有素材不要音轨：iPhone 的 .MOV 带 apple_apac（空间音频），ffmpeg 解不了，会让整段切片失败
    a = ["-map", "0:a?", "-c:a", "aac"] if audio else ["-an"]
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{offset_s:.3f}", "-i", str(src),
            "-t", f"{dur_s:.3f}", "-map", "0:v:0", *vf, "-c:v", "libx264", "-crf", "16",
            "-preset", "medium", "-pix_fmt", "yuv420p", *a, str(out)]


def cut(src: Path, offset_s: float, dur_s: float, out: Path, max_height: int | None = None,
        audio: bool = True) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cut_cmd(src, offset_s, dur_s, out, max_height, audio), check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE / "clips.yaml")
    ap.add_argument("--own-manifest", type=Path, default=OWN_MANIFEST, help="自有素材清单")
    ap.add_argument("--footage", type=Path, default=DEFAULT_FOOTAGE, help="自有素材原片根目录")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--only", nargs="*", help="只处理这些片段编号")
    ap.add_argument("--download", action="store_true", help="用 yt-dlp 下载片段区间（见上方版权说明）")
    a = ap.parse_args(argv)

    data = a.data.expanduser()
    if data.resolve().is_relative_to(HERE.parent.resolve()):
        sys.exit("数据目录不能放在仓库里（视频不进仓库）。")
    clips = load_manifests(a.manifest, a.own_manifest)
    todo = [c for cid, c in clips.items() if not a.only or cid in a.only]
    missing, own_missing = [], []
    footage = a.footage.expanduser()
    for c in todo:
        out = data / "clips" / f"{c['id']}.mp4"
        meta_p = out.with_suffix(".meta.json")
        if out.exists() and meta_p.exists():
            print(f"[跳过] {c['id']} 已存在")
            continue
        if is_own(c):
            src = find_own(footage, c["file"]) if footage.is_dir() else None
            if src is None:
                own_missing.append(c)
                continue
            cut(src, c["start_s"], c["duration_s"], out, MAX_HEIGHT, audio=False)
            info = probe(out)
            meta = {"clip_id": c["id"], "video_id": Path(c["file"]).stem, "source_file": c["file"],
                    "start_ms": int(round(c["start_s"] * 1000)), "duration_ms": int(round(c["duration_s"] * 1000)),
                    **info}
            meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[完成] {c['id']}  {info['width']}x{info['height']} @ {info['fps']}fps → {out}")
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
    if own_missing:
        state = "找不到这个目录（硬盘没接上？）" if not footage.is_dir() else "目录里没有对应文件"
        print(f"\n以下自有素材在 {footage} 里找不到（{state}），可用 --footage 指定原片根目录：")
        for c in own_missing:
            print(f"  {c['id']:16s} {c['file']}")
    if missing:
        print("\n以下片段缺少原视频，请放到", data / "raw", "（文件名 <video_id>.mp4）或加 --download：")
        for c in missing:
            print(f"  {c['id']:14s} https://www.youtube.com/watch?v={c['video_id']}&t={c['start_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
