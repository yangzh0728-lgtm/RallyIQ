"""Copy finished videos from a batch run directory into another store (e.g. the local demo).

    python scripts/import_runs.py --run ~/rallyiq-data/runs/own-v12 --dest ~/rallyiq-data/demo \\
        [--only own_20260529_0006 ...] [--meta-dir ~/rallyiq-data/own/full] [--dry-run]

A run directory is what ``real_run.py`` / the batch scripts write: ``results.json`` plus a store
in ``data/`` (``rallyiq.db`` and ``videos/<video_id>/{source.mp4, artifacts/*.json}``). For each
succeeded entry of ``results.json`` this registers the source video in the destination store
under the same ``video_id`` (so re-running skips what is already there), copies the video
meta, every artifact, and the finished job, so the web UI shows it as analysed.

The video gets a name its owner recognises: the original camera file name from
``<meta-dir>/<clip_id>.meta.json`` (``source_file``) when present, else the clip id.
Nothing in the source run is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import STAGES  # noqa: E402
from rallyiq.store import Store  # noqa: E402


def display_name(clip_id: str, meta_dir: Path | None) -> str:
    if meta_dir:
        p = meta_dir / f"{clip_id}.meta.json"
        if p.exists():
            src = json.loads(p.read_text(encoding="utf-8")).get("source_file")
            if src:
                return f"{Path(src).stem} ({clip_id}){Path(src).suffix.lower() or '.mp4'}"
    return f"{clip_id}.mp4"


def import_video(src: Store, dst: Store, video_id: str, filename: str, dry_run: bool = False) -> str:
    """Copy one video with its meta, artifacts and latest job. Returns 'imported' / 'present' / reason."""
    if dst.get_video(video_id) is not None:
        return "present"
    job = src.latest_job(video_id)
    if job is None or job.state != "succeeded":
        return f"not finished ({job.state if job else 'no job'})"
    path = src.video_path(video_id)
    if path is None or not path.exists():
        return "source video missing"
    if dry_run:
        return "would import"
    dst.register_video(video_id, filename, path)
    meta = src.get_video_meta(video_id)
    if meta is not None:
        dst.set_video_meta(meta)
    for name in STAGES + ["calibration_manual"]:
        data = src.artifact_bytes(video_id, name)
        if data is None:
            continue
        dst.blobs.put_bytes(dst.artifact_key(video_id, name), data, "application/json")
        if name in STAGES:
            dst._set_artifact_flag(video_id, name, True)
    copy = dst.create_job(video_id, [s.name for s in job.stages if s.state != "skipped"])
    dst.update_job(copy.model_copy(update={"state": job.state, "stages": job.stages,
                                           "started_at": job.started_at, "finished_at": job.finished_at,
                                           "error": job.error}))
    return "imported"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run directory with results.json and data/")
    ap.add_argument("--dest", type=Path, required=True, help="destination store root (e.g. the demo data dir)")
    ap.add_argument("--only", nargs="*", help="clip ids to import (default: all succeeded)")
    ap.add_argument("--meta-dir", type=Path, help="folder with <clip_id>.meta.json for original file names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run = a.run.expanduser()
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    src = Store(run / "data")
    dst = Store(a.dest.expanduser())
    meta_dir = a.meta_dir.expanduser() if a.meta_dir else None
    for r in results:
        if a.only and r["clip_id"] not in a.only:
            continue
        if r.get("state") != "succeeded" or not r.get("video_id"):
            print(f"{r['clip_id']}: skipped ({r.get('state')})")
            continue
        name = display_name(r["clip_id"], meta_dir)
        print(f"{r['clip_id']} -> {r['video_id']} '{name}': "
              f"{import_video(src, dst, r['video_id'], name, a.dry_run)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
