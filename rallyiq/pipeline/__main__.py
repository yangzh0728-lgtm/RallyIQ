"""CLI.

    python -m rallyiq.pipeline analyze VIDEO [--demo] [--from STAGE]   # offline, prints artifact dir
    python -m rallyiq.pipeline worker [--idle-exit 300]                # queue worker
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ..contracts import STAGES
from ..store import Store
from .runner import run_job, stages_from
from .worker import Worker


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rallyiq.pipeline")
    ap.add_argument("--data", help="data dir (default $RALLYIQ_DATA or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="import a video and run the pipeline synchronously")
    a.add_argument("video", type=Path)
    a.add_argument("--demo", action="store_true", help="stub stages emit synthetic data")
    a.add_argument("--from", dest="first", choices=STAGES, default="probe")
    w = sub.add_parser("worker", help="run queued jobs forever")
    w.add_argument("--demo", action="store_true")
    w.add_argument("--idle-exit", type=float, default=None, metavar="SECONDS",
                   help="exit after the queue has been empty this long (on-demand GPU containers)")
    w.add_argument("--lease", type=float, default=300.0, metavar="SECONDS",
                   help="a running job whose worker has not heart-beaten for this long is re-queued")
    w.add_argument("--max-attempts", type=int, default=2)
    w.add_argument("--poll", type=float, default=2.0, metavar="SECONDS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    store = Store(args.data)
    config = {"demo": True} if args.demo else {}

    if args.cmd == "analyze":
        video_id, dest = store.create_video(args.video.name, args.video.suffix)
        shutil.copyfile(args.video, dest)
        store.register_video(video_id, args.video.name, dest)
        job = run_job(store, store.create_job(video_id, stages_from(args.first)), config)
        for s in job.stages:
            print(f"{s.name:12s} {s.state:10s} {s.error or ''}")
        print(f"video_id={video_id} artifacts={store.video_dir(video_id) / 'artifacts'}")
        return 0 if job.state == "succeeded" else 1

    Worker(store, poll_s=args.poll, config=config, lease_s=args.lease, max_attempts=args.max_attempts,
           idle_exit_s=args.idle_exit, single_process=False).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
