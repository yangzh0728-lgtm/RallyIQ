"""Re-score the TrackNet public eval from saved shuttle candidates (no network run).

Candidates come from ``python -m rallyiq.perception.shuttle VIDEO --peak-threshold 0.1
--candidates DIR/CID.json``; labels and manifest from ``scripts/public_eval.py tracknet``.

    python scripts/tracknet_retrack.py                                  # current defaults
    python scripts/tracknet_retrack.py --tracker weak_short_min_travel_diag_per_s=0.03
    python scripts/tracknet_retrack.py --threshold 0.15

Predictions are written to a scratch directory (default ~/rallyiq-data/diag/tracknet_retrack),
never over the public eval's own preds/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import benchmark  # noqa: E402
from rallyiq.contracts import VideoMeta  # noqa: E402
from rallyiq.perception.shuttle.export import to_benchmark_prediction  # noqa: E402
from rallyiq.perception.shuttle.stage import build_track, tracker_config  # noqa: E402

DATA = Path.home() / "rallyiq-data"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=DATA / "diag" / "tracknet_cands")
    ap.add_argument("--public", type=Path, default=DATA / "public" / "tracknet", help="labels/ and manifest.json")
    ap.add_argument("--out", type=Path, default=DATA / "diag" / "tracknet_retrack")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--tracker", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)

    overrides = dict(kv.split("=", 1) for kv in a.tracker)
    cfg = tracker_config(overrides, a.threshold)
    labels = a.public / "labels"
    manifest = json.loads((a.public / "manifest.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for cid in manifest:
        lab = json.loads((labels / f"{cid}.json").read_text())
        doc = json.loads((a.candidates / f"{cid}.json").read_text())
        v, (w, h) = lab["video"], doc["frame_size"]
        cands = {int(f): [tuple(c) for c in x["candidates"] if c[2] >= a.threshold] for f, x in doc["frames"].items()}
        n = max(round(lab["range_ms"][1] * v["fps"] / 1000), (max(cands) + 1) if cands else 0)
        meta = VideoMeta(video_id=cid, filename=cid, fps=v["fps"], frame_count=n, width=v["width"],
                         height=v["height"], duration_ms=lab["range_ms"][1])
        track, _ = build_track(meta, cands, w, h, doc.get("detector"), cfg)
        (a.out / f"{cid}.json").write_text(json.dumps(to_benchmark_prediction(track, meta, cid, 0)))
    rep = benchmark.run(labels, a.out, manifest, None, False)
    print(json.dumps({"threshold": a.threshold, "tracker": overrides, "shuttle": rep["overall"]["shuttle"]},
                     ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
