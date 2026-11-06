"""Fine-tune the shuttle heatmap network on reviewed labels from evaluation/labels/.

    # all reviewed labels: split=train clips train, labeled split=eval clips validate
    python scripts/finetune_shuttle.py --out weights/shuttle/finetuned.pt

    # only one side-court clip labeled so far: first 70% trains, the rest validates
    python scripts/finetune_shuttle.py --split-clip ko26_court3:0.7 --out weights/shuttle/side.pt

    # then compare on the real pipeline
    RALLYIQ_SHUTTLE_WEIGHTS=weights/shuttle/side.pt python scripts/real_run.py --only ko26_court3

Labels come from ``python -m evaluation.tools.label_tool`` (only ``status: reviewed`` files
unless ``--include-draft``; a draft prefilled from the model would teach the model its own
mistakes). Shuttle entries are the supervised frames: ``visible: true`` with ``xy`` is a
positive, ``visible: false`` a negative; frames without an entry are ignored.

Validation clips are kept out of training. Note that the default validation set is the
labeled ``split: eval`` clips, i.e. the benchmark: picking the best epoch on it makes the
benchmark slightly optimistic, so prefer ``--split-clip`` or a separate ``--val`` clip when
there is enough data.

Memory: each clip's labeled range is held decoded at 512x288 (about 0.44 MB per frame,
so ~1.6 GB for 2 minutes at 30 fps).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import schema

DEFAULT_CLIPS = Path.home() / "rallyiq-data" / "bwf" / "clips"
LABELS = ROOT / "evaluation" / "labels"
MANIFEST = ROOT / "evaluation" / "clips.yaml"


def clip_labels(doc: dict, start_ms: int) -> dict[int, tuple[float, float] | None]:
    """Shuttle labels by frame index of the clip video."""
    fps = float(doc["video"]["fps"])
    out: dict[int, tuple[float, float] | None] = {}
    for s in doc.get("shuttle", []):
        f = int(round((s["t_ms"] - start_ms) * fps / 1000.0))
        if f < 0:
            continue
        out[f] = (float(s["xy"][0]), float(s["xy"][1])) if s["visible"] and s.get("xy") else None
    return out


def split_by_time(labels: dict, frac: float) -> tuple[dict, dict]:
    frames = sorted(labels)
    cut = frames[int(len(frames) * frac)] if frames else 0
    return {f: v for f, v in labels.items() if f < cut}, {f: v for f, v in labels.items() if f >= cut}


def select(labels_dir: Path, clips_dir: Path, manifest: dict, include_draft: bool, train_ids: list[str],
           val_ids: list[str], split_clip: str | None):
    from rallyiq.perception.shuttle.training import TrainClip

    docs = {cid: d for cid, d in schema.load_dir(labels_dir).items()
            if d.get("shuttle") and (include_draft or d.get("status") == "reviewed")}
    train, val, skipped = [], [], []
    split_id, frac = (split_clip.split(":") + ["0.7"])[:2] if split_clip else (None, "0.7")
    for cid, doc in sorted(docs.items()):
        video, meta_p = clips_dir / f"{cid}.mp4", clips_dir / f"{cid}.meta.json"
        if not video.exists() or not meta_p.exists():
            skipped.append(f"{cid} (no video in {clips_dir})")
            continue
        start_ms = int(json.loads(meta_p.read_text(encoding="utf-8"))["start_ms"])
        labels = clip_labels(doc, start_ms)
        cam = (manifest.get(cid, {}).get("conditions") or {}).get("camera")

        def mk(lab, suffix=""):
            return TrainClip(cid + suffix, video, float(doc["video"]["fps"]), lab, cam)

        if cid == split_id:
            a, b = split_by_time(labels, float(frac))
            train.append(mk(a, f"[:{frac}]"))
            val.append(mk(b, f"[{frac}:]"))
        elif cid in val_ids:
            val.append(mk(labels))
        elif cid in train_ids or (not train_ids and manifest.get(cid, {}).get("split") == "train"):
            train.append(mk(labels))
        elif not val_ids and manifest.get(cid, {}).get("split") == "eval":
            val.append(mk(labels))
        else:
            skipped.append(f"{cid} (not selected)")
    if split_id and split_id not in docs:
        raise SystemExit(f"--split-clip {split_id}: no {'' if include_draft else 'reviewed '}labels with shuttle entries")
    return train, val, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="starting checkpoint (default: the installed one)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    ap.add_argument("--train", nargs="*", default=[], help="clip ids to train on (default: split=train clips)")
    ap.add_argument("--val", nargs="*", default=[], help="clip ids to validate on (default: labeled split=eval)")
    ap.add_argument("--split-clip", help="ID[:FRAC] train on the first FRAC of this clip, validate on the rest")
    ap.add_argument("--include-draft", action="store_true", help="also use labels not marked reviewed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=300, help="steps per epoch")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", help="cuda | mps | cpu")
    ap.add_argument("--train-bn", action="store_true", help="also update BatchNorm statistics")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rallyiq.perception.shuttle.detector import find_weights
    from rallyiq.perception.shuttle.training import TrainConfig, train

    weights = find_weights(a.weights)
    if weights is None:
        raise SystemExit("no starting weights found (see rallyiq/perception/shuttle/README.md)")
    manifest = schema.load_manifest(MANIFEST)
    tr, va, skipped = select(a.labels, a.clips.expanduser(), manifest, a.include_draft, a.train, a.val, a.split_clip)
    for s in skipped:
        print(f"skip {s}")
    if not tr:
        raise SystemExit("no training clips with labels: label a split=train clip, or use --train / --split-clip")
    print("train: " + ", ".join(f"{c.clip_id} ({len(c.labels)} frames)" for c in tr))
    print("val:   " + (", ".join(f"{c.clip_id} ({len(c.labels)} frames)" for c in va) or "none"))
    cfg = TrainConfig(epochs=a.epochs, steps_per_epoch=a.steps, batch_size=a.batch_size, lr=a.lr,
                      device=a.device, freeze_bn=not a.train_bn)
    report = train(weights, tr, va, a.out.expanduser(), cfg)
    a.out.expanduser().with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
