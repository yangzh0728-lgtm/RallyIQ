"""Train the scikit-learn stroke-type classifier (rallyiq/strokes/learned.py) and compare it with the rules.

    pip install -e .[strokes-ml]

    # quick check on synthetic rallies (says nothing about real video)
    python scripts/learn_strokes.py --source synthetic --out ~/rallyiq-data/models/strokes_synth.joblib

    # .json output = numpy-only model; rallyiq/strokes/models/strokes_v2.json is the bundled one
    # ShuttleSet clips built and run by scripts/build_train_set.py (cached artifacts, no video work)
    python scripts/learn_strokes.py --source shuttleset \\
        --data ~/rallyiq-data/public/shuttleset_train ~/rallyiq-data/public/shuttleset_train_v3 \\
        --out ~/rallyiq-data/models/strokes.joblib

    # score a saved model on the held-out matches (build_train_set.py --split test)
    python scripts/learn_strokes.py --source shuttleset --data ~/rallyiq-data/public/shuttleset_test \\
        --eval ~/rallyiq-data/models/strokes.joblib

Every predicted shot that lies within --tol frames of a labelled ShuttleSet stroke (nearest first,
one to one) becomes one example: the ``features.py`` evidence -> the labelled type. Labels come from
``shot_type`` in the label files (written by build_train_set.py since this script exists) or, for
older label files, straight from the ShuttleSet set CSVs (--shuttleset).

Model: ``HistGradientBoostingClassifier`` (missing evidence stays NaN). Scored with ``GroupKFold``
by match, so the same match never sits in training and validation. The rules (``rules_v1``) are
scored on the same shots for comparison. Output: the .joblib model and a .md report next to it.

Use it in the pipeline with ``{"strokes": {"classifier": "sklearn", "sklearn": {"model": "<path>"}}}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rallyiq.contracts import EventsResult, PlayerTracks, ShuttleTrack  # noqa: E402
from rallyiq.strokes import classify, shot_features  # noqa: E402
from rallyiq.strokes import learned as L  # noqa: E402
from rallyiq.strokes.taxonomy import from_shuttleset  # noqa: E402


class Example:
    """One clip's matched shots: features, model inputs per feature set, labelled types, rules' prediction."""

    def __init__(self, group: str, feats, X: dict[str, np.ndarray], y: list[str], rules: list[str | None]):
        self.group, self.feats, self.X, self.y, self.rules = group, feats, X, y, rules


def _from_clip(group: str, events: EventsResult, shuttle, players, fps: float, truth: dict[str, str]) -> Example:
    all_feats = shot_features(events, shuttle, players, fps)
    keep = [i for i, f in enumerate(all_feats) if f.shot_id in truth]
    feats = [all_feats[i] for i in keep]
    # context features look at each shot's neighbours, so compute them on the whole clip first
    X = {fs: L.full_matrix(all_feats, events, shuttle, players, fs)[keep] for fs in L.FEATURE_SETS}
    rules = {s.shot_id: s.stroke.shot_type for s in classify(events, shuttle, players, fps).shots}
    return Example(group, feats, X, [truth[f.shot_id] for f in feats], [rules.get(f.shot_id) for f in feats])


# ------------------------------------------------------------------ synthetic
def synthetic_examples(n_seeds: int) -> list[Example]:
    from rallyiq.strokes.synthetic import make_rallies

    out = []
    variants = [{}, {"dropout": 0.15}, {"frame_jitter": 2}, {"px_noise": 3.0, "dropout": 0.1, "frame_jitter": 1}]
    for seed in range(n_seeds):
        v = make_rallies(n_rallies=8, seed=seed, **variants[seed % len(variants)], video_id=f"synth{seed}")
        out.append(_from_clip(f"seed{seed}", v.events, v.shuttle, v.players, v.fps, v.truth))
    return out


# ------------------------------------------------------------------ ShuttleSet
def _label_types(label: dict[str, Any], entry: dict[str, Any], cid: str, set_dir: Path | None) -> list[tuple[float, str]]:
    """(t_ms, type) of every labelled stroke, in label time.

    With ``set_dir`` the types are re-read from the set CSVs (so a taxonomy fix applies without
    rebuilding the label files); otherwise the label files' ``shot_type`` is used."""
    typed = [(h["t_ms"], h["shot_type"]) for h in label["hits"] if h.get("shot_type")]
    if set_dir is None:
        return typed
    from evaluation.public import shuttleset as SS

    set_csv = set_dir / entry["video"] / f"{cid.split('_', 1)[1]}.csv"
    if not set_csv.exists():
        return []
    r0, r1 = label["range_ms"]
    out = []
    for s in SS.load_strokes(set_csv):
        t = int(round(s["frame"] * 1000 / float(entry["label_fps"])))
        code = from_shuttleset(s["type"])[0] if s.get("type") else None
        if code and r0 <= t <= r1:
            out.append((t, code))
    return out


def shuttleset_examples(data: Path, set_dir: Path | None, tol_frames: float) -> list[Example]:
    from evaluation import schema

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    out = []
    for cid, entry in sorted(index.items()):
        if entry.get("state") != "succeeded":
            continue
        art = Path(entry["artifacts"])
        lp = data / "labels" / f"{cid}.json"
        if not lp.exists() or not (art / "events.json").exists():
            print(f"[skip] {cid}: no label or no events artifact")
            continue
        label = schema.load(lp)
        types = _label_types(label, entry, cid, set_dir)
        if not types:
            print(f"[skip] {cid}: no stroke types (pass --shuttleset or rebuild the labels)")
            continue
        fps = float(json.loads((art / "probe.json").read_text(encoding="utf-8"))["fps"])
        events = EventsResult.model_validate_json((art / "events.json").read_text(encoding="utf-8"))
        shuttle = ShuttleTrack.model_validate_json((art / "shuttle.json").read_text(encoding="utf-8"))
        pp = art / "players.json"
        players = PlayerTracks.model_validate_json(pp.read_text(encoding="utf-8")) if pp.exists() else None
        offset = float(label["range_ms"][0])
        tol = tol_frames * 1000.0 / fps + 1e-6
        pairs = sorted((abs(offset + s.frame * 1000.0 / fps - t), s.shot_id, j)
                       for s in events.shots for j, (t, _) in enumerate(types)
                       if abs(offset + s.frame * 1000.0 / fps - t) <= tol)
        truth: dict[str, str] = {}
        used = set()
        for _, sid, j in pairs:
            if sid in truth or j in used:
                continue
            truth[sid] = types[j][1]
            used.add(j)
        print(f"{cid}: {len(truth)}/{len(types)} labelled strokes matched to predicted shots")
        out.append(_from_clip(str(entry.get("match_id", cid)), events, shuttle, players, fps, truth))
    return out


# ------------------------------------------------------------------ report
def _rules_line(ex: list[Example]) -> str:
    y = [t for e in ex for t in e.y]
    r = [t for e in ex for t in e.rules]
    acc = sum(a == b for a, b in zip(y, r)) / len(y) if y else 0.0
    return f"rules_v1 on the same shots: accuracy {acc:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("synthetic", "shuttleset"), required=True)
    ap.add_argument("--data", type=Path, nargs="+", help="shuttleset: build_train_set.py output folder(s)")
    ap.add_argument("--features", choices=tuple(L.FEATURE_SETS), default="v2",
                    help="base = the rules' evidence only; v2 = plus rally context, direction, pose, early flight")
    ap.add_argument("--shuttleset", type=Path, help="ShuttleSet/set dir: read stroke types from the set CSVs")
    ap.add_argument("--seeds", type=int, default=24, help="synthetic: number of videos")
    ap.add_argument("--tol", type=float, default=3, help="frames between a predicted shot and its labelled stroke")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", type=Path, help="where to write the trained model (.joblib)")
    ap.add_argument("--eval", type=Path, help="score this saved model instead of training")
    a = ap.parse_args(argv)

    if a.source == "synthetic":
        ex = synthetic_examples(a.seeds)
    else:
        if a.data is None:
            ap.error("--data is required for --source shuttleset")
        set_dir = None
        if a.shuttleset or not all(d.expanduser().joinpath("labels").exists() for d in a.data):
            import public_eval

            set_dir = public_eval.shuttleset_dir(a.shuttleset)
        ex = [e for d in a.data for e in shuttleset_examples(d.expanduser(), set_dir, a.tol)]
    ex = [e for e in ex if e.feats]
    if not ex:
        print("no labelled shots found")
        return 1
    y = [t for e in ex for t in e.y]
    groups = [e.group for e in ex for _ in e.y]
    print(f"\n{len(y)} shots, {len(set(groups))} groups, types: {dict(Counter(y).most_common())}")
    print(_rules_line(ex))

    if a.eval:
        blob = L.load(a.eval)
        model = blob["model"]
        X = np.vstack([e.X[blob["feature_set"]] for e in ex])
        pred = model.predict(X)
        acc = float(np.mean(pred == np.asarray(y, dtype=object)))
        print(f"{a.eval.name}: accuracy {acc:.3f}")
        return 0

    X = np.vstack([e.X[a.features] for e in ex])
    model, report = L.train(X, y, groups, L.TrainConfig(folds=a.folds))
    md = (f"# Stroke-type model ({L.SKLEARN_NAME}, features: {a.features}, source: {a.source})\n\n"
          f"{_rules_line(ex)}\n\n{report.markdown()}")
    print("\n" + md)
    if a.out:
        out = a.out.expanduser()
        meta = {"source": a.source, "n": len(y), "cv": {"accuracy": report.accuracy, "macro_f1": report.macro_f1}}
        if out.suffix == ".json":  # numpy-only JSON trees (what the repository ships)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(L.export_json(model, a.features, meta), separators=(",", ":")), encoding="utf-8")
        else:
            L.save(model, out, meta, feature_set=a.features)
        out.with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"saved {out} (+ {out.with_suffix('.md').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
