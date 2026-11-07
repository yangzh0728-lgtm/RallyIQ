"""Train the pose-sequence hit model (rallyiq/events/pose_hits.py) and measure what it recovers.

    pip install -e .[strokes-ml]          # scikit-learn, only for training

    # train on the ShuttleSet training clips (cached artifacts from scripts/build_train_set.py)
    python scripts/learn_pose_hits.py train \\
        --data ~/rallyiq-data/public/shuttleset_train ~/rallyiq-data/public/shuttleset_train_v3
    #   -> rallyiq/events/models/pose_hits.json (+ .md report)

    # hit F1 without / with pose recovery on the held-out matches
    python scripts/learn_pose_hits.py eval --data ~/rallyiq-data/public/shuttleset_test_v3

Only the events stage is re-run, from cached shuttle / pose artifacts: minutes, no video work.

Examples: every pose sample of each player (near / far) inside a labelled rally (from 1.5 s before
its first hit); positive when a labelled hit by that side lies within 2 frames. Model:
``HistGradientBoostingClassifier`` exported to JSON trees (numpy at run time). Thresholds of
``recover`` are tuned end to end on out-of-fold predictions (5 folds by match), maximising hit F1
(±3 frames, same side) on top of the shuttle-based detector with its current scorer.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import learn_hits as LH  # noqa: E402
from rallyiq.events import pose_hits as PH  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402


def samples(c: LH.Clip) -> tuple[np.ndarray, np.ndarray]:
    ser = PH.side_series(c.poses)
    fm = 1000.0 / c.fps
    lab = [(h["t_ms"] - c.offset, h.get("side")) for h in c.label["hits"]]
    spans = [(r["start_ms"] - c.offset - 1500, r["end_ms"] - c.offset + 500) for r in c.label.get("rallies", [])]
    if not spans and lab:
        spans = [(min(t for t, _ in lab) - 1500, max(t for t, _ in lab) + 1500)]
    X, y = [], []
    for side, se in ser.items():
        lt = np.array([t for t, s in lab if s == side])
        for b in se.bodies:
            if any(a <= b.t_ms <= e for a, e in spans):
                X.append(PH.window_features(se, b.t_ms, fm))
                y.append(1.0 if len(lt) and np.abs(lt - b.t_ms).min() <= 2 * fm + 1 else 0.0)
    return np.asarray(X, float).reshape(-1, len(PH.FEATURES)), np.asarray(y)


def fit(clips: list[LH.Clip], max_iter: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.vstack([c.Xp for c in clips])
    y = np.concatenate([c.yp for c in clips])
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(X, y)


def load(dirs: list[Path], cfg: EventsConfig) -> list[LH.Clip]:
    out = []
    for d in dirs:
        for c in LH.load_clips(d.expanduser(), "index.json"):
            c.cid = f"{d.name}/{c.cid}"
            c.Xp, c.yp = samples(c)
            c.base = c.run(replace(cfg, pose_hits="off"))
            out.append(c)
    return out


def hit_score(clips: list[LH.Clip], models: dict[str, PH.PoseHitModel] | None, rc: PH.RecoverConfig | None):
    by = {}
    for c in clips:
        rs = copy.deepcopy(c.base)
        if models is not None:
            PH.recover(rs, c.poses, models[c.cid], rc, 1000.0 / c.fps)
        ser = PH.side_series(c.poses)
        by[c.cid] = [{"t_ms": c.t_label(h.frame), "side": PH._side_of(h, ser), "ok": True} for r in rs for h in r.hits]
    return LH.score(clips, by)


def _row(name: str, s: dict) -> str:
    return f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['f1_ignore_side']:.3f} |"


HEAD = "| | precision | recall | hit F1 | F1 (any side) |\n|---|---|---|---|---|"


def cmd_train(a: argparse.Namespace) -> int:
    cfg = EventsConfig(hit_scorer=a.scorer)
    clips = load(a.data, cfg)
    groups = sorted({c.group for c in clips})
    k = min(a.folds, len(groups))
    folds = [set(groups[i::k]) for i in range(k)]
    oof: dict[str, PH.PoseHitModel] = {}
    for f in folds:
        m = PH.PoseHitModel(PH.export_hgb(fit([c for c in clips if c.group not in f], a.max_iter)))
        for c in clips:
            if c.group in f:
                oof[c.cid] = m
    base = hit_score(clips, None, None)
    best = None
    for st, gt in itertools.product([0.5, 0.6, 0.7, 2.0], [0.2, 0.3, 0.4, 0.5, 0.6, 2.0]):
        rc = PH.RecoverConfig(serve_threshold=st, gap_threshold=gt)
        s = hit_score(clips, oof, rc)
        print(f"serve {st} gap {gt}: F1 {s['f1']:.4f}")
        if best is None or s["f1"] > best[0]["f1"]:
            best = (s, {"serve_threshold": st, "gap_threshold": gt})
    final = fit(clips, a.max_iter)
    n_pos = int(sum(c.yp.sum() for c in clips))
    spec = PH.export_hgb(final, best[1], {"trained_on": {"clips": len(clips), "matches": len(groups),
                                                         "samples": int(sum(len(c.yp) for c in clips)),
                                                         "positives": n_pos}})
    out = a.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    md = (f"# Pose hit model\n\n- clips {len(clips)} from {len(groups)} matches; pose samples "
          f"{spec['trained_on']['samples']} ({n_pos} positive)\n- chosen thresholds: {best[1]}\n\n"
          f"Out-of-fold hit accuracy on the training matches ({k} folds by match), shuttle scorer `{a.scorer}`:\n\n"
          f"{HEAD}\n{_row('shuttle-based detector', base)}\n{_row('+ pose recovery', best[0])}\n")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\nsaved {out}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    cfg = EventsConfig(hit_scorer=a.scorer)
    clips = load(a.data, cfg)
    m = PH.get_model(str(a.model))
    if m is None:
        print(f"no model at {a.model}")
        return 1
    models = {c.cid: m for c in clips}
    s0 = hit_score(clips, None, None)
    s1 = hit_score(clips, models, None)
    print(f"{HEAD}\n{_row('shuttle-based detector', s0)}\n{_row('+ pose recovery', s1)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, nargs="+", required=True, help="build_train_set.py output folder(s)")
        p.add_argument("--scorer", default="auto", help="shuttle hit scorer (EventsConfig.hit_scorer)")
        if name == "train":
            p.add_argument("--out", type=Path, default=PH.BUNDLED)
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--max-iter", type=int, default=200)
        else:
            p.add_argument("--model", type=Path, default=PH.BUNDLED)
    a = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
