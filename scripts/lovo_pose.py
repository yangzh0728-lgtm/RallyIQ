"""Pose-swing model retrained with the amateur labels, evaluated leave-one-own-video-out end to end.

For held-out video V (or "none" = final model):
  1. pose hit model P_V (pose_hits.py features, HGB) on ShuttleSet train pose samples + the amateur pose samples of the
     other 12 videos (positives: a labelled in-play hit by that side within 3 frames; unrefined qa6 times and
     unsure labels are excluded from both classes);
  2. hit candidates of ShuttleSet train + the other amateur videos re-collected with swing features from P_V;
  3. hit scorer (v2+all, aw3, 5 seeds) trained on them; V scored with the full events pipeline (own settings).

    ~/RallyIQ-app/.venv/bin/python scripts/lovo_pose.py samples            # pose samples cache
    ~/RallyIQ-app/.venv/bin/python scripts/lovo_pose.py fold own_20260529_0003
    ~/RallyIQ-app/.venv/bin/python scripts/lovo_pose.py report
(scikit-learn is needed for step 1; the ~/RallyIQ-app venv has it.)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import amateur_hits as AH  # noqa: E402
import learn_hits as LH  # noqa: E402
import learn_pose_hits as LP  # noqa: E402
import train_hits_v3 as T  # noqa: E402
from rallyiq.events import learned as L  # noqa: E402
from rallyiq.events import pose_hits as PH  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402

W = Path("~/Desktop/RallyIQ/train_v3").expanduser()
OUT = W / "ckpt" / "lovo_pose"
LABELS = W / "labels" / "amateur_v2.json"
SS_TRAIN = W / "inputs" / "public" / "shuttleset_train_v3"
TH = {"min_candidate_conf": 0.1, "keep_threshold": 0.3, "ok_threshold": 0.7}


def am_pose_samples(c: AH.AmClip) -> tuple[np.ndarray, np.ndarray]:
    ser = PH.side_series(c.poses)
    fm = 1000.0 / c.fps
    pos = [(x["t"] * 1000.0, x["side"]) for x in c.truth["labels"] if x["kind"] == "pos" and not x.get("unrefined") and x.get("side")]
    skip = [x["t"] * 1000.0 for x in c.truth["labels"] if x["kind"] == "ign" or x.get("unrefined")]
    X, y = [], []
    for side, se in ser.items():
        lt = np.array([t for t, s in pos if s == side])
        for b in se.bodies:
            if not c.in_window(b.t_ms / 1000.0, pad=0.5):
                continue
            near = len(lt) and np.abs(lt - b.t_ms).min() <= 3 * fm + 1
            if not near and any(abs(t - b.t_ms) <= 300 for t in skip):
                continue
            X.append(PH.window_features(se, b.t_ms, fm))
            y.append(1.0 if near else 0.0)
    return np.asarray(X, float).reshape(-1, len(PH.FEATURES)), np.asarray(y)


def cmd_samples(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    am = {}
    for c in AH.load_clips(W / "inputs", LABELS):
        am[c.cid] = am_pose_samples(c)
        print(c.cid, am[c.cid][0].shape, int(am[c.cid][1].sum()), flush=True)
    ss = []
    for c in LH.load_clips(SS_TRAIN, "index.json"):
        ss.append(LP.samples(c))
    X = np.vstack([s[0] for s in ss]); y = np.concatenate([s[1] for s in ss])
    print("ShuttleSet", X.shape, int(y.sum()))
    pickle.dump({"am": am, "ss": (X, y)}, open(OUT / "pose_samples.pkl", "wb"))
    return 0


def fit_pose(held: str, am_weight: float) -> Path:
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = pickle.load(open(OUT / "pose_samples.pkl", "rb"))
    Xs, ys = d["ss"]
    parts = [(Xs, ys, np.ones(len(ys)))] + [(X, y, np.full(len(y), am_weight)) for k, (X, y) in d["am"].items() if k != held]
    X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); w = np.concatenate([p[2] for p in parts])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=20,
                                       random_state=0).fit(X, y, sample_weight=w)
    spec = PH.export_hgb(m, json.loads(PH.BUNDLED.read_text())["thresholds"],
                         {"trained_on": {"shuttleset_samples": int(len(ys)), "amateur_held_out": held, "am_weight": am_weight}})
    p = OUT / f"pose_{held}.json"
    p.write_text(json.dumps(spec, separators=(",", ":")))
    return p


def cmd_fold(a) -> int:
    held = a.video
    pose_path = fit_pose(held, a.pose_am_weight)
    L.SWING_MODEL = str(pose_path)
    PH._load.cache_clear()
    cfg = replace(EventsConfig(), **T.V2_EVENT)
    clips = AH.load_clips(W / "inputs", LABELS)
    for c in clips:
        if c.cid != held:
            c.collect_am(cfg)
    ss = LH.load_clips(SS_TRAIN, "index.json")
    for c in ss:
        c.collect(cfg, 3)
        c.w = np.ones(len(c.y))
    ns = argparse.Namespace(am_weight=3.0, hidden=16, l2=1e-2, swing_dropout=0.0)
    sc = T.train_on(ss, [c for c in clips if c.cid != held], ns, T.FEATURE_SETS["v2+all"], list(range(5)))
    spec = {**sc.spec, "thresholds": {**T.V2_EVENT, **TH}, "swing_model": pose_path.name}
    mp = OUT / f"scorer_{held}.json"
    mp.write_text(json.dumps(spec))
    if held != "none":
        c = next(c for c in clips if c.cid == held)
        preds = c.predict_full(replace(EventsConfig(), **{**AH.AMATEUR_BASE, "hit_scorer": str(mp)}))
        (OUT / f"preds_{held}.json").write_text(json.dumps(preds))
    print("done", held, flush=True)
    return 0


def cmd_report(a) -> int:
    clips = AH.load_clips(W / "inputs", LABELS)
    preds = {c.cid: json.loads((OUT / f"preds_{c.cid}.json").read_text()) for c in clips}
    md, js = AH.report(clips, preds)
    print(md)
    (W / "logs" / "lovo_pose.json").write_text(json.dumps(js, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("samples")
    f = sub.add_parser("fold")
    f.add_argument("video")
    f.add_argument("--pose-am-weight", type=float, default=3.0)
    sub.add_parser("report")
    a = ap.parse_args()
    return {"samples": cmd_samples, "fold": cmd_fold, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
