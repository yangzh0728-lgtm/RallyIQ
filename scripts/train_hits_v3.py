"""Hit scorer v3: train on ShuttleSet + sam's amateur videos, validate leave-one-amateur-video-out and grouped by
ShuttleSet match, compare feature sets / sizes / domain weights, write the chosen model.

    python scripts/train_hits_v3.py collect --out ~/Desktop/RallyIQ/train_v3/ckpt/cands   # candidates + features (cached)
    python scripts/train_hits_v3.py cv --cache ... --features v3 --hidden 16 --l2 1e-2 --am-weight 3
    python scripts/train_hits_v3.py fit --cache ... --features v3 ... --out model.json

Candidates come from the rules (learn_hits.Clip.collect). ShuttleSet positives: nearest candidate within 3 frames
of a labelled hit. Amateur positives: nearest candidate within 0.2 s of a positive label; candidates outside the
labelled windows or next to an ignored label get weight 0 (see amateur_hits.AmClip.collect_am).

Scores: ShuttleSet hit F1 (learn_hits.score: +-3 frames, same side) and amateur hit F1 (amateur_hits.score_clip:
+-0.25 s), both from the fast end-to-end simulation of the hit filter on out-of-fold probabilities.
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import amateur_hits as AH  # noqa: E402
import learn_hits as LH  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402
from rallyiq.events.learned import ALL_FEATURES, FEATURES, HitScorer  # noqa: E402

# Candidate/rally parameters the bundled v2 scorer was trained with (its thresholds block); kept for v3.
V2_EVENT = {"window_ms": 100.0, "min_shuttle_conf": 0.3, "max_hitter_dist_rel": 1.2, "static_ms": 400.0,
            "min_rally_ms": 700.0, "hit_frame_offset": 1}
DIR_FEATURES = ("dir_in_y", "dir_out_y", "dir_in_x", "dir_out_x")
SWING = ("p_swing_hitter", "p_swing_other")
GEOM = ("toward_in", "away_out", "log_speed_ratio", "wrist_dist", "track_density")
FEATURE_SETS = {
    "v2": FEATURES,
    "v2+": FEATURES + GEOM,
    "v2+sw": FEATURES + SWING,
    "v2+all": FEATURES + GEOM + SWING,
    "v3": tuple(f for f in FEATURES if f not in DIR_FEATURES) + GEOM,
    "v3+sw": tuple(f for f in FEATURES if f not in DIR_FEATURES) + GEOM + SWING,
    "v3-dens+sw": tuple(f for f in FEATURES if f not in DIR_FEATURES) + tuple(g for g in GEOM if g != "track_density") + SWING,
}


# ------------------------------------------------------------------ data
# input root of the amateur runs/truth (train_v4 sets TRAIN_INPUTS to its own inputs/ with the re-timed qa6 truth)
INPUTS = Path(os.environ.get("TRAIN_INPUTS", "~/Desktop/RallyIQ/train_v3/inputs")).expanduser()

def collect(a: argparse.Namespace) -> int:
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for msc in a.min_shuttle_conf:
        cfg = replace(EventsConfig(), **{**V2_EVENT, "min_shuttle_conf": msc, **json.loads(a.overrides or "{}")})
        p = out / f"am_msc{msc}.pkl"
        if not p.exists():
            t0 = time.time()
            clips = AH.load_clips(a.root.expanduser(), a.labels.expanduser())
            rows = []
            for c in clips:
                c.collect_am(cfg)
                n_gt = sum(1 for x in c.truth["labels"] if x["kind"] == "pos")
                # recall ceiling inside windows: positives with a candidate within 0.25 s
                cand_t = c.t[c.w > 0] if len(c.t) else np.zeros(0)
                reach = sum(1 for x in c.truth["labels"] if x["kind"] == "pos" and len(cand_t) and np.min(np.abs(cand_t - x["t"])) <= 0.25)
                print(f"  {c.cid}: cands {len(c.y)} (in windows {int((c.w > 0).sum())}, pos {int(c.y.sum())}), ceiling {reach}/{n_gt}", flush=True)
                rows.append(c)
            for c in rows:  # drop heavy members before pickling
                c.shuttle = c.poses = c.players = None
            pickle.dump(rows, open(p, "wb"))
            print(f"amateur msc={msc}: {len(rows)} clips in {time.time() - t0:.0f}s -> {p}", flush=True)
        p = out / f"ss_msc{msc}.pkl"
        if not p.exists() and not a.amateur_only:
            t0 = time.time()
            for split, d in (("train", a.ss_train), ("test", a.ss_test)):
                clips = LH.load_clips(d.expanduser(), "index.json") if split == "train" else LH.load_clips(d.expanduser(), "results.json")
                for c in clips:
                    c.collect(cfg, 3)
                    c.shuttle = c.poses = c.players = None
                pickle.dump(clips, open(out / f"ss_{split}_msc{msc}.pkl", "wb"))
                n_gt = sum(c.recall_ceiling[1] for c in clips)
                n_r = sum(c.recall_ceiling[0] for c in clips)
                print(f"ShuttleSet {split} msc={msc}: {len(clips)} clips, cands {sum(len(c.y) for c in clips)}, "
                      f"ceiling {n_r}/{n_gt} in {time.time() - t0:.0f}s", flush=True)
            p.write_text("done")
    return 0


def relabel_unrefined(c, match_s: float = 0.35, ambiguous_s: float = 0.4) -> None:
    """qa6 hits with unrefined (8 fps) times: an unmatched one takes its nearest candidate within ``match_s`` as
    the positive; other candidates within ``ambiguous_s`` of it get weight 0 instead of being negatives."""
    for g in c.truth["labels"]:
        if g["kind"] != "pos" or not g.get("unrefined") or not len(c.t):
            continue
        d = np.abs(c.t - g["t"])
        near = np.where(d <= ambiguous_s)[0]
        if not len(near):
            continue
        if not any(c.y[i] == 1 and d[i] <= 0.2 for i in near):
            i = int(near[np.argmin(d[near])])
            if d[i] <= match_s and c.w[i] > 0:
                c.y[i] = 1
        for i in near:
            if c.y[i] == 0:
                c.w[i] = 0.0


def load(cache: Path, msc_am: float, msc_ss: float, unrefined: bool = False):
    am = pickle.load(open(cache / f"am_msc{msc_am}.pkl", "rb"))
    if unrefined:
        truth = AH.load_truth(INPUTS, None)
        for c in am:
            if c.cid in truth:
                c.truth = truth[c.cid]
                relabel_unrefined(c)
    ss = pickle.load(open(cache / f"ss_train_msc{msc_ss}.pkl", "rb"))
    te = pickle.load(open(cache / f"ss_test_msc{msc_ss}.pkl", "rb"))
    for c in ss + te:
        c.w = np.ones(len(c.y))
    return am, ss, te


# ------------------------------------------------------------------ model
def fit(X, y, w, feats, hidden, l2, seed=0, epochs=1500):
    return LH.fit_mlp(X, y, hidden, l2, seed=seed, epochs=epochs, features=feats, w=w)


def fit_gbt(X, y, w, feats, seed=0, max_iter=200, lr=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2=1.0):
    """HistGradientBoostingClassifier (needs scikit-learn: run with ~/RallyIQ-app/.venv/bin/python) -> JSON spec."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from rallyiq.gbt import export_hgb
    Xf = X[:, [ALL_FEATURES.index(n) for n in feats]]
    m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=max_leaf_nodes,
                                       min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=seed)
    m.fit(Xf, y, sample_weight=w)
    g = export_hgb(m)
    return {"kind": "gbt", "features": list(feats), "baseline": g["baseline"][0], "trees": [it[0] for it in g["trees"]]}


MODEL_KIND = "mlp"
GBT_ARGS: dict = {}


def fit_ens(X, y, w, feats, hidden, l2, seeds):
    if MODEL_KIND == "gbt":
        specs = [fit_gbt(X, y, w, feats, seed=s, **GBT_ARGS) for s in seeds]
        if len(specs) == 1:
            return specs[0]
        return {"features": list(feats), "members": [{k: v for k, v in sp.items() if k != "features"} for sp in specs]}
    specs = [fit(X, y, w, feats, hidden, l2, seed=s) for s in seeds]
    if len(specs) == 1:
        return specs[0]
    return {"features": list(feats), "members": [{k: s[k] for k in ("mean", "std", "layers")} for s in specs],
            "hidden": hidden, "l2": l2, "seeds": list(seeds)}


def stack(clips, w_scale=1.0):
    X = np.vstack([c.X for c in clips])
    y = np.concatenate([c.y for c in clips])
    w = np.concatenate([c.w for c in clips]) * w_scale
    m = w > 0
    return X[m], y[m], w[m]


def train_on(ss, am, a, feats, seeds):
    Xs, ys, ws = stack(ss, 1.0) if ss else (np.zeros((0, len(ALL_FEATURES))), np.zeros(0), np.zeros(0))
    Xa, ya, wa = stack(am, a.am_weight) if am else (np.zeros((0, len(ALL_FEATURES))), np.zeros(0), np.zeros(0))
    X, y, w = np.vstack([Xs, Xa]), np.concatenate([ys, ya]), np.concatenate([ws, wa])
    q = getattr(a, "swing_dropout", 0.0)
    if q > 0:  # copies with the pose-swing evidence "unknown" (0.5): the hitter may be untracked or hidden
        X2 = X.copy()
        X2[:, [ALL_FEATURES.index(f) for f in SWING]] = 0.5
        X, y, w = np.vstack([X, X2]), np.concatenate([y, y]), np.concatenate([w * (1 - q), w * q])
    return HitScorer(fit_ens(X, y, w, feats, a.hidden, a.l2, seeds))


def oof(am, ss, a, feats, seeds, ss_folds=5):
    """Out-of-fold probabilities: each amateur video held out (trained on all ShuttleSet + other amateur videos);
    ShuttleSet held out by match group in ``ss_folds`` folds (trained on the other ShuttleSet folds + all amateur)."""
    p_am, p_ss = {}, {}
    use_am = a.am_weight > 0
    for c in am:
        sc = train_on(ss if a.ss_weight > 0 else [], [o for o in am if o.cid != c.cid] if use_am else [], a, feats, seeds)
        p_am[c.cid] = sc.predict(c.X) if len(c.X) else np.zeros(0)
    if ss_folds:
        for f in LH.folds_by_group(ss, ss_folds):
            held = {c.cid for c in f}
            sc = train_on([c for c in ss if c.cid not in held], am if use_am else [], a, feats, seeds)
            for c in f:
                p_ss[c.cid] = sc.predict(c.X) if len(c.X) else np.zeros(0)
    return p_am, p_ss


def am_score(am, probs, cfg, which=None):
    rows = {c.cid: AH.score_clip(c, AH.simulate_am(c, probs[c.cid] if probs else None, cfg)) for c in am
            if which is None or c.truth["set"] == which}
    return AH.summarize(rows) if rows else None


def ss_score(ss, probs, cfg):
    return LH.score(ss, {c.cid: LH.simulate(c, probs[c.cid] if probs else None, cfg) for c in ss})


def tune(am, ss, p_am, p_ss, base, objective="am"):
    """Grid over (min_candidate_conf, keep_threshold); returns best per objective: am = qa6+am7 F1, ss = ShuttleSet F1."""
    res = []
    for mc in (0.05, 0.1, 0.15, 0.2, 0.3):
        for keep in np.round(np.arange(0.15, 0.76, 0.05), 2):
            cfg = replace(base, min_candidate_conf=mc, keep_threshold=float(keep), ok_threshold=max(0.7, float(keep)))
            r = {"mc": mc, "keep": float(keep)}
            if p_am is not None:
                s = am_score(am, p_am, cfg)
                r.update(am_f1=s["f1"], qa6_f1=am_score(am, p_am, cfg, "qa6")["f1"], am7_f1=am_score(am, p_am, cfg, "am7")["f1"])
            if p_ss is not None and p_ss:
                r["ss_f1"] = ss_score(ss, p_ss, cfg)["f1"] or 0.0
            res.append(r)
    return res


def cmd_cv(a: argparse.Namespace) -> int:
    cache = a.cache.expanduser()
    am, ss, _ = load(cache, a.msc_am, a.msc_ss, a.unrefined)
    feats = FEATURE_SETS[a.features]
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    t0 = time.time()
    p_am, p_ss = oof(am, ss, a, feats, seeds, a.ss_folds)
    base = replace(EventsConfig(), **{**V2_EVENT, **json.loads(a.overrides or "{}")})
    grid = tune(am, ss, p_am, p_ss, base)
    best_am = max(grid, key=lambda r: r["am_f1"])
    best_ss = max(grid, key=lambda r: r.get("ss_f1", 0)) if p_ss else None
    # joint: maximise amateur F1 subject to ShuttleSet F1 within 1 point of its own best
    joint = best_am
    if p_ss:
        ok = [r for r in grid if r["ss_f1"] >= best_ss["ss_f1"] - 0.01]
        joint = max(ok, key=lambda r: r["am_f1"])
    out = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "func"},
           "best_am": best_am, "best_ss": best_ss, "joint": joint, "secs": round(time.time() - t0)}
    print(json.dumps(out), flush=True)
    if a.log:
        with open(a.log.expanduser(), "a") as fh:
            fh.write(json.dumps(out) + "\n")
    if a.save_oof:
        pickle.dump({"p_am": p_am, "p_ss": p_ss}, open(a.save_oof.expanduser(), "wb"))
    return 0


def cmd_fit(a: argparse.Namespace) -> int:
    cache = a.cache.expanduser()
    am, ss, te = load(cache, a.msc_am, a.msc_ss, a.unrefined)
    feats = FEATURE_SETS[a.features]
    sc = train_on(ss if a.ss_weight > 0 else [], am if a.am_weight > 0 else [], a, feats, list(range(a.seed0, a.seed0 + a.seeds)))
    spec = sc.spec
    th = json.loads(a.thresholds) if a.thresholds else {}
    spec["thresholds"] = {**V2_EVENT, **json.loads(a.overrides or "{}"), **th}
    spec["trained_on"] = {"dataset": "ShuttleSet (CoachAI-Projects, MIT) train v3 + sam's own videos (13, labelled windows)",
                          "features": a.features, "hidden": a.hidden, "l2": a.l2, "seeds": a.seeds,
                          "am_weight": a.am_weight, "msc_am": a.msc_am, "msc_ss": a.msc_ss}
    out = a.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, separators=(",", ":")))
    print(f"model -> {out}")
    return 0


def cmd_lovo(a: argparse.Namespace) -> int:
    """Full events pipeline (amateur settings: swing veto, min_no_hit_rally_ms, ...) on each amateur video with a
    model trained without that video (all ShuttleSet + the other amateur videos)."""
    cache = a.cache.expanduser()
    am, ss, _ = load(cache, a.msc_am, a.msc_ss, a.unrefined)
    feats = FEATURE_SETS[a.features]
    th = {**V2_EVENT, **json.loads(a.overrides or "{}"), **(json.loads(a.thresholds) if a.thresholds else {})}
    raw = {c.cid: c for c in AH.load_clips(INPUTS, Path(a.labels).expanduser())}
    tmp = Path(a.tmp).expanduser()
    tmp.mkdir(parents=True, exist_ok=True)
    preds = {}
    for c in am:
        sc = train_on(ss if a.ss_weight > 0 else [], [o for o in am if o.cid != c.cid] if a.am_weight > 0 else [], a, feats,
                      list(range(a.seed0, a.seed0 + a.seeds)))
        spec = {**sc.spec, "thresholds": th}
        mp = tmp / f"lovo_{c.cid}.json"
        mp.write_text(json.dumps(spec))
        cfg = replace(EventsConfig(), **{**AH.AMATEUR_BASE, "hit_scorer": str(mp), **json.loads(a.extra or "{}")})
        preds[c.cid] = raw[c.cid].predict_full(cfg)
    md, js = AH.report(list(raw.values()), preds)
    print(md)
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(js, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--root", type=Path, default=Path("~/Desktop/RallyIQ/train_v3/inputs"))
    c.add_argument("--labels", type=Path, default=Path("~/Desktop/RallyIQ/train_v3/labels/amateur_v1.json"))
    c.add_argument("--ss-train", type=Path, default=Path("~/Desktop/RallyIQ/train_v3/inputs/public/shuttleset_train_v3"))
    c.add_argument("--ss-test", type=Path, default=Path("~/Desktop/RallyIQ/train_v3/inputs/public/shuttleset_test_v3"))
    c.add_argument("--min-shuttle-conf", type=float, nargs="+", default=[0.3, 0.15])
    c.add_argument("--amateur-only", action="store_true")
    c.add_argument("--overrides", help="JSON EventsConfig overrides on top of V2_EVENT (candidate search)")
    for name in ("cv", "fit", "lovo"):
        p = sub.add_parser(name)
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--features", default="v3", choices=sorted(FEATURE_SETS))
        p.add_argument("--hidden", type=int, default=16)
        p.add_argument("--l2", type=float, default=1e-2)
        p.add_argument("--seeds", type=int, default=1)
        p.add_argument("--seed0", type=int, default=0, help="first seed (seeds seed0..seed0+seeds-1)")
        p.add_argument("--am-weight", type=float, default=3.0, help="weight of an amateur row (0 = ShuttleSet only)")
        p.add_argument("--ss-weight", type=float, default=1.0, help="0 = amateur only")
        p.add_argument("--msc-am", type=float, default=0.15)
        p.add_argument("--msc-ss", type=float, default=0.3)
        p.add_argument("--model", default="mlp", choices=["mlp", "gbt"])
        p.add_argument("--overrides", help="JSON EventsConfig overrides the candidates were collected with")
        p.add_argument("--gbt", help="JSON kwargs of fit_gbt (max_iter, lr, max_leaf_nodes, min_samples_leaf, l2)")
        p.add_argument("--unrefined", action="store_true", help="qa6 unrefined label times: looser positives (relabel_unrefined)")
        p.add_argument("--swing-dropout", type=float, default=0.0, help="share of training weight on copies with swing=0.5")
        if name == "cv":
            p.add_argument("--ss-folds", type=int, default=5)
            p.add_argument("--log", type=Path)
            p.add_argument("--save-oof", type=Path)
        elif name == "fit":
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--thresholds", help="JSON with min_candidate_conf/keep_threshold/ok_threshold")
        else:
            p.add_argument("--thresholds", help="JSON with min_candidate_conf/keep_threshold/ok_threshold")
            p.add_argument("--tmp", default="~/Desktop/RallyIQ/train_v3/ckpt/lovo_models")
            p.add_argument("--extra", help="JSON EventsConfig overrides on top of the amateur settings")
            p.add_argument("--labels", default="~/Desktop/RallyIQ/train_v3/labels/amateur_v1.json")
            p.add_argument("--json")
    a = ap.parse_args(argv)
    global MODEL_KIND, GBT_ARGS
    MODEL_KIND = getattr(a, "model", "mlp")
    GBT_ARGS = json.loads(getattr(a, "gbt", None) or "{}")
    return {"collect": collect, "cv": cmd_cv, "fit": cmd_fit, "lovo": cmd_lovo}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
