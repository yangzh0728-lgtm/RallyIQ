"""Train the court-level stroke model (rallyiq/strokes/court.py) on the ShuttleSet annotation.

    python scripts/learn_court_strokes.py <CoachAI-Projects>/ShuttleSet/set --out rallyiq/strokes/models/strokes_court_v2.json

No video is needed: ShuttleSet gives, for every stroke of 44 singles matches, the type, the hit frame,
the hitter position (image pixels + one court homography per match) and whether the contact was
above the net. Training adds position noise and contact flips so the model tolerates the errors of
our own measurements. Our hit detector also misses hits (DJI_0009: 19-41% of them), which breaks the
sequence features (flight time, previous/next shot), so training mixes copies of the data with 0, 15
and 30% of the hits dropped (``--drop-hit``). Reports grouped (by match) 5-fold CV, with and without
sequence smoothing, on full rallies and on rallies with 30% of the hits missing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools

from rallyiq.strokes import court as CM
from rallyiq.strokes.taxonomy import from_shuttleset

UNIT_M = 13.4 / 660  # ShuttleSet homographies map the doubles court to x 25..325, y 150..810


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _secs(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def load_rallies(set_dir: Path) -> list[dict]:
    homs = {r["video"]: np.asarray(ast.literal_eval(r["homography_matrix"]), float)
            for r in _rows(set_dir / "homography.csv")}
    out = []
    for m in _rows(set_dir / "match.csv"):
        H = homs.get(m["video"])
        if H is None:
            continue

        def world(x, y, H=H):
            if x is None or y is None:
                return None
            p = H @ [x, y, 1.0]
            return ((p[0] / p[2] - 25) * UNIT_M, (p[1] / p[2] - 150) * UNIT_M)

        for sc in sorted((set_dir / m["video"]).glob("set*.csv")):
            R = [r for r in _rows(sc) if r.get("rally") and _f(r.get("frame_num"))]
            est = [_f(r["frame_num"]) / _secs(r["time"]) for r in R if r.get("time") and _secs(r["time"]) > 0]
            if not est:
                continue
            fps = min((25.0, 30.0, 50.0, 60.0), key=lambda x: abs(x - statistics.median(est)))
            by = defaultdict(list)
            for r in R:
                by[int(float(r["rally"]))].append(r)
            for rs in by.values():
                rs.sort(key=lambda r: _f(r["frame_num"]))
                strokes = []
                for r in rs:
                    t = (r.get("type") or "").strip()
                    hh = _f(r.get("hit_height"))
                    strokes.append({"type": from_shuttleset(t)[0] if t else None,
                                    "t_ms": _f(r["frame_num"]) * 1000.0 / fps,
                                    "loc": world(_f(r.get("player_location_x")), _f(r.get("player_location_y"))),
                                    "high": None if hh is None else float(hh == 1.0)})
                out.append({"match": m["id"], "strokes": strokes})
    return out


def examples(rallies, rng=None, pos_noise=0.0, flip=0.0, drop_loc=0.0, drop_hit=0.0):
    fl = defaultdict(list)
    for r in rallies:
        s = r["strokes"]
        fl[r["match"]] += [b["t_ms"] - a["t_ms"] for a, b in itertools.pairwise(s)]
    med = {k: statistics.median(v) for k, v in fl.items() if v}
    X, y, g, ridx = [], [], [], []
    for i, r in enumerate(rallies):
        s = r["strokes"]
        if rng is not None and drop_hit:
            s = [st for st in s if rng.random() >= drop_hit]  # hits our detector misses
        locs, high = [], []
        for st in s:
            loc = st["loc"]
            if loc is not None and rng is not None and pos_noise:
                loc = (loc[0] + rng.normal(0, pos_noise), loc[1] + rng.normal(0, pos_noise))
            if loc is not None and rng is not None and rng.random() < drop_loc:
                loc = None  # our far player often has no court position at the hit
            locs.append(loc)
            h = np.nan if st["high"] is None else st["high"]
            if rng is not None and not np.isnan(h):
                if rng.random() < flip:
                    h = 1.0 - h
                h = float(np.clip(0.85 * h + 0.15 * (1 - h) + rng.normal(0, 0.05), 0, 1))  # model-like soft p
            high.append(h)
        rows = CM.rally_features(locs, [st["t_ms"] for st in s], high, med.get(r["match"], 900.0))
        for st, row in zip(s, rows):
            if st["type"]:
                X.append(row), y.append(st["type"]), g.append(r["match"]), ridx.append(i)
    return np.asarray(X), np.asarray(y), np.asarray(g), np.asarray(ridx)


def sequence_stats(rallies, classes):
    K = len(classes)
    ix = {c: i for i, c in enumerate(classes)}
    A, S0 = np.ones((K, K)), np.ones(K)
    for r in rallies:
        s = [st["type"] for st in r["strokes"]]
        if s and s[0] in ix:
            S0[ix[s[0]]] += 1
        for a, b in itertools.pairwise(s):
            if a in ix and b in ix:
                A[ix[a], ix[b]] += 1
    return S0 / S0.sum(), A / A.sum(1, keepdims=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pos-noise", type=float, default=0.4)
    ap.add_argument("--flip", type=float, default=0.12)
    ap.add_argument("--drop-loc", type=float, default=0.0, help="share of hitter positions dropped in training (0.15 and 0.3 were worse on DJI_0009)")
    ap.add_argument("--drop-hit", type=float, nargs="+", default=[0.0, 0.15, 0.3],
                    help="one training copy per value, with that share of the hits removed from each rally")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--seq-weight", type=float, default=0.0,
                    help="rally smoothing at run time (0 = off: it costs ~6 points once hits are missed)")
    a = ap.parse_args(argv)
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.model_selection import GroupKFold

    from rallyiq.gbt import export_hgb

    rallies = load_rallies(a.set_dir)
    parts = [examples(rallies, np.random.default_rng(i), a.pos_noise, a.flip, a.drop_loc, d)
             for i, d in enumerate(a.drop_hit)]
    X = np.vstack([p[0] for p in parts])
    y, g = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    classes = sorted(set(y.tolist()))  # scikit-learn orders classes this way
    print(f"{len(rallies)} rallies, {len(parts[0][1])} strokes, {len(set(g))} matches, {len(a.drop_hit)} training copies")

    def fit(Xt, yt):
        return HGB(max_iter=a.iters, learning_rate=0.1, max_leaf_nodes=a.leaves, l2_regularization=1.0,
                   random_state=0).fit(Xt, yt)

    start, trans = sequence_stats(rallies, classes)
    prior = np.asarray([np.mean(parts[0][1] == c) for c in classes])
    evals = {"full rallies": examples(rallies, np.random.default_rng(101), a.pos_noise, a.flip, a.drop_loc, 0.0),
             "30% hits missing": examples(rallies, np.random.default_rng(102), a.pos_noise, a.flip, a.drop_loc, 0.3)}
    matches = np.unique(g)
    cv = {}
    for name, (Xv, yv, gv, rv) in evals.items():
        P = np.zeros((len(yv), len(classes)))
        for _, te_m in GroupKFold(5).split(matches, matches, matches):
            held = matches[te_m]
            m = fit(X[~np.isin(g, held)], y[~np.isin(g, held)])
            te = np.isin(gv, held)
            P[te] = m.predict_proba(Xv[te])[:, [list(m.classes_).index(c) for c in classes]]
        S = P.copy()
        for i in np.unique(rv):
            k = rv == i
            S[k] = CM.smooth(P[k], start, trans, prior, 1.0)
        cv[name] = (float(np.mean(np.asarray(classes)[P.argmax(1)] == yv)), float(np.mean(np.asarray(classes)[S.argmax(1)] == yv)))
        print(f"grouped 5-fold CV, {name}: {cv[name][0]:.3f} per shot, {cv[name][1]:.3f} with rally smoothing")
    acc, acc_seq = cv["full rallies"]

    if a.out:
        m = fit(X, y)
        order = [list(m.classes_).index(c) for c in classes]
        assert order == list(range(len(classes))), "class order"
        depths = X[:, CM.FEATURES.index("hd")]
        spec = {"kind": "strokes_court", "features": list(CM.FEATURES), **export_hgb(m),
                "depth_quantiles": [round(float(v), 4) for v in np.quantile(depths[~np.isnan(depths)], np.linspace(0.05, 0.95, 19))],
                "start": start.round(5).tolist(), "trans": trans.round(5).tolist(), "prior": prior.round(5).tolist(),
                "seq_weight": a.seq_weight,
                "meta": {"source": "ShuttleSet (CoachAI-Projects, MIT)", "strokes": len(parts[0][1]), "matches": len(set(g)),
                         "cv_acc": round(float(acc), 3), "cv_acc_seq": round(float(acc_seq), 3),
                         "cv_acc_missing30": round(cv["30% hits missing"][0], 3),
                         "cv_acc_missing30_seq": round(cv["30% hits missing"][1], 3), "drop_hit": a.drop_hit,
                         "pos_noise_m": a.pos_noise, "drop_loc": a.drop_loc, "contact_flip": a.flip, "counts": dict(Counter(parts[0][1].tolist()))}}
        a.out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
