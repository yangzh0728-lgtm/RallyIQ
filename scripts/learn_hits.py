"""Learn the hit scorer (rallyiq/events/learned.py) from labelled clips, tune its thresholds, and
compare rules vs learned on held-out clips.

    # 1. train on the ShuttleSet training clips built by scripts/build_train_set.py
    python scripts/learn_hits.py train --data ~/rallyiq-data/public/shuttleset_train
    #    -> rallyiq/events/models/hit_scorer.json (+ .md report next to it)
    # 2. before/after on the held-out matches scored by scripts/public_eval.py
    python scripts/learn_hits.py eval --data ~/rallyiq-data/public/shuttleset

Only the events stage is re-run (cached shuttle/players artifacts), so both steps take minutes.

Training: every candidate the rules propose (after non-maximum suppression, before any filtering)
becomes one example; it is positive when a labelled hit lies within --tol frames. Model and
hyper-parameters are chosen by grouped cross-validation (folds by match, so the same match never
appears in training and validation). Thresholds (min_candidate_conf, keep_threshold, ok_threshold)
are tuned end-to-end on out-of-fold probabilities, maximising hit F1 under the benchmark rule
(±3 frames, same side). The rally parameters are left as they are.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import benchmark, from_contracts, schema  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from rallyiq.contracts import PlayerTracks, ShuttleTrack  # noqa: E402
from rallyiq.events import hits as H  # noqa: E402
from rallyiq.events.adapters import events_to_contract, poses_from_contract, shuttle_from_contract  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402
from rallyiq.events.learned import BUNDLED, FEATURES, HitScorer  # noqa: E402
from rallyiq.events.pipeline import segment_events  # noqa: E402

PLAYER_IDS = {"P1", "P2", "P3", "P4"}


# ------------------------------------------------------------------ data
class Clip:
    def __init__(self, cid: str, art: Path, label: dict[str, Any], group: str):
        self.cid, self.art, self.label, self.group = cid, art, label, group
        meta = json.loads((art / "probe.json").read_text(encoding="utf-8"))
        self.fps, self.duration_ms, self.video_id = float(meta["fps"]), float(meta["duration_ms"]), meta["video_id"]
        self.offset = int(label["range_ms"][0])
        sh = ShuttleTrack.model_validate_json((art / "shuttle.json").read_text(encoding="utf-8"))
        pp = art / "players.json"
        self.players = PlayerTracks.model_validate_json(pp.read_text(encoding="utf-8")) if pp.exists() else None
        self.shuttle = shuttle_from_contract(sh)
        self.poses = poses_from_contract(self.players)
        self.match_format = self.players.match_format if self.players is not None else "singles"
        self.rallies: list[dict[str, Any]] = []
        self.X = np.zeros((0, len(FEATURES)))
        self.y = np.zeros(0)

    def t_label(self, frame: int) -> float:
        return self.offset + frame * 1000.0 / self.fps

    def run(self, cfg: EventsConfig):
        return segment_events(self.shuttle, self.poses, cfg, video_end_ms=self.duration_ms)

    def collect(self, cfg: EventsConfig, tol_frames: float) -> None:
        """Candidates per rally with features and labels (scorer off: rule confidences kept)."""
        sink: list = []
        rallies = segment_events(self.shuttle, self.poses, replace(cfg, hit_scorer="off"),
                                 video_end_ms=self.duration_ms, collect=sink)
        by_id = {r.rally_id: {"start_ms": r.start_ms, "cands": [], "serve": None} for r in rallies}
        X = []
        for k, (rid, x, c, pt) in enumerate(sink):
            e = {"k": k, "c": c, "t": self.t_label(pt.frame), "frame": pt.frame, "rule": c.confidence}
            if H.R_INFERRED_SERVE in c.reasons:
                by_id[rid]["serve"] = e
            else:
                by_id[rid]["cands"].append(e)
            X.append(x)
        self.rallies = list(by_id.values())
        self.X = np.asarray(X, float).reshape(-1, len(FEATURES))
        # positives: nearest candidate to each labelled hit, within tolerance (one-to-one)
        tol = tol_frames * 1000.0 / self.fps + 1e-6
        ts = [None] * len(sink)
        for r in self.rallies:
            for e in r["cands"] + ([r["serve"]] if r["serve"] else []):
                ts[e["k"]] = e["t"]
        pairs = sorted((abs(t - g["t_ms"]), i, j) for j, g in enumerate(self.label["hits"])
                       for i, t in enumerate(ts) if abs(t - g["t_ms"]) <= tol)
        y = np.zeros(len(sink))
        ys = np.full(len(sink), np.nan)  # positives with a labelled side: 1 = near end hit it
        used_i, used_j = set(), set()
        for _, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            y[i] = 1
            gs = self.label["hits"][j].get("side")
            if gs in ("near", "far"):
                ys[i] = 1.0 if gs == "near" else 0.0
        self.y = y
        self.ys = ys
        self.recall_ceiling = (len(used_j), len(self.label["hits"]))


def load_clips(data: Path, index_name: str) -> list[Clip]:
    out = []
    if index_name == "index.json":  # build_train_set.py
        index = json.loads((data / "index.json").read_text(encoding="utf-8"))
        items = [(cid, Path(v["artifacts"]), str(v["match_id"])) for cid, v in index.items() if v.get("state") == "succeeded"]
    else:  # public_eval.py output: every labelled clip whose prediction points at a stored video
        # (results.json only lists the clips of the last run, e.g. with --skip-run, so it is not used)
        items = []
        for lp in sorted((data / "labels").glob("*.json")):
            pp = data / "preds" / lp.name
            vid = json.loads(pp.read_text(encoding="utf-8")).get("source_video_id") if pp.exists() else None
            if vid:
                items.append((lp.stem, data / "data" / "videos" / vid / "artifacts", lp.stem))
    for cid, art, group in sorted(items):
        lp = data / "labels" / f"{cid}.json"
        if not lp.exists() or not (art / "shuttle.json").exists():
            print(f"[skip] {cid}: no label or no shuttle artifact")
            continue
        out.append(Clip(cid, art, schema.load(lp), group))
    return out


# ------------------------------------------------------------------ fast end-to-end simulation
def simulate(clip: Clip, probs: np.ndarray | None, cfg: EventsConfig,
             p_near: np.ndarray | None = None) -> list[dict[str, Any]]:
    """The tail of hits.detect_hits (filter, serve, alternation) on cached candidates.
    ``probs`` None = rule confidences. Returns predicted hits in label time with side and status."""
    out_hits = []
    for r in clip.rallies:
        def conf(e):
            return float(probs[e["k"]]) if probs is not None else e["rule"]

        kept = []
        for e in r["cands"]:
            c = replace(e["c"], prox=list(e["c"].prox), reasons=list(e["c"].reasons), confidence=round(conf(e), 3))
            if c.confidence >= cfg.min_candidate_conf and (probs is not None or e["rule"] > 0):
                if p_near is not None:
                    H.apply_side_model(c, float(p_near[e["k"]]), cfg.side_threshold)
                kept.append((c, e))
        s = r["serve"]
        if s is not None:
            needed = not kept or kept[0][1]["t"] - (clip.offset + r["start_ms"]) > cfg.min_hit_gap_ms
            c = replace(s["c"], prox=list(s["c"].prox), reasons=list(s["c"].reasons), confidence=round(conf(s), 3))
            if needed and (probs is None or c.confidence >= cfg.min_candidate_conf):
                if p_near is not None:
                    H.apply_side_model(c, float(p_near[s["k"]]), cfg.side_threshold)
                kept.insert(0, (c, s))
        cands = [c for c, _ in kept]
        ev = {id(c): e for c, e in kept}
        cands = H._alternating_subset(cands, cfg)
        H._reattribute(cands, cfg)
        H._check_alternation(cands)
        for c in cands:
            b = c.best
            hitter = b.player_id if c.hitter_known and b.player_id in PLAYER_IDS else None
            ok = c.confidence >= cfg.ok_threshold and not c.reasons and hitter is not None
            out_hits.append({"t_ms": ev[id(c)]["t"], "side": b.pose.side if hitter else None, "ok": ok})
    return out_hits


def score(clips: list[Clip], hits_by_clip: dict[str, list[dict]], tol_frames: int = M.HIT_TOL_FRAMES) -> dict[str, Any]:
    st = M.EventStats()
    ok_n = ok_tp = 0
    side = {"side_ok": 0, "side_missing": 0, "side_wrong": 0}
    old = M.HIT_TOL_FRAMES
    M.HIT_TOL_FRAMES = tol_frames
    try:
        for c in clips:
            hs = hits_by_clip[c.cid]
            st.merge(M.eval_hits(c.label, {"hits": hs}))
            g = [h for h in c.label["hits"] if M._in_range(h["t_ms"], c.label)]
            fm = M.frame_ms(c.label)
            for i, j, _ in M._greedy_match(g, hs, tol_frames * fm + 1e-6, same_side=False):
                gs, ps = g[i].get("side"), hs[j].get("side")
                k = "side_ok" if (not gs or gs == ps) else ("side_missing" if ps is None else "side_wrong")
                side[k] += 1
            oks = [h for h in hs if h["ok"]]
            ok_n += len(oks)
            ok_tp += M.eval_hits(c.label, {"hits": oks}).tp
    finally:
        M.HIT_TOL_FRAMES = old
    s = st.summary()
    s["ok"] = ok_n
    s["ok_precision"] = ok_tp / ok_n if ok_n else None
    s.update(side)  # hits found at the right time: is their near/far side right, missing or wrong?
    return s


# ------------------------------------------------------------------ model
def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int, l2: float, seed: int = 0, epochs: int = 1500,
            lr: float = 0.01) -> dict[str, Any]:
    """Full-batch Adam on mean BCE + l2; returns a learned.HitScorer spec."""
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    sizes = [X.shape[1]] + ([hidden] if hidden else []) + [1]
    P = []
    for a, b in zip(sizes, sizes[1:]):
        P += [rng.normal(0, 1 / math.sqrt(a), (a, b)), np.zeros(b)]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    n = len(y)
    for t in range(1, epochs + 1):
        acts = [Z]
        h = Z
        for k in range(0, len(P), 2):
            h = h @ P[k] + P[k + 1]
            if k < len(P) - 2:
                h = np.tanh(h)
            acts.append(h)
        p = 1 / (1 + np.exp(-acts[-1][:, 0]))
        g = ((p - y) / n)[:, None]
        grads = [None] * len(P)
        for k in range(len(P) - 2, -1, -2):
            a_in = acts[k // 2]
            grads[k] = a_in.T @ g + l2 * P[k]
            grads[k + 1] = g.sum(0)
            if k > 0:
                g = (g @ P[k].T) * (1 - acts[k // 2] ** 2)
        for i, (p_, g_) in enumerate(zip(P, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g_
            v[i] = 0.999 * v[i] + 0.001 * g_ ** 2
            P[i] = p_ - lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    layers = [[P[k].tolist(), P[k + 1].tolist()] for k in range(0, len(P), 2)]
    return {"features": list(FEATURES), "mean": mean.tolist(), "std": std.tolist(), "layers": layers,
            "hidden": hidden, "l2": l2}


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def folds_by_group(clips: list[Clip], k: int) -> list[list[Clip]]:
    groups = sorted({c.group for c in clips})
    random.Random(0).shuffle(groups)
    gi = {g: i % k for i, g in enumerate(groups)}
    return [[c for c in clips if gi[c.group] == f] for f in range(k)]


def oof_probs(clips: list[Clip], folds: list[list[Clip]], hidden: int, l2: float) -> dict[str, np.ndarray]:
    out = {}
    for f in folds:
        if not f:
            continue
        held = {c.cid for c in f}
        tr = [c for c in clips if c.cid not in held]
        spec = fit_mlp(np.vstack([c.X for c in tr]), np.concatenate([c.y for c in tr]), hidden, l2)
        sc = HitScorer(spec)
        for c in f:
            out[c.cid] = sc.predict(c.X) if len(c.X) else np.zeros(0)
    return out


def oof_side(clips: list[Clip], folds: list[list[Clip]], l2: float = 1e-2) -> dict[str, np.ndarray]:
    """Out-of-fold P(near end hit it) from a logistic model trained on positives with a labelled side."""
    out = {}
    for f in folds:
        if not f:
            continue
        held = {c.cid for c in f}
        tr = [c for c in clips if c.cid not in held]
        X = np.vstack([c.X[~np.isnan(c.ys)] for c in tr])
        y = np.concatenate([c.ys[~np.isnan(c.ys)] for c in tr])
        sc = HitScorer(fit_mlp(X, y, 0, l2))
        for c in f:
            out[c.cid] = sc.predict(c.X) if len(c.X) else np.zeros(0)
    return out


def tune_side(clips: list[Clip], probs: dict[str, np.ndarray], p_near: dict[str, np.ndarray], cfg: EventsConfig):
    best = None
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
        c2 = replace(cfg, side_threshold=thr)
        s = score(clips, {c.cid: simulate(c, probs[c.cid], c2, p_near[c.cid]) for c in clips})
        if best is None or (s["f1"] or 0) > (best[1]["f1"] or 0):
            best = (thr, s)
    return best


def tune_thresholds(clips: list[Clip], probs: dict[str, np.ndarray], base: EventsConfig):
    best = None
    for mc in (0.05, 0.1, 0.15, 0.2, 0.3):
        for keep in np.arange(0.2, 0.75, 0.05):
            for ok in (0.5, 0.6, 0.7, 0.8, 0.9):
                if ok < keep:
                    continue
                cfg = replace(base, min_candidate_conf=mc, keep_threshold=round(float(keep), 2), ok_threshold=ok)
                s = score(clips, {c.cid: simulate(c, probs[c.cid], cfg) for c in clips})
                # hit F1 first; among near-ties prefer more trustworthy "ok" hits
                key = (round(s["f1"] or 0, 3), s["ok_precision"] or 0)
                if best is None or key > best[0]:
                    best = (key, {"min_candidate_conf": mc, "keep_threshold": round(float(keep), 2), "ok_threshold": ok}, s)
    return best[1], best[2]


# ------------------------------------------------------------------ commands
def cmd_train(a: argparse.Namespace) -> int:
    clips = load_clips(a.data.expanduser(), "index.json")
    if len(clips) < 5:
        print(f"only {len(clips)} usable clips in {a.data}", file=sys.stderr)
        return 1
    base = EventsConfig()
    for c in clips:
        c.collect(base, a.tol)
    X = np.vstack([c.X for c in clips])
    y = np.concatenate([c.y for c in clips])
    n_gt = sum(c.recall_ceiling[1] for c in clips)
    n_reach = sum(c.recall_ceiling[0] for c in clips)
    L = ["# Hit scorer training report", "",
         f"- clips: {len(clips)} from {len({c.group for c in clips})} matches; labelled hits {n_gt}",
         f"- candidates: {len(y)} ({int(y.sum())} positive); candidate recall ceiling (±{a.tol} frames): "
         f"{n_reach / max(n_gt, 1):.1%}", ""]
    print("\n".join(L), flush=True)

    rules = score(clips, {c.cid: simulate(c, None, base) for c in clips})
    folds = folds_by_group(clips, a.folds)
    results = []
    for hidden in a.hidden:
        for l2 in a.l2:
            probs = oof_probs(clips, folds, hidden, l2)
            ll = logloss(np.concatenate([probs[c.cid] for c in clips]), y)
            print(f"hidden={hidden} l2={l2}: out-of-fold logloss {ll:.4f}", flush=True)
            results.append((ll, hidden, l2, probs))
    results.sort(key=lambda r: r[0])
    ll, hidden, l2, probs = results[0]
    th, cv = tune_thresholds(clips, probs, base)
    p_near = oof_side(clips, folds)
    side_thr, cv_side = tune_side(clips, probs, p_near, replace(base, **th))
    use_side = side_thr <= 1.0 and (cv_side["f1"] or 0) > (cv["f1"] or 0)
    # also: rules with tuned thresholds only (does the model add anything beyond re-thresholding?)
    rules_t = max(((score(clips, {c.cid: simulate(c, None, replace(base, keep_threshold=k, min_candidate_conf=mc))
                                  for c in clips}), k, mc)
                   for k in (0.35, 0.4, 0.45, 0.5, 0.55) for mc in (0.2, 0.3, 0.4)),
                  key=lambda r: r[0]["f1"] or 0)

    def row(name: str, s: dict[str, Any]) -> str:
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        return (f"| {name} | {f(s['precision'])} | {f(s['recall'])} | {f(s['f1'])} | {f(s['f1_ignore_side'])} | "
                f"{s['ok']} | {f(s['ok_precision'])} |")

    L += ["## Cross-validated hit accuracy on the training matches (grouped by match)", "",
          "| scorer | precision | recall | F1 | F1 (any side) | ok hits | ok precision |", "|---|---|---|---|---|---|---|",
          row("rules (current defaults)", rules),
          row(f"rules, re-thresholded (keep={rules_t[1]}, min={rules_t[2]})", rules_t[0]),
          row(f"learned (hidden={hidden}, l2={l2}), out-of-fold", cv),
          row(f"learned + side model (threshold {side_thr}), out-of-fold", cv_side), "",
          f"Side model {'used' if use_side else 'not used'}: by side among correctly timed hits "
          f"right {cv_side['side_ok']}, missing {cv_side['side_missing']}, wrong {cv_side['side_wrong']}", "",
          f"Chosen thresholds: {json.dumps(th)}", "",
          "Hits found at the right time, by near/far side (learned): "
          f"right {cv['side_ok']}, missing {cv['side_missing']}, wrong {cv['side_wrong']}; "
          f"rules: right {rules['side_ok']}, missing {rules['side_missing']}, wrong {rules['side_wrong']}", "",
          "Model search (out-of-fold log-loss): " + ", ".join(f"h{h}/l2={l}: {x:.4f}" for x, h, l, _ in results), ""]
    spec = fit_mlp(X, y, hidden, l2)
    if use_side:
        m = np.concatenate([~np.isnan(c.ys) for c in clips])
        spec["side"] = fit_mlp(X[m], np.concatenate([c.ys for c in clips])[m], 0, 1e-2)
        th = {**th, "side_threshold": side_thr}
        cv = cv_side
    spec["thresholds"] = th
    spec["trained_on"] = {"dataset": "ShuttleSet (CoachAI-Projects, MIT)", "clips": len(clips),
                          "matches": len({c.group for c in clips}), "candidates": int(len(y)), "positives": int(y.sum()),
                          "tol_frames": a.tol, "cv": {k: cv[k] for k in ("precision", "recall", "f1", "ok_precision")}}
    out = a.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[5:]))
    print(f"model: {out}\nreport: {md}")
    return 0


def _events_preds(clips: list[Clip], cfg: EventsConfig, preds: Path) -> None:
    preds.mkdir(parents=True, exist_ok=True)
    for c in clips:
        ev = events_to_contract(c.video_id, c.run(cfg), c.match_format)
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            for name in ("probe", "players", "calibration", "shuttle"):
                if (c.art / f"{name}.json").exists():
                    (t / f"{name}.json").symlink_to(c.art / f"{name}.json")
            (t / "events.json").write_text(ev.model_dump_json(), encoding="utf-8")
            schema.save(from_contracts.convert(t, c.cid, c.offset), preds / f"{c.cid}.json")


def cmd_eval(a: argparse.Namespace) -> int:
    data = a.data.expanduser()
    clips = load_clips(data, "results.json")
    mp = data / "manifest.json"
    full = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    # only clips with artifacts are scored (a clip without them would count as all misses for every variant)
    manifest = {c.cid: full.get(c.cid, {"id": c.cid, "split": "eval", "conditions": {"source": data.name}})
                for c in clips}
    models = [m.expanduser() for m in a.model]
    out = data / a.out_name
    variants = {"rules": EventsConfig(hit_scorer="off")}
    variants.update({m.stem: EventsConfig(hit_scorer=str(m)) for m in models})
    L = [f"# Hits: rules vs learned scorer on {len(clips)} held-out clips ({data.name})", "",
         f"clips: {', '.join(c.cid for c in clips)}", "", "models: " + ", ".join(f"`{m}`" for m in models), ""]
    summary = {}
    per_clip: dict[str, dict[str, Any]] = {}
    for name, cfg in variants.items():
        _events_preds(clips, cfg, out / name)
        rep = benchmark.run(data / "labels", out / name, manifest, None, False)
        old = M.HIT_TOL_FRAMES
        try:
            M.HIT_TOL_FRAMES = 6
            loose = benchmark.run(data / "labels", out / name, manifest, None, False)["overall"]["hits"]
        finally:
            M.HIT_TOL_FRAMES = old
        o = rep["overall"]
        summary[name] = {"hits": o["hits"], "hits_6f": loose, "rallies": o["rallies"]}
        for cid, cs in rep["clips"].items():
            per_clip.setdefault(cid, {})[name] = cs["hits"]["f1"]
    f = lambda x: "—" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    L += ["| scorer | hit P | hit R | hit F1 | F1 any side | F1 ±6 frames | rally F1 |", "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        h = s["hits"]
        L.append(f"| {name} | {f(h['precision'])} | {f(h['recall'])} | {f(h['f1'])} | {f(h['f1_ignore_side'])} | "
                 f"{f(s['hits_6f']['f1'])} | {f(s['rallies']['f1'])} |")
    L += ["", "Hit F1 per clip:", "", "| clip | " + " | ".join(summary) + " |", "|---|" + "---|" * len(summary)]
    for cid, row in sorted(per_clip.items()):
        L.append(f"| {cid} | " + " | ".join(f(row.get(n)) for n in summary) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("\n".join(L))
    print(f"\nreport: {out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=Path("~/rallyiq-data/public/shuttleset_train"))
    t.add_argument("--out", type=Path, default=BUNDLED)
    t.add_argument("--tol", type=float, default=3, help="frames between a candidate and a labelled hit for a positive")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--hidden", type=int, nargs="+", default=[0, 8, 16])
    t.add_argument("--l2", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, default=Path("~/rallyiq-data/public/shuttleset"))
    e.add_argument("--model", type=Path, nargs="+", default=[BUNDLED], help="one or more scorer JSON files")
    e.add_argument("--out-name", default="learn_eval", help="output folder inside --data")
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    raise SystemExit(main())
