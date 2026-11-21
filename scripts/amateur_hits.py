"""Hit detection and hitter attribution on sam's own (amateur) videos, against fully labelled windows.

    python scripts/amateur_hits.py eval --root ~/Desktop/RallyIQ/train_v3/inputs \
        --labels ~/Desktop/RallyIQ/train_v3/labels/amateur_v1.json [--set hit_scorer=off] [--model m.json]

Two label sets, both "every hit inside the windows is labelled":
- qa6: ``hits_truth_v2.json`` (+ windows of ``hits_truth.json``) on the six own-v12 runs. Positive = sure and
  in_play True; dead-ball hits (in_play False) are negatives; unsure / in_play None are ignored.
- am7: the reviewed pre-labels (``amateur_v1.json``) on seven runs. Positive = reviewed_sure / reviewed_unsure;
  not_a_hit = negative. Windows are the labelled rallies.

A prediction inside a window is a true hit when it lies within ``TOL_S`` of a positive (one-to-one, nearest
first). False hits = unmatched predictions + predictions on a negative ("dead-time" false hits are the latter
plus any false hit while no rally is labelled as in play). Side is near/far of the hitter; hitter right is
the exact pipeline player id where the label has one (am7).

Only the events stage runs (cached shuttle/players artifacts, ``.json`` or ``.json.gz``).
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import learn_hits as LH  # noqa: E402
from rallyiq.contracts import CourtCalibration, PlayerTracks, ShuttleTrack  # noqa: E402
from rallyiq.events import hits as H  # noqa: E402
from rallyiq.events.adapters import calibration_segments, poses_from_contract, shuttle_from_contract  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402
from rallyiq.events.learned import ALL_FEATURES  # noqa: E402
from rallyiq.events.pipeline import segment_events  # noqa: E402

TOL_S = 0.25
# qa6 hits whose time was never refined (align_src "unrefined": 8 fps sheets, stated +-0.15 s) may be matched up to
# this far in the "loose" score (reported next to the strict one).
UNREFINED_TOL_S = 0.4
# Settings of the own-footage runs (PR #56 branch), applied on top of whatever is being compared.
AMATEUR_BASE = {"swing_veto": 0.1, "swing_veto_keep_exchange_ms": 2700.0, "min_no_hit_rally_ms": 2500.0, "landed_ms": 0.0}

QA6 = {
    "own_20260529_0006": "vid_97a8ee07f5fe", "own_20260529_0003": "vid_f52af81fca49",
    "own_20260529_0004": "vid_355e0f0cc3a6", "own_20260529_0008": "vid_b4eb017d39e6",
    "own_20260906_0615": "vid_efbc5a6de4af", "own_20260906_0620": "vid_df0e89955052",
}
AM7_R1 = ("own_20260324_0245", "own_20260310_01", "own_20260320_0239", "own_20260302_07")
AM7_R2 = ("own_20260324_0241", "own_20260906_0616", "own_20260906_0622")


def read_json(d: Path, name: str) -> Optional[str]:
    for p in (d / f"{name}.json", d / f"{name}.json.gz"):
        if p.exists():
            return gzip.open(p, "rt", encoding="utf-8").read() if p.suffix == ".gz" else p.read_text(encoding="utf-8")
    return None


def art_dir(root: Path, key: str) -> Path:
    if key in QA6:
        return root / "own_v12" / QA6[key] / "artifacts"
    if key in AM7_R1:
        return root / "prelabels" / "amateur_prelabels" / "artifacts" / key
    return root / "prelabels" / "amateur_prelabels_r2" / "artifacts" / key.replace("own_2026", "")


def load_truth(root: Path, labels: Optional[Path]) -> dict[str, dict[str, Any]]:
    """key -> {"set", "windows" [(s, e)] seconds, "labels" [{t, kind pos|neg|ign, side, hitter}]}."""
    out: dict[str, dict[str, Any]] = {}
    v1 = json.loads((root / "hits_truth.json").read_text())
    v2 = json.loads((root / "hits_truth_v2.json").read_text())
    for key, wins in v1["windows"].items():
        labs = []
        for h in v2["hits"]:
            if h["clip"] != key:
                continue
            kind = "pos" if (h["sure"] and h["in_play"] is True) else ("neg" if h["in_play"] is False else "ign")
            labs.append({"t": float(h["t"]), "kind": kind, "side": h.get("side"), "hitter": None,
                         "unrefined": h.get("align_src") == "unrefined"})
        out[key] = {"set": "qa6", "windows": [tuple(w) for w in wins], "labels": labs}
    if labels is not None:
        d = json.loads(labels.read_text())
        for key, hs in d["hits"].items():
            labs = []
            for h in hs:
                st = h["status"]
                kind = "neg" if st == "not_a_hit" else "pos"
                labs.append({"t": float(h["t"]), "kind": kind, "side": h.get("side"), "hitter": h.get("hitter"),
                             "stroke": h.get("stroke"), "status": st})
            out[key] = {"set": "am7", "windows": [tuple(w) for w in d["windows"][key]], "labels": labs}
    return out


class AmClip(LH.Clip):
    """learn_hits.Clip for an amateur run: gz artifacts, labels in seconds, training rows only inside windows."""

    def __init__(self, key: str, art: Path, truth: dict[str, Any]):
        self.cid = self.group = key
        self.art = art
        self.truth = truth
        meta = json.loads(read_json(art, "probe"))
        self.fps, self.duration_ms, self.video_id = float(meta["fps"]), float(meta["duration_ms"]), meta["video_id"]
        self.offset = 0
        sh = ShuttleTrack.model_validate_json(read_json(art, "shuttle"))
        pj = read_json(art, "players")
        self.players = PlayerTracks.model_validate_json(pj) if pj else None
        self.shuttle = shuttle_from_contract(sh)
        self.poses = poses_from_contract(self.players)
        self.match_format = self.players.match_format if self.players is not None else "singles"
        cj = read_json(art, "calibration")
        self.cal_segments = calibration_segments(CourtCalibration.model_validate_json(cj) if cj else None)
        pos = [x for x in truth["labels"] if x["kind"] == "pos"]
        self.label = {"range_ms": [0, 0], "hits": [{"t_ms": x["t"] * 1000.0, "side": x["side"]} for x in pos]}
        self.rallies = []
        self.X = np.zeros((0, len(ALL_FEATURES)))
        self.y = np.zeros(0)
        self.w = np.zeros(0)
        # pose side lookup for full-pipeline predictions
        self.side_seq: dict[str, list[tuple[int, str]]] = {}
        for f in self.poses:
            for p in f.players:
                if p.side in ("near", "far"):
                    self.side_seq.setdefault(p.player_id, []).append((f.frame, p.side))

    def in_window(self, t_s: float, pad: float = 0.0) -> bool:
        return any(a - pad <= t_s <= b + pad for a, b in self.truth["windows"])

    def collect_am(self, cfg: EventsConfig, tol_s: float = 0.2) -> None:
        """Candidates + features; y = 1 nearest candidate to a positive within ``tol_s``; weight 0 outside the
        windows and for candidates within TOL_S of an ignored label (unsure / in_play None)."""
        self.collect(cfg, tol_s * self.fps)
        n = len(self.y)
        t = np.zeros(n)
        for r in self.rallies:
            for e in r["cands"] + ([r["serve"]] if r["serve"] else []):
                t[e["k"]] = e["t"] / 1000.0
        self.t = t
        w = np.array([1.0 if self.in_window(x) else 0.0 for x in t])
        ign = [x["t"] for x in self.truth["labels"] if x["kind"] == "ign"]
        for i in range(n):
            if self.y[i] == 0 and any(abs(t[i] - g) <= TOL_S for g in ign):
                w[i] = 0.0
        self.w = w

    def side_at(self, pid: Optional[str], frame: int) -> Optional[str]:
        seq = self.side_seq.get(pid or "")
        if not seq:
            return None
        i = bisect.bisect_left(seq, (frame, ""))
        best = min((seq[j] for j in (i - 1, i) if 0 <= j < len(seq)), key=lambda s: abs(s[0] - frame))
        return best[1] if abs(best[0] - frame) <= 60 else None

    def predict_full(self, cfg: EventsConfig) -> list[dict[str, Any]]:
        out = []
        for r in self.run(cfg):
            for h in r.hits:
                side = self.side_at(h.hitter_id, h.frame)
                if side is None and h.hitter_court_xy is not None:
                    side = "near" if h.hitter_court_xy[1] < 6.7 else "far"
                out.append({"t": h.t_ms / 1000.0, "hitter": h.hitter_id, "side": side, "conf": h.confidence})
        return out


def simulate_am(clip: AmClip, probs: Optional[np.ndarray], cfg: EventsConfig) -> list[dict[str, Any]]:
    """learn_hits.simulate on cached candidates, plus the hitter id."""
    out = []
    for r in clip.rallies:
        def conf(e):
            return float(probs[e["k"]]) if probs is not None else e["rule"]
        kept = []
        for e in r["cands"]:
            c = replace(e["c"], prox=list(e["c"].prox), reasons=list(e["c"].reasons), confidence=round(conf(e), 3))
            if c.confidence >= cfg.min_candidate_conf and (probs is not None or e["rule"] > 0):
                kept.append((c, e))
        s = r["serve"]
        if s is not None:
            needed = not kept or kept[0][1]["t"] - r["start_ms"] > cfg.min_hit_gap_ms
            c = replace(s["c"], prox=list(s["c"].prox), reasons=list(s["c"].reasons), confidence=round(conf(s), 3))
            if needed and (probs is None or c.confidence >= cfg.min_candidate_conf):
                kept.insert(0, (c, s))
        ev = {id(c): e for c, e in kept}
        for c in H.resolve_sequence([c for c, _ in kept], cfg):
            b = c.best
            hitter = b.player_id if (b is not None and c.hitter_known) else None
            out.append({"t": ev[id(c)]["t"] / 1000.0, "hitter": hitter, "side": b.pose.side if hitter else None})
    return out


def score_clip(clip: AmClip, preds: list[dict[str, Any]], tol: float = TOL_S, loose: bool = False) -> dict[str, int]:
    labs = clip.truth["labels"]
    pr = [p for p in preds if clip.in_window(p["t"])]

    def tol_of(g):
        return max(tol, UNREFINED_TOL_S) if (loose and g.get("unrefined")) else tol
    pairs = sorted((abs(p["t"] - g["t"]), i, j) for i, p in enumerate(pr) for j, g in enumerate(labs)
                   if abs(p["t"] - g["t"]) <= tol_of(g))
    up, ul, m = set(), set(), []
    for _, i, j in pairs:
        if i in up or j in ul:
            continue
        up.add(i)
        ul.add(j)
        m.append((i, j))
    s = {"pos": sum(1 for g in labs if g["kind"] == "pos"), "pred": len(pr), "tp": 0, "fp": 0, "dead_fp": 0,
         "side_n": 0, "side_ok": 0, "hitter_n": 0, "hitter_ok": 0}
    for i, j in m:
        k = labs[j]["kind"]
        if k == "pos":
            s["tp"] += 1
            if labs[j].get("side") and pr[i]["side"] is not None:
                s["side_n"] += 1
                s["side_ok"] += pr[i]["side"] == labs[j]["side"]
            if labs[j].get("hitter"):
                s["hitter_n"] += 1
                s["hitter_ok"] += pr[i]["hitter"] == labs[j]["hitter"]
        elif k == "neg":
            s["fp"] += 1
            s["dead_fp"] += 1
    s["fp"] += len(pr) - len(m)
    s["fn"] = s["pos"] - s["tp"]
    return s


def summarize(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
    t = {k: sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))}
    p = t["tp"] / max(t["tp"] + t["fp"], 1)
    r = t["tp"] / max(t["pos"], 1)
    t.update(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
             side_acc=t["side_ok"] / max(t["side_n"], 1), hitter_acc=t["hitter_ok"] / max(t["hitter_n"], 1))
    return t


def parse_set(items: list[str]) -> dict[str, Any]:
    known = {f.name: f for f in fields(EventsConfig)}
    out: dict[str, Any] = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in known:
            raise SystemExit(f"unknown EventsConfig field {k}")
        cur = getattr(EventsConfig(), k)
        out[k] = (v.lower() in ("1", "true", "yes", "on")) if isinstance(cur, bool) else (type(cur)(v) if not isinstance(cur, str) else v)
    return out


def load_clips(root: Path, labels: Optional[Path], only: Optional[list[str]] = None) -> list[AmClip]:
    truth = load_truth(root, labels)
    return [AmClip(k, art_dir(root, k), v) for k, v in truth.items() if not only or k in only]


def report(clips: list[AmClip], preds: dict[str, list[dict]]) -> tuple[str, dict[str, Any]]:
    rows = {c.cid: score_clip(c, preds[c.cid]) for c in clips}
    lines = ["| video | set | pos | pred | TP | FP | dead FP | FN | F1 | side | hitter |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in clips:
        s = summarize({c.cid: rows[c.cid]})
        lines.append(f"| {c.cid} | {c.truth['set']} | {s['pos']} | {s['pred']} | {s['tp']} | {s['fp']} | {s['dead_fp']} | {s['fn']} | "
                     f"{s['f1']:.3f} | {s['side_ok']}/{s['side_n']} | {s['hitter_ok']}/{s['hitter_n']} |")
    out = {}
    for name in ("qa6", "am7", "all"):
        sel = {k: v for k, v in rows.items() if name == "all" or next(c for c in clips if c.cid == k).truth["set"] == name}
        if not sel:
            continue
        s = summarize(sel)
        out[name] = s
        lines.append(f"| **{name}** | | {s['pos']} | {s['pred']} | {s['tp']} | {s['fp']} | {s['dead_fp']} | {s['fn']} | "
                     f"**{s['f1']:.3f}** | {s['side_ok']}/{s['side_n']} | {s['hitter_ok']}/{s['hitter_n']} |")
    loose = {c.cid: score_clip(c, preds[c.cid], loose=True) for c in clips if c.truth["set"] == "qa6"}
    if loose:
        s = summarize(loose)
        out["qa6_loose"] = s
        lines.append(f"| qa6, unrefined labels +-{UNREFINED_TOL_S} s | | {s['pos']} | {s['pred']} | {s['tp']} | {s['fp']} | {s['dead_fp']} | "
                     f"{s['fn']} | **{s['f1']:.3f}** | {s['side_ok']}/{s['side_n']} | |")
    return "\n".join(lines), {"total": out, "per_video": rows, "per_video_qa6_loose": loose}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--root", type=Path, default=Path("~/Desktop/RallyIQ/train_v3/inputs"))
    ap.add_argument("--labels", type=Path, help="reviewed pre-labels (amateur_v1.json); without it only qa6")
    ap.add_argument("--set", action="append", default=[], help="EventsConfig override name=value (on top of AMATEUR_BASE)")
    ap.add_argument("--json", type=Path, help="write the summary here")
    a = ap.parse_args(argv)
    clips = load_clips(a.root.expanduser(), a.labels.expanduser() if a.labels else None)
    cfg = replace(EventsConfig(), **{**AMATEUR_BASE, **parse_set(a.set)})
    preds = {c.cid: c.predict_full(cfg) for c in clips}
    md, js = report(clips, preds)
    print(md)
    if a.json:
        a.json.write_text(json.dumps(js, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
