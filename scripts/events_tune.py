"""Tune the hit / rally rules against public labels, offline (seconds per setting, no models).

    python scripts/events_tune.py                     # ~/rallyiq-data/public/shuttleset
    python scripts/events_tune.py --dir <out> --passes 2 --json tune.json

Reads the saved shuttle track and players of each clip (found through preds/<clip>.json ->
source_video_id -> data/videos/<id>/artifacts), re-runs only ``segment_events`` with other
``EventsConfig`` values and scores hits (+-3 frames, side ignored, because side depends on
calibration) and rallies against labels/<clip>.json.

Search: coordinate descent from the defaults over GRID, maximising hit F1 + 0.5 * rally F1.
To show whether a setting generalises, it also tunes on half of the clips and scores the
other half (both ways). Prints the signed timing offsets of near-miss hits too: a constant
bias there is a timing problem, not a detection one.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import metrics as M  # noqa: E402
from evaluation import schema  # noqa: E402
from rallyiq.contracts import PlayerTracks, ShuttleTrack, VideoMeta  # noqa: E402
from rallyiq.events.adapters import events_to_contract, poses_from_contract, shuttle_from_contract  # noqa: E402
from rallyiq.events.config import EventsConfig  # noqa: E402
from rallyiq.events.pipeline import segment_events  # noqa: E402

GRID: dict[str, list[Any]] = {
    "window_ms": [70.0, 100.0, 130.0, 170.0],
    "min_turn_deg": [20.0, 25.0, 35.0, 50.0],
    "speed_jump_ratio": [1.4, 1.8, 2.5],
    "min_hit_gap_ms": [200.0, 280.0, 400.0, 550.0],
    "max_hitter_dist_rel": [0.6, 0.9, 1.3, 2.0],
    "min_candidate_conf": [0.2, 0.3, 0.4, 0.5],
    "keep_threshold": [0.35, 0.45, 0.55, 0.65],
    "end_gap_ms": [800.0, 1200.0, 2000.0],
    "static_ms": [400.0, 600.0, 1000.0],
    "min_rally_ms": [700.0, 1500.0, 2500.0],
    "min_rally_points": [8, 15, 30],
    "min_shuttle_conf": [0.1, 0.2, 0.35],
}


class Clip:
    def __init__(self, cid: str, label: dict[str, Any], art: Path, offset_ms: float):
        self.cid, self.label, self.offset_ms = cid, label, offset_ms
        self.meta = VideoMeta.model_validate_json((art / "probe.json").read_text())
        self.shuttle = shuttle_from_contract(ShuttleTrack.model_validate_json((art / "shuttle.json").read_text()))
        pp = art / "players.json"
        players = PlayerTracks.model_validate_json(pp.read_text()) if pp.exists() else None
        self.poses = poses_from_contract(players)
        self.fmt = players.match_format if players is not None else "singles"

    def predict(self, cfg: EventsConfig) -> dict[str, Any]:
        rallies = segment_events(self.shuttle, self.poses, cfg, video_end_ms=float(self.meta.duration_ms))
        ev = events_to_contract(self.meta.video_id, rallies, self.fmt)
        fps = float(self.meta.fps)
        t = lambda f: self.offset_ms + f * 1000.0 / fps  # noqa: E731
        return {"hits": [{"t_ms": t(s.frame), "side": None} for s in ev.shots],
                "rallies": [{"start_ms": t(r.start_frame), "end_ms": t(r.end_frame)}
                            for r in ev.rallies if r.end_frame > r.start_frame]}


def load_clips(d: Path) -> list[Clip]:
    out = []
    for lp in sorted((d / "labels").glob("*.json")):
        pp = d / "preds" / f"{lp.stem}.json"
        if not pp.exists():
            continue
        pred = json.loads(pp.read_text(encoding="utf-8"))
        art = d / "data" / "videos" / pred["source_video_id"] / "artifacts"
        if not (art / "shuttle.json").exists():
            print(f"[skip] {lp.stem}: no {art / 'shuttle.json'}")
            continue
        out.append(Clip(lp.stem, schema.load(lp), art, float(pred["range_ms"][0])))
    return out


def score(clips: list[Clip], cfg: EventsConfig) -> dict[str, Any]:
    tp = fp = fn = 0
    rally = M.RallyStats()
    for c in clips:
        p = c.predict(cfg)
        st = M.eval_hits(c.label, p)
        tp += st.tp_any_side
        fp += st.tp + st.fp - st.tp_any_side
        fn += st.tp + st.fn - st.tp_any_side
        rally.merge(M.eval_rallies(c.label, p))
    hit_f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    r_f1 = rally.summary()["f1"] or 0.0
    return {"hit_f1": round(hit_f1, 4), "hit_p": round(tp / (tp + fp), 4) if tp + fp else None,
            "hit_r": round(tp / (tp + fn), 4) if tp + fn else None, "rally_f1": round(r_f1, 4),
            "objective": round(hit_f1 + 0.5 * r_f1, 4)}


def descend(clips: list[Clip], passes: int, log: bool = True) -> tuple[EventsConfig, dict[str, Any]]:
    cfg = EventsConfig()
    best = score(clips, cfg)
    for _ in range(passes):
        changed = False
        for name, values in GRID.items():
            for v in values:
                if getattr(cfg, name) == v:
                    continue
                cand = replace(cfg, **{name: v})
                s = score(clips, cand)
                if s["objective"] > best["objective"] + 1e-4:
                    cfg, best, changed = cand, s, True
                    if log:
                        print(f"  {name}={v}: {s}", flush=True)
        if not changed:
            break
    return cfg, best


def offsets(clips: list[Clip], cfg: EventsConfig) -> list[float]:
    """Signed offset (frames, pred - label) of the nearest prediction within 15 frames of each label hit."""
    out = []
    for c in clips:
        fm = M.frame_ms(c.label)
        pt = np.array([h["t_ms"] for h in c.predict(cfg)["hits"]])
        for h in c.label["hits"]:
            if pt.size and M._in_range(h["t_ms"], c.label):
                d = (pt - h["t_ms"]) / fm
                k = int(np.argmin(np.abs(d)))
                if abs(d[k]) <= 15:
                    out.append(float(d[k]))
    return out


def _diff(cfg: EventsConfig) -> dict[str, Any]:
    base = asdict(EventsConfig())
    return {k: v for k, v in asdict(cfg).items() if base[k] != v}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / "rallyiq-data" / "public" / "shuttleset")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    clips = load_clips(a.dir.expanduser())
    if not clips:
        print("no clips with labels, preds and artifacts", file=sys.stderr)
        return 1
    print("clips:", [c.cid for c in clips])
    base = score(clips, EventsConfig())
    print("defaults:", base, "per clip:", {c.cid: score([c], EventsConfig())["hit_f1"] for c in clips})
    off = offsets(clips, EventsConfig())
    hist = np.histogram(off, bins=np.arange(-15.5, 16.5, 1.0))[0].tolist() if off else []
    print(f"near-miss offsets (pred - label, frames) median {np.median(off) if off else None}, "
          f"histogram -15..+15: {hist}")

    print("\n== tune on all clips")
    cfg, best = descend(clips, a.passes)
    print("best:", best, "changes:", _diff(cfg))
    print("per clip:", {c.cid: score([c], cfg)["hit_f1"] for c in clips})
    off2 = offsets(clips, cfg)
    print(f"near-miss offsets after: median {np.median(off2) if off2 else None}")

    halves = (clips[0::2], clips[1::2])
    folds = []
    for k, (tr, te) in enumerate((halves, halves[::-1])):
        print(f"\n== fold {k}: tune on {[c.cid for c in tr]}, test on {[c.cid for c in te]}")
        fcfg, fbest = descend(tr, a.passes, log=False)
        r = {"train": [c.cid for c in tr], "test": [c.cid for c in te], "changes": _diff(fcfg),
             "train_score": fbest, "test_default": score(te, EventsConfig()), "test_tuned": score(te, fcfg),
             "test_all_tuned": score(te, cfg)}
        print(json.dumps(r, ensure_ascii=False))
        folds.append(r)
    if a.json:
        a.json.write_text(json.dumps({"defaults": base, "best": best, "changes": _diff(cfg), "folds": folds,
                                      "offsets_default": off, "offsets_best": off2}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
