"""Score hit timing and hitter attribution of a finished run against hand labels, re-running only the events stage.

    # DJI_0009 singles (labels: events_labeled.json with P1 = far, P2 = near)
    python scripts/hitter_eval.py ~/rallyiq-data/eval_dji/run60/data/videos/<vid> \
        --labels runs/2026-11-08-dji0009-labels/events_labeled.json
    # 2026-03-02 doubles (labels: gt_0302_06.json; NG/NM/FG/FM, F = far side)
    python scripts/hitter_eval.py <video_dir> --labels notes/doubles/gt_0302_06.json
    # try settings
    python scripts/hitter_eval.py <video_dir> --labels ... --set hitter_scale=mixed --set same_side_min_gap_ms=600

Uses ``artifacts/{probe,players,shuttle,calibration}.json``; no model runs, a few seconds on CPU.
``--saved`` scores ``artifacts/events.json`` as it is instead of re-running.

Reported (labels are matched to hits by time, within ``--tol-ms``):
- hit recall / precision (precision only for singles labels, which list every hit of the labelled rallies);
- side right: near/far of the hitter among matched hits (``F`` labels count for side only);
- player right (doubles): the exact labelled player, through ``--map`` (pipeline id -> label);
- timing: median and mean of (predicted - labelled) ms;
- alternation: share of consecutive predicted hits in a rally whose side changes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import CourtCalibration, EventsResult, PlayerTracks, ShuttleTrack, VideoMeta  # noqa: E402
from rallyiq.events.adapters import (  # noqa: E402
    calibration_segments,
    events_to_contract,
    poses_from_contract,
    shuttle_from_contract,
)
from rallyiq.events.config import EventsConfig  # noqa: E402
from rallyiq.events.pipeline import off_court_views, segment_events  # noqa: E402

# Run 3 of own_20260302_06: pipeline ids of the labelled players (P3 was a spectator).
DEFAULT_DOUBLES_MAP = {"P1": "NG", "P2": "NM", "P4": "FG"}
NEAR = {"NG", "NM"}
FAR = {"FG", "FM", "F"}


def _load(path: Path, model):
    return model.model_validate_json(path.read_text()) if path.exists() else None


def parse_set(items: list[str]) -> dict[str, Any]:
    known = {f.name: f for f in fields(EventsConfig)}
    out: dict[str, Any] = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in known:
            raise SystemExit(f"unknown EventsConfig field {k}")
        cur = getattr(EventsConfig(), k)
        out[k] = (v.lower() in ("1", "true", "yes", "on")) if isinstance(cur, bool) else type(cur)(v)
    return out


def run_events(video_dir: Path, overrides: dict[str, Any]) -> tuple[EventsResult, Optional[PlayerTracks], VideoMeta]:
    art = video_dir / "artifacts"
    meta = _load(art / "probe.json", VideoMeta)
    players = _load(art / "players.json", PlayerTracks)
    shuttle = _load(art / "shuttle.json", ShuttleTrack)
    calib = _load(art / "calibration.json", CourtCalibration)
    if meta is None or shuttle is None:
        raise SystemExit(f"{art}: needs probe.json and shuttle.json")
    cfg = replace(EventsConfig(), **overrides)
    off = off_court_views(calibration_segments(calib), cfg.court_view_min_share) if cfg.mask_off_court_views else []
    rallies = segment_events(
        shuttle_from_contract(shuttle), poses_from_contract(players), cfg,
        video_end_ms=float(meta.duration_ms), off_court=off,
    )
    fmt = players.match_format if players is not None else "singles"
    return events_to_contract(meta.video_id, rallies, fmt), players, meta


def side_lookup(players: Optional[PlayerTracks]) -> dict[str, list[tuple[int, str]]]:
    """player id -> [(frame, side)] sorted, for the side of a hitter at a hit."""
    out: dict[str, list[tuple[int, str]]] = {}
    for fr in (players.frames if players else []):
        for p in fr.players:
            if p.side in ("near", "far"):
                out.setdefault(p.player_id, []).append((fr.frame, p.side))
    return out


def side_at(lookup: dict[str, list[tuple[int, str]]], pid: Optional[str], frame: int, hitter_court=None) -> Optional[str]:
    seq = lookup.get(pid or "")
    if seq:
        i = bisect.bisect_left(seq, (frame, ""))
        best = min((seq[j] for j in (i - 1, i) if 0 <= j < len(seq)), key=lambda s: abs(s[0] - frame))
        if abs(best[0] - frame) <= 60:
            return best[1]
    if hitter_court is not None:
        return "near" if hitter_court[1] < 6.7 else "far"
    return None


def load_labels(path: Path) -> tuple[str, list[dict[str, Any]], list[tuple[int, int]]]:
    """(kind, labels [{t_ms, side, player}], labelled spans in ms (singles) )."""
    d = json.loads(path.read_text())
    if "detections" in d:  # doubles review of detected hits
        labs = []
        for x in d["detections"]:
            lab = x["label"]
            if lab in ("U",):
                continue
            labs.append({
                "t_ms": x["t_ms"], "frame": x.get("contact_frame", x["frame"]), "label": lab,
                "hit": lab != "X", "side": "near" if lab in NEAR else ("far" if lab in FAR else None),
                "player": lab if lab in ("NG", "NM", "FG", "FM") else None, "conf": x.get("conf"),
            })
        return "doubles", labs, []
    # singles contract labels: P1 = far, P2 = near (DJI_0009 labels.py)
    labs = [{"t_ms": s["t_ms"], "frame": s["frame"], "label": s["hitter"], "hit": True,
             "side": {"P1": "far", "P2": "near"}.get(s["hitter"]), "player": None, "rally": s["rally_id"]}
            for s in d["shots"]]
    spans = [(r["start_ms"], r["end_ms"]) for r in d["rallies"]]
    return "singles", labs, spans


def match(pred: list[dict], labs: list[dict], tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by time difference."""
    pairs = sorted(
        ((abs(p["t_ms"] - l["t_ms"]), i, j) for i, p in enumerate(pred) for j, l in enumerate(labs)
         if abs(p["t_ms"] - l["t_ms"]) <= tol),
    )
    used_p, used_l, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_p or j in used_l:
            continue
        used_p.add(i)
        used_l.add(j)
        out.append((i, j))
    return out


def score(ev: EventsResult, players: Optional[PlayerTracks], labels: Path, tol: float, offset_ms: float,
          id_map: dict[str, str]) -> dict[str, Any]:
    kind, labs, spans = load_labels(labels)
    look = side_lookup(players)
    rally_of = {sid: r.rally_id for r in ev.rallies for sid in r.shot_ids}
    pred = []
    for s in ev.shots:
        pred.append({
            "t_ms": s.t_ms + offset_ms, "frame": s.frame, "hitter": s.hitter, "rally": rally_of.get(s.shot_id),
            "side": side_at(look, s.hitter, s.frame, s.hitter_court), "conf": s.confidence,
        })
    # alternation over all predicted hits
    alt_n = alt_ok = 0
    for a, b in zip(pred, pred[1:]):
        if a["rally"] and a["rally"] == b["rally"] and a["side"] and b["side"]:
            alt_n += 1
            alt_ok += a["side"] != b["side"]

    if kind == "singles":
        pad = tol
        in_span = [p for p in pred if any(s - pad <= p["t_ms"] <= e + pad for s, e in spans)]
        m = match(in_span, labs, tol)
        hit_labs = labs
        pr = in_span
    else:
        m = match(pred, labs, tol)
        hit_labs = [l for l in labs if l["hit"]]
        pr = pred
    tp = [(pr[i], labs[j]) for i, j in m if labs[j]["hit"]]
    fp_x = sum(1 for i, j in m if not labs[j]["hit"])  # matched a "not a hit" label
    side_n = sum(1 for p, l in tp if l["side"] and p["side"])
    side_ok = sum(1 for p, l in tp if l["side"] and p["side"] == l["side"])
    pl_n = sum(1 for p, l in tp if l["player"])
    pl_ok = sum(1 for p, l in tp if l["player"] and id_map.get(p["hitter"] or "") == l["player"])
    conf_tp = [(p, l) for p, l in tp if l.get("conf") in ("high", "med")]
    dts = [p["t_ms"] - l["t_ms"] for p, l in tp]
    # alternation inside labelled singles rallies
    out = {
        "kind": kind,
        "pred_hits": len(pred),
        "labels_hit": len(hit_labs),
        "matched_hits": len(tp),
        "recall": round(len(tp) / len(hit_labs), 3) if hit_labs else None,
        "side_right": f"{side_ok}/{side_n}",
        "side_acc": round(side_ok / side_n, 3) if side_n else None,
        "alternation": round(alt_ok / alt_n, 3) if alt_n else None,
        "dt_median_ms": round(statistics.median(dts)) if dts else None,
        "dt_mean_ms": round(statistics.mean(dts)) if dts else None,
    }
    if kind == "singles":
        out["precision"] = round(len(tp) / len(pr), 3) if pr else None
        out["pred_in_labelled_rallies"] = len(pr)
    else:
        out["matched_not_hit"] = fp_x
        out["player_right"] = f"{pl_ok}/{pl_n}"
        cs_n = sum(1 for p, l in conf_tp if l["side"] and p["side"])
        cs_ok = sum(1 for p, l in conf_tp if l["side"] and p["side"] == l["side"])
        out["side_right_confident"] = f"{cs_ok}/{cs_n}"
    out["errors"] = [
        {"t_ms": l["t_ms"], "label": l["label"], "pred": p["hitter"], "pred_side": p["side"], "dt": p["t_ms"] - l["t_ms"]}
        for p, l in tp if l["side"] and p["side"] != l["side"]
    ]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], help="EventsConfig override name=value")
    ap.add_argument("--saved", action="store_true", help="score artifacts/events.json without re-running")
    ap.add_argument("--tol-ms", type=float, default=200.0)
    ap.add_argument("--offset-ms", type=float, default=0.0, help="added to predicted t_ms (clip start in label time)")
    ap.add_argument("--map", default=None, help="doubles id map, e.g. P1=NG,P2=NM,P4=FG")
    ap.add_argument("--errors", action="store_true")
    a = ap.parse_args(argv)
    vd = a.video_dir.expanduser()
    if a.saved:
        art = vd / "artifacts"
        ev = _load(art / "events.json", EventsResult)
        players = _load(art / "players.json", PlayerTracks)
    else:
        ev, players, _ = run_events(vd, parse_set(a.set))
    id_map = dict(DEFAULT_DOUBLES_MAP)
    if a.map:
        id_map = dict(kv.split("=") for kv in a.map.split(","))
    res = score(ev, players, a.labels.expanduser(), a.tol_ms, a.offset_ms, id_map)
    errs = res.pop("errors")
    print(json.dumps(res, ensure_ascii=False))
    if a.errors:
        for e in errs:
            print("  side wrong:" if e["pred_side"] else "  side unknown:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
