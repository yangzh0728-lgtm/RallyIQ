"""Train the shot attribute heads (rallyiq/strokes/detail.py) and measure them on held-out matches.

    pip install -e .[strokes-ml]          # scikit-learn, only for training

    python scripts/learn_stroke_attrs.py \\
        --data ~/rallyiq-data/public/shuttleset_train ~/rallyiq-data/public/shuttleset_train_v3 \\
        --test ~/rallyiq-data/public/shuttleset_test_v3
    #   -> rallyiq/strokes/models/stroke_attrs.json (+ .md report)

Every predicted shot within --tol frames of a labelled ShuttleSet stroke (one to one, nearest
first) becomes an example. Labels, all from the ShuttleSet set CSVs:

* ``hand``: ``backhand`` = 1 -> backhand, else ``aroundhead`` = 1 -> around_head, else forehand;
* ``contact``: ``hit_height`` 1 -> high (above the net), 2 -> low;
* ``direction``: hit and landing points on the same or opposite halves of the centre line
  (image coordinates, centre line interpolated between the labelled court corners); shots
  within 8 % of the court width of the centre line are left out (neither straight nor cross).

Pace reference speeds (court speed hitter -> receiver, 33rd / 67th percentile per family) come
from the training shots. Only the events / strokes evidence is recomputed from cached artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rallyiq.contracts import EventsResult, PlayerTracks, ShuttleTrack  # noqa: E402
from rallyiq.gbt import export_hgb  # noqa: E402
from rallyiq.strokes import detail as DT  # noqa: E402
from rallyiq.strokes import shot_features  # noqa: E402
from rallyiq.strokes.taxonomy import from_shuttleset  # noqa: E402

HEADS = ("hand", "contact", "direction")
HEAD_FEATURES = {"hand": DT.HAND_HEAD_FEATURES}  # heads not listed use every feature
CENTRE_MARGIN = 0.08


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _centre(court: list[dict], y: float) -> tuple[float, float] | None:
    if not court or "corners" not in court[0]:
        return None
    fl, fr, nr, nl = court[0]["corners"]
    if nl[1] == fl[1]:
        return None
    t = (y - fl[1]) / (nl[1] - fl[1])
    left, right = fl[0] + t * (nl[0] - fl[0]), fr[0] + t * (nr[0] - fr[0])
    return (left + right) / 2.0, right - left


def labels_of(row: dict[str, str], court: list[dict]) -> dict[str, str | None]:
    bh, ah, hh = _f(row.get("backhand")), _f(row.get("aroundhead")), _f(row.get("hit_height"))
    out: dict[str, str | None] = {
        "hand": "backhand" if bh == 1 else "around_head" if ah == 1 else "forehand",
        "contact": {1: "high", 2: "low"}.get(int(hh)) if hh else None,
        "direction": None,
    }
    hx, hy, lx, ly = (_f(row.get(k)) for k in ("hit_x", "hit_y", "landing_x", "landing_y"))
    if None not in (hx, hy, lx, ly):
        c1, c2 = _centre(court, hy), _centre(court, ly)
        if c1 and c2 and c1[1] > 0 and c2[1] > 0:
            a, b = (hx - c1[0]) / c1[1], (lx - c2[0]) / c2[1]
            if abs(a) >= CENTRE_MARGIN and abs(b) >= CENTRE_MARGIN:
                out["direction"] = "cross" if a * b < 0 else "straight"
    return out


def load(data: Path, set_dir: Path, tol_frames: float) -> list[dict]:
    from evaluation import schema

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    out = []
    for cid, e in sorted(index.items()):
        if e.get("state") != "succeeded":
            continue
        art, lp = Path(e["artifacts"]), data / "labels" / f"{cid}.json"
        csv_path = set_dir / e["video"] / f"{cid.split('_', 1)[1]}.csv"
        if not lp.exists() or not (art / "events.json").exists() or not csv_path.exists():
            continue
        label = schema.load(lp)
        r0, r1 = label["range_ms"]
        lfps = float(e["label_fps"])
        rows = []
        with csv_path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                fr = _f(r.get("frame_num"))
                if fr is not None and r0 <= fr * 1000.0 / lfps <= r1:
                    rows.append((fr * 1000.0 / lfps, r))
        fps = float(json.loads((art / "probe.json").read_text(encoding="utf-8"))["fps"])
        ev = EventsResult.model_validate_json((art / "events.json").read_text(encoding="utf-8"))
        sh = ShuttleTrack.model_validate_json((art / "shuttle.json").read_text(encoding="utf-8"))
        pp = art / "players.json"
        pl = PlayerTracks.model_validate_json(pp.read_text(encoding="utf-8")) if pp.exists() else None
        feats = shot_features(ev, sh, pl, fps)
        X = DT.attr_matrix(ev, sh, pl, feats, fps)
        t_of = {s.shot_id: r0 + s.frame * 1000.0 / fps for s in ev.shots}
        tol = tol_frames * 1000.0 / fps + 1e-6
        pairs = sorted((abs(t_of[f.shot_id] - t), i, j) for i, f in enumerate(feats)
                       for j, (t, _) in enumerate(rows) if abs(t_of[f.shot_id] - t) <= tol)
        ui, uj = set(), set()
        for _, i, j in pairs:
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            row = rows[j][1]
            out.append({"group": str(e.get("match_id", cid)), "x": X[i], "speed": feats[i].court_speed_mps,
                        "type": from_shuttleset(row.get("type", ""))[0], **labels_of(row, label.get("court"))})
        print(f"{data.name}/{cid}: {len(ui)}/{len(rows)} labelled strokes matched", flush=True)
    return out


def _cols(head: str) -> list[int]:
    names = DT.attr_names()
    return [names.index(n) for n in HEAD_FEATURES.get(head, names)]


def fit(rows: list[dict], head: str):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rs = [r for r in rows if r[head] is not None]
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=10,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0).fit(
        np.vstack([r["x"] for r in rs])[:, _cols(head)], [r[head] for r in rs])


def score(model, rows: list[dict], head: str) -> dict:
    rs = [r for r in rows if r[head] is not None]
    if not rs:
        return {"n": 0}
    y = np.asarray([r[head] for r in rs], dtype=object)
    p = model.predict(np.vstack([r["x"] for r in rs])[:, _cols(head)])
    classes = sorted(set(y))
    recall = {c: float(np.mean(p[y == c] == c)) for c in classes}
    return {"n": len(y), "accuracy": float(np.mean(p == y)), "balanced": float(np.mean(list(recall.values()))),
            "majority": Counter(y).most_common(1)[0][1] / len(y), "recall": recall, "counts": dict(Counter(y))}


def pace_reference(rows: list[dict]) -> dict[str, list[float]]:
    by: dict[str, list[float]] = {}
    for r in rows:
        fam = DT.FAMILY.get(r["type"] or "", r["type"])
        if fam and r["speed"] is not None and np.isfinite(r["speed"]):
            by.setdefault(fam, []).append(float(r["speed"]))
    return {k: [round(float(np.percentile(v, 33)), 2), round(float(np.percentile(v, 67)), 2)]
            for k, v in sorted(by.items()) if len(v) >= 15}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="training clips (build_train_set.py output)")
    ap.add_argument("--test", type=Path, nargs="*", default=[], help="held-out clips, scored only")
    ap.add_argument("--shuttleset", type=Path, help="ShuttleSet/set dir (default: the public_eval.py checkout)")
    ap.add_argument("--tol", type=float, default=3)
    ap.add_argument("--out", type=Path, default=DT.BUNDLED)
    a = ap.parse_args(argv)
    import public_eval

    set_dir = public_eval.shuttleset_dir(a.shuttleset)
    train = [r for d in a.data for r in load(d.expanduser(), set_dir, a.tol)]
    test = [r for d in a.test for r in load(d.expanduser(), set_dir, a.tol)]
    models = {h: fit(train, h) for h in HEADS}
    res = {h: score(models[h], test, h) for h in HEADS} if test else {}
    spec = {"kind": "stroke_attrs", "name": DT.ATTRS_NAME, "features": list(DT.attr_names()),
            "heads": {h: export_hgb(m) for h, m in models.items()},
            "head_features": {h: list(f) for h, f in HEAD_FEATURES.items()}, "pace": pace_reference(train),
            "meta": {"trained_on": {"shots": len(train), "matches": len({r["group"] for r in train})},
                     "test": {h: {k: v for k, v in s.items() if k != "recall"} for h, s in res.items()}}}
    out = a.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    lines = [f"# Shot attribute model ({DT.ATTRS_NAME})", "",
             f"- trained on {len(train)} shots from {spec['meta']['trained_on']['matches']} ShuttleSet matches",
             f"- held out: {len(test)} shots from {len({r['group'] for r in test})} other matches", "",
             "| attribute | n (test) | accuracy | balanced accuracy | always-majority | recall per class |",
             "|---|---|---|---|---|---|"]
    for h, s in res.items():
        if s.get("n"):
            rec = ", ".join(f"{c} {v:.2f} (n {s['counts'][c]})" for c, v in s["recall"].items())
            lines.append(f"| {h} | {s['n']} | {s['accuracy']:.3f} | {s['balanced']:.3f} | {s['majority']:.3f} | {rec} |")
    lines += ["", f"Pace reference speeds (m/s, 33rd / 67th percentile per family): {spec['pace']}", ""]
    md = "\n".join(lines)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
