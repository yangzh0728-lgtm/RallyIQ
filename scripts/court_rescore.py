"""Re-run only court calibration on the public-eval clips and compare with the saved predictions.

    python scripts/court_rescore.py                 # ~/rallyiq-data/public/shuttleset

For each clip with a label and a prediction: calibrate clips/<clip>.mp4 with the current code,
swap the court entries of preds/<clip>.json for the new ones (in memory; nothing is written
except court_rescore.json), and print court success / corner error before and after, plus the
symmetry breakdown from public_diag (share within 10 px as labelled vs under any symmetry).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import public_diag as D  # noqa: E402

from evaluation import from_contracts, schema  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from rallyiq.perception.calibration.stage import to_contract_segment  # noqa: E402
from rallyiq.perception.calibration.tracker import calibrate_video  # noqa: E402


def court_entries(clip: Path, pred: dict) -> list[dict]:
    fps = float(pred["video"]["fps"])
    n = len(pred["court"]) or int(round((pred["range_ms"][1] - pred["range_ms"][0]) * fps / 1000))
    segs = calibrate_video(str(clip), frame_count=n)
    per_frame: list = [None] * n
    for s in segs:
        c = from_contracts._corners(to_contract_segment(s.start_frame, s.end_frame, s.calibration).model_dump())
        for f in range(max(0, s.start_frame), min(n - 1, s.end_frame) + 1):
            per_frame[f] = c
    t0 = pred["range_ms"][0]
    return [{"t_ms": int(round(t0 + f * 1000.0 / fps)), "visible": c is not None, "corners": c}
            for f, c in enumerate(per_frame)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / "rallyiq-data" / "public" / "shuttleset")
    a = ap.parse_args(argv)
    d = a.dir.expanduser()
    rows, before, after = [], M.CourtStats(), M.CourtStats()
    for lp in sorted((d / "labels").glob("*.json")):
        if lp.name.startswith("._"):  # macOS metadata files on exFAT
            continue
        pp, clip = d / "preds" / f"{lp.stem}.json", d / "clips" / f"{lp.stem}.mp4"
        if not (pp.exists() and clip.exists()):
            continue
        gt, old = schema.load(lp), json.loads(pp.read_text(encoding="utf-8"))
        new = {**old, "court": court_entries(clip, old)}
        b, n = M.eval_court(gt, old), M.eval_court(gt, new)
        before.merge(b)
        after.merge(n)
        r = {"clip": lp.stem, "before": b.summary(), "after": n.summary(),
             "sym_before": D.court(gt, old), "sym_after": D.court(gt, new)}
        rows.append(r)
        print(json.dumps({"clip": r["clip"],
                          "success": [b.summary()["success_rate"], n.summary()["success_rate"]],
                          "median_err": [b.summary()["median_corner_err_px"], n.summary()["median_corner_err_px"]],
                          "mean_err": [b.summary()["mean_corner_err_px"], n.summary()["mean_corner_err_px"]],
                          "ok_any_sym_after": r["sym_after"].get("share_ok_any_sym")}), flush=True)
    tot = {"before": before.summary(), "after": after.summary()}
    print("overall:", json.dumps(tot))
    (d / "court_rescore.json").write_text(json.dumps({"overall": tot, "clips": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
