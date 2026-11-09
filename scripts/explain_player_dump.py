"""Print, for sampled frames of a ``dump_player_candidates.py`` file, who the filters keep and who is tracked.

    python scripts/explain_player_dump.py DUMP.npz CALIBRATION.json [--every 30] [--start 0] [--end 1800]

For each sampled frame: every candidate the filters keep (foot on court in metres, player score, reasons,
box), people found only by the far-court second pass (src 1, not merged), and the tracker's players.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallyiq.contracts import CourtCalibration  # noqa: E402
from rallyiq.perception.players.filtering import FilterConfig, build_candidates  # noqa: E402
from rallyiq.perception.players.models import PersonDetection  # noqa: E402
from rallyiq.perception.players.pose import same_person  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("calibration")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1800)
    a = ap.parse_args()
    z = np.load(a.dump)
    cal = CourtCalibration.model_validate_json(Path(a.calibration).read_text())
    print("dropped runs:", list(z["dropped"]))
    tracks = sorted({(str(p), int(t)) for p, t in zip(z["trk_pid"], z["trk_track"])})
    for p, t in tracks:
        m = (z["trk_pid"] == p) & (z["trk_track"] == t)
        fr = z["trk_frame"][m]
        fc = z["trk_foot_court"][m]
        print(f"track {p}:{t} frames {fr.min()}-{fr.max()} n={m.sum()} median foot {np.round(np.nanmedian(fc, 0), 1)}")
    cfg = FilterConfig()
    for f in range(a.start, a.end + 1, a.every):
        seg = cal.segment_for(f)
        H = np.asarray(seg.homography, np.float64) if seg is not None and seg.homography is not None else None
        m0 = (z["frame"] == f) & (z["src"] == 0)
        dets = [PersonDetection(b.copy(), k.copy(), float(s)) for b, k, s in zip(z["bbox"][m0], z["kps"][m0], z["score"][m0])]
        cands = build_candidates(dets, H, (1920, 1080), cfg)
        print(f"--- frame {f}: {m0.sum()} detections, {len(cands)} candidates")
        for c in cands:
            fc = np.round(c.foot_court, 1) if c.foot_court is not None else None
            print(f"  cand foot {fc} score {c.player_score:.2f} det {c.det.score:.2f} {c.reasons} box {np.round(c.det.bbox).astype(int).tolist()}")
        m1 = (z["frame"] == f) & (z["src"] == 1)
        for b, k, s in zip(z["bbox"][m1], z["kps"][m1], z["score"][m1]):
            d = PersonDetection(b.copy(), k.copy(), float(s))
            if not any(same_person(d, e) for e in dets):
                print(f"  far-pass only: det {s:.2f} box {np.round(b).astype(int).tolist()}")
            elif not any(np.abs(d.bbox - e.bbox).max() < 3 for e in dets):
                hit = [np.round(e.bbox).astype(int).tolist() for e in dets if same_person(d, e)]
                print(f"  far-pass merged into full-frame box {hit[0]}: det {s:.2f} box {np.round(b).astype(int).tolist()}")
        mt = z["trk_frame"] == f
        for p, t, b, fc in zip(z["trk_pid"][mt], z["trk_track"][mt], z["trk_bbox"][mt], z["trk_foot_court"][mt]):
            print(f"  TRACK {p}:{t} foot {np.round(fc, 1)} box {np.round(b).astype(int).tolist()}")


if __name__ == "__main__":
    main()
