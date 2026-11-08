"""Swing evidence of every hit of a finished run, to choose the events ``swing_veto`` rule.

    python scripts/diag_hits.py --run ~/rallyiq-data/own/run_0529_court14 --clip own0529_0006

For each detected hit: frame, hitter and side, whether it is the rally's first hit, the
shuttle's image position, the pose model's best ``p_swing`` for the hitter's side within
``--window`` ms and the fastest wrist there (body spans per second). Join it with what a
reviewer saw to pick ``swing_veto`` / ``swing_veto_keep_wrist_v`` / ``swing_veto_keep_serve``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts import PlayerTracks
from rallyiq.events.adapters import poses_from_contract
from rallyiq.events.pose_hits import get_model, side_series, swing_evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--window", type=float, default=250.0)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    a = ap.parse_args(argv)

    run = a.run.expanduser()
    res = next(r for r in json.loads((run / "results.json").read_text()) if r["clip_id"] == a.clip)
    art = run / "data" / "videos" / res["video_id"] / "artifacts"
    players = PlayerTracks.model_validate_json((art / "players.json").read_text())
    events = json.loads((art / "events.json").read_text())
    poses = poses_from_contract(players)
    series = side_series(poses)
    model = get_model(a.model)
    if model is None:
        print("no pose model", file=sys.stderr)
        return 1
    fr = [p.frame for p in poses[:50]]
    ts = [p.t_ms for p in poses[:50]]
    frame_ms = (ts[-1] - ts[0]) / (fr[-1] - fr[0]) if len(fr) > 1 and fr[-1] > fr[0] else 1000 / 30

    first = {}
    for s in events.get("shots", []):
        first.setdefault(s.get("rally_id"), s["frame"])
    rows = []
    print("frame  hitter side  first  shuttle_px       p_swing  wrist_v_max")
    for s in events.get("shots", []):
        side = None
        for ser in series.values():
            b = ser.near(float(s["t_ms"]), 120.0)
            if b is not None and b.player_id == s.get("hitter"):
                side = ser.side
        p, wv = swing_evidence(series, float(s["t_ms"]), side, model, a.window, frame_ms)
        px = s.get("shuttle_px")
        row = {"frame": s["frame"], "hitter": s.get("hitter"), "side": side,
               "first": first.get(s.get("rally_id")) == s["frame"],
               "shuttle_px": [round(v) for v in px] if px else None,
               "p_swing": round(p, 3), "wrist_v_max": round(wv, 2)}
        rows.append(row)
        print(f"{row['frame']:>5}  {row['hitter'] or '-':>6} {side or '-':>4}  {'yes' if row['first'] else '-':>5}  "
              f"{str(row['shuttle_px']):<15}  {row['p_swing']:7.3f}  {row['wrist_v_max']:11.2f}")
    if a.json:
        a.json.expanduser().write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
