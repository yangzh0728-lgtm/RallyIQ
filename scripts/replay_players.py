"""Replay player filtering + tracking from a ``dump_player_candidates.py`` file, without the pose model or video.

    python scripts/replay_players.py DUMP.npz CALIBRATION.json [--match-format doubles] [--every 150]

Runs ``track_players`` with the code on PYTHONPATH on the recorded detections (and torso colours), then prints
a per-track summary, coverage numbers, and the same numbers for the tracks recorded in the dump (the code that
made the dump), so two versions of the tracker can be compared on the same frames.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallyiq.contracts import CourtCalibration  # noqa: E402
from rallyiq.perception.players import runner as R  # noqa: E402
from rallyiq.perception.players.geometry import PlayArea  # noqa: E402
from rallyiq.perception.players.models import NET_Y_M, PersonDetection  # noqa: E402

LINES = PlayArea(margin_side=0.0, margin_end=0.0)


def coverage(rows: list[tuple[int, str, np.ndarray]], frames: list[int]) -> dict[str, float]:
    """rows: (frame, player_id, foot_court). Share of frames with 4 players, with two on each half, with every
    player's team on its own half (team A near), and with someone tracked standing >= 0.3 m outside the lines."""
    by = defaultdict(list)
    for f, pid, fc in rows:
        by[f].append((pid, fc))
    n = max(len(frames), 1)
    four = sum(len(by[f]) == 4 for f in frames)
    halves = sum(
        sum(fc[1] < NET_Y_M for _, fc in by[f]) == 2 and sum(fc[1] >= NET_Y_M for _, fc in by[f]) == 2 for f in frames
    )
    teams = sum(
        bool(by[f]) and all((pid in ("P1", "P3")) == (fc[1] < NET_Y_M) for pid, fc in by[f]) for f in frames
    )
    off = sum(any(LINES.distance_outside(fc) >= 0.3 for _, fc in by[f]) for f in frames)
    out = {"4 players": four, "2+2 halves": halves, "teams by half": teams, "someone off court": off}
    return {k: round(float(v) / n, 3) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("calibration")
    ap.add_argument("--match-format", default="doubles")
    ap.add_argument("--every", type=int, default=150, help="print the tracked players every n frames (0: never)")
    ap.add_argument("--max-cands", type=int, default=None, help="candidates kept per frame (filter and tracker)")
    ap.add_argument("--still-smooth-s", type=float, default=None, help="tracker still_smooth_s (0: raw position)")
    a = ap.parse_args()
    z = np.load(a.dump)
    cal = CourtCalibration.model_validate_json(Path(a.calibration).read_text())
    m0 = z["src"] == 0 if "src" in z else np.ones(len(z["frame"]), bool)
    frames = sorted(set(int(f) for f in z["frame"][m0]) | set(int(f) for f in z["trk_frame"]))
    fps = float(z["fps"])
    per_frame: dict[int, list[PersonDetection]] = defaultdict(list)
    hist: dict[int, np.ndarray] = {}
    for b, k, s, h, f in zip(z["bbox"][m0], z["kps"][m0], z["score"][m0], z["hist"][m0], z["frame"][m0]):
        d = PersonDetection(b.copy(), k.copy(), float(s))
        per_frame[int(f)].append(d)
        hist[id(d)] = None if np.isnan(h).any() else h

    img = np.zeros((1080, 1920, 3), np.uint8)

    class Replay:
        def __call__(self, image, far_roi=None):
            return list(per_frame.get(cur["f"], []))

    cur = {"f": 0}

    def gen():
        for f in frames:
            cur["f"] = f
            yield f, f * 1000.0 / fps, img

    def homography_for(frame: int):
        seg = cal.segment_for(frame)
        h = seg.homography if seg is not None and seg.status != "undeterminable" else None
        return np.asarray(h, dtype=np.float64) if h is not None else None

    R.colour_histogram = lambda image, det: hist.get(id(det))
    cfg = R.PlayerTrackingConfig(match_format=a.match_format)
    cfg.tracker.fps = fps
    if a.max_cands is not None:
        cfg.filter.max_candidates = a.max_cands
        cfg.tracker.max_doubles_candidates = a.max_cands
    if a.still_smooth_s is not None and hasattr(cfg.tracker, "still_smooth_s"):
        cfg.tracker.still_smooth_s = a.still_smooth_s
    res = R.track_players(gen(), Replay(), homography_for, cfg)

    new_rows = [(r.frame, p.player_id, np.asarray(p.foot_court)) for r in res for p in r.players
                if p.foot_court is not None]
    tracks = defaultdict(list)
    for r in res:
        for p in r.players:
            if p.foot_court is not None:
                tracks[(p.player_id, p.track_id)].append((r.frame, np.asarray(p.foot_court)))
    print("replayed tracks:")
    for (pid, tid), v in sorted(tracks.items()):
        fr = [f for f, _ in v]
        print(f"  {pid}:{tid} frames {min(fr)}-{max(fr)} n={len(v)} median foot {np.round(np.median([c for _, c in v], 0), 1)}")
    if a.every:
        for r in res:
            if r.frame % a.every == 0:
                print(f"  frame {r.frame}: " + ", ".join(
                    f"{p.player_id}:{p.track_id} {[round(float(x), 1) for x in p.foot_court]}" for p in r.players))
    old_rows = [(int(f), str(p), fc) for f, p, fc in zip(z["trk_frame"], z["trk_pid"], z["trk_foot_court"])
                if p and np.all(np.isfinite(fc))]
    print("recorded (dump, before off-court runs are dropped):", coverage(old_rows, frames))
    print("replayed (now, final output):                    ", coverage(new_rows, frames))


if __name__ == "__main__":
    main()
