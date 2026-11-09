"""Save every person the players stage sees, so tracking can be replayed without the pose model.

Runs the same pose pass as the ``players`` stage (RTMO, far-court boost, court
from ``calibration.json``) and writes to one ``.npz``:

* every detection handed to the filters (``src`` 0): box, COCO-17 keypoints,
  score and torso colour histogram;
* every person the far-court second pass found before it was merged with the
  full-frame pass (``src`` 1), so a far player lost in the merge shows up;
* the tracker output before the post-processing that drops off-court runs
  (``trk_*``), and which (player, track) runs that step dropped.

Tracker and filter changes can then be tested on a machine without the weights
or the video::

    python scripts/dump_player_candidates.py VIDEO CALIBRATION.json OUT.npz [--max-frame N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallyiq.contracts import CourtCalibration  # noqa: E402
from rallyiq.perception.players import runner as R  # noqa: E402
from rallyiq.perception.players.appearance import H_BINS, S_BINS, V_BINS, colour_histogram  # noqa: E402
from rallyiq.perception.players.pose import RTMOOnnx, find_pose_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("calibration")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frame", type=int, default=None, help="stop after this frame")
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--match-format", default="doubles")
    a = ap.parse_args()

    cal = CourtCalibration.model_validate_json(Path(a.calibration).read_text())
    est = RTMOOnnx(find_pose_model(a.pose_model))
    n_hist = H_BINS * S_BINS * V_BINS
    rows: dict[str, list] = {k: [] for k in ("frame", "bbox", "kps", "score", "hist", "src")}
    cur: dict = {}

    def record(dets, src: int) -> None:
        for d in dets:
            h = colour_histogram(cur["img"], d) if src == 0 else None
            rows["frame"].append(cur["frame"])
            rows["bbox"].append(np.asarray(d.bbox, np.float32))
            rows["kps"].append(np.asarray(d.keypoints, np.float32))
            rows["score"].append(d.score)
            rows["hist"].append(h if h is not None else np.full(n_hist, np.nan, np.float32))
            rows["src"].append(src)

    def homography_for(frame: int):
        seg = cal.segment_for(frame)
        h = seg.homography if seg is not None and seg.status != "undeterminable" else None
        return np.asarray(h, dtype=np.float64) if h is not None else None

    def frames():
        t0 = time.time()
        for i, t_ms, img in R.iter_video(a.video, a.stride):
            if a.max_frame is not None and i > a.max_frame:
                break
            cur["frame"], cur["img"] = i, img
            if i % 300 == 0:
                print(f"frame {i}  {time.time() - t0:.0f}s", flush=True)
            yield i, t_ms, img

    real_build = R.build_candidates

    def recording_build(dets, *args, **kw):
        record(dets, 0)
        return real_build(dets, *args, **kw)

    real_roi = RTMOOnnx._infer_roi

    def recording_roi(self, img, roi, full):
        x1, y1, x2, y2 = (int(v) for v in roi[:4])
        crop = img[y1:y2, x1:x2]
        raw = self._infer(crop) if crop.size else []
        for d in raw:
            d.bbox[[0, 2]] += x1
            d.bbox[[1, 3]] += y1
            d.keypoints[:, 0] += x1
            d.keypoints[:, 1] += y1
        record(raw, 1)
        return real_roi(self, img, roi, full)

    trk: dict[str, list] = {k: [] for k in ("frame", "pid", "track", "bbox", "foot_court")}
    dropped: list[tuple[str, int]] = []
    real_drop = R.drop_offcourt_runs

    def recording_drop(out, *args, **kw):
        before = {(p.player_id, p.track_id) for f in out for p in f.players}
        for f in out:
            for p in f.players:
                trk["frame"].append(f.frame)
                trk["pid"].append(p.player_id or "")
                trk["track"].append(p.track_id)
                trk["bbox"].append(np.asarray(p.bbox, np.float32))
                fc = p.foot_court
                trk["foot_court"].append(np.asarray(fc, np.float32) if fc is not None else np.full(2, np.nan, np.float32))
        n = real_drop(out, *args, **kw)
        after = {(p.player_id, p.track_id) for f in out for p in f.players}
        dropped.extend(sorted(before - after, key=str))
        return n

    R.build_candidates = recording_build
    RTMOOnnx._infer_roi = recording_roi
    R.drop_offcourt_runs = recording_drop
    cfg = R.PlayerTrackingConfig(stride=a.stride, match_format=a.match_format)
    cfg.tracker.fps = R.video_fps(a.video)
    R.track_players(frames(), est, homography_for, cfg)
    np.savez_compressed(
        a.out,
        frame=np.asarray(rows["frame"], np.int32),
        bbox=np.asarray(rows["bbox"], np.float32).reshape(-1, 4),
        kps=np.asarray(rows["kps"], np.float32).reshape(-1, 17, 3),
        score=np.asarray(rows["score"], np.float32),
        hist=np.asarray(rows["hist"], np.float32).reshape(-1, n_hist),
        src=np.asarray(rows["src"], np.int8),
        trk_frame=np.asarray(trk["frame"], np.int32),
        trk_pid=np.asarray(trk["pid"]),
        trk_track=np.asarray(trk["track"], np.int32),
        trk_bbox=np.asarray(trk["bbox"], np.float32).reshape(-1, 4),
        trk_foot_court=np.asarray(trk["foot_court"], np.float32).reshape(-1, 2),
        dropped=np.asarray([f"{p}:{t}" for p, t in dropped]),
        fps=np.float64(cfg.tracker.fps),
        stride=np.int32(a.stride),
    )
    print(f"{len(rows['frame'])} detections, dropped runs {dropped} -> {a.out}")


if __name__ == "__main__":
    main()
