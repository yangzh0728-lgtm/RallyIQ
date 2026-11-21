# Hit scorer v3 (train_v3, 2026-11-21)

Bundled as `hit_scorer.json`; the previous model is kept as `hit_scorer_v2.json` (+ `.md`).

- Model: mean of 5 one-hidden-layer MLPs (16 tanh units, l2 1e-2, seeds 0-4), features = the 27 v2 features +
  hitter-relative shuttle motion (`toward_in`, `away_out`), `log_speed_ratio`, `wrist_dist`, `track_density`, and
  the pose-sequence swing probability of the best player's side / the other side within 250 ms (`p_swing_hitter`,
  `p_swing_other`, from the bundled `pose_hits.json`, which was trained on ShuttleSet train only).
- Training data: ShuttleSet train v3 (88 clips, 3025 candidates) + sam's own videos inside fully labelled windows
  (13 videos: qa6 = hits_truth_v2 on the six own-v12 runs, am7 = reviewed pre-labels `amateur_v1.json` on seven runs;
  ~1000 candidates, weight 3 per row; dead-ball pickups/passes are negatives).
- Thresholds: the v2 candidate/rally parameters (window_ms 100, min_shuttle_conf 0.3, max_hitter_dist_rel 1.2,
  static_ms 400, min_rally_ms 700, hit_frame_offset +1) and min_candidate_conf 0.1, keep_threshold 0.3, ok 0.7.
- Trained with `scripts/train_hits_v3.py fit --cache <cands> --features v2+all --am-weight 3 --seeds 5`.

| set (metric) | rules | v2 | **v3** |
|---|---|---|---|
| ShuttleSet test v3, 14 clips (hit F1, ±3 frames, same side) | 64.9 | 77.0 | **78.7** |
| ShuttleSet test: same side twice in a row (labels 0%) | 16.3% | 11.3% | **5.4%** |
| own videos qa6, 126 blind-labelled hits (F1, ±0.25 s; leave-one-video-out) | 0.493 | 0.556 | **0.600** |
| qa6 with the unrefined 8 fps label times matched up to 0.4 s | 0.568 | 0.648 | **0.686** |
| own videos am7, 164 reviewed pre-labels (F1; labels come from v2's own detections) | 0.734 | **0.839** | 0.822 |
| all own videos (F1) | 0.630 | 0.723 | **0.725** |
| dead-time false hits (detections on a labelled pickup/pass) | 33 | 45 | **24** |

Own-video numbers use the own-footage settings (swing_veto 0.1, swing_veto_keep_exchange_ms 2700,
min_no_hit_rally_ms 2500, landed_ms 0). Paired bootstrap over videos, v2 -> v3: qa6 +0.044 (A5) to +0.054 (single
seed), 95% CI about [0.00, +0.10]; am7 -0.017, CI [-0.11, +0.05]. Seed sets 0-4 / 5-9 / 10-14 give all-video F1
0.725 / 0.722 / 0.725. Cost: the swing features run the pose model per candidate: the events stage of a 14 min
video takes 12.7 s instead of 1.8 s (only when the scorer uses them). Full log: train_v3/LOG.md, REPORT.md.
