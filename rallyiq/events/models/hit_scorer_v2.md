# Hit scorer training report

- event parameters: {"window_ms": 100.0, "min_shuttle_conf": 0.3, "max_hitter_dist_rel": 1.2, "static_ms": 400.0, "min_rally_ms": 700.0, "hit_frame_offset": 1}
- clips: 88 from 37 matches; labelled hits 1738
- candidates: 3020 (1474 positive); candidate recall ceiling (±3 frames): 84.8%

## Cross-validated hit accuracy on the training matches (grouped by match)

| scorer | precision | recall | F1 | F1 (any side) | ok hits | ok precision |
|---|---|---|---|---|---|---|
| rules (hit frame offset +1) | 0.617 | 0.710 | 0.660 | 0.742 | 986 | 0.845 |
| rules, re-thresholded (keep=0.55, min=0.4) | 0.644 | 0.697 | 0.670 | 0.740 | 975 | 0.845 |
| learned (hidden=16, l2=0.01), out-of-fold | 0.830 | 0.696 | 0.757 | 0.829 | 813 | 0.941 |
| learned + side model (threshold 1.01), out-of-fold | 0.830 | 0.696 | 0.757 | 0.829 | 813 | 0.941 |

Side model not used: by side among correctly timed hits right 1208, missing 78, wrong 39

Chosen thresholds: {"min_candidate_conf": 0.3, "keep_threshold": 0.65, "ok_threshold": 0.7}

Hits found at the right time, by near/far side (learned): right 1208, missing 78, wrong 39; rules: right 1233, missing 93, wrong 62

Model search (out-of-fold log-loss): h16/l2=0.01: 0.2462, h8/l2=0.01: 0.2467, h8/l2=0.001: 0.2809, h16/l2=0.001: 0.2834, h0/l2=0.001: 0.2845, h0/l2=0.0001: 0.2854, h0/l2=0.01: 0.2881, h8/l2=0.0001: 0.3732, h16/l2=0.0001: 0.5266

## Held-out ShuttleSet matches (6 matches, 14 clips, never trained on)

| variant | hit P | hit R | hit F1 | F1 any side | rally F1 |
|---|---|---|---|---|---|
| rules | 61.4% | 63.0% | 62.2% | 71.8% | 82.1% |
| learned v1e (35 clips) | 84.1% | 61.9% | 71.3% | 79.4% | 82.1% |
| **this model (v2)** | 84.1% | 65.8% | **73.8%** | 80.3% | **85.2%** |
| v2 with default event parameters | 75.7% | 64.2% | 69.5% | 82.9% | 82.1% |
