# train_v3 log (autonomous run, started 2026-11-21 19:10 UTC)

## Setup
- Worktree `repo/` on branch claude/train-v3 from origin/claude/project-thread-ah2yye, then merged origin/main
  (hitter_eval.py, doubles hitter fixes) and origin/claude/project-thread-lfd1cx (swing_veto*, min_no_hit_rally_ms, landed_ms,
  score_hits.py). Both merges clean; tests/events 67 passed.
- hits_truth_v2.json not found on this Mac (no project-file tool here). Using qa_rallies/hits_truth.json: 153 sure hits
  (33+26+17+36+38+3), the same count the brief gives for v2. Asked parent to relay v2 if it differs.
- DJI_0009 events_labeled.json not found anywhere on the Mac or T7 (searched / and mdfind). Item 4 skipped unless relayed.
- CPU: 14 cores -> all jobs nice -n 10, <= 7 threads.
- hits_truth_v2.json relayed by the parent (md5 265f765f... ok) -> inputs/. Positives = sure & in_play True (126), negatives =
  in_play False (13, all 0620), ignored = unsure or in_play None. Windows from hits_truth.json.
- Inputs copied: ShuttleSet train/test v3 artifacts+labels (index paths rewritten to local), own-v12 artifacts (6 videos),
  both pre-label folders, and H.264 clips of every labelled window +-3 s (inputs/videos, index.json).

## Baselines (2026-11-21 ~19:40)
ShuttleSet test v3 (14 clips, learn_hits eval, +-3 frames same side): rules F1 64.9 (P 64.7 R 65.0); bundled v2 F1 **77.0** (P 87.2 R 68.9), rally F1 85.2.

Amateur (scripts/amateur_hits.py, tol 0.25 s, base settings swing_veto 0.1 / keep_exchange 2700 / min_no_hit_rally 2500 / landed 0;
am7 = pre-review statuses, v0):
| set | rules F1 | v2 F1 | v2 P / R | v2 dead FP | v2 side | v2 hitter |
|---|---|---|---|---|---|---|
| qa6 (126 blind-labelled hits) | 0.493 | **0.556** | 0.541 / 0.571 | 9 | 65/72 | - |
| am7 (177 hits, v0) | 0.750 | **0.873** | 0.808 / 0.949 | 24 | 164/167 | 161/165 |
Caveat: am7 labels were built from the v2 detector's own detections (+4 added), so its recall is optimistic; qa6 is the
unbiased set (labelled blind from 8 fps sheets). Decisions weight qa6 first.

## A. Labels (done ~20:55)
- Reviewed all 108 unsure pre-labels from per-hit sheets (overview at the hit, 4 hitter crops at -0.12/0/+0.12/+0.24 s,
  overview at the next hit), review/sheets/. Decisions with a one-line reason each: labels/review_decisions.txt.
- Result: 13 not_a_hit (dead-ball pickups/passes: 0616 x6, 0239 x2, 0310 x2, 0302_07, 0241, 0622), 95 real hits
  (5 fully sure, 90 with hitter or stroke unconfirmed). Hitter changed on 6 (0310 14.08 P2->P4; 0622 376.48 P1->P4;
  0241 284.18/285.17/286.77 -> untracked partner (null); 0241 138.42 kept P4 but unsure). Stroke changed/cleared on ~35
  (e.g. serves typed drop/block/push -> short_serve x5; lunging "drops" -> lift/null; 0302_07 369.57 drop -> smash).
- labels/amateur_v1.json (scripts/build_amateur_v1.py): 204 entries, 61 reviewed_sure, 103 reviewed_unsure, 40 not_a_hit.
  Review sheets with the v1 label written on: review/v1_reviewed/<video>_NN.jpg.
- Amateur eval on v1 labels: rules qa6 0.493 / am7 0.734; v2 qa6 0.556 / am7 0.839 (am7 dead FP 36 for v2 now that the
  13 dead-ball detections are negatives).

## B. Hit scorer v3 — experiments
Candidates cached per min_shuttle_conf (ckpt/cands, ckpt/cands2 with swing features). scripts/train_hits_v3.py.
CV = amateur leave-one-video-out (13 videos; model trained on all ShuttleSet train + other 12) + ShuttleSet train
5-fold grouped by match (trained on other folds + all amateur). Thresholds (min_candidate_conf, keep) tuned on the
OOF probabilities: "joint" = best amateur F1 with ShuttleSet OOF F1 within 1 pt of its best. Fast simulation of the
hit filter (no swing veto / min_no_hit_rally), so numbers are below the full-pipeline ones.

Reference in the same simulation: bundled v2 amateur 0.678 (qa6 0.565, am7 0.749), ShuttleSet test 0.771; rules amateur 0.596.

Findings:
- min_shuttle_conf 0.15 vs 0.3 changes almost nothing on amateur videos (+0..7 candidates per video, ceilings equal);
  CV am F1 0.703 (0.15) vs 0.708 (0.3). Kept 0.3. The brief's suspicion (amateur shuttle points dropped) does not
  hold for these runs: the misses are frames with no detection at all (qa6 candidate ceiling 99/126; 0620 13/21).
- Direction features: dropping dir_in/out_x/y (v3, camera-independent) did NOT help: amateur 0.685-0.695 vs 0.705-0.714
  with them; ShuttleSet OOF also lower. The hitter-relative replacements (toward_in/away_out/...) add ~+0.5..1 pt at most.
  Kept the original direction features. (The amateur cameras here are mostly high/side; the low-rear case may still flip.)
- Amateur rows in training (weight 3): amateur +2.7 pt (0.678 -> 0.705), mostly am7; qa6 +1.
- Pose swing features (p_swing of the best player's side / other side from the bundled pose_hits model, +-250 ms):
  ShuttleSet OOF 0.769 -> 0.777-0.781, qa6 0.578 -> 0.590-0.602. Cost: events stage ~4-5x slower (32 s -> 155 s for
  13 amateur videos). pose_hits was trained on ShuttleSet train only, so no leak into amateur/ShuttleSet test.
- Size/regularisation: h8/h16/h32 within noise; l2 1e-3 clearly worse (-3..4 pts), 3e-2 worse (-1.5). hidden 0 (logistic) -3.
- am-weight 1/3/6: within ~1 pt; 10 hurts ShuttleSet (-2 pts).

### Full pipeline, leave-one-amateur-video-out (scripts/train_hits_v3.py lovo, scripts/lovo_sweep.py)
Each amateur video scored with a model trained without it, full events stage with the amateur settings.
| model | thresholds (mc/keep, veto) | qa6 F1 (P/R) | am7 F1 (P/R) | all | dead FP |
|---|---|---|---|---|---|
| rules | - | 0.493 | 0.734 | 0.630 | 33 |
| bundled v2 | 0.3/0.65, 0.1 | 0.556 (.54/.57) | 0.839 (.75/.95) | 0.723 | 45 |
| v2 feats, SS v3 only | 0.3/0.65, 0.1 | 0.571 | 0.825 | 0.719 | 36 |
| v2 feats + amateur aw3 (D) | 0.1/0.3, 0.1 | 0.592 | 0.806 | 0.710 | 25 |
| v2+geom aw3 (E) | 0.1/0.3, 0.1 | 0.591 | 0.813 | 0.713 | 24 |
| v2+swing aw3 | 0.3/0.15, 0.1 | 0.614 | 0.805 | 0.722 | 22 |
| v2+geom+swing aw3 (A) | 0.15/0.3, 0.1 | 0.610 | 0.820 | 0.728 | 22 |
| A, swing dropout 0.5 (F) | 0.1/0.3, 0.1 | 0.597 | 0.820 | 0.722 | 24 |
Threshold sweep for A (veto 0.1): keep 0.15 -> qa6 .602 am7 .794; keep 0.3 -> .602/.821; veto 0 costs am7 3-4 pts.
Paired bootstrap over videos, bundled v2 -> A: qa6 +0.054 [95% CI +0.002, +0.107], am7 -0.019 [-0.107, +0.049], all +0.006.

Why am7 drops: am7 pre-labels ARE the bundled v2 detections (plus 4 added), so v2 is favoured by construction (its
misses are mostly not labelled; another model's extra true hits count as false). Still, ~25 confirmed hits that v2
found are lost by A: most have low pose-swing evidence (p_swing 0.02-0.2): serves, net strokes, and players the
tracker does not follow (0241's partners are untracked; the tracked "near" player is a spectator). Swing-dropout
augmentation (F) does not recover them; v2 features without swing (D) lose them too, so it is mainly the amateur
rows + lower threshold regime, not the swing feature.

### More variants (full pipeline LOVO unless noted)
- ShuttleSet-only training with the new features (H5 v2+all, I5 v2+swing; 5 seeds): qa6 0.566, am7 0.81-0.82.
  -> the amateur rows are what lift qa6 (+3.4 pts); the am7 level (~0.81-0.82) is the same for every non-v2 model.
- Pose-based hit recovery on top (pose_hits=auto): bundled qa6 0.554 / am7 0.831; A5 qa6 0.595 -> no help, rejected.
- Swing veto 0.05 / 0.2 instead of 0.1 with A5: worse (0.719 / 0.709 all). keep 0.25/0.3/0.35: flat (0.723-0.725).
- 5-seed ensemble A5 vs single seed A: same (0.725 vs 0.728 all) - kept for stability.
- Label-time noise: 58 of 126 qa6 positives have unrefined 8 fps times (+-0.15 s). All 25 qa6 positives without any
  candidate within 0.25 s are unrefined ones; their nearest candidates sit 0.28-0.85 s away while the shuttle is
  visible, so part of the qa6 "misses" is label timing. Added a loose qa6 score (unrefined labels matched up to
  0.4 s): rules 0.568, bundled v2 0.648, A5 0.686.
- Treating unrefined qa6 labels as looser positives in training (J5, --unrefined): qa6 0.595 / loose 0.680: no help.
- Rendered time strips for the 58 unrefined hits (review/qa6_strips) to refine the times by eye; the crops are too
  small to read the contact reliably at 60 fps, so the times were left as they are (loose score used instead).

### Candidate-generation search (D1, 2026-11-21 ~21:40 UTC)
10 random overrides of the rule candidate parameters (min_turn_deg, speed_jump_ratio, end_gap_ms, max_window_ms,
min_hit_gap_ms, max_hitter_dist_rel, window_ms), each re-collected (amateur + ShuttleSet) and cross-validated with
the v2+all / aw3 scorer (sim, thresholds tuned). Default: amateur 0.723 (qa6 .609, am7 .811), SS OOF .773.
Best others: speed_jump 1.4 + min_turn 28 -> 0.716 (SS .777); end_gap 2000 + min_turn 20 + speed 1.6 -> 0.712
(qa6 .614); all others 0.681-0.709. More candidates raise the ShuttleSet ceiling a little (1500 vs 1475 of 1738) but
not the amateur F1. Rejected; the v2 candidate parameters stay.

### Size / weight sweep, full pipeline LOVO, 5 seeds each (D2)
| variant | all | qa6 | am7 | qa6 loose |
|---|---|---|---|---|
| A5 (h16, l2 1e-2, aw3) | **0.725** | **0.600** | 0.822 | **0.686** |
| h8 | 0.719 | 0.597 | 0.814 | 0.683 |
| h32 | 0.724 | 0.600 | 0.820 | 0.686 |
| aw1 | 0.708 | 0.591 | 0.798 | 0.672 |
| aw6 | 0.714 | 0.579 | 0.816 | 0.672 |
| l2 3e-3 | 0.718 | 0.584 | 0.822 | 0.675 |
Seed sets 5-9 / 10-14 of A5: 0.722 / 0.725. A5 kept.

### Strokes, final (eval_prelabels.py on amateur_v1, 130 labels with a stroke, statuses reviewed_sure+unsure)
- saved events (v2 hits of the pre-label runs): court_v1 73/126 (58%), court_v2 85/126 (67%)
- events re-run with v2 + own-footage settings: court_v1 69/125 (55%), court_v2 81/125 (65%)
- events re-run with v3: court_v1 60/112 (54%), court_v2 69/112 (62%) (13 fewer labelled hits matched)
- 61 fully trusted labels only (saved events): court_v1 38/57 (67%), court_v2 41/57 (72%)
Not enough trusted stroke labels (61 < ~150) for the planned fine-tune; skipped.

### Bundled
v3 = A5 refit on everything -> rallyiq/events/models/hit_scorer.json (v2 kept as hit_scorer_v2.json).
ShuttleSet test v3 (learn_hits eval): rules 64.9, v2 77.0, v3 78.7 (P 84.1 R 73.9), same-side-twice 5.4%.
Full test suite: same 24 failures + 11 errors as the base commit (sqlite/server/shuttle-weights env issues), none new.

### qa6 unrefined label times (D3)
Tighter 13-frame strips (-0.30..+0.30 s, step 0.05) for the 58 unrefined qa6 hits: review/qa6_strips2/. I read the
first 12 (clip 0003): 6 contacts found at offsets -0.30, -0.10, -0.08, -0.05, 0.00, +0.10 s; 6 show no contact within
+-0.3 s (player standing still, serve preparation, far hitter too small). So the unrefined times are often off by
more than the 0.25 s tolerance or are not visible contacts at all; I stopped there and did not change the truth file
(labels/qa6_time_refine.txt has the 12 readings). The loose qa6 score (0.4 s for these) is the fairer view; a proper
fix is sam re-timing these 58 hits in the label tool.

### Round-3 labels = held-out test (D4, ~22:30 UTC)
17 new rallies in 6 am7 videos (0239 had no eligible rally), picked at random from v2 and v3 segmentations, none used in
training. Every detection of v2 OR v3 inside them (108) reviewed from sheets review/r3/ (prev overview | hit overview |
4 crops | next overview): 71 hits, 27 not hits (passes, pickups, juggling, a neighbour court, a spectator), 10 unsure
(ignored). labels/r3/decisions.txt, merged into labels/amateur_v2.json. Because both detectors' outputs were labelled,
this set does not favour either (unlike am7 v1). Hits neither found are not labelled.
| scorer | F1 | P | R | dead FP | hitter id | side |
|---|---|---|---|---|---|---|
| rules | 0.644 | 0.53 | 0.82 | 15 | 51/54 | 55/58 |
| v2 | 0.783 | 0.70 | 0.89 | 27 | 54/57 | 61/63 |
| **v3** | **0.879** | **0.89** | 0.87 | **8** | 56/59 | 59/62 |
Paired bootstrap over the 6 videos, v2 -> v3: +0.097, 95% CI [-0.001, +0.189], P(>0) 0.97.
Note: the hitter id is wrong or untracked on several doubles hits (0241, 0622 partners), not counted when the label
hitter is null.

### Owner asked (via thread) to use the remaining budget for heavier training. Added (D5-D7):
- v3b: same recipe + round-3 labels (amateur_v2.json; 361 positives). LOVO full pipeline on all 13 videos with v2 labels:
  | scorer | qa6 | qa6 loose | am7+r3 | all | dead FP |
  |---|---|---|---|---|---|
  | v2 (bundled before) | 0.556 | 0.648 | 0.822 | 0.735 | 72 |
  | v3 A5 (trained without r3) | 0.600 | 0.686 | 0.830 | 0.750 | 33 |
  | **v3b B5** | **0.610** | **0.695** | **0.838** | **0.759** | **27** |
- Pose-swing model retrained with amateur pose samples (13-fold nested LOVO: pose model, candidate features and scorer all
  without the held-out video; scripts/lovo_pose.py): qa6 0.565, am7 0.800, all 0.717 -> worse than the ShuttleSet-only
  pose model (0.759). Rejected. (Amateur pose positives are few, 950, and noisy in time.)

### qa6 unrefined times, second pass (22 of 58 read now)
11 contacts found at offsets -0.30..+0.20 s (median about -0.10), 11 outside +-0.3 s or not visible (two look like
the same hit labelled twice, e.g. 0004 162.40/162.60 far/near 0.2 s apart). So about half of the unrefined qa6 times
are further than the 0.25 s tolerance from the contact: qa6 strict F1 underestimates every model, and the misses
there are partly label error. Readings in labels/qa6_time_refine.txt; truth file left unchanged.

### v3b error bars (D8)
Four disjoint 5-seed sets (0-4, 5-9, 10-14, 15-19), full LOVO: all-video F1 0.759 / 0.758 / 0.758 / 0.759, qa6 0.610 in all,
dead FP 27-28. Seed variance is negligible next to the between-video variance (bootstrap CIs about ±0.05-0.10).
Grid (72 configs, sim): best 0.742 vs chosen 0.735 (rank 6), within noise. Chosen config kept.
Final bundled: v3b (models/hit_scorer_v3b_B5.json == repo rallyiq/events/models/hit_scorer.json). Run end ~22:45 UTC.
