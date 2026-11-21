# RallyIQ train-v3 report (autonomous run, 2026-11-21)

Branch `claude/train-v3` (pushed, not merged) = `claude/project-thread-ah2yye` + `main` + `claude/project-thread-lfd1cx`
(swing veto / min_no_hit_rally_ms / landed_ms settings), plus this run's commits. Everything else lives in
`~/Desktop/RallyIQ/train_v3/` (inputs copied from the T7, labels, review sheets, checkpoints, logs). Full experiment log: `LOG.md`.

## Before / after

| metric | before | after | notes |
|---|---|---|---|
| ShuttleSet test v3 hit F1 (14 clips, ±3 frames, same side) | 77.0 (v2) | **78.7** (v3) | P 87.2→84.1, R 68.9→73.9; "same side twice" 11.3%→5.4% (labels 0%) |
| Own videos qa6: 126 blind-labelled in-play hits, F1 ±0.25 s, leave-one-video-out | 0.556 | **0.600** | rules 0.493. Bootstrap over videos: +0.044..0.054, 95% CI ≈ [0.00, +0.10] |
| qa6, unrefined 8 fps label times matched up to 0.4 s | 0.648 | **0.686** | 58 of the 126 qa6 times were never refined (±0.15 s) |
| Own videos am7: 164 reviewed pre-label hits, F1 ±0.25 s, LOVO | **0.839** | 0.822 | biased toward v2: these labels are v2's own detections (+4 added) |
| All 13 own videos, hit F1 | 0.723 | **0.725** | flat overall; v3 trades recall on am7 for precision |
| Dead-time false hits (detections on a labelled pickup/pass) | 45 | **24** | rules 33 |
| **Held-out round 3**: 17 new rallies, every v2 or v3 detection reviewed (71 hits, 27 not), F1 ±0.25 s | 0.783 | **0.879** | P 0.70→0.89, R 0.89→0.87, dead-time false hits 27→8; bootstrap +0.097 [−0.001, +0.189] |
| Hitter side right (matched hits, all own videos) | 219/228 (96%) | 198/207 (96%) | qa6 alone 65/72 → 68/75 |
| Hitter id right (am7, doubles + singles) | 146/150 (97%) | 129/132 (98%) | ids from the same run, so this mostly checks the hit/hitter pairing |
| Stroke type, court_v2, 125-130 reviewed hits (events with v2 hits) | 65% (court_v1: 55%) | 62% with v3 hits (69/112) | fewer labelled hits are matched with v3 (112 vs 125); not significant (±4.5 pts) |
| Stroke type on the 61 fully trusted labels (saved events) | court_v1 67% | court_v2 72% | 41/57 matched |
| **Bundled v3b** (v3 recipe + round-3 labels), LOVO on all 13 videos, 361 hits | v2 0.735 / v3 0.750 | **0.759** | qa6 0.556→0.610, dead-time false hits 72→27; ShuttleSet test 78.3 |

"Before" = the bundled v2 scorer (now `hit_scorer_v2.json`), "after" = v3. The bundled `hit_scorer.json` is v3b:
the same recipe retrained after round 3 added 17 labelled rallies (last row; LOVO with the round-3 labels). Own-video numbers use the
own-footage settings (swing_veto 0.1, swing_veto_keep_exchange_ms 2700, min_no_hit_rally_ms 2500, landed_ms 0) and
recompute events from the own-v12 / pre-label artifacts. Each own video is scored by a model trained without it.

Decision rule from the plan: keep a model only if amateur hit F1 improves and ShuttleSet test F1 drops < 1 point.
v3 passes on the unbiased sets (qa6 +4.4, CI just above 0; round-3 held-out rallies +9.7) and on ShuttleSet (+1.7);
on the biased am7 set it is -1.7 (CI [-0.11, +0.05]); on all 13 v1-labelled videos it is flat (+0.2). I bundled v3 and kept v2 next to it; if sam prefers
recall on footage like the pre-label videos, `hit_scorer=<path to hit_scorer_v2.json>` switches back.

## What was done

**A. Labels.** I reviewed the 108 "unsure" pre-labels from per-hit frame sheets (overview at the hit, four crops of the
hitter at -0.12/0/+0.12/+0.24 s, overview at the next hit): 13 were not hits (pickups, passes, a let-go), 95 are real.
Hitter changed on 6 (three are untracked partners in 0241), stroke changed or cleared on about 35 (five serves; lunging
"drops" that are lifts or net shots; one drop that is a smash). A later cross-check against the neighbouring labels
caught two of my own serve calls (0302_07 478.03 is a block after a smash; 0241 333.42 is not a serve) and fixed them.
Output `labels/amateur_v1.json` (61 reviewed_sure, 103 reviewed_unsure, 40 not_a_hit; every reviewed entry has a
one-line reason), decisions in `labels/review_decisions.txt`, and one sheet per 3 hits with my label written on in
`review/v1_reviewed/` for sam to spot errors.

**Round 3 (held-out test).** After choosing v3 I labelled 17 new rallies (6 videos, never used in training): every
detection of v2 or v3 inside them, 108 in all (71 hits, 27 not hits, 10 unsure), so the set favours neither model
(`labels/r3/decisions.txt`, sheets `review/r3/`, merged into `labels/amateur_v2.json`).

**B. Hit detector v3.** Candidates from the rules, as before; the scorer is a 5-seed MLP ensemble (16 tanh units,
l2 1e-2) over the 27 v2 features plus hitter-relative shuttle motion (toward/away from the best player), speed
ratio, wrist distance, track density, and the pose-sequence swing probability of the hitter's side and the other side
within 250 ms (from the bundled `pose_hits` model, trained on ShuttleSet train only). Trained on ShuttleSet train v3 +
the 13 labelled own videos (amateur rows weight 3; outside the labelled windows weight 0; dead-ball pickups are
negatives). Validation: leave-one-own-video-out with the full events pipeline, ShuttleSet held out by match (5 folds)
for model selection, ShuttleSet test only at the end. Code: `scripts/train_hits_v3.py`, `scripts/amateur_hits.py`,
`scripts/lovo_sweep.py`, `scripts/boot_compare.py`; model support in `rallyiq/events/learned.py` (column selection,
ensembles, optional GBT); the swing features are only computed when the scorer uses them (events stage of a 14 min
video: 1.8 s → 12.7 s with v3).

**C. Hitter and stroke type.** Hitter side/id are reported above (no regression). Only 61 stroke labels are fully
trusted (< ~150), so I did not fine-tune the court stroke model; `scripts/eval_prelabels.py` gained `--trusted`,
`--stroke-sure-only` and `--rerun-events` for this evaluation. The known errors are confirmed on the reviewed labels:
lift typed net_shot (6-9×, the largest group), serves typed push/net_shot (3-5×), long serves never predicted
long_serve (all 3), smashes typed drop/net_kill or lost (5-7×).

## Tried and rejected (numbers in LOG.md)

- **Dropping the image-direction features** (camera-independent v3 set): worse on both domains (amateur sim −1..2 pts,
  ShuttleSet OOF −0.5..1). These cameras are mostly high or side; the low-rear flip may still matter on other footage.
- **min_shuttle_conf 0.15 instead of 0.3**: adds 0-7 candidates per own video and no F1. The qa6 misses are not low-
  confidence points: all 25 positives without a candidate are unrefined 8 fps label times, with the shuttle visible.
- **Per-domain threshold / calibration**: with amateur rows in training, one threshold (keep 0.3, min 0.1) is best for
  both domains; a separate amateur threshold gave nothing in the full pipeline (keep 0.15-0.45 flat within 1 pt).
- **Gradient-boosted trees** (scikit-learn HGB exported to JSON, as LightGBM-style GBDT): +1 pt in the simulation, but
  −3 pts in the full pipeline at every threshold (0.644-0.694 all-video F1).
- **Blend of v2 and v3** (mean probability): 0.718-0.725, no gain over v3 alone.
- **Pose-based hit recovery** (`pose_hits=auto`) on top: −0.5 pt with both scorers.
- **Swing-feature dropout** (train copies with "unknown" swing): no help (0.716-0.722).
- **Looser positives for unrefined qa6 times**: no help (0.717).
- **Amateur weight 1 / 6 / 10, hidden 0 / 8 / 32, l2 1e-3 / 3e-2**: within noise or worse (see D below and LOG.md).
- **Pose-swing model retrained with amateur pose samples** (nested leave-one-video-out, `scripts/lovo_pose.py`):
  all-video F1 0.717 vs 0.759 with the ShuttleSet-only pose model. The 950 amateur positives are few and noisy in time.
- **Large grid** (72 configs: features v2+all / v2+swing, hidden 8-32, l2 3e-3..2e-2, amateur weight 2-4; simulation):
  the best is 0.742 vs 0.735 for the chosen config (rank 6), within single-seed noise; a 20-seed ensemble gives the
  same ShuttleSet result as 5 seeds (78.3); four disjoint 5-seed sets give LOVO all-video F1 0.758-0.759. Kept
  h16 / l2 1e-2 / aw3 / 5 seeds.
- **Candidate-parameter search** (10 random changes of min_turn_deg, speed_jump_ratio, end_gap_ms, windows, min hit
  gap, hitter distance): none beats the v2 parameters on amateur F1 (0.681-0.716 vs 0.723 in the simulation).
- **Splitting merged rallies for the stroke model** (treat a hit after a > 2.5-5 s pause as a serve): no change in
  stroke accuracy; of 20 such "first" hits only 12 are serves (the real serve is often missed). Reverted.

## Remaining errors

1. **Hits only v2 found** (am7 recall 0.95 → 0.80): serves, net strokes and partners the tracker does not follow
   (0241: the tracked "near player" is a spectator), where the pose swing evidence is near 0. Fix upstream: player
   tracking of doubles partners and spectators by the bleachers, and serve detection.
2. **qa6 label times**: 99/126 positives have a rule candidate within 0.25 s. I read 22 of the 58 unrefined times
   on 13-frame strips (`review/qa6_strips2/`): half have a visible contact at -0.30..+0.20 s, half are further than
   0.3 s from it or not visible (one pair looks like the same hit labelled twice). Strict qa6 F1 therefore
   underestimates every model; re-timing these 58 hits in the label tool would make qa6 a sharper benchmark.
3. **Stroke types**: lift vs net_shot at the net, serve types (flick/long serves), overheads on side views.
   Needs ~150+ trusted stroke labels before a fine-tune is worth it.
4. **DJI_0009** (`events_labeled.json`, 58 hits) was not on this Mac or the T7, so it is not in any number here.

## References

- Hit-frame detection from shuttle trajectory and pose: *Automated Hit-frame Detection for Badminton Match Analysis*
  (arXiv 2307.16000); the candidate + learned-scorer design and the pose-swing evidence follow its idea of combining
  shuttle direction change with the player's swing.
- Shuttle tracking that produces the trajectories: TrackNet (arXiv 1907.03698), TrackNetV3 (MMAsia 2023), TrackNetV4
  (arXiv 2409.14543); MonoTrack (CVPRW 2022, arXiv 2204.01899) for monocular court-space reasoning and hit detection
  in amateur-style footage.
- Data: ShuttleSet (KDD 2023, arXiv 2306.04948; CoachAI-Projects, arXiv 1907.12888) for training/test clips and the
  stroke taxonomy of the court model.
- Pose: RTMO (CVPR 2024, arXiv 2312.07526) is the pose model in the players stage the swing features read; TemPose
  (CVPRW 2023) and BST (arXiv 2502.21085) motivate pose-sequence features for strokes and swings.
- Domain shift between broadcast and amateur footage: DANN (arXiv 1409.7495). Here handled by mixing weighted
  amateur rows and camera-agnostic features rather than adversarial training (13 videos are too few for DANN).
- Gradient-boosted trees: LightGBM (NeurIPS 2017); tried via scikit-learn's histogram GBDT, rejected.
