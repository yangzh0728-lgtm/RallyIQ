# train-v6 log

2026-11-27 07:30 UTC start. Worktree repo/ on claude/train-v6 from origin/claude/train-v5 (432824f). PLAN.md written.

## A1 hitter truth (doubles)
- review/hl/<key>_NN.jpg: every labelled hit of 0310_01, 0239, 0241, 0622 (137) with raw pose detections (score>=0.2) and v5 ids.
- 30 labels without a hitter id: I picked the hitter from 3-frame strips (review/hl/none_*): 25 now tracked in v5, 2 hidden
  behind a partner with no detection at the hit (0241 262.86, 0622 308.44), 1 detected but untracked (0241 286.77), 1 ambiguous (0310_01 305.64).
- Checked the 9 label-id rows where v5's pick disagreed (review/hl/chk_*): 6 were truth errors from the v4->v5 id remap
  (0241 134.87/136.67/278.06/289.82, 0622 82.98/312.14), 2 ambiguous (label itself unsure: 0310_01 14.08, 0239 138.89), 1 kept.
  All decisions with reasons in scripts/htruth.py -> labels/hitter_px.json (134 usable hits).
- Reproduced v5 C4h predictions exactly (rescore with train_v4/ckpt/lovo_B5).
- Result: hitter has no track at the hit: v4 tracks 22/134 (16.4%) -> v5 4/134 (3.0%).
  Attribution among matched hits: v4 80/86 (93.0%) -> v5 99/103 (96.1%). Remaining v5 errors: 0310_01 393.08 (partner),
  0310_01 474.79 (side), 0241 285.17 (partner), 0241 286.77 (untracked far boy).
  => partner tracking is no longer the main limit; hit recall is (31 of 134 doubles hits not detected).

## A2 partner choice (commit 25a8d8e)
- ckpt/hdiag_C4h.json: per matched doubles hit, proximity and pose-model p_swing of the picked player and partner.
- Rule partner_swing (events/pose_hits.py, config partner_swing_margin=0.2, partner_swing_prox_margin=0.3): plateau
  0.1-0.2 x 0.3-0.5 all give 101/102; Δswing 0.3 fixes 1.
- v6_A2 (bundled v3b / LOVO lovo_B5, v5 tracks): F1 all 0.859 / other 0.909 / qa6 0.695 (identical to v5);
  hitter attribution 99/103 -> 101/103 (96.1% -> 98.1%), untracked 4/134 unchanged.
- 13 hitters moved in all; 11 outside the truth set checked by eye (review/ps/pair*.jpg): new right 6
  (0622 413.45, 462.96, 480.58, 990.99 lean, 0239 228.46, 0615 81.51 lean), old right 3 (0622 125.66 clear,
  0310_01 136.87 lean, 0615 219.35 lean), unclear 2 (0310_01 26.38, 262.56). Kept (net +5 on 113 looked at).
- Player tracking itself not changed -> off-court counts = v5. Low-threshold / re-acquisition ideas not pursued:
  only 2 of 134 truth hitters are detected-but-unassigned and 2 have no detection at all (hidden behind the partner).
- FN analysis (ckpt/fn_C4h.json): of 75 missed labelled hits, 38 have a track point at a wrist (<0.35 heights) within
  0.1 s, 32 within 1 height, 3 far, 2 no track -> misses are decided in events/scorer, not by the detector.

## A3 0245 white wall (S4, S15, S19) - no fix
- review/w45/S*.jpg: candidates and track at -0.25..+0.25 s. In S15 the v5 track sits on a static spot on the door glass
  just before the hit; the shuttle is not visible to me at the contact either (white shuttle, white wall, blur).
- Racket-local zoom (scripts/zoom.py: detector on an 800x450 ROI around the hitter, i.e. 2.4x the usual resolution, peak
  threshold 0.1): only peaks on the player's body/racket head (review/w45/S15_zoom.jpg). No shuttle found -> not
  shipped. S4/S19: the shuttle is not resolvable at the contact either; left for B (hard-example fine-tuning).

## A4 next-court shuttle between rallies - gate rejected
- All 8 rows fall inside a detected rally segment (the foreign flight itself extends/creates the segment), so a plain
  rally gate does nothing (scripts/rallygate.py).
- Hit-anchored gate (keep points from first hit - pre to last hit + post, and across hit gaps <= G), evaluated on the
  v5 hand-labelled tracklets with position match (scripts/gate_eval.py): of 1,960 foreign frames still in the v5 track it
  removes 45-60%, but it also removes 3.7-11% of 6,115 hand-checked frames of ours (undetected hits, serve tosses).
  v5 rule was <= 2.6% own frames lost -> rejected. The 8 rows stay (between rallies, no hits created).
- 'activity' feature for ownership (any of our wrists swinging during the tracklet): useless, our players' wrists
  always move (foreign 94% vs ours 99.5% above 1 swing unit); AUC 0.9736 -> 0.9737. Not shipped.

## B shuttle detector fine-tune (ft1)
- Labels: 377 gap frames inside hit-to-hit flights of the 11 non-qa6 labelled videos, read by me on 100 px crops
  (labels/shuttle_v6/hp_decisions.txt): 206 accepted, 40 visible but off the estimate, 131 not visible.
  Easy positives = v5 track points in those flights that lie on a smooth quadratic (<= 5 px), minus frames with a
  hand-labelled next-court tracklet; two random 48-point samples checked by eye (review/hp/easy*_00.jpg; 5 bad in the first
  sample, all > 5 px residual -> filter added; second sample clean).
- Split: 0239 and 0622 held out whole + the last 25% of every other video's labelled span -> val (3,481 frames, 81 hard);
  train 5,360 frames (104 hard, weight 4). TrackNet Professional rehearsal: 60 clips x 150 frames train, 10 val (guard).
- scripts/b_train.py: from heatmap.pt (dbf2ae6beab0), AdamW 1e-4, BN frozen, 8 x 300 steps, bs 4, 1/3 TrackNet, select the
  best own-val epoch with TrackNet val F1 >= base - 0.005. Picked epoch 8.
- own val F1@0.2: 0.919 -> 0.962 (hard 0.575 -> 0.894, easy 0.926 -> 0.964); TrackNet val 0.956 -> 0.954.
- TrackNet public test (30 clips, re-detected): shipped heatmap.pt F1 90.6 / acc 88.3 -> ft1 92.6 / 90.5.
  (v5's 95.0/93.0 came from candidates of the older loop/base.pt; same code on today's heatmap.pt gives 90.6.)
- ShuttleSet local test-match clips (6, re-detected, bundled scorer): hit F1 61.4 -> 65.4, rules 45.3 -> 52.0.
