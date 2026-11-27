# train-v10 plan (start 2026-11-27 ~18:00 UTC; claude/train-v10 from claude/train-v9 c1f464f). Finish by ~07:00 UTC.
Rules as v9 (pre-register here before results; nested LOVO over the 11 non-qa6 own videos, per-video bootstrap; fresh set
test-only; qa6 untouched; ShuttleSet must not drop). Time boxes: R 6 h (checkpoint at 3 h: if the sequence model trails
the v7 pipeline by > 0.05 F1 on the own LOVO, stop R and record), S 4 h (checkpoint at 2 h, same rule vs 77% LOVO), T 2 h.

## R: temporal sequence hit detector (pre-registered)
- Frames resampled to ~30 Hz (stride 2 on 60 fps). Per frame 34 inputs: shuttle (visible, x, y in image units, dx, dy,
  confidence) and per side (near / far) the player nearest the shuttle: wrists relative to the shuttle in body heights (4),
  hip centre relative to the shuttle (2), body height / image height, wrist speed, present flag (= 10 x 2 + 6 + 8 spare =
  bucket of 34; exact list in scripts/r_feats.py).
- Model: single-stage dilated TCN (MS-TCN style, Farha & Gall CVPR 2019, arXiv 1903.01945), 10 layers, 64 channels,
  kernel 3, dilations 1..512; heads: hit (per frame, target = 1 within +-1 frame of a labelled hit) and side (near/far).
- Training: pretrain on the 88 ShuttleSet train clips; fine-tune on own labelled windows (frames outside windows masked)
  with LOVO over the 11 am7 videos; 3 seeds; Adam 1e-3, 30 + 30 epochs, windows of 256 frames.
- Decode: peak picking on the hit probability, threshold and min gap chosen inside the training folds (inner selection on
  the fold's own training videos' predictions); hitter = nearest player of the predicted side.
- Comparison (single test): sequence-model hits vs the v7 pipeline (0.860 with nested dead-ball models) on the 17 videos with
  per-video bootstrap; plus fresh set (0.821), qa6, ShuttleSet test 14 clips (78.9). Ship only if CI > 0 and no gate drops.
- Fallback (also pre-registered, only if the standalone model fails): the sequence model's max hit probability within
  +-2 frames of a candidate as one extra input to a v3b-recipe scorer retrain (nested LOVO), same keep rule.

## S: BST-style stroke classifier (pre-registered)
- Per labelled hit: a window of -0.5..+1.0 s of the same frame features + the hitter's full keypoints; small transformer
  (2 layers, 64 d) or TCN pooled, pretrained on ShuttleSet strokes (taxonomy as court_v2), fine-tuned LOVO on the 197 trusted
  own rows (+ unsure rows at weight 0.5); test on the fresh set (78 typed). Compare with court_v2 + residual (77.2% LOVO on
  the same rows, 55.6% fresh). Ship only if LOVO CI > 0 and fresh not worse.

References: MS-TCN (Farha & Gall, CVPR 2019, arXiv 1903.01945); BST (arXiv 2502.21085); TemPose (CVPRW 2023);
ShuttleSet (arXiv 2306.04948); plus earlier rounds' references.

## R checkpoint 1 (~0.5 h in)
Model A (as pre-registered): own 17 videos 0.789 vs v7 0.860 (CI [-0.110, -0.039]); fresh 0.740 vs 0.821; qa6 0.667 vs
0.692; ShuttleSet test 86.4 (LH.score, +-3 frames same side) vs v7 baseline (computed the same way below).
Pre-declared before running: ONE variant B (own weight 6, 3000 fine-tune steps, 96 channels); then the stacked-feature
fallback regardless of B's result, with whichever of A/B has the higher nested own F1 (chosen on am7 only).

## S checkpoint 1 (~0.3 h into S)
S-A (as pre-registered): own trusted LOVO 71.4% vs court+residual 77.2% (CI [-0.114, -0.008]); fresh 55.6% = 55.6% on the 63
matched; ShuttleSet test 70.1%. Trails.
Pre-declared single variant S-B (before running): the residual layer gets the seq classifier's out-of-fold log-probabilities
(12 inputs) in addition to its 4 features (same L2 0.003, same LOVO). Fresh is evaluated only if S-B wins on own LOVO.

## R2 pre-registration (broadcast profile only; written before any R2 result)
- Model: the TCN of config A trained on the 88 ShuttleSet train clips only (no own data), 3 seeds, 3000 steps.
- Decode: peak threshold and min gap chosen on ShuttleSet train out-of-fold predictions (4 folds, same recipe) — nothing chosen
  on the 14 test clips. Thresholds {0.2..0.7}, gaps {0.2, 0.28, 0.36 s}.
- Integration: the events pipeline keeps its rally segmentation; inside each rally (+-0.3 s) the hits are replaced by the TCN
  peaks; hitter = nearest player of the predicted side (fallback: nearest player). Used only when the pipeline profile is
  broadcast/unset AND `events.hit_model = "seq"`... (decision on the default below).
- Test: the integrated pipeline on the 14 ShuttleSet test clips (6 held-out matches) with the same benchmark scorer as the
  78.9 gate (learn_hits eval), per-match bootstrap of the F1 difference (5,000), keep if CI > 0. Own footage: profile=own must
  give identical predictions (unit test + rerun of the 17 videos). Runtime measured on this Mac per minute of video.

## R2 result 1 and R2b pre-registration
R2 as pre-registered (TCN peaks only inside the rules pipeline's rallies +-0.3 s), after fixing a frame-time bug (fps was
rounded from the shuttle timestamps; 30 fps clips drifted ~300 ms per 30 s): 14 test clips F1 0.815 vs rules 0.780, per-match
CI [-0.016, +0.069] -> fails the keep rule. The unrestricted TCN on the same clips scores 0.886 (seen; not pre-registered).
R2b (decided before any further test look): choose the integration mode {inside rallies +-0.3 s, inside rallies +-1.0 s,
unrestricted with rallies rebuilt from the TCN hits} on the ShuttleSet TRAIN clips with the out-of-fold TCN probabilities; then
run the test clips once with the chosen mode. Report both R2 and R2b; ship only if R2b's per-match CI > 0. This is a second
look at the test matches, stated as such.
R2b outcome: on ShuttleSet train OOF the modes score r03 0.869, r10 0.867, unrestricted 0.863 -> r03 chosen, which is R2's
mode, so the test result stays 0.815 vs 0.780, CI [-0.016, +0.069]: not shipped.
