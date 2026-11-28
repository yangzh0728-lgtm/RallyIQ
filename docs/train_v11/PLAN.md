# train-v11 plan (start 2026-11-27 ~22:00 UTC; claude/train-v11 from claude/train-v10 fc0ba15)

Goal: more fully labelled own videos (whole length, not windows), then retrain what the evidence supports.
Rules as v7-v10: pre-register here before looking at any result; fresh set (own_20260529_0005/0007/0009) test-only;
qa6 labels untouched and nothing selected on qa6 (as before, qa6 videos still take part in leave-one-video-out like in
every earlier scorer retrain); ShuttleSet test must not drop; ship only if the per-video bootstrap CI of the gain is > 0
and no gate drops.

## Data (step 1-2)
- 8 never-labelled own videos with a verifiable court (others: no usable court, see LOG.md): own_20260310_03,
  own_20260310_06, own_20260302_06 (doubles), own_20260906_0623, own_20250820_7695/7696/7697/7698 (new hall, iPhone).
- Pipeline: claude/train-v11 code, profile=own + the own batch settings (runs/*/queue.json, scripts/batch_v11.py).
- Pre-labels: every production hit (P), every low-threshold scorer candidate (L, 0.1), every rules candidate (R), and a gap
  probe (G) for every > 1.6 s stretch of a rally without a row. Each row checked by Claude on contact sheets
  (REVIEW_PROTOCOL.md), with a reason. Second pass: an independent re-review of a random 10% of rows (seed 11), agreement
  reported (hit/not-a-hit, side, hitter, stroke).
- Label file labels/amateur_v11.json = amateur_v6_newids (hits) / amateur_v9 (strokes) for the 11 old videos + the new ones.

## H: hit scorer, v3b recipe retrained on the larger set (pre-registered)
- Recipe exactly as v3b / retrain8.sh: train_hits_v3.py collect/lovo/fit, features v2+all, hidden 16, l2 0.01, 5 seeds,
  am-weight 3, ShuttleSet train v3, thresholds {min_candidate_conf 0.1, keep 0.3, ok 0.7}; own labels = old 17 + new 8.
- Baseline arm: the shipped v3b scorer with its leave-one-video-out models (train_v4/ckpt/lovo_B5; videos it never saw use
  full.json, which includes all 8 new videos). Candidate arm: the retrained recipe, leave-one-video-out over all labelled
  own videos (each video scored by a model that never saw it); fresh set scored by the all-video fit.
- Both arms: same tracks (shipped tracker on the saved candidates, peak 0.3), production events settings, the same
  nested dead-ball models (train_v7/ckpt/dead_lovo_aug; full model for videos outside its training).
- Primary metric: hit F1 pooled over all 25 own videos (17 old + 8 new), per-video paired bootstrap (5,000) of the
  difference. Gates: fresh set F1 not lower, qa6 not lower by more than one hit's worth (0.005), ShuttleSet test (14 clips)
  not lower than the shipped 78.9 by more than 0.5. Also reported: old 17, new 8 separately.
- No hyper-parameter search (no selection inside the folds is needed).

## S: own stroke residual refit (pre-registered)
- Shipped residual (court_v2 + own residual, features rise/fall/oh/first, L2 0.003, unsure rows weight 0.5) refit on the old
  trusted rows + the new rows. Baseline arm: same residual fit on the old rows only. Both leave-one-video-out over every
  video with typed rows (old and new), scored on trusted (sure) rows of the held-out video; fresh set (78 typed) with each
  arm's all-video fit. Ship if the LOVO CI > 0 and fresh not lower.

## Anything else
Only if the H/S results point to it, and pre-registered here before running.

## Addendum 2026-11-28 ~08:00 UTC (sam: "增加训练内容"; written before any H/S/C result)
- A (rescue courts): 0 rescued by me (LOG.md); CLICK_ME.md asks sam for ~10 min of corner clicks on 8 positions. Videos that
  arrive in time are run and labelled exactly like the 8 and join the same tests; a list of which ones made it goes in
  the report. If none arrive before the retrain starts, H/S/C run on the 8.
- B (public data): no download (licences unclear or broadcast-only, see LOG.md / REPORT.md); nothing changes in H/S.
- C (viewpoint augmentation, pre-registered): each of the 88 ShuttleSet train clips gets 2 synthetic copies: an exact
  camera-rotation homography R = K Rot K^-1 (f = 0.9 x width; yaw U[-15, 15] deg, pitch U[-10, 10], roll U[-3, 3]) applied
  to every image-space quantity (shuttle x/y, player boxes, keypoints, foot points, calibration: H_new = H_old R^-1, corners);
  points leaving the image are dropped; then own-footage-like detection loss: bursts (start p = 0.03 per frame, length
  U{2..10} frames) + independent drop 0.10 + Gaussian noise sigma = 2 px (1080p scale). Labels unchanged (same times / sides).
  A rotation cannot move the camera lower (sam's cameras are low behind the baseline), so this covers orientation and detection
  quality, not height; court-space re-projection of the shuttle needs its unknown height, so it is not attempted.
  Arm C = H's recipe and own labels + the 176 augmented clips in the ShuttleSet train set. Same LOVO evaluation, same gates
  and the same ship rule against the shipped baseline. If both H and C pass, the one with the higher pooled own F1 ships
  (a post-hoc choice between two pre-registered arms, stated as such).
- The stroke residual (S) does not use augmented ShuttleSet rows: it is an own-footage correction on top of the
  ShuttleSet-trained court model, and adding broadcast rows would remove what it corrects.
