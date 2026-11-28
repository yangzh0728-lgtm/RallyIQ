# train-v11 log (started ~22:00 UTC 2026-11-27; claude/train-v11 from claude/train-v10 fc0ba15). Plan: PLAN.md.

## Step 1: candidates
- T7 mounted, read only. Unlabelled own videos listed from ~/Desktop/RallyIQ/videos_2026_status.csv + T7 folders (cand/grid*.jpg).
- Usable court (verified by overlay at 4 times, courts/checks/): own_20260310_03 / _06 (sam's clicks p02/p04),
  own_20260302_06 (existing Claude doubles court), own_20260906_0623 (SIFT transfer from 0621's points),
  own_20250820_7695 (my fit: 4 line intersections + 2 net-post feet, ICP snap to paint, residual 0-2 px on lines),
  7696/7697/7698 (SIFT transfer from 7695 every 10 s; same hall, same camera spot).
- Rejected: Rowland 0312_*, Wilson 0319/0420, Walnut 0320/0324 others, practice 0250-0253, Fairmont 0302_01/03/04/05,
  Diamond Bar 0310_02/04/05: wood floors with overlapping lines, auto calibration failed and sam skipped all of them as
  "too hard" on the click page; my attempt on 0312_03 could not identify the lines reliably. With Toby IMG_35xx: ultra-wide lens
  distortion (a homography fit on 7 points leaves 120-130 px residuals). Wilson 0233: crowded team practice. 0324_0243:
  moving camera. Short (< 60 s) and first-person clips skipped.
- Sources: 1080p H.264 (transcodes from ~/rallyiq-data/own/full stream-copied; iPhone/4K re-encoded with videotoolbox) in src/.

## Review method pilot (before labelling the new videos)
- Tools: scripts/items.py (P/L/R/G rows), scripts/sheet.py (4 panels per row, 6 rows per sheet), REVIEW_PROTOCOL.md.
- Pilot on own_20260906_0617 (already labelled in amateur_v9 windows; reviewer did not see those labels): 51 rows.
  Inside the v9 windows: 13 of 13 v9 hits found (times within 0.15 s); 1 extra hit (63.76, v9 says not_a_hit; on the sheet the
  rising track has no swing: I side with v9 -> protocol now requires a racket at the shuttle). Not-a-hit rows agree.
  Strokes agree on 8 of 13; disagreements are serve length (2, sheet shows a high serve: pilot looks right) and lift/drive,
  drop/clear (1 each). 17 of 26 hits marked unsure:stroke -> strokes from the rear camera stay mostly "unsure", as in v9.
- One hit had no row (noted by the reviewer) -> protocol: `missed_hit` lines, refined on a second sheet.

## Step 1 runs (resumed 2026-11-28 ~07:00 UTC)
- Batch (4 parallel) finished 6 videos: 0302_06 (doubles, 39 pipeline hits), 0310_03 (players stage nearly empty:
  players_both_frac 0.003, so hitter ids are mostly missing there), 0310_06, 7695, 7696, 7697 (108-119 hits each).
- 0623 and 7698 failed in the shuttle stage ("MPS command buffer ... victim of GPU error/recovery": 4 GPU jobs at once);
  re-run 2 at a time (runs/E, runs/F).
- Review rows: P/L/R/G 111-253 per video, sheets in review/sheets/<KEY>/; 10 review agents (one per <= 21 sheets), each
  given REVIEW_PROTOCOL.md and only the sheets/items (no prior labels).
- inputs/ = inputs_C (old 17, symlinks) + new runs; harness scripts/tune.py reproduces v7 exactly on it:
  v3b all 0.8604 / other 0.9102 / qa6 0.6920.
- Refine pass (30 gap/missed-hit rows, frames at -0.3/-0.1/+0.1/+0.3): 26 hit, 3 not, 1 unsure. Follow-ups fixed by me in
  labels/raw/own_20260310_06.txt: row 2 -> hit 4.20 (P2 lift); rows 32/33 (36.17/36.80) -> unsure_hit (full frames show the
  near player walking, racket down; detections likely from the neighbouring court).
- 10% second pass (119 rows, random per video, seed "11-<KEY>", independent reviewer that saw only the sampled rows, each row on
  its own): hit / no-hit agreement 97 of 97 rows that both passes decided (22 rows: one or both passes unsure); on the 51
  common hits contact time within 0.15 s 51/51, side 51/51, hitter id 51/51; stroke 38/46 (83%) where both typed it, 22/22
  where both marked the stroke sure. Caveat: both passes are Claude reviewers (correlated errors possible); sam's spot-check
  is the independent check. (ckpt/agree_pass2.txt)
- Fresh-set harness check: scripts/tune.py on train_v6 d_run inputs + labels/fresh_d.json reproduces v7 fresh 0.8208.
- Production pipeline (v3b, bundled models) on the first 6 new videos vs the new labels: F1 0.874
  (7695 0.919, 7696 0.951, 7697 0.943, 0302_06 0.738, 0310_06 0.727, 0310_03 0.779).

## A: rescuing videos without a court (2026-11-28, after sam's "增加训练内容")
- With Toby IMG_3504-3507 (~45 min, iPhone 1080p30): plumb-line check (white-line mask, undistort grid k1 in [-0.5, 0.1],
  Hough length score) gives no consistent k1 across the 4 files (best k1 0.00 / -0.50 / -0.20 / -), and the long floor lines
  near the left edge measure straight (slope -1.70 vs -1.69 along 450 px). So lens distortion is small; my earlier diagnosis
  (130 px residual) came from wrong line identities. The real problem: a diagonal corner view over 3-4 side-by-side courts
  whose baselines are collinear; the automatic post fitter (fit_court_posts.py, 62k hypotheses) and my hand reading both
  give inconsistent assignments (rescue/toby/). Not rescued; no undistort step added (not needed for these files).
- Diamond Bar 0310_02/04/05: floor-only SIFT transfer from 0310_03/06 courts finds 3-13 inliers (unusable); frames show the
  camera on the net line beside a post with players on both sides and people sitting on court (rescue/g_0310_04.jpg).
- Walnut 0320_0237: crowded team practice (10+ people on and around 3 courts); 0239's court does not transfer (other gym end,
  5 floor inliers). practice 0251: camera at the net post looking along the net (rescue view in courts/g_own_20260322_0251.jpg),
  near/far halves are left/right of the image: the pipeline's side model does not apply.
- Fairmont 0302_01/03/04/05 (blue tape courts like the labelled 0302_07): floor-only SIFT transfer of hs0302_07 gives 6-8
  inliers; the post+line fitter with the blue-tape mask finds no consistent hypothesis on 0302_05 (diagonal view, singles
  court partly hidden behind spectators' chairs).
- Not examined closely (thumbnails only, cand/grid1-2.jpg): Rowland 0312_01/02, Wilson 0319_0232/0234/0235, 0420_0255-0257,
  Walnut 0320_0236/0238, 0324_0240/0242, practice 0250/0252/0253. Thumbnails show the same kinds of view (crowded team
  practice, sideline / net-line camera, overlapping basketball lines); I stopped after the closer checks above all failed.
- Result: 0 rescued this round. Honest limit: these need sam's eye on which court is being played (or 4 clicks per video).

## C: augmented ShuttleSet clips built (scripts/aug_ss.py, seed 11)
- 88 succeeded train clips x 2 copies -> aug/ss_train_aug (266 clips incl. originals; 174 MB). Shuttle points kept after rotation +
  loss: ~55-75% of the original. Loader check (learn_hits.load_clips + collect): candidates and recall ceiling unchanged in
  kind (st02_set1: 15/17/14 candidates, ceiling 9/10 in all three). Calibration consistency: H_new maps the moved corners
  back to (0,0)/(6.1,0)/(6.1,13.4)/(0,13.4) exactly.
