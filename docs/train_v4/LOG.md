# train_v4 log (autonomous run, leg 2, started 2026-11-22 04:40 UTC)

## Setup
- Worktree `repo/` = branch claude/train-v4 from origin/claude/train-v3 at ecbc2e9 (includes the coordinator's db.py fix).
- inputs/: symlinks to train_v3/inputs (own_v12, prelabels, public, videos, hits_truth*.json) except `hits_truth_v2.json`,
  which points at labels/qa6_retimed.json. New copies from the T7: inputs/new_runs/ (own-v12 artifacts + source.mp4 of
  four unlabelled 0906 videos: 0617, 0618, 0619, 0621), inputs/sources/ (source.mp4 of the 7 am7 videos),
  inputs/shuttleset_set/ (ShuttleSet annotation CSVs for the court stroke model).
- Jobs: nice -n 10, single-threaded BLAS, at most ~6 processes (14 cores).

## 1. qa6 re-timing (done ~04:55 UTC)
- Sheets review/qa6_retime/NNN.jpg: overview at the label time with the shuttle track over +-1 s (blue before, orange
  after, labelled every 0.2 s, rule candidates circled) + 16 hitter-side crops at -0.8..+0.7 s. The wider window and the
  drawn track made the contact readable where the 13-frame strips of leg 1 were not.
- Readings: labels/qa6_retime_readings.txt (58 rows, one-line reason each; 12 of leg 1's readings were kept unchanged).
  44 re-timed (offsets -0.40..+0.55 s; 21 of them > 0.25 s away), 2 also changed side (#95 far smash, #145 far lift),
  14 set to ignored: duplicates of a neighbouring label (#65, #122, #247, #251), dead balls / pickups (#73, #262),
  no contact of the labelled side (#9, #45, #46, #47, #125, #139, #172, #189).
- labels/qa6_retimed.json (scripts/build_qa6_retimed.py): qa6 now has 112 positives (was 126).
- Re-score, same predictions (scripts/rescore.py; LOVO models from leg 1 for v3/v3b, bundled v2):
| scorer | qa6 old truth | qa6 re-timed | qa6 re-timed P / R | am7+r3 | all 13 videos (re-timed) |
|---|---|---|---|---|---|
| rules | 0.493 | 0.578 | .47 / .74 | 0.706 | 0.664 |
| v2 | 0.556 | 0.650 | .60 / .71 | 0.822 | 0.768 |
| v3 (LOVO) | 0.600 | 0.706 | .67 / .75 | 0.830 | 0.788 |
| v3b (LOVO) | 0.610 | **0.718** | .69 / .75 | 0.838 | **0.798** |
  The gap v2 -> v3b on qa6 grows from +5.4 to +6.8 pts once label timing is fixed. The "loose" score is now nearly the
  same as the strict one (0.743 vs 0.718), as expected. All later qa6 numbers use the re-timed truth.

## 2. Labels
- Round 4 (held-out style, new venue/angle videos): 14 rallies in 4 unlabelled 0906 videos (0617 singles, 0618/0619/0621
  side camera; 0619 is listed as doubles but played as singles), chosen at random from the v3b segmentation; every v2 or
  v3b detection inside them reviewed on new sheets (scripts/sheets.py: overview with the shuttle path to the next hit, 3
  hitter crops, overview at the next hit) -> review/r4/. Decisions labels/r4/decisions.txt: 68 hits, 17 not hits,
  3 unsure; 32 stroke labels I am sure of, 22 more typed but unsure. One hitter error found (0621 148.05 near hit
  attributed to far P4); one swing detected twice (0621 149.55/149.98); two neighbour-court shuttles.
- Stroke pass over leg-1 hits without a trusted stroke (review/strokes1/, 4 hits per sheet with the current label in the
  title; decisions labels/strokes1_decisions.txt, one reason each): 114 of 174 read, 35 now trusted, 67 strokes changed
  or cleared (e.g. 0245 "cross_net"/"drop" that are lifts/clears; 0622 push vs lift; serves typed push). Skipped:
  0310_01 (four tiny players on a distant doubles court), most of 0302_07 (spectator in front), rest of 0241 and 0239.
- qa6 stroke pass (review/strokes_qa6/) stopped after 4 hits (doubles side view, 1 trusted): low yield.
- Round 4b/4c: 10 more rallies in 0619 (singles play) and 0621, same procedure (review/r4b, review/r4c; decisions in
  labels/r4b, labels/r4c): 0619 39 detections -> 25 hits, 12 not hits (5 neighbour-court shuttles, 2 pickups, walking),
  2 unsure; 0621 15 -> 13 hits, 1 not hit (shuttle held in the hand before a serve), 1 unsure. One more hitter error
  (0619 161.26: far player's lift given to the near player) and one serve only v2 found (0619 184.55).
- labels/amateur_v3.json (scripts/build_amateur_v3.py): 11 videos, 341 positives, 97 negatives, 264 hits with a stroke,
  **145 fully trusted strokes** (leg 1: 61): lift 54, smash 26, short_serve 19, clear 17, net_shot 15, long_serve 4,
  block 4, drop 3, drive 2, push 1. Short of the 150 goal by 5; the 119 typed-but-unsure strokes are used with a lower
  weight in step 4. Doubles hits: round 4 adds 3 doubles-format videos (0618/0621 are 2-v-2 courts filmed side-on).

## 3/4 interleaved (05:05-05:35 UTC)
### Round-4 held-out hit scores (4 videos never used in training; bundled models = held out)
scripts/subset_score.py on labels/amateur_v3.json, round-4 videos only (106 hits, every v2 or v3b detection labelled):
| scorer | F1 | P | R | false (all on labelled non-hits) | side | hitter |
|---|---|---|---|---|---|---|
| rules | 0.738 | 0.65 | 0.85 | 48 | 89/90 | 89/90 |
| v2 | 0.845 | 0.78 | 0.93 | 28 | 98/98 | 98/98 |
| **v3b** | **0.923** | **0.89** | **0.96** | **13** | 100/102 | 100/102 |
Second independent held-out confirmation of v3b (after leg 1's round 3: 0.783 -> 0.879).
All 17 videos (LOVO for the 13 old ones, bundled for the 4 new): rules 0.680, v2 0.785, v3b 0.828.

### Post-filter toggles on v3b (LOVO models, all 17 videos, re-timed qa6)
| variant | all F1 | qa6 | am | false |
|---|---|---|---|---|
| v3b (veto 0.1, self-feed 2000) | **0.828** | 0.718 | 0.865 | 85 |
| swing_veto 0 | 0.806 | 0.705 | 0.841 | 114 |
| swing_veto_keep_serve | 0.822 | 0.715 | 0.858 | 92 |
| self_feed_max_gap_ms 0 | 0.828 | 0.718 | 0.865 | 85 |
None helps. Miss census (v3b LOVO, leg-1 labels, 68 misses): 14 serves (first labelled hit of a window), 9 untracked
doubles partners (0241, 0622), 9 without any candidate within 0.25 s, 36 with a candidate the scorer rated low; 20 of the
misses have a candidate with p >= 0.3 that is still lost (traced next).

### Stroke types (step 4)
- Rows: scripts/stroke_rows.py -> ckpt/stroke_rows.pkl: 240 labelled hits matched to an events shot (events re-run
  with bundled v3b + own settings; 138 trusted). Per row: the 17 court features, the video-model probabilities, and
  new own-footage features: rise / fall (shuttle climb to its highest image point before the next hit and drop to the
  end of the flight, in hitter heights) and oh (highest wrist above highest shoulder at contact, torso lengths).
  Medians by class: lift rise 1.37 vs net_shot 0.45 / short_serve 0.31; oh: clear 0.74, smash 0.68, lift -0.61.
- Same rows for the 14 ShuttleSet test clips (stroke types from the ShuttleSet CSVs by frame, 216 matched):
  ckpt/stroke_rows_ss.pkl, to check that a change does not break broadcast footage.
- Baseline court_v2 (x video model, as in production): own videos 67.4% trusted (93/138), 58.3% all typed;
  ShuttleSet test clips 71.3% (154/216). Retraining the ShuttleSet-only model reproduces court_v2 exactly.
- A: HGB fine-tune with amateur rows (scripts/court_ft.py, LOVO): aw 2/5/10/20 -> 69.6/71.0/71.0/71.0%; unsure
  weight 0/0.5/1 at aw10 -> 72.5/71.0/71.7%; + overhead feature 73.2%; + rise/fall 74.6%; + both 75.4% (aw10),
  76.8% (aw20). The remaining top error stays lift -> net_shot (7).
- B: residual stage (scripts/court_resid.py): logits = log p(court x video) + W z + b with z = (rise, fall, oh, first,
  missing flags), L2 lam, trained on own rows only (unsure rows weight 0.5), LOVO:
  lam 0.001/0.003/0.01/0.03 -> trusted 77.5/77.5/75.4/75.4%, all typed 70.0/69.6/66.7/64.2%.
  ShuttleSet test clips with the all-own fit: 71.3 -> 63.0/65.3/66.2/67.1%. Subsets (lam 0.003): oh+first 76.8% own /
  66.7% SS; rise+fall 75.4 / 66.2. So the gain does not transfer to broadcast footage (image-space features + own
  class mix in b): the stage is opt-in for own footage only.
- Chosen: B, lam 0.003 -> rallyiq/strokes/models/strokes_residual_own.json, strokes option residual=auto|path
  (default off; court.own_shot_features computes z in production; scripts/check_residual.py confirms identical features).
  Own videos LOVO: 67.4 -> **77.5%** trusted (+14 hits of 138), 58.3 -> 69.6% all typed. Errors left: lift -> net_shot 4,
  short_serve -> net_shot 3, clear -> lift 2, long_serve -> lift 2.

## 3. Missed hits, continued (05:35-05:55 UTC)
- Trace (scripts/trace_miss.py, 32 misses with a candidate p >= 0.3): almost all are removed in hits.resolve_sequence by
  the alternation subset, not by the scorer or the swing veto. Typical serve cases: (a) a false candidate by the server
  2-3 s before the serve (feed/pickup) takes the chain, so the serve (same side) is dropped (0302_07 368.13); (b) the
  return is attributed to the server's side (receiver ambiguous in proximity), and the DP keeps the stronger of serve and
  return (0239 135.75, 0006 182.5). Untracked partners: the candidate gets the nearest tracked player's side and loses the
  same way (0241 285.17, 0622 1022.65).
- Tried (LOVO models, all 17 videos, all F1 / qa6 / other; base 0.828 / 0.718 / 0.865):
  ok_threshold 0.6/0.55/0.5/0.45 (strong candidates kept regardless of alternation): 0.828/0.828/0.824/0.823.
  New alternation_free_gap_ms (same side allowed after a gap) 1500/2000/2500/3000: 0.821/0.829/0.827/0.828
  (2000: qa6 +1.5 but held-out round 4 0.923 -> 0.909). Repair before the DP (hand-over to the other side, else drop;
  same_side_pre_gap_ms 1000/1500): 0.811/0.811 (more false hits). Existing same_side_min_gap_ms after the DP
  700/1000/1400: 0.830/0.830/0.828 (round 4 unchanged). free2000 + post1000: 0.831.
  Everything is within +-0.3 pt on all videos and nothing improves the held-out round 4, so no change is kept (the two
  new options were removed again). Leg 1 already found pose-based recovery (pose_hits=auto) unhelpful.
- Conclusion: the remaining misses are attribution errors (which side hit), not detection errors; a fix needs a better
  hitter/side signal for doubles partners (tracking), not looser sequence rules.

## 5. Error bars and speed
- Paired bootstrap over videos (scripts/boot.py, labels amateur_v3, re-timed qa6), v2 -> v3b:
  qa6 0.650 -> 0.718 (+0.068, 95% CI [+0.026, +0.113]); other 11 videos 0.829 -> 0.865 (+0.037, [-0.018, +0.078]);
  round 4 (4 new videos) 0.845 -> 0.923 (+0.078, [+0.053, +0.098]); all 17 0.785 -> 0.828 (+0.043, [+0.002, +0.077]).
  rules -> v3b all: +0.148 [+0.112, +0.186].
- Strokes, residual vs court_v2 (trusted, LOVO): +0.101, 95% CI over videos [+0.031, +0.159].
- Events speed (commit 692f726): profile of a 16-min video showed 83% of the 19.9 s in the swing features: per-tree
  GBT evaluation in Python (458k tree walks) and recomputing per-frame pose features for overlapping windows. Packed
  forest walk (all trees at once, tests/test_gbt_forest.py) + memoised pose steps: 19.9 -> 5.1 s; all 17 own videos'
  events in 81 s; predictions identical on 2672 hits. Leg 1's 1.8 s (v2, no swing) -> 12.7 s (v3) is now ~3-4 s.

## Hit scorer retrained on 17 videos (v3c)
- New candidate cache ckpt/cands4 (amateur_v3 labels, re-timed qa6; ShuttleSet caches copied from leg 1's cands3),
  scripts/train_hits_v3.py now takes TRAIN_INPUTS. Same recipe as v3b (v2+all features, h16, l2 1e-2, aw3, 5 seeds,
  thresholds 0.1/0.3/0.7). LOVO over all 17 videos (logs/lovo_C5s0/s5): all F1 0.830 / 0.829 (two seed sets), qa6
  0.714, other 0.871 / 0.869; vs v3b 0.828 (qa6 0.718, other 0.865). Bootstrap v3b -> v3c: all +0.002 [-0.009, +0.013],
  round 4 +0.008 [+0.001, +0.021], qa6 -0.004 [-0.016, +0.007]. ShuttleSet test 78.3 -> 78.1.
- Decision: equivalent; v3b stays bundled. v3c is kept as train_v4/models/hit_scorer_v3c_C5.json. 106 more labelled
  hits (+30%) did not move the hit F1, which agrees with step 3: the remaining errors are side/hitter attribution in
  the sequence rules, not the candidate score.
- ShuttleSet test (learn_hits eval, inputs/ss_test -> symlinks + own output folder): rules 64.9, v2 77.0, v3b 78.3
  (same as leg 1 with the faster code), v3c 78.1.

## Review sheets for sam
review/final/<video>_NN.jpg (68 sheets, 4 rows each): every round-4/4b/4c detection with MY LABEL (hit / NOT A HIT /
UNSURE, hitter/side, stroke, "(sure)") next to the detector's output and my one-line reason; and every leg-1 hit whose
stroke I re-read (MY STROKE vs the leg-1 label). Earlier working sheets: review/r4*, review/strokes1, review/qa6_retime.

## Side head (05:xx) and rear-camera strokes
- Side head (P(near end hit it), logistic on v2+all, trained on ShuttleSet + other own videos, LOVO; scripts/side_head.py):
  95% right on 441 labelled own positives, but it agrees with the nearest player almost always (2 disagreements >= 0.7
  in 0241, the worst doubles video), so attaching it changes nothing: all F1 0.830 at side_threshold 0.6-0.9, with or
  without the Viterbi side decoding (side_decode=false alone: 0.826, qa6 0.714 -> 0.687). Rejected.
- Stroke labels on the rear DJI camera (0529_0006, qa6 hits): only 2 of 8 strokes readable with confidence at this
  distance; stopped (labels/strokes_qa6_decisions.txt). The stroke residual is therefore validated on side/oblique
  cameras (11 videos) only; on the rear camera it is untested.
- Sanity check of the residual on unlabelled shots (class mix of all events shots, base -> residual):
  rear camera 0529 (784 shots): net_shot 26 -> 15%, lift 16 -> 31%, short_serve 13 -> 18%, block 9 -> 3%;
  side camera 0619/0621 (183): net_shot 25 -> 15%, lift 16 -> 29%. The same shift on both cameras, towards the
  labelled own mix (lift is 31% of the labelled own strokes); nothing collapses on the rear camera.
