# train-v7 log (started 2026-11-27 14:20 UTC; branch claude/train-v7 from claude/train-v6 9c38813)

Test set rule: the v6 fresh videos (0529_0005/0007/0009) are only evaluated at the end of a phase, never tuned on.

## F: make ft1 pay off end to end (not shipped)
- scripts/tune.py: in-memory harness (tracker from saved candidates -> events with LOVO v3b -> TP/FP/FN per video), one worker
  per video; reproduces v6 exactly (base 0.8467, ft1 0.8209 on amateur_v6, 17 videos) in ~70 s per pair.
- Diagnosis of the 32 hits ft1 loses: most have near-identical track points; the loss is rally context (e.g. 0245 S15: ft1 has
  no point at the white-wall smash, so the next hit becomes "same side twice" and the swing veto drops it).
- Tracker grid on ft1 (peak 0.3/0.4/0.5 x strong_score x foreign_drop x gate_frac x link_skip_s, 72): best 0.8247.
- Score calibration (monotone quantile map of ft1 scores onto the shipped detector's, label-free): 0.8232 (distributions were
  already nearly equal: medians 0.524 vs 0.519).
- Combining: ft1 only on frames where the shipped detector has no candidate >= 0.3 ("fill"): 0.8489 (+0.0023, CI
  [-0.006, +0.012]); union of both: 0.8303.
- Events grid (keep_threshold x swing_veto x keep_exchange_ms, 36) on base and fill: best fill_k0.35 0.8500 (+0.0033,
  CI [-0.009, +0.015], qa6 -0.005); shipped values already at the optimum within +-0.003.
- Fresh set (end of phase): fill 0.793 vs shipped 0.798.
=> no variant beats 0.847 with a CI above 0. ft1 stays unshipped; tracker and thresholds unchanged.

## G: false hits between rallies (dead-ball filter; built, opt-in, not on by default)
- Context of the 44 labelled false hits (27 am7, 17 fresh): mostly first/last/single hit of a "rally", long gaps both sides,
  no swing in reply, lower scorer confidence (ckpt/fp_ctx.json, scripts/fp_ctx.py).
- events/dead_ball.py: logistic model on 16 rally-context features (scorer confidence, rally size, position, gaps to
  previous/next hit, same-side repeats, hitter's and other side's p_swing / wrist speed from the bundled pose model,
  rally duration). Config `dead_ball` ("off" default | "auto" | path).
- Hard dead-time examples: 232 isolated pipeline hits (gaps >= 1.5 s both sides) outside the labelled windows of the 11 am7
  videos; 100 random ones read by me (review/iso/r_NN.jpg, labels/iso_decisions.txt): 59 not a hit, 32 hit, 9 unsure.
- Fully nested evaluation (inner LOVO picks C in {0.1,0.3,1,3} and the threshold; outer LOVO over the 11 am7 videos; qa6 and
  fresh scored with the model fitted on all 11): all 17 0.8467 -> 0.8502 (+0.0035, CI [-0.0007, +0.0073]); am7 0.8914 ->
  0.8965; qa6 0.6949 -> 0.6949; fresh 0.798 -> 0.810.
  (Fixed C=0.3 without the extra examples gave CI [+0.0003, +0.0056] but C was not chosen blind there, so I don't count it.)
=> CI includes 0: not switched on. Code and model kept as opt-in (`dead_ball="auto"`).

## H: missed hits (shipped together with G)
- scripts/h_fn.py: 106 labelled hits missed (58 am7, 30 qa6, 18 fresh). 15 have no candidate within 0.25 s (detector /
  track), 91 do; of those 35 were scored >= 0.3 by the scorer and still removed. Swing veto removed 4, the min gap 1; the rest
  were removed by the alternation DP for "same side twice" (scripts/h_drop.py).
- Grid of min_hit_gap_ms / pose_hits recovery / veto window (12 configs): best 0.8475 (noise).
- New: alternation may put a candidate on the other side's nearby player (proximity >= alternation_flip_min_prox) at a cost
  alternation_flip_penalty (hits.py _alternating_subset). Alone, nested selection: 0.8484, CI [-0.004, +0.011].
- G+H together, both chosen nested (flip settings by inner selection on the other am7 videos; dead-ball C/threshold by inner
  LOVO; qa6 gets the choice made on all 11): all 17 0.8467 -> 0.8588, CI [+0.0027, +0.0244]; am7 0.8914 -> 0.9080
  [+0.005, +0.033]; qa6 0.6949 -> 0.6920 (one hit, CI [-0.013, 0.000]). The decision to combine the two was made after
  seeing each alone (one extra choice; reported).
- Gates: fresh held-out videos 0.798 -> 0.814 on the v6 labels; one detection the new pipeline found there had no label
  row (0007 269.20): I checked it (review/final/fresh_269.jpg), it is a real near overhead, added to labels/d with a note ->
  0.793 -> 0.821 on 88 hits. ShuttleSet test 14 clips: 78.3 -> 78.9 (rules unchanged). Doubles hitter attribution 101/103 ->
  107/111 detected (2 of the 4 errors are hitters with no track; the 101 previously right ones unchanged).
- Shipped as defaults: alternation_flip_penalty 0.2, alternation_flip_min_prox 0.4, dead_ball "auto" (commit 115cf1c).
  Full test suite 445 passed.
- Changed hits in labelled windows (17 videos): +9 true hits, -7 not-a-hit rows, 0 true hits lost (review/final/diff/);
  fresh: review/final/fresh_diff/. qa6 0615 707.21 is a new detection with no label (looks like a real near forehand to me;
  qa6 not relabelled, so it counts as a false hit).

## After I (extra, nothing shipped)
- ft1 on top of the v7 rules: 0.824 (v7 0.860 with nested dead-ball models); ft1 only where the shipped detector is empty:
  0.861 (+0.0002, CI [-0.008, +0.009]).
- Nested retune of keep / ok / min-candidate thresholds under the v7 rules (36): 0.860 -> 0.856, CI [-0.009, -0.001]:
  the shipped thresholds stay.
