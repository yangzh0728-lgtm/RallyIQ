# train-v10 log (started ~18:00 UTC; claude/train-v10 from claude/train-v9 c1f464f). Plan + pre-registrations: PLAN.md.

## R: rally sequence hit detector (TCN)
- scripts/r_feats.py: per-frame (~30 Hz) 27 inputs for 122 clips (88 ShuttleSet train, 14 test, 17 own, 3 fresh): shuttle
  (visible, x, y, dx, dy, confidence, frames since seen) and per side the player nearest the shuttle (wrists and hips relative
  to the shuttle in body heights, body height, wrist speed, wrist above shoulder, present).
- scripts/r_train.py: dilated TCN (8 layers, dilations 1..128, 64 ch; MS-TCN-style single stage), hit + side heads;
  pretrain on ShuttleSet train, LOVO fine-tune on own windows (outside windows masked), 3 seeds. scripts/r_eval.py: peak
  picking, threshold chosen on the other am7 videos' out-of-fold predictions; hitter side from the side head.
- A (pre-registered): own 17 videos 0.789 vs v7 0.860 (CI [-0.110, -0.039]); am7 0.825 vs 0.910; qa6 0.667 vs 0.692; fresh
  0.740 vs 0.821; ShuttleSet test (LH.score, +-3 frames, same side) **86.4 vs 78.0** for the v7 pipeline scored the same way.
- B (pre-declared variant: own weight 6, 3000 steps, 96 ch): own 0.765, am7 0.796, fresh 0.781, ShuttleSet **89.0**.
- => standalone sequence model loses on own footage (~0.07) but wins clearly on broadcast. Not shipped standalone.

## S: stroke classifier on pose + shuttle windows
- scripts/s_data.py: 46-frame windows (-0.5..+1.0 s) x 28 inputs for 1,644 ShuttleSet train strokes, 251 test, 333 own rows
  (v9 labels), 78 fresh. scripts/s_train.py: temporal CNN + attention pooling, ShuttleSet pretrain, LOVO fine-tune, 5 seeds.
- S-A: own trusted LOVO 71.4% vs court_v2 + residual 77.2% on the same 189 rows (CI [-0.114, -0.008]); fresh 55.6% = 55.6%
  (63 matched); ShuttleSet test 70.1%.
- S-B (pre-declared: residual layer with the classifier's out-of-fold log-probabilities as extra inputs): 75.7% < 77.2%.
- => not shipped; court_v2 + own residual stays.

## R fallback: TCN probability stacked into the scorer (not shipped)
- Out-of-fold TCN probabilities for every training row (own: LOVO models; ShuttleSet train: 4-fold models), feature seq_p =
  max TCN P(hit) within +-70 ms; v3b recipe retrained (v2+all+seq, 5 seeds, LOVO).
- Nested evaluation (v7 rules, nested dead-ball models): 0.860 -> 0.843, CI [-0.036, -0.001]; qa6 0.692 -> 0.670. Not shipped.

## R2: TCN as the broadcast-profile hit detector (not shipped)
- ShuttleSet-only TCN (88 train clips, 3 seeds); threshold 0.4 / gap 0.36 s chosen on ShuttleSet train out-of-fold F1 (0.863);
  exported to a numpy-only module (events/seqhits.py, kept in ckpt/unshipped_code/; numpy vs torch max diff 5e-7).
- Integrated (pre-registered: TCN peaks inside the rules pipeline's rallies +-0.3 s) on the 14 held-out ShuttleSet test clips
  (6 matches): F1 0.815 vs 0.780, per-match CI [-0.016, +0.069] (matches: +0.08, -0.09, +0.01, +0.02, +0.03, +0.10).
  A frame-time bug (rounded fps -> 300 ms drift on 30 fps clips) was found and fixed before this number.
- R2b (pre-registered after R2): mode chosen on ShuttleSet train OOF = the same rally-restricted mode (0.869 vs 0.863
  unrestricted) -> same test result -> fails. The unrestricted TCN scores 0.886 on the test clips (seen, not selected).
- Runtime: the TCN adds nothing measurable (14 clips, 9.3 min of video: 1.4 s vs 1.6 s for the events stage); feature
  extraction + numpy inference ~0.1 s per video minute. Own footage unaffected by construction (rules stay for profile own).
- Repo changes reverted; code, weights and diff saved under train_v10/ckpt/unshipped_code/ for a future pre-registered test on
  new held-out broadcast matches.

## 18:45 UTC follow-up 1: frame-time bug located
Only in my TCN integration (seqhits.apply used t = frame*1000/fps with fps estimated from integer-ms steps, ~1% off ->
~300 ms drift per 30 s). Shipped code: events/pipeline._fps_ms is only a per-frame duration (windows, quality, one-frame
hit offset; 0.3 ms error); every absolute frame<->time conversion (contracts VideoMeta.t_ms/frame_at, strokes features,
calibration tracker, web CorrectionPanel/corrections/Timeline/Workbench) uses the probe's exact fps. Nothing to fix in
shipped code; no hit F1 changes. seqhits in ckpt/unshipped_code already interpolates frame->time from observed pairs.
## 18:47 follow-up 2 pre-registered (PLAN.md), kf_bcast.py started.
## 18:58 follow-up 2 result (harness, pre-registered): 43 matches (102 clips; one of 44 matches has no loadable clip)
rules 0.7869, seq 0.8626, diff +0.0758, per-match CI [+0.0627, +0.0897]. Per fold thr/gap chosen inside: (0.4,.28) (0.2,.36)
(0.5,.36) (0.3,.28) (0.3,.28). Held-out ShuttleSet test clips alone: rules 0.780, seq 0.876.
## 19:03 production-path check (kf_export.py): prod 0.7856, diff -0.001 CI [-0.013, +0.011]: harness and shipped code disagree.
Cause 1: segment_events masks the shuttle in off-court views before the model sees it (model trained on raw track) -> fixed
(raw_shuttle). Still 0.785. Cause 2 (kf_sidecheck.py): time-only matches equal (prod tp_any 1683 vs harness 1678); the gap
is side: hitter() fell back to any player / None when the nearest pose frame had no player on the side head's side.
Fixed: nearest pose frame within 1 s that has a player on that side. -> prod 0.8112, diff +0.0244, CI [+0.0118, +0.0370].
Remaining gap to the harness is side again (no player track near the hit), the same limit the rules path has.
Final export: 3 seeds on all 102 clips, thr/gap (0.5, 0.2) chosen on pooled 5-fold OOF. Full suite: 486 passed, 12 skipped.
