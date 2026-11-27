# RallyIQ train-v6: doubles hitter, detector fine-tuning on own footage, fresh test videos (2026-11-27)

Branch `claude/train-v6` = `claude/train-v5` + this work (not merged, nothing on main). Work folder
`~/Desktop/RallyIQ/train_v6/` (PLAN.md, LOG.md with every step, labels/, review/final/ with INDEX.md, scripts/, ckpt/).
Plan and keep rules: `PLAN.md`. A change ships only if it wins on its own target **and** loses nothing on the gate
(17 own videos with leave-one-video-out scorers and per-video bootstrap CI, qa6 untouched, ShuttleSet, TrackNet).

## Summary

| item | v5 | v6 | shipped? |
|---|---|---|---|
| Doubles: true hitter has no player track at the hit (134 hand-anchored hits) | 4 (3.0%); v4 had 22 (16.4%) | 4 (3.0%) | – |
| Doubles: hitter attribution on detected hits | 99/103 (96.1%) | **101/103 (98.1%)** | **yes** (partner-swing rule) |
| Hit F1, 17 own videos, revised labels v6 (526 hits) | 0.847 | 0.847 | (rule does not move hits) |
| Hit F1, the 11 non-qa6 videos / qa6 | 0.891 / 0.695 | 0.891 / 0.695 | |
| Hit F1 on the v5 labels (amateur_v4) | 0.859 | 0.859 | |
| ShuttleSet test, 14 clips, hit F1 (rules) | 78.3 (65.0) | 78.3 (65.0) | |
| TrackNet public test, shuttle F1 / acc | tracks unchanged | tracks unchanged | |
| Fresh test: 3 never-labelled videos, 87 hits labelled by me | – | F1 **0.798** (P 0.80, R 0.79) | new held-out set |
| Sheet errors a / b / c / m (68 v5 sheets) | 2 / 8 / 0 / 3 | 2 / 8 / 0 / 3 (tracks unchanged) | |
| Off-court player frames | v5 | = v5 (players unchanged) | |
| Detector fine-tuned on own hard examples (ft1): own per-frame F1 | 0.919 | 0.962 (hard frames 0.575 -> 0.894) | **no**, see below |
| ft1: hit F1 end to end, 17 videos | 0.847 | 0.821 [-0.045, -0.008] | no |
| Hit scorer retrained on revised labels (v3b recipe, LOVO) | 0.847 | 0.834 [-0.028, +0.007] | no (v3b stays) |
| Stroke type, landing | unchanged inputs | not retrained | – |

## Phase A: the three v5 leftovers

**A1 Measure (doubles partner).** The hitter ids in the labels are pipeline ids of one run, so they break whenever the
tracks change. I made a pixel-anchored truth: for every labelled hit of the 4 labelled doubles videos (0310_01, 0239,
0241, 0622) the hitter's box at the hit frame (`labels/hitter_px.json`). 30 labels had no hitter; I picked the hitter on
3-frame strips with every raw pose detection numbered (`review/final/hitter/none_*`). I also checked the 9 label ids the
pipeline disagreed with (`chk_*`): 6 were wrong after the v4->v5 id remap and were fixed. Result: v5 already solved partner
*tracking* (no track at the hit: v4 22/134, v5 4/134, of which 2 hidden behind the partner with no detection at all), so
the remaining limit was *attribution*.

**A2 Fix.** Of the 4 attribution errors, 2 were partners at the net where the nearest wrist at the contact frame belonged
to the wrong one while the pose model clearly saw the other swing. New rule (`events/pose_hits.partner_swing`, config
`partner_swing_margin=0.2`, `partner_swing_prox_margin=0.3`): in doubles, a shuttle-based hit goes to the partner when the
partner's p_swing is >= 0.2 higher and their proximity is at most 0.3 lower. 99 -> 101/103, none broken; hit F1 identical;
11 other moved hits outside the truth set checked by eye: 6 better, 3 worse, 2 unclear. Unit tests added. The ideas from
the brief for tracking (lower threshold in the team's half, re-acquisition, gap bridging) were not built: only 2/134 truth
hitters are detected but unassigned, so there is nothing left for them to gain.

**A3 0245 white wall (S4, S15, S19).** Re-checked every detector candidate around the three hits and ran the detector on a
racket-local crop at 2.4x resolution with a 0.1 peak threshold: only peaks on the body / racket head. The shuttle is not
visible to me at those contacts either. Not fixable by search; left to B.

**A4 Next-court shuttle between rallies (8 rows).** All 8 fall inside a detected rally segment (the foreign flight itself
extends it), so a plain rally gate does nothing. A hit-anchored gate (keep points from first hit - pre to last hit + post)
removes 45-60% of the remaining foreign frames but also 3.7-11% of 6,115 hand-checked frames of ours (v5 bar: 2.6%).
Rejected. An "any of our players swinging" ownership feature did not separate (AUC 0.9736 -> 0.9737). The 8 rows stay; they
create no hits.

## Phase B: shuttle detector fine-tuning on own footage

- Labels (all mine): 377 frames where the track has a 1-8 frame gap inside a hit-to-hit flight, read on 100 px crops:
  206 shuttle visible at the interpolated spot, 40 visible elsewhere, 131 not visible. Easy positives: v5 track points in
  those flights that lie on a smooth path (two random samples of 48 read by eye; the first showed 5 off-shuttle points, all
  > 5 px off a smooth path, so that filter was added). qa6 never used. Held out: 0239 and 0622 whole, plus the last 25% of
  every other video. Rehearsal: 70 TrackNet Professional clips (OHEM-style weighting of hard positives x4; LwF-style replay).
- ft1 (AdamW 1e-4, BN frozen, 8 x 300 steps, a third TrackNet): own held-out per-frame F1@0.2 0.919 -> **0.962**
  (hard 0.575 -> 0.894); TrackNet val 0.956 -> 0.954. At the production peak threshold 0.3: TrackNet public test F1/acc
  93.9/91.8 -> 94.5/92.2; ShuttleSet 6 local test-match clips hit F1 61.1 -> 61.9, rally F1 73.9 -> 81.8.
- End to end it loses. With the harness made identical to production (per-segment background medians, threshold 0.3; it
  then reproduces the v5 candidates and v5's 0.716 on 0310_01 exactly): 17 videos 0.847 -> 0.821 [-0.045, -0.008];
  with the hit scorer retrained on ft1 tracks still 0.821 [-0.050, -0.003]; fresh videos 0.798 -> 0.739. Tried thresholds
  0.4 / 0.5 and "ft1 only fills the base detector's empty frames" (0.852 vs 0.855 on 5 videos): no win. Not shipped.
  Lesson: the tracker and scorer are tuned on the shipped detector's score distribution; a better per-frame detector needs
  the tracker re-tuned with it, which is a bigger job than this round.
- Note on the v5 TrackNet number (95.0 / 93.0): it came from saved candidates of an older checkpoint (loop/base.pt). The
  shipped heatmap.pt measured the same way at threshold 0.3 gives 93.9 / 91.8; the tracks are unchanged by v6.

## Phase C: downstream retraining

- Hit scorer, v3b recipe (5 seeds, `v2+all`, amateur weight 3), LOVO, on the v6 labels: 0.834 vs bundled v3b 0.847
  ([-0.028, +0.007]; qa6 0.656 vs 0.695). v3b stays bundled (same result as in v5).
- Hitter attribution: the partner rule above.
- Stroke type (court model + own residual) and landing / playing-style features: their inputs (tracks, players) are
  unchanged because ft1 was rejected, so there is nothing to retrain; v5 measured no stroke gain from cleaner tracks.

## Phase D: fresh held-out videos

own_20260529_0005, 0007 and 0009 (Quest courts, singles; run by own-v12 but never labelled; copied to the Desktop, players
replayed with the v5/v6 code from new pose dumps). Three 40 s windows per video at 20 / 50 / 80 % of its length, chosen by
position only. I read every pipeline hit and every rules candidate in them (52 sheets, 246 rows): 87 hits, 22 unsure.
v6: 69 TP, 17 FP, 18 FN, **F1 0.798** (0005 0.857, 0007 0.737, 0009 0.806). Most false hits are between rallies (a player
tapping, scooping or tossing the shuttle over while the other is not ready); most misses are fast net exchanges or a far
player hidden behind the near one; 7 rows track a next-court shuttle between rallies.

## Labels

- `labels/amateur_v6.json` = amateur_v4 + 30 rows (`labels/v6_decisions.txt`): detections of the re-detected tracks in
  labelled windows with no label, read on `review/final/unlabelled_detections/`: 14 hits, 9 not hits, 7 unsure. qa6 untouched.
- `labels/hitter_px.json`: doubles hitter boxes (reasons in `scripts/htruth.py`).
- `labels/shuttle_v6/`: detector hard-example decisions. `labels/d/`: fresh-video labels, one reason per row.

## Remaining known issues

See `review/final/INDEX.md`: the 8 between-rally next-court rows, 0245 S4/S15/S19, 2 doubles hits still credited to the
wrong partner and 2 hitters hidden at the contact, and on the fresh videos 17 false hits / 18 misses as above.

## References

PLAN.md lists the 16 references (the 12 of train_v3 plus ByteTrack arXiv 2110.06864 and OC-SORT arXiv 2203.14360, which
motivated the low-score association / re-acquisition options of A2, OHEM arXiv 1604.03540 for the hard-example weighting,
and Learning without Forgetting arXiv 1606.09282 for the TrackNet rehearsal in B).
