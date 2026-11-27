# RallyIQ train-v7: tracker retune for the fine-tuned detector, dead balls, and alternation (2026-11-27)

Branch `claude/train-v7` = `claude/train-v6` + this work (not merged; nothing on main). Work folder `~/Desktop/RallyIQ/train_v7/`
(LOG.md with every step, labels/, review/final/ with INDEX.md, scripts/, ckpt/). The v6 fresh videos (0529_0005/0007/0009)
were used only as a held-out test at the end of each phase.

## Summary

| item | v6 | v7 | shipped? |
|---|---|---|---|
| Hit F1, 17 own videos (amateur_v6 labels, 526 hits), fully nested selection | 0.847 | **0.859**, 95% CI of gain over videos [+0.003, +0.024] | **yes** |
| ... the 11 non-qa6 videos | 0.891 | **0.908** [+0.005, +0.033] | |
| ... qa6 (blind, untouched) | 0.695 | 0.692 (one hit; CI [-0.013, 0.000]) | |
| Fresh held-out videos (88 hits) | 0.793 | **0.821** | |
| ShuttleSet test, 14 clips | 78.3 | **78.9** | |
| Doubles hitter attribution (detected hits) | 101/103 | 107/111 | |
| Labelled-window changes | – | +9 true hits, −7 false hits, 0 true hits lost | |
| ft1 detector + joint tracker / threshold retune (≈150 configs) | 0.821 | best 0.850, CI [-0.009, +0.015] | no |

Shipped (commit 115cf1c), events stage only; shuttle tracks and players unchanged:
1. **Alternation side flip** (`hits._alternating_subset`, `alternation_flip_penalty=0.2`, `alternation_flip_min_prox=0.4`):
   the dynamic programme that enforces near/far alternation may put a candidate on the other side's player who is near the
   shuttle, at a cost. 35 confidently scored true hits had been removed as "same side twice": in fast exchanges the nearest
   wrist at the contact belongs to the player who just hit.
2. **Dead-ball filter** (`events/dead_ball.py`, `dead_ball="auto"`): a logistic model on 16 rally-context features (gaps to
   the neighbouring hits, rally size and position, scorer confidence, both sides' swing evidence) removes isolated hits nobody
   answers: bouncing the shuttle on the racket, scooping it off the floor, tapping it over between rallies. Trained on the 11
   non-qa6 videos plus 100 dead-time detections I labelled (59 not a hit / 32 hit / 9 unsure).

## F: making the fine-tuned detector (ft1) pay off — not shipped

- In-memory harness (`scripts/tune.py`) reproduces v6 exactly (0.847 / 0.821) in about a minute per configuration.
- The 32 hits ft1 loses mostly have the same track points; the loss is rally context (e.g. one missing white-wall point makes
  the next hit a same-side repeat that the swing veto drops).
- Tracker grid on ft1 (peak threshold, strong score, foreign-court drop, gating, link skip; 72): best 0.825.
- Mapping ft1's scores onto the shipped detector's distribution: 0.823 (the distributions were already close).
- ft1 only on frames where the shipped detector has nothing: 0.849; union of both: 0.830.
- Events thresholds (keep, swing veto, exchange window; 72 with base and fill): best 0.850, CI [-0.009, +0.015]; fresh 0.793.
- Verdict: no configuration clears the keep rule; ft1 stays unshipped.

## G: false hits between rallies

The dead-ball filter alone, fully nested (inner LOVO picks C and threshold): 0.847 → 0.850, CI [-0.001, +0.007]; fresh
0.798 → 0.810; qa6 unchanged. Not enough alone; shipped together with H (below).

## H: missed fast exchanges and hidden players

106 labelled hits are missed: 15 have no candidate at all, 56 are scored low, 35 are scored >= 0.3 and then removed (4 by the
swing veto, 1 by the minimum gap, the rest by the alternation rule). A grid of minimum gap, pose-based recovery and veto window
gave nothing (best 0.8475). The side-flip rule alone: 0.848, CI [-0.004, +0.011]. Together with the dead-ball filter, both
selected inside the leave-one-video-out loop: 0.859, CI [+0.003, +0.024] — the only result this round that clears the rule.
Caveat: deciding to combine the two was one more choice made after seeing each alone.

## I: sheets

`review/final/INDEX.md`: every hit that changed (27 rows on the 17 videos, 13 on the fresh set), checked by me, with per-sheet
counts; the 100 dead-time labels; remaining issues.

## Labels

- `labels/iso_decisions.txt`: 100 isolated detections outside the labelled windows (training only).
- Fresh set: one row added after v7 found it (0007 269.20, real overhead; documented); the score is given with and without it
  (0.814 on the v6 rows vs 0.798; 0.821 vs 0.793 with it).
- qa6 untouched (one new qa6 detection looks like a real hit to me but stays counted as false).

## After phase I

ft1 on top of the new rules still loses (0.824); a nested threshold retune under the new rules is worse (0.856, CI below 0). Nothing more shipped.

## Remaining

Missed hits with no candidate (detector / track) or a low score; 7 between-rally next-court rows; 4 doubles hitter errors.
