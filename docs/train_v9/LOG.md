# train-v9 log (started 2026-11-27 17:40 UTC; claude/train-v9 from claude/train-v8 f799199). Plan: PLAN.md.

## N: own stroke labels and a depth-robust residual (labels kept, no model shipped)
- Reviewed all 268 labelled am7 hits whose stroke was not sure (review/n/s_NN.jpg, contact + next-hit frame with the flight;
  labels/n_decisions.txt, one reason each): 157 typed (66 confirmed, 68 newly typed, 23 corrected, e.g. 0621 109.48 was
  short_serve, actually a long serve), 111 left unknown (tangled/partial tracks, doubles net exchanges, rear-camera depth).
  Sure rows 146 -> 197 (typed 292 -> 360 of 414). The brief's 400 trusted rows are not supported by what is visible; I did
  not guess. qa6 not typed.
- Rows rebuilt on the v7 events (scripts/n_rows.py): 333 typed matched (189 trusted).
- Pre-registered residual v2 (+ time of flight, receiver depth / movement, early image speed, wrist speed), lambda nested:
  trusted LOVO 74.1% vs shipped-feature residual 77.2% (court_v2 alone 64.6%). Not shipped.
- N2 (pre-registered): the shipped residual refit on the larger v9 rows vs the bundled one on the fresh set: 52.4% vs 55.6%
  (sure 60.9% vs 65.2%, n=23). Not shipped; the bundled residual stays.

## O: playing-style stability (shipped: uncertainty in the report)
- scripts/o_style.py: 14 own videos (11 am7 + 3 fresh), full-video v7 events + strokes with the own residual, style features
  per player (22 player-sessions with >= 20 classified shots; 29-121 shots each), 1,000 rally-bootstrap resamples.
- Median 95% CI width in points: smash 18.7, net 20.1, clear 15.5, drop 11.4, lift/block 23.7, attack 20.2, defense 11.4.
  The prototype ranges in report/style.py are 10-25 points wide, so one session rarely separates two styles.
- Same player across sessions: the near player in the S.YANG jersey (0616 P2, 0619 P2, 0621 P1): smash 18 [9,26] / 17 [8,29]
  / 21 [10,31]; net 24 [16,37] / 35 [23,47] / 17 [6,29]; clear 13 / 9 / 10; attack 26 / 22 / 24. All three intervals overlap
  for every share: consistent with one stable style, and also with real differences of 10 points that a session cannot show.
- Shipped (540c5fa): style share metrics carry "95% 区间 lo–hi%（按回合重抽样）" in their note (200 seeded resamples,
  only when the style is reported). Test added. Proposal (not built): pool sessions of the same player for the style verdict
  (needs a cross-video player identity, which the app does not have yet).

## P: missed net-exchange hits (nothing shipped)
- Pre-registered rule (PLAN.md): candidates with the best player's foot within 1.5 m of the net and a wrist within 0.35
  heights of the shuttle admitted from scorer prob 0.05 (not 0.1) and with keep threshold 0.2 (not 0.3) in the alternation
  programme. It fired on 22 of 585 candidates in 0619 and changed no hit on any of the 17 videos (0.8604 both). Reverted.
- Why it can't help: (v7 h_fn) of the 106 missed hits, 15 have no candidate and 56 are scored < 0.3; those low-scored net
  contacts mostly have the wrist further than 0.35 heights from the tracked shuttle point (tracking lag at the contact),
  and the ones below 0.1 are below even 0.05. A real fix needs the contact itself to be seen better (detector/track),
  which v6-v7 showed the current tracker cannot use yet.
