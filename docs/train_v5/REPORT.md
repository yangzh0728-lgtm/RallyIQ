# RallyIQ train-v5: neighbour-court shuttles, trails and bystanders (2026-11-25/26)

Branch `claude/train-v5` = `claude/train-v4` + this work. Not merged; nothing pushed to main. Work folder
`~/Desktop/RallyIQ/train_v5/` (inputs copied from the T7, pose dumps, labels, sheets, logs). Review sheets for sam:
`review/final/` (68 sheets, `INDEX.md` with per-sheet counts, `check_v5.txt` with my reading of every sheet).

## Summary

| item | before | after |
|---|---|---|
| sam's marks (0616_00 S118/S119, 0616_01 row 1, 0616_03 S130, 0617_00 r4#0-3, 0241_01 P3n) | wrong | all fixed except the NEXT panel of S130 (between rallies) |
| Sheet rows, (a) right shuttle but trail from another court (264 rows) | 29 | **2** |
| Sheet rows, (b) circle on another court's shuttle at the hit or NEXT | 41 | **8** (all between rallies) |
| Sheet rows, (c) a spectator / bystander boxed as a player | 37 | **0** |
| Sheet rows, our shuttle missed at the hit (new problem) | – | 3 (0245, white walls / window light) |
| Hand-labelled next-court trajectory frames still kept in the track (12,489) | 12,453 (99.7%) | **2,084 (17%)** |
| Our trajectory frames lost doing so (6,379 hand-checked) | 7 | 165 (2.6%) |
| Hit-to-hit pseudo-truth flights, frames on the wrong shuttle (6,442) | 67 | **13** |
| Player boxes > 0.3 m past a sideline in labelled windows (9 replayed videos) | 5,963 frames, 33 runs | **167 frames, 4 runs** |
| Hit F1, all 17 own videos, revised labels (512 hits), bundled v3b | 0.796 | **0.859** (95% CI of the gain over videos [+0.021, +0.112]) |
| Hit F1, the 11 non-qa6 videos | 0.822 | **0.909** [+0.037, +0.150] |
| Hit F1, qa6 (6 videos, sam's blind labels, unchanged) | 0.714 | 0.695 [−0.072, +0.009] |
| ShuttleSet test hit F1 (14 clips) | 78.3 | **78.3** (rules 65.0 / 65.0) |
| TrackNet public test, shuttle F1 / accuracy (29 clips) | 95.0 / 93.0 | **95.0 / 93.0** |
| Stroke type, own videos, trusted labels, residual on (LOVO) | 77.4% | 77.2% (no change) |

## What was wrong

1. **Neighbour-court shuttle chosen over ours (b: S118, S119, 0616_01 row 1).** The detector had seen the real
   shuttle in every case (it was in `shuttle_candidates.json`). The tracker linked candidates into tracklets and kept
   the heaviest by score sum. The next court's rally was often longer and brighter, and the court lock could not
   reject it: from a camera behind our near corner, the next court's air space overlaps ours in the image, so the
   region test gave it full weight. At S119 a linked tracklet that wandered between a racket, the net and the
   neighbour's shuttle outweighed, by length alone, the clean flight from P2n's racket. The "static spot above the
   net" sam saw was part of that wandering tracklet, not a fixed light.
2. **Trail mismatch (a: 0617_00 r4#0-3, 0616_03 S130).** The sheet drew every track point in the time window, so
   when the track switched to another court between two of our hits, the trail showed the other shuttle. That is
   partly the drawing and partly the same tracker problem (the track itself contained the other court's points).
3. **Bystanders as players (0241_01 P3n).** Three causes: (i) the 0241 and 0245 court lines were about 1 m off (the
   old manual corners), so people by the bleachers mapped inside the play area; (ii) 0618, 0619, 0620 and 0621 are
   singles but were tracked as doubles (format decided on the first 3-6 s, when onlookers stood by the net), so the
   two spare slots went to whoever was nearest; (iii) a free slot could be refilled with any person within the run-off,
   including one standing past the sideline or across the court, so P3n hopped between spectators.
4. **A doubles bug found on the way.** The swing evidence for the hit scorer (`p_swing_hitter`) read only the
   lower-id player of each side. Renumbering the players (fix 3) moved the hitter out of that slot and halved the
   swing evidence on 0622/0241. Now every player of the side is read.

## What changed (code, all on `claude/train-v5`)

- `rallyiq/perception/shuttle/ownership.py` (new): tells our rally from another court's using our players only:
  (1) a shuttle flies in the vertical plane through two contact points, so its image x stays between our outermost
  players (plus reach); (2) it turns only at a swinging wrist of ours. Logistic weights fitted on 1,072 hand-labelled
  tracklets of the 17 own videos (223 next-court, my labels from mosaics in `review/tk/`) plus all ShuttleSet training
  clips (as "ours"); leave-one-video-out AUC 0.98. A third input, *concurrency* (share of the time with two
  shuttles in flight), switches the test off where only one court is playing, so broadcast footage is unaffected.
- `tracker.py`: `_ownership` drops trajectories and cut-out flights another court is playing and down-weights
  doubtful ones; `_challenge` lets a clean hit-to-hit flight take its frames back from a longer tracklet that
  scores lower point by point. Unit tests: `tests/perception/shuttle/test_ownership.py`.
- Sheets (`scripts/sheets5.py`): the trail is the hit's own flight (runs continuous with the shuttle at the hit,
  carried over gaps only where the next run continues the motion).
- Players: format probe 450 frames and people past a sideline not counted (`match_format.py`); a slot following
  someone past the sideline (>0.6 m for 1.5 s, or a little for 4 s) is released and the spot ignored; no
  re-acquisition jump over 5 m; spectator-run filter on by default (`tracker.py`, `postprocess.py`, `runner.py`).
- Court lines re-fitted for 0241 and 0245 from the painted lines (`inputs/fixcal/`); used by the replays.
- Events: per-player swing series in doubles (`events/pose_hits.py`, `strokes/detail.py`).

## Measurements

### Labels

Better tracks exposed real hits the old tracks hid, so a label pass was needed before any F1 comparison. Three
rounds, every row read from a sheet, one-line reason each (`labels/v5_decisions.txt`):

1. hits the new pipeline found in labelled windows with no label (16 videos except qa6);
2. the same after the doubles fix (0241, 0310_01, 0621, 0622);
3. for fairness, every detection of the **old** pipeline in labelled windows with no label (11 rows).

Result `labels/amateur_v4.json`: 400 positives (was 341), 107 negatives (was 97), 59 added hits, 7 retimed, 3
dropped duplicates, 4 status changes. qa6 (sam's blind set) was **not** touched. Caveat: the new pipeline found
more of the new hits, so the revised labels still favour it somewhat. On the unrevised v3 labels the new tracks score
lower (0.812 → 0.773, [−0.066, −0.013]): their extra detections are counted as false there.

### Hits (bundled v3b; LOVO models for the 13 leg-1 videos, bundled for the 4 round-4 ones)

| set | old tracks | new tracks + players |
|---|---|---|
| all 17, F1 (P / R), revised labels | 0.796 | **0.859** |
| other 11 | 0.822 | **0.909** |
| qa6 | 0.714 | 0.695 |
| dead-ball false hits | 50 | **36** |
| hitter side right | 383/396 | 432/440 |

Largest per-video gains: 0241 0.58 → 0.87, 0616 0.74 → 0.96, 0617 0.74 → 0.96, 0621 0.83 → 0.97, 0310_01 0.67 → 0.80.
Losses: 0615 0.94 → 0.75 (two hits at 709.9 and 710.3 s now missed), 0006 0.76 → 0.72 (one hit).

**Retraining** the scorer on the cleaned tracks and revised labels (same v3b recipe, 5 seeds, LOVO) scored 0.838,
0.022 lower than keeping v3b [−0.035, −0.010]. v3b stays bundled.

### Strokes and landing

Stroke rows rebuilt on the new tracks (270 matched labelled hits, 136 trusted). Court model alone 67.9% → 66.9%,
with the own-footage residual 77.4% → 77.2% (lam 0.003); ShuttleSet test clips 71.3% either way (residual 66.2/64.4%).
The cleaner trails did not change stroke accuracy, so the shipped residual is kept. Landing is not a trained model
here (it is a rule on the track inside `strokes/features.py`), and there is no landing ground truth; on the last
shot of every labelled rally the share of landing points more than 1 m off our court went from 57% to 49%.

## Remaining errors (from my reading of all 68 sheets)

- 8 rows where the circle is on another court's shuttle, all **between rallies** when nobody of ours is hitting
  (0617_02 r4#9, 0618_00 r4#0/#1, 0618_03 r4#13, 0619_10 r4#14, 0619_14 r4b#34, 0616_01 S122 NEXT, 0616_03 S130 NEXT).
  With no hitter of ours there is no evidence either way; these do not create hits.
- 2 trails that are a next-court arc crossing between our two players (0619_08 r4b#19/#20, between rallies).
- 3 hits where our shuttle is missed (0245 S4, S15, S19): a white shuttle against white walls and window light.
- 0615 lost two quick hits; untracked doubles partners are still the main hit-attribution limit (0622, 0241).
- 165 of 6,379 hand-checked frames of our own shuttle are dropped (2.6%), mostly short pieces near the frame edge.

## How to reproduce

`scripts/` in the work folder: `rt.py` (retrack from saved candidates), `eval_tr.py` (track metrics), `dump.py` +
`replay.py`/`make_players.py` (pose once, replay tracking), `build_inputs.py`, `rescore.py`, `boot5.py`,
`render_final5.py` + `index.py` (sheets), `count_ab.py`, `bystat.py`, `scripts5/stroke_rows.py`, `court_resid.py`.
Full log of decisions in `labels/v5_decisions.txt` and `review/final/check_v5.txt`.
