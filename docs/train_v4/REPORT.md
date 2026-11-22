# RallyIQ train-v4 report (autonomous run, leg 2, 2026-11-22)

Branch `claude/train-v4` = `claude/train-v3` (PR #84, including the coordinator's db.py fix ecbc2e9) plus this
leg's commits. Not merged; nothing pushed to main. Work folder: `~/Desktop/RallyIQ/train_v4/` (inputs copied from the
T7, labels, review sheets, checkpoints, logs). Full experiment log: `LOG.md`. Leg 1: `train_v3/REPORT.md`.

## Summary

| item | result |
|---|---|
| qa6 benchmark | 58 unrefined hit times re-read from frames: 44 re-timed, 2 sides fixed, 14 dropped (duplicates, dead balls, no contact). With fixed labels, v3b's qa6 F1 is **0.718** (0.610 with the old labels). |
| Labels | Round 4: 24 new rallies in 4 videos never labelled before (106 hits, 30 non-hits). **146 fully trusted stroke labels**, up from 61; 265 hits have a stroke type. |
| Held-out hit F1 on the 4 new videos | v2 0.845 → **v3b 0.923**: a second independent confirmation of v3b. |
| Missed hits | Diagnosed: most confident misses are dropped by the alternation rule after a wrong side or hitter attribution. About 15 variants tried (sequence rules, serve handling, swing veto, a learned side head); none helped, so none is kept. |
| Stroke types | New opt-in own-footage stage. Leave-one-video-out: **67.4% → 77.5%** on trusted labels, 58.3% → 69.6% on all typed labels. Off by default because it costs 5–8 points on broadcast clips. |
| Events speed | **19.9 s → 5.1 s** on a 16-minute video, with identical output. |
| Hit scorer retrained on all 17 videos (v3c) | Same F1 as v3b (+0.002, CI [−0.009, +0.013]). v3b stays bundled. |

## Before / after

### Hits

Truth for every row below: re-timed qa6, `labels/amateur_v3.json`, F1 at ±0.25 s. Own-footage settings
(swing_veto 0.1, keep_exchange 2700 ms, min_no_hit_rally 2500 ms). Each of the 13 leg-1 videos is scored by a model
trained without it (leg-1 leave-one-video-out models). The 4 round-4 videos are scored by the bundled models, which
never saw them.

| set | rules | v2 (before leg 1) | **v3b (bundled)** | v3c (17-video retrain, LOVO) |
|---|---|---|---|---|
| qa6, 6 videos, 112 hits | 0.578 | 0.650 | **0.718** | 0.714 |
| other 11 own videos, 341 hits | 0.716 | 0.829 | **0.865** | 0.871 |
| round 4: 4 new videos, 106 hits (held out) | 0.738 | 0.845 | **0.923** (P 0.89, R 0.96) | 0.932 |
| all 17 videos, 453 hits | 0.680 (P 0.53, R 0.83) | 0.785 (P 0.66, R 0.88) | **0.828** (P 0.75, R 0.84) | 0.830 |
| false hits on labelled non-hits (pickups, passes, neighbouring court) | 59 | 100 | **40** | 41 |
| hitter side right (matched hits) | 364/376 | 388/397 | 367/380 | 370/381 |
| hitter id right (am videos) | 281/289 | 298/305 | 286/294 | 287/293 |
| ShuttleSet test v3, 14 clips (±3 frames, same side) | 64.9 | 77.0 | **78.3** | 78.1 |

Paired bootstrap over videos, v2 → v3b: qa6 **+0.068** [95% CI +0.026, +0.113]; other videos +0.037 [−0.018, +0.078];
round 4 **+0.078** [+0.053, +0.098]; all 17 **+0.043** [+0.002, +0.077]. Rules → v3b, all 17 videos: +0.148 [+0.112, +0.186].

Effect of the re-timing alone (same predictions, qa6 F1, old → re-timed labels): rules 0.493 → 0.578, v2 0.556 → 0.650,
v3 0.600 → 0.706, v3b 0.610 → 0.718. Under the old labels, about half the "misses" were wrong label times, so
every model looked worse. The v2 → v3b gap grows from +5.4 to +6.8 points.

### Stroke types

Production combination of the court model and the video model (geometric mean). Each own video is scored by a model
fitted without it.

| set | court_v2 (before) | court_v2 + own residual (after, opt-in) |
|---|---|---|
| own videos, 138 trusted labels (11 videos) | 67.4% (93) | **77.5%** (107); gain 95% CI over videos [+3.1, +15.9] pts |
| own videos, all 240 typed labels | 58.3% | **69.6%** |
| ShuttleSet test clips, 216 strokes | 71.3% | 65.3%, so the stage stays **off** for broadcast footage |

Remaining stroke errors (trusted, after): lift → net_shot 4, short_serve → net_shot 3, clear → lift 2, long_serve → lift 2.
Before: lift → push 7, lift → net_shot 7, short_serve → net_shot 4, short_serve → push 3.

### Speed

Events stage of one 16-minute own video: 19.9 s → **5.1 s**. All 17 own videos now take 81 s in one process. Predictions
are identical on 2672 hits; ShuttleSet test numbers are unchanged. Leg 1 had grown this stage from 1.8 s (v2) to 12.7 s (v3).

## What was done

1. **Benchmark hygiene.** I built new sheets (`review/qa6_retime/`): the shuttle path over ±1 s drawn on the overview
   (labelled every 0.2 s, rule candidates circled) plus 16 hitter crops from −0.8 to +0.7 s. With the drawn track
   and the wider window, contacts that leg 1's 13-frame strips could not show became readable. Readings, with a reason for
   each, are in `labels/qa6_retime_readings.txt`; the result is `labels/qa6_retimed.json`.
2. **Labels.** Round 4/4b/4c covers 24 rallies in 0617 (singles), 0618 and 0621 (doubles courts, side camera) and 0619
   (side camera). I reviewed every v2 or v3b detection inside them, so the set favours neither model. Separately, I re-read
   the strokes of 114 leg-1 hits that had no trusted type: 36 are now trusted and 71 types were changed or cleared.
   Merged file: `labels/amateur_v3.json`. **Review sheets for sam: `review/final/<video>_NN.jpg`** (68 sheets, my label
   and reason written on every row).
3. **Missed hits.** A trace of the 32 misses that had a candidate scored p ≥ 0.3 shows them removed by the rally-level
   alternation rule, not by the scorer. Two patterns: a false "hit" by the server 2–3 s before the serve takes the
   serve's place, or the return is credited to the server's side and only the stronger of the two survives.
   Untracked doubles partners fail the same way. I tried about 15 variants: ok_threshold 0.45–0.6, a new gap-tolerant
   alternation, a repair before or after alternation, swing-veto changes, and a learned side head (P(near end hit it),
   95% right, but it almost always agrees with the nearest player). All landed within ±0.3 pt on 17 videos and none improved round 4, so none is kept.
4. **Stroke types.** `scripts/train_v4/stroke_rows.py` builds rows for each labelled hit: the court features, the
   video-model probabilities and three new own-footage features. *rise* and *fall* are the shuttle's climb to its highest image
   point before the next hit and its drop afterwards, in hitter heights; *oh* is the highest wrist above the highest shoulder at
   contact. Medians: lift rises 1.37 hitter heights versus 0.45 for net_shot; overheads have oh around +0.7, lifts −0.6.
   I tried two designs:
   (A) fine-tuning the court GBT with weighted own rows, best 76.8% (aw20 + rise + oh);
   (B) a small residual layer, logits = log p(base) + W·z + b, L2-regularised and fitted on own rows only, 77.5%.
   I kept B because it leaves the ShuttleSet-trained model unchanged. Both lose 5–8 points on ShuttleSet clips (the
   features are image-space for one camera family), so B ships as the strokes option `residual=auto` (model file
   `rallyiq/strokes/models/strokes_residual_own.json`), default off, to be set in the own-footage batch settings.
5. **Speed and error bars.** A profile showed 83% of the events time in the pose-swing features. Two exact rewrites
   fix it: all GBT trees are now walked together instead of one Python loop per tree (`rallyiq/gbt.py`, with a test),
   and per-frame pose features are memoised. Error bars come from paired bootstraps over videos (`scripts/train_v4/boot.py`).
   The 5-seed ensembles were re-run as two disjoint seed sets for v3c: 0.830 / 0.829, so seed noise is negligible (leg 1 found the same).
6. **Retrain (v3c).** Same recipe on all 17 videos with the corrected and new labels (+30% own hits). It is equivalent
   to v3b (table above), so v3b stays bundled; v3c is kept in `train_v4/models/hit_scorer_v3c_C5.json`.

Commits on `claude/train-v4`: 94aa8dc (strokes residual stage), 692f726 (events speed), bbdcbbf (scripts),
plus docs and labels.

## Caveats

- Round-4 labels review only the v2 and v3b detections. Hits that neither model found are not labelled, so recall there
  is optimistic for both. Round 3 in leg 1 had the same design.
- The stroke residual is validated on side and oblique cameras (11 videos). On the rear DJI camera (0529) I could read only
  2 of 8 strokes with confidence, so it is untested there. On unlabelled 0529 shots it shifts the class mix the same way
  as on side views (net_shot 26 → 15%, lift 16 → 31%) and nothing collapses.
- 146 trusted strokes is just short of the 150 target. Serve types (long_serve 4, short_serve 19) are still thin.
- `tests/contracts/test_contracts.py::test_contract_files_up_to_date` fails locally on the base commit too (stale
  MatchStats/MatchReport schema files). The rest of the suite passes: 431 passed, 12 skipped.

## Next steps

1. Hitter and side attribution in doubles: track partners and spectators near the court. This is now the main limit
   on hit F1: missed serves and partner hits are dropped by alternation after a wrong attribution.
2. sam: check the sheets in `review/final/` (round-4 hit/non-hit calls and stroke types) and `review/qa6_retime/` with
   `labels/qa6_retime_readings.txt`.
3. If the own-footage batches should use the stroke residual, add `residual: auto` to their strokes options; it is
   untested on the rear camera.

## References (same 12 as leg 1, where they apply)

- Hit-frame detection from shuttle trajectory and pose: *Automated Hit-frame Detection for Badminton Match Analysis*
  (arXiv 2307.16000). It is the basis of the candidate + learned scorer and the pose-swing evidence whose cost this leg cut.
- Shuttle trajectories: TrackNet (arXiv 1907.03698), TrackNetV3 (MMAsia 2023), TrackNetV4 (arXiv 2409.14543). The
  rise/fall stroke features read those trajectories. MonoTrack (CVPRW 2022, arXiv 2204.01899) covers court-space reasoning on amateur-style footage.
- Data and stroke taxonomy: ShuttleSet (KDD 2023, arXiv 2306.04948) and CoachAI-Projects (arXiv 1907.12888). ShuttleSet
  trains the court model and provides the stroke check on the 14 test clips.
- Pose: RTMO (CVPR 2024, arXiv 2312.07526) supplies the keypoints behind the overhead feature. TemPose (CVPRW 2023) and
  BST (arXiv 2502.21085) motivate pose features for stroke type.
- Domain shift: DANN (arXiv 1409.7495). Here it is handled by a small residual layer fitted on own footage and switched on
  only for that footage, because the shifted features do not transfer to broadcast views.
- Gradient-boosted trees: LightGBM (NeurIPS 2017). The court and pose models are GBTs exported to JSON, and their evaluator is what got faster.
