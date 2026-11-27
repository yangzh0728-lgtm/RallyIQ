# RallyIQ train-v10: learned rally sequence models (2026-11-27)

Branch `claude/train-v10` = `claude/train-v9` + the broadcast-profile sequence hit model (below) + docs. Work folder `~/Desktop/RallyIQ/train_v10/`
(PLAN.md with every pre-registration, LOG.md, scripts/, ckpt/ incl. unshipped_code/). Fresh set test-only, qa6 untouched.

| experiment | baseline | result | shipped? |
|---|---|---|---|
| R: TCN rally hit detector, own 17 videos (nested) | 0.860 | 0.789 (A) / 0.765 (B), CI [-0.110, -0.039] | no |
| R: same, fresh held-out videos | 0.821 | 0.740 / 0.781 | no |
| R: same, ShuttleSet test (standalone, own-fine-tuned) | 0.780 | 0.864 / 0.890 | (see R2) |
| R fallback: TCN probability stacked into the v3b scorer | 0.860 | 0.843, CI [-0.036, -0.001] | no |
| R2: ShuttleSet-only TCN as broadcast hit detector, integrated, 14 held-out test clips | 0.780 | 0.815, per-match CI [-0.016, +0.069] | no |
| R3: same model, grouped 5-fold by match over 43 ShuttleSet matches (102 clips), evaluation decode | 0.787 | 0.863, per-match CI [+0.063, +0.090] | |
| R3: same, the shipped code path (segment_events, hitter on the model's side) | 0.787 | **0.811**, CI [+0.012, +0.037] | **yes, broadcast profile only** |
| S: pose + shuttle window stroke classifier, own trusted LOVO | 77.2% (court_v2 + residual) | 71.4%, CI [-0.114, -0.008] | no |
| S: same, fresh set (63 matched) | 55.6% | 55.6% | no |
| S-B: classifier probabilities as residual inputs | 77.2% | 75.7% | no |

Findings worth keeping:
- A small dilated TCN over the whole rally (MS-TCN style) is much better than the candidate + scorer pipeline on broadcast
  footage (ShuttleSet test 0.86-0.89 standalone vs 0.78) and needs no extra runtime, but on sam's own footage it is 0.07 F1
  worse: the own labelled data (≈ 400 hits in 11 videos, windows only) are too few to adapt it.
- Integrated into the pipeline's rally segmentation it kept only part of that gain. The follow-up (R3, below) found why and
  shipped it for broadcast footage only.
- The learned stroke classifier does not beat the court model + own residual on own footage.

## Follow-ups (after the 18:43 UTC brief)

**Frame-time bug.** Only in my TCN integration: it turned frame numbers into times with an fps estimated from integer-ms
steps (30.3 instead of 29.97 / 30), ~300 ms drift per 30 s. Shipped code uses that rounded value only as a per-frame duration
(window sizes, one-frame offsets; 0.3 ms error); every absolute frame <-> time conversion (contracts, strokes, calibration,
web app, review scripts) uses the probe's exact fps. Nothing to fix in shipped code, so no hit F1 changes. The sequence model
maps frames to times from the observed timestamps; a regression test covers 29.97 fps over a minute.

**R3: grouped K-fold by match (pre-registered in PLAN.md).** 5 folds over the 43 ShuttleSet matches with clips (102 clips);
per fold the TCN is retrained on the other matches only and its threshold / gap chosen by an inner 3-fold on those matches;
peaks kept inside the rules pipeline's rallies +-0.3 s. Evaluation decode: 0.787 -> 0.863, CI [+0.063, +0.090]. I then ran
the real pipeline code with the same fold models: at first no gain (0.786). Two integration bugs: the model read the
off-court-masked shuttle track instead of the raw one it was trained on, and the hitter was any nearest player instead of
the player on the side the model predicts. Fixed to match the specification: **0.811, CI [+0.012, +0.037]**. The rest of the
gap is hits with no player track nearby (the rules path loses the same). Shipped: `events.hit_model` ("rules" default,
"seq"), `rallyiq/events/seqhits.py` (numpy inference, 1.5 MB weights trained on all 102 clips, 3 seeds),
`RALLYIQ_PIPELINE_PROFILE=broadcast` turns it on. `own` and unset are unchanged (rules), so the 17 own videos, the fresh set
and qa6 are untouched by construction. Tests: `tests/events/test_seqhits.py`, profile test; full suite 486 passed, 12 skipped.

## What would move the needle next (estimates, not measured)

| idea | why (evidence from v6-v10) | expected gain | cost |
|---|---|---|---|
| More fully labelled own videos (≈ 10 more, whole length, I pre-label, sam spot-checks) | Own hit data are ≈ 400 hits in windows of 11 videos. The TCN that beats the rules by +0.08 on broadcast (thousands of hits) loses 0.07 on own footage; every scorer retrain in v5-v8 also trailed v3b by 0.01-0.03, a small-data symptom. | hit F1 +0.01 to +0.04 (TCN or retrained scorer finally usable); stroke residual +2-5 pts | ≈ 2-3 h of my review per video; sam ≈ 15 min per video |
| Side (or second) camera | Rear camera: 111 of 268 doubtful strokes stay unknown (depth along the line of sight), missed net hits lack a visible contact, far player hidden behind the near one. | stroke type +5-10 pts on own footage; net-exchange recall up, maybe +0.01-0.02 hit F1 | tripod + phone; sync by the hit sound (audio onsets already work, ~86 ms offset); 2-view code ≈ 1-2 days |
| Cross-video player identity | Style shares from one session have 95% intervals 11-24 pts wide; pooling a player's sessions narrows them roughly by 1/sqrt(n). | 3 sessions: 6-14 pt intervals, style verdicts become separable | re-ID embedding (jersey / pose) + naming in the app ≈ 1-2 days |
| 60 fps recording | Most remaining misses are fast net exchanges (two contacts < 0.4 s apart, shuttle 1-3 frames). | small, +0.005-0.015 hit F1 (guess) | a camera setting; models already take fps from the probe |
| Detector + tracker re-tuned jointly | ft1 detector wins per frame (0.919 -> 0.962) but the tracker / scorer are tuned to the old score distribution; 150 configs gave at most +0.003. | ≤ +0.01 | large; low priority |
