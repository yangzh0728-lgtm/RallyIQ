# RallyIQ train-v10: learned rally sequence models (2026-11-27)

Branch `claude/train-v10` = `claude/train-v9` + docs only (no model shipped). Work folder `~/Desktop/RallyIQ/train_v10/`
(PLAN.md with every pre-registration, LOG.md, scripts/, ckpt/ incl. unshipped_code/). Fresh set test-only, qa6 untouched.

| experiment | baseline | result | shipped? |
|---|---|---|---|
| R: TCN rally hit detector, own 17 videos (nested) | 0.860 | 0.789 (A) / 0.765 (B), CI [-0.110, -0.039] | no |
| R: same, fresh held-out videos | 0.821 | 0.740 / 0.781 | no |
| R: same, ShuttleSet test (standalone, own-fine-tuned) | 0.780 | 0.864 / 0.890 | (see R2) |
| R fallback: TCN probability stacked into the v3b scorer | 0.860 | 0.843, CI [-0.036, -0.001] | no |
| R2: ShuttleSet-only TCN as broadcast hit detector, integrated, 14 held-out test clips | 0.780 | 0.815, per-match CI [-0.016, +0.069] | no |
| S: pose + shuttle window stroke classifier, own trusted LOVO | 77.2% (court_v2 + residual) | 71.4%, CI [-0.114, -0.008] | no |
| S: same, fresh set (63 matched) | 55.6% | 55.6% | no |
| S-B: classifier probabilities as residual inputs | 77.2% | 75.7% | no |

Findings worth keeping:
- A small dilated TCN over the whole rally (MS-TCN style) is much better than the candidate + scorer pipeline on broadcast
  footage (ShuttleSet test 0.86-0.89 standalone vs 0.78) and needs no extra runtime, but on sam's own footage it is 0.07 F1
  worse: the own labelled data (≈ 400 hits in 11 videos, windows only) are too few to adapt it.
- Integrated into the pipeline's rally segmentation it keeps only part of that gain (0.815); the integration mode chosen on
  training data was the restrictive one. A future test on new held-out broadcast matches is the honest next step; the code,
  weights and diff are in ckpt/unshipped_code/.
- The learned stroke classifier does not beat the court model + own residual on own footage.
