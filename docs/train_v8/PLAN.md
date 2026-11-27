# train-v8 plan (started 2026-11-27 16:45 UTC; branch claude/train-v8 from claude/train-v7 43c2d9d)

Rules as v7: nested leave-one-video-out over the 11 non-qa6 own videos (inner loop picks every setting), per-video bootstrap
95% CI on the 17-video hit F1 gain; ship only if the CI is above 0 and the fresh set (0529_0005/0007/0009, test only),
qa6 (untouched) and ShuttleSet (14 clips) don't drop beyond noise. I label and check everything myself.

**Pre-registered combination.** Phase J's features are evaluated as ONE pre-declared package: the shipped v7 pipeline with
the v3b-recipe scorer retrained on (v3b features + audio features + both-sides pose features). Variants (audio only, pose
only) are reported for information but cannot be shipped instead of the package unless the package itself fails and the
variant is then evaluated as a new, separately pre-registered test on the fresh set only once. K and L are evaluated on their
own targets (stroke accuracy, hitter attribution), each as a single pre-declared change.

## J (~5 h) missed hits: audio + pose
- Audio is present in all 20 own videos (AAC in the 1080p transcodes the pipeline ran on; same duration as the probe).
  Extract mono 22.05 kHz; onset strength by spectral flux on a log-magnitude STFT with a high-pass band (> 2 kHz, the
  racket-shuttle "pop") and a local-max filter (SuperFlux, Böck & Widmer 2013); per candidate: max onset strength and
  its time offset within +-80 ms, local onset-to-background ratio, count of onsets in +-0.5 s (other courts).
- Check alignment on labelled hits first (onset peak vs label time histogram) before any training.
- Pose both sides: hitter-side and other-side wrist speed / p_swing in +-250 ms (already partly in v3b) plus their max over
  +-120 ms of the candidate.
- Retrain scorer (v3b recipe, LOVO, 5 seeds); keep v3b if the package doesn't win.

## K (~4 h) stroke types / playing style
- Evaluate court_v2 + own-footage residual on the trusted stroke labels (LOVO) and on the fresh set (I label strokes for
  the 87 fresh hits). Try adding audio/pose features to the residual. Keep only if it wins.

## L (~3 h) far player hidden behind the near one (doubles)
- Track interpolation through occlusion for the far partner; re-measure the 4 doubles attribution errors.

## M (~2 h) sheets, INDEX.md, REPORT.md, docs/train_v8/, push.

## References (new for v8)
- S. Böck, G. Widmer, "Maximum filter vibrato suppression for onset detection", DAFx 2013 (SuperFlux).
- S. Böck, F. Krebs, M. Schedl, "Evaluating the online capabilities of onset detection methods", ISMIR 2012.
- J. P. Bello et al., "A tutorial on onset detection in music signals", IEEE TASLP 2005.
- Audio-visual event localisation: Y. Tian et al., "Audio-Visual Event Localization in Unconstrained Videos", ECCV 2018.
  (I did not verify a specific badminton hit-sound paper offline, so none is cited; the onset method above is generic.)
- Earlier rounds' 16 references (train_v6/PLAN.md) still apply (TrackNet family, RTMO, ShuttleSet, BST, TemPose, ...).

## J2 (pre-registered 2026-11-27 ~18:40 UTC, after the J package failed and before any J2 result)
v3b stays the base scorer. A logistic "audio layer" re-scores each candidate: p = sigmoid(w0 + w1*logit(p_v3b) + w2*a_peak +
w3*a_off + w4*a_count) for candidates with audio (has_audio=1; without audio p = p_v3b). Fitted on the amateur candidate rows
only (weight as in v3b training), leave-one-video-out over the 11 non-qa6 videos for the 11 held-out evaluations, the fit on
all 11 for qa6 / fresh / ShuttleSet (ShuttleSet has no audio, so it is unchanged by construction). Single test, same keep rule.

## K pre-registration (2026-11-27, before any K variant result)
Baseline: court_v2 + video GBT + own residual (rise, fall, oh, first) as shipped (opt-in), evaluated LOVO on the trusted
own stroke labels (v5 rows rebuilt on the v7 events) and on my 78 typed fresh hits. Single variant: the same residual with
two added inputs, the hit-sound peak (a_peak, smashes are loud) and the hitter's wrist speed (wv_hitter), same L2 and
lambda as shipped. Keep only if LOVO accuracy on trusted rows improves with a per-video bootstrap CI above 0 and the fresh
set is not worse.

## L pre-registration (before any L result)
Finding first: the pipeline already bridges a hidden partner (events gap fill), and with bridged boxes counted as the
truth identity, 0241 286.77 is already right; only 0622 308.44 (hidden far jumper, picked partner standing still) is wrong.
Metric change (not a model change): hmetric counts a truth hitter as that player when the player's box interpolated over a
<= 1.5 s gap overlaps the truth box (IoU > 0.3).
Single model test: partner_swing may also hand the hit to a *bridged* (hidden) partner when the picked player's p_swing
is < 0.1 and the bridged partner's proximity is not lower. Keep if doubles attribution improves and no detected hit
gets worse; hit F1 must be unchanged.
