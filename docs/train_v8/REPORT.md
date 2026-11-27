# RallyIQ train-v8: hit sound, stroke types, hidden partners (2026-11-27)

Branch `claude/train-v8` = `claude/train-v7` + this work (not merged; nothing on main). Work folder `~/Desktop/RallyIQ/train_v8/`
(PLAN.md with every pre-registration, LOG.md, labels/, review/final/INDEX.md, scripts/, ckpt/). The v6 fresh videos were
test-only; qa6 untouched. Every combination or variant was written into PLAN.md before its result was seen.

## Summary

| item | before | after | shipped? |
|---|---|---|---|
| Hit F1, 17 own videos (v7 pipeline, nested dead-ball models) | 0.860 | J package 0.855 [-0.021, +0.009]; J2 0.858 [-0.016, +0.013] | no |
| Stroke type, trusted own rows (138), LOVO | 65.9% (court_v2) | 76.1% with the own residual, CI [+5, +16] pts; +audio/wrist variant 73.2% | residual: already in code, now reachable (below) |
| Stroke type, fresh held-out hits (78 typed by me) | 44.4% | 55.6% with the residual | |
| Doubles hitter attribution (bridged-aware metric) | 108/111 | 108/111 (pre-registered bridged-partner rule changed nothing) | no |
| Own-footage pipeline profile | residual never on in the API / CLI | `RALLYIQ_PIPELINE_PROFILE=own` turns it on | **yes** (35a3879) |

## J: missed hits with the hit sound and pose
- All 20 own videos carry audio. The racket "pop" is real: at labelled hits the strongest > 2 kHz onset (SuperFlux-style
  flux, Böck & Widmer 2013) is 7.6x the local background (median), vs 4.2x at not-a-hit rows and random times; it arrives
  ~86 ms after the labelled contact frame.
- `rallyiq/events/audio.py` + six scorer features (hit-sound peak / offset / onset count / audio present, both sides' wrist
  speed), plumbed through `segment_events(audio=...)`. The bundled v3b scorer ignores them, so nothing changes by default.
- Pre-registered package (v3b recipe retrained with audio + wrist speed): 0.855 vs v3b 0.860. Second pre-registered test
  (v3b kept, logistic audio layer on its logit): 0.858. Neither wins; audio adds ~0.008 to a retrained scorer, but every
  retrain still trails v3b (as in v5 and v6). The code is kept as experimental (unused).

## K: stroke types
- I typed 78 of the 88 fresh hits (23 sure; rear camera, so depth is often ambiguous).
- The own-footage residual (train_v4) is confirmed on new data: fresh 44% -> 56%, trusted LOVO 66% -> 76%.
- Pre-registered variant with hit sound and wrist speed in the residual: worse (73.2%). Not shipped.
- **Shipped:** the residual was never actually on in the path sam uses: the demo API's worker and the pipeline CLI ran with
  an empty config. New `rallyiq/pipeline/profiles.py`: `RALLYIQ_PIPELINE_PROFILE=own` (or `--profile own`) sets
  `strokes.residual=auto`; unset or `broadcast` keeps the defaults, so ShuttleSet / broadcast runs are unchanged
  (the residual costs 5-8 points there). Tests in `tests/pipeline/test_profiles.py`. To use it after merging: restart the
  API with `RALLYIQ_PIPELINE_PROFILE=own`. The own batch template on the T7 (`runs/templates/batch_own.py`) also needs
  `"strokes": {"residual": "auto"}` in its CFG; I did not edit it (other thread's files).

## L: far partner hidden behind the near one
The events stage already bridges a hidden player over short gaps. Counting bridged boxes as the truth identity, 132/134
doubles truth hitters are identified and 108/111 detected hits credited correctly; 0241 286.77 (v6's "untracked") was right
all along. Only 0622 308.44 is a true occlusion error. The pre-registered bridged-partner rule changed no prediction; reverted.

## Remaining
Missed hits (fast net exchanges, hidden far player), 7 between-rally next-court rows, 3 doubles attribution errors, stroke
depth ambiguity on the rear camera.

## References
Böck & Widmer, DAFx 2013 (SuperFlux); Böck, Krebs & Schedl, ISMIR 2012; Bello et al., IEEE TASLP 2005 (onset detection);
Tian et al., ECCV 2018 (audio-visual event localisation); plus the 16 earlier references (train_v6/PLAN.md).
