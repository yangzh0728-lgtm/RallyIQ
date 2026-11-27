# train-v8 log (started 2026-11-27 16:45 UTC; claude/train-v8 from claude/train-v7 43c2d9d). Plan and pre-registration: PLAN.md.

## J: missed hits with audio and pose (nothing shipped)
- All 20 own videos carry AAC audio in the 1080p transcodes the pipeline ran on (same duration as the probe); the copies on
  the Desktop had been stripped of audio. Extracted mono 22.05 kHz (audio/), onset strength (scripts/audio_onset.py,
  SuperFlux-style, > 2 kHz) -> ckpt/onset/.
- Alignment on 414 labelled hits: the strongest onset within +-0.2 s lies at a median +0.086 s after the label (IQR
  +0.017..+0.111); peak/background median 7.6 at hits vs 4.2 at not-a-hit rows and at random window times.
- Code: rallyiq/events/audio.py (onsets, features), 6 new scorer features (wv_hitter, wv_other, a_peak, a_off, a_count,
  has_audio), audio plumbed through segment_events(audio=...) and AmClip (ONSET_DIR). The bundled v3b scorer ignores them
  (verified: 0.8604 unchanged).
- Pre-registered package (v3b recipe retrained on v3b features + audio + wrist speed, LOVO): 0.8547 vs v3b 0.8604,
  CI [-0.021, +0.009]; qa6 0.684 vs 0.692. Not shipped.
- For information (same cache, LOVO logs with in-sample dead-ball model): plain retrain 0.845, + audio 0.853, + pose only
  see logs/lovo_v2_all_p.log, + both 0.856. Audio helps a retrained scorer by ~0.008 but every retrain still trails v3b.
- J2 (pre-registered after J failed): v3b kept, logistic audio layer on its logit (lovo_J2): 0.8582 vs 0.8604,
  CI [-0.016, +0.013]; qa6 0.697 vs 0.692. Not shipped.

## K: stroke types (nothing new shipped; residual confirmed)
- Fresh set: I typed 78 of the 88 fresh hits (labels/fresh_strokes.txt, review/strokes2/; 23 marked sure; rear camera, depth
  often unreadable). court_v2 + video GBT 44.4% -> with the shipped own residual 55.6% (sure: 65.2% both, n=23).
- Trusted own rows rebuilt on the v7 events (138 trusted / 276): LOVO 65.9% -> residual 76.1%, CI over videos
  [+0.050, +0.158] (matches v4/v5).
- Pre-registered variant (residual + hit-sound peak + hitter wrist speed): 73.2%, worse than 76.1%. Not shipped.
- Note for sam: the residual is still opt-in (`strokes.residual=auto`); on own footage it is clearly better (also on the
  fresh set), on broadcast it costs points, so the own-footage batch settings should keep it on.
- Own-footage path check: the demo API's in-process worker ran with an empty config and the pipeline CLI likewise, so
  uploads never got the residual; the own batch template (~/rallyiq-data/runs/templates/batch_own.py, on the T7, not
  edited) passes events settings but no strokes options. Added rallyiq/pipeline/profiles.py: RALLYIQ_PIPELINE_PROFILE=own
  (or `--profile own`) sets strokes.residual=auto; unset/broadcast unchanged. Tests: tests/pipeline/test_profiles.py.
  Takes effect after merge when the API is restarted with RALLYIQ_PIPELINE_PROFILE=own; running 8000/5173/watcher untouched.
  The batch template would need `"strokes": {"residual": "auto"}` in its CFG (left for sam / the other thread).

## L: far partner hidden behind the near one (nothing shipped)
- Hitter metric now also counts a truth hitter whose box is bridged over a <= 1.5 s gap (scripts/hmetric.py). With it:
  132/134 truth hitters identified, attribution 108/111 (97.3%) on v7. 0241 286.77 (v6 "untracked") is the bridged P2 and
  the pipeline already credits P2. 0622 308.44: the hidden far jumper P3 is bridged over 1.53 s (just above the
  pre-registered 1.5 s), picked hitter P1 is wrong.
- Pre-registered rule (hand a hit to a bridged partner when the picked player's p_swing < 0.1 and the bridged partner is at
  least as close): no prediction changed (F1 0.862 both, attribution identical) -> reverted.
- Remaining attribution errors (3): 0239 135.75 and 0622 379.01 (partners both visible, nearest wrist wrong, no swing
  difference), 0622 308.44 (occlusion).
