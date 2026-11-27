# train-v9 plan (started 2026-11-27 17:40 UTC; claude/train-v9 from claude/train-v8 f799199). Ends ~07:00 UTC.
Rules as v8: pre-registration here before results; nested LOVO over the 11 non-qa6 own videos with per-video bootstrap CI;
fresh set test-only; qa6 untouched (no stroke typing on qa6 either); I label and check everything.

## N (~6 h) own stroke data + depth-robust own stroke model
Labels: review every labelled hit of the 11 am7 videos whose stroke is not yet "sure" (146 typed-unsure + 122 untyped) on
contact + next-hit sheets; type it and mark sure only when the flight and the receiver make it unambiguous; rear-camera
depth ambiguity stays unsure. Target 400 trusted rows was set by the brief; I report what the evidence supports.
Pre-registered model test (single): residual v2 = the shipped residual inputs (rise, fall, oh, first) plus
time-of-flight to the next hit, receiver's court-y at the next hit and its displacement, apex height above contact
(image, in body heights), mean image speed of the first 0.2 s, hitter wrist-above-head and wrist speed; same L2 fit,
lambda chosen by inner LOVO from {0.001, 0.003, 0.01}. Baseline: shipped residual. Keep if trusted LOVO accuracy gain has a
per-video CI above 0, fresh set not worse, and ShuttleSet unchanged under the broadcast profile (residual is own-profile only).

## O (~3 h) playing-style stability
Per player and session: stroke mix, court coverage, attack/defence ratio; bootstrap over rallies for CIs; same player across
sessions. Propose / add uncertainty display if cheap.

## P (~2 h) one pre-registered attempt at missed net-exchange hits (to be written here before it runs).

## Q (~2 h) sheets, INDEX.md, REPORT.md, docs/train_v9, push, final report.

## N result + N2 pre-registration (before running N2)
N: residual v2 (added tof, receiver depth/move, early speed, wrist speed): trusted LOVO 74.1% vs shipped-feature residual 77.2%
on the same 189 trusted rows -> not shipped.
N2 (single test): refit the shipped residual (same 4 features, lambda 0.003) on the expanded v9 rows (333 typed, 189 trusted)
and compare it with the bundled residual on the fresh set (78 typed, 23 sure), the only data neither has seen. Ship the refit
model file if fresh accuracy is not lower on either all typed or sure rows and higher on at least one.

## P pre-registration (before any P result)
Missed net-exchange hits: lower the candidate admission only near the net. Single change: in hits.detect_hits, a candidate
whose shuttle point lies within 1.5 m of the net (court y in [5.2, 8.2]) and within 0.35 player heights of a wrist keeps
min_candidate_conf 0.05 instead of 0.1 (scorer unchanged). Evaluate nested? There is one setting, no tuning: plain LOVO-scored
17 videos (v3b LOVO models + nested dead-ball models) with per-video bootstrap; keep if CI > 0, fresh not worse, ShuttleSet
not worse.
Amendment (written before running, after reading the code): the shuttle has no court position in the air, so "near the net"
= the best player's foot within 1.5 m of the net (|foot y - 6.7| <= 1.5) and wrist distance <= 0.35 heights. The change is
both admission (0.05 instead of the scorer's 0.1) and the alternation weight (keep_threshold 0.2 instead of 0.3) for such
candidates; one fixed setting, no tuning.
