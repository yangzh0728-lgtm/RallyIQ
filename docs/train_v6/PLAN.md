# train-v6: 24-hour plan (started 2026-11-27 07:30 UTC)

Branch `claude/train-v6` from `claude/train-v5` (432824f). Work folder `~/Desktop/RallyIQ/train_v6/` (LOG.md, ckpt/,
labels/, review/; Desktop, not T7). Never main, never merge. qa6 (sam's blind set) is never relabelled or tuned on.
Demo API 8000, web 5173 and the autoimport watcher stay up; the other T7 thread's files are not touched.

Common evaluation for every experiment ("the gate"):
- hits: 17 own videos on the revised v5 labels (`amateur_v4.json`, plus v6 additions recorded in `labels/v6_decisions.txt`),
  bundled scorer / leave-one-video-out where a model is retrained, paired per-video bootstrap 95% CI (5,000 draws);
  reported for all 17, the 11 non-qa6 and qa6 separately;
- hitter attribution on the labelled doubles hits (new pixel-anchored hitter truth, see A1);
- ShuttleSet test hit F1 (14 clips) and TrackNet public test shuttle F1 / accuracy (29 clips): must not drop;
- error counts per type from the contact sheets (wrong court at hit / NEXT, foreign trail, bystander boxed, our shuttle
  missed, partner untracked) and off-court player frames: must stay at or below v5.

Keep rule (all phases): a change is kept only if its own target metric improves with a CI that excludes 0 (or, where the
sample is too small for that, improves on every video it touches) **and** no gate metric gets worse beyond its CI.
Otherwise it is reverted and the result is still reported.

## A. v5 leftovers (~8 h)

A1 (1.5 h) Measure: pixel-anchored hitter truth for every labelled doubles hit (0310_01, 0239, 0241, 0622; 0615 has no
hitter labels and is qa6). Truth = the hitter's box at the hit frame, taken from the reviewed label id where there is one
and labelled by me from frame crops where the label says "untracked". Metrics: hitter accuracy, share of hits whose true
hitter has no track (no box with IoU > 0.3 for ±3 frames), split by cause (no detection / detection below threshold /
detection present but not assigned).
A2 (3 h) Fix partner tracking, replaying saved pose dumps (score ≥ 0.2 kept): (i) lower detection threshold only inside
the team's half of the court, (ii) re-acquire a lost slot from the last position / appearance, (iii) bridge short gaps,
(iv) swing evidence on both partners (done in v5). Stop rule: hitter accuracy up with the off-court counts ≤ v5.
A3 (1.5 h) 0245 white-wall shuttle (S4, S15, S19): first check candidates (was the shuttle detected at all?), then
racket-local search at hit time / contrast normalisation; detector fine-tuning moves to B. Stop rule: the 3 rows fixed
with ShuttleSet / TrackNet unchanged.
A4 (2 h) Next-court shuttle between rallies (8 rows): suppress tracks outside rally segments or ones that never pass
between our players. Stop rule: rows fixed, no labelled hit lost, own-trajectory frames lost ≤ v5's 2.6%.

## B. Shuttle detector fine-tuning on own footage (~6 h)

Hard positives and negatives mined across all 17 own videos (track/candidate disagreements, lost-at-hit frames, white
background, next-court shuttles as negatives for *our* court only through ownership, not the detector). Every mined frame
checked by me on crops before use; labels with reasons in `labels/shuttle_v6/`. Fine-tune TrackNet with ShuttleSet mixed
in (replay to avoid forgetting), small LR. Held-out own frames by video (2 videos never trained on). Stop rule: own
held-out F1 up and ShuttleSet / TrackNet public F1 not below the current weights (95.0 / 93.0); otherwise discard.

## C. Downstream retrain (~5 h)

On the kept tracks/players: hit scorer (v3b recipe, 5 seeds, LOVO), hitter attribution, stroke type (court model +
own-footage residual), landing / playing-style features. Keep rule as above; v3b and the shipped residual stay if they win.

## D. New own videos as a fresh test (~3 h)

Find own videos not yet labelled (Desktop folders, autoimport inbox, T7 runs read-only). Run the pipeline on copies,
pre-label hits myself from sheets, check every sheet, and score v5 vs v6 on them as a held-out set nobody tuned on.

## E. Sheets and report (~2 h)

Regenerate every affected contact sheet into `review/final/`, read each one, INDEX.md with changed sheets, per-sheet
error counts and remaining issues; REPORT.md (copied to `docs/train_v6/`); push.

Checkpoints: LOG.md after each step, commit + push after each kept change. If a usage limit hits, resume from LOG.md.

## References

1. Automated Hit-frame Detection for Badminton Match Analysis, arXiv 2307.16000 (hit candidates + pose swing).
2. TrackNet, arXiv 1907.03698 (shuttle heatmap detector).
3. TrackNetV3, MMAsia 2023 (trajectory rectification; inpainting gaps).
4. TrackNetV4, arXiv 2409.14543 (motion attention; background/contrast robustness for A3/B).
5. MonoTrack, CVPRW 2022, arXiv 2204.01899 (monocular court-space reasoning, amateur footage).
6. ShuttleSet, KDD 2023, arXiv 2306.04948 (data, stroke taxonomy).
7. CoachAI-Projects, arXiv 1907.12888.
8. RTMO, CVPR 2024, arXiv 2312.07526 (pose model in the players stage).
9. TemPose, CVPRW 2023 (pose sequences for strokes).
10. BST, arXiv 2502.21085 (badminton stroke transformer, pose + shuttle).
11. DANN, arXiv 1409.7495 (domain shift; here a residual layer on own footage).
12. LightGBM, NeurIPS 2017.
New for v6:
13. ByteTrack, ECCV 2022, arXiv 2110.06864: associate low-score detections in a second pass (A2 (i)).
14. OC-SORT, CVPR 2023, arXiv 2203.14360: observation-centric re-update after occlusion, recovering lost tracks (A2 (ii)-(iii)).
15. Online Hard Example Mining, CVPR 2016, arXiv 1604.03540 (B).
16. Learning without Forgetting, arXiv 1606.09282 / rehearsal with source data (B: ShuttleSet replay during fine-tuning).
