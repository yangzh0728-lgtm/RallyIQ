# events: rally segmentation and hit candidates (rules v1)

Stage entry: `rallyiq.events.stage:run(ctx) -> EventsResult`. It reads
`ctx.shuttle` (ShuttleTrack) and `ctx.players` (PlayerTracks) and emits raw
model predictions; human corrections are layered on top by `rallyiq/api/corrections.py`.
All thresholds live in `config.py` and can be overridden per run with
`ctx.config["events"] = {"end_gap_ms": 1500, ...}`.

## Rules

Distances are measured in **player heights** (H) of the nearest player, so the
far player (often half the size on screen) is judged on their own scale. Times
are in milliseconds, so nothing depends on the frame rate.

1. **Clean the track.** Drop lone detections far from both neighbours (false positives).
2. **Rallies.** A rally runs while the shuttle is seen and moving. It ends when the
   shuttle is unseen for `end_gap_ms` (`shuttle_lost`), stays still for `static_ms`
   (`shuttle_static`, e.g. on the floor), or the video ends (`video_end`).
   The rally starts at the last still point before the shuttle moves (the serve).
3. **Hit candidates.** At every point compare the incoming and outgoing velocity
   (least-squares over `window_ms`, reaching across occlusions up to `max_window_ms`):
   a turn of at least `min_turn_deg`, or a speed-up of at least `speed_jump_ratio`.
   Slow-downs alone do not count (drag slows the shuttle all the time).
4. **Pick the contact frame** by non-maximum suppression over `min_hit_gap_ms`,
   ranking by velocity impulse |v_out − v_in| weighted by racket proximity.
5. **Hitter.** Distance from the shuttle to each player's wrists and extrapolated
   racket head (wrist + k·(wrist − elbow)). Beyond `max_hitter_dist_rel` H nobody hit
   it, so the candidate is dropped (landing, lob apex, false detection).
6. **Alternation.** A small dynamic program keeps the candidates that best
   explain a rally in which the hitting side alternates (the player in singles, the
   team in doubles): a weak hit survives when it is needed between two hits of the
   other side, a weak duplicate is dropped. In doubles the hitter is the nearest of
   the four players; if that player's team just hit and a player of the other team
   is nearly as close (within `2·hitter_margin`), the other team's player is used
   and the hit carries `hitter_from_alternation`.
7. **Serve fallback.** If the shuttle is first seen already at the server's racket,
   a hit is proposed there for review.

## Confidence and status

`confidence = 0.35·change + 0.35·proximity + 0.15·sharpness + 0.15·track_quality`.
A hit is `ok` only with confidence ≥ `ok_threshold`, a known hitter and no reason
codes; everything else is `needs_review`. A rally is `ok` only if all its hits are.
Rules never set `winner`.

Reason codes (stable strings for the review UI):

| code | meaning |
| --- | --- |
| `hitter_ambiguous` | two players about equally close to the shuttle |
| `wrist_not_visible` | no confident wrist; shoulders used instead |
| `shuttle_occluded_at_hit` | no observation right before/after the hit |
| `interpolated_shuttle_at_hit` | the hit point was filled in by the tracker |
| `low_track_quality` | few confident detections around the hit |
| `weak_direction_change` | the trajectory barely changes |
| `teammate_ambiguous` | doubles: both partners about equally close to the shuttle |
| `hitter_from_alternation` | doubles/singles: nearest player broke alternation, next-nearest of the other side used |
| `same_hitter_consecutive` | two hits in a row by the same player (singles) or team (doubles) |
| `inferred_from_rally_start` | serve proposed from the rally start, not seen |
| `no_pose_data` | no pose near this frame; hitter unknown |
| `hitter_identity_unresolved` | nearest track has no P1–P4 identity yet |
| `no_hits_detected` / `possible_missed_hit` / `hits_need_review` | rally-level |

## Synthetic benchmark

`synthetic.py` renders singles or doubles (`doubles=True`) rallies (shuttle + COCO-17 poses, behind-baseline
camera) with ground truth; `metrics.py` scores hits with the README rule (±3
frames, same player). Current numbers (10 seeds × 4 rallies):

| condition | precision | recall | F1 |
| --- | --- | --- | --- |
| clean | 0.99 | 0.99 | 0.99 |
| 10% dropout + 2% false detections | 0.98 | 0.98 | 0.98 |
| 30% of hits occluded | 0.95 | 0.96 | 0.96 |
| occluded + tracker interpolation | 0.99 | 0.99 | 0.99 |
| 25 fps | 0.98 | 0.99 | 0.98 |
| heavy noise (4 px, 15% dropout, 3% false) | 0.80 | 0.95 | 0.87 |

These numbers only show the rules behave as designed; they say nothing yet
about real video. Re-tune on the first real clips.
