# Court stroke model (court_v1)

Trained by `scripts/learn_court_strokes.py` on the ShuttleSet annotation (CoachAI-Projects, MIT):
35,009 strokes, 3,683 rallies, 44 singles matches. No video is used. Features are court positions,
flight time relative to the video median, and high/low contact of this, the previous and the next shot
(see `rallyiq/strokes/court.py`). Training adds 0.4 m position noise and 12% contact flips.

- grouped (by match) 5-fold CV: 0.775 per shot, 0.777 with rally smoothing

On sam's DJI_0009 3-minute clip (58 hand-checked strokes, `evaluation/stroke_labels/dji0009_stroke_types.json`,
hits taken from the labelled events):

| classifier | all 58 | 35 "sure" labels |
| --- | --- | --- |
| gbt_v3 (video model alone) | 26 (45%) | 19 (54%) |
| gbt_v3 + court_v1 + rally smoothing | 37 (64%) | 26 (74%) |

The far player's foot positions come out ~1.5 m too deep on this camera; `court.py` quantile-maps
each side's hitter depths onto ShuttleSet's before scoring, which accounts for most of the gain of
the court model on this clip (24 -> 30 of 58 for the court model alone).
