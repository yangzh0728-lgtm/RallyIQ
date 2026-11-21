# Court stroke model (court_v2)

Trained by `scripts/learn_court_strokes.py` on the ShuttleSet annotation (CoachAI-Projects, MIT):
35,009 strokes, 3,683 rallies, 44 singles matches. No video is used. Features are court positions,
flight time relative to the video median, and high/low contact of this, the previous and the next shot
(see `rallyiq/strokes/court.py`). Training adds 0.4 m position noise and 12% contact flips, and uses
three copies of the data with 0, 15 and 30% of the hits removed from each rally, because our hit
detector misses 20-40% of the hits on amateur video and every miss shifts the sequence features.
Rally smoothing is off (`seq_weight` 0): it assumes no hit is missing in between.

Grouped (by match) 5-fold CV, same evaluation data for both:

| | full rallies | 30% of hits missing |
| --- | --- | --- |
| court_v1 + rally smoothing (before) | 77.4% | 58.7% |
| court_v2, no smoothing | 77.2% | 68.5% |

Tried on the same CV and not used: larger trees / more iterations / lower learning rate (within
±0.3 points), a second-stage model on the neighbours' probabilities (no gain), smoothing with chains
split at long flights or with a missed-hit transition matrix (both below no smoothing once hits are
missing), training noise matched to a weaker contact head (within ±0.5 points).

DJI_0009 3-minute clip (58 hand-checked strokes, `evaluation/stroke_labels/dji0009_stroke_types.json`;
retraining with another seed moves these by up to ±3):

| hits from | court_v1 + smoothing | court_v2 |
| --- | --- | --- |
| hand labels | 32/58 | 30/58 |
| auto, learned hit scorer (34 of 58 found) | 13 | 16 |
| auto, rule hit scorer (47 of 58 found) | 24 | 24 |
