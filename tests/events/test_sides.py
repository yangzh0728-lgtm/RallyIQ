from rallyiq.events.sides import viterbi_sides


def test_alternation_overrides_weak_proximity():
    # middle hit: the near player is somewhat closer (far player undetected), neighbours are clear near hits
    logits = [-6.0, -1.2, -6.0]
    assert viterbi_sides(logits, [800, 800], 0.08, 0.35, 2000) == ["near", "far", "near"]


def test_strong_proximity_wins():
    logits = [-6.0, -6.0, 6.0]
    assert viterbi_sides(logits, [800, 800], 0.08, 0.35, 2000)[:2] == ["near", "near"]


def test_long_gap_relaxes_alternation():
    logits = [-6.0, -1.5]
    assert viterbi_sides(logits, [800], 0.08, 0.35, 2000) == ["near", "far"]
    assert viterbi_sides(logits, [3000], 0.08, 0.35, 2000) == ["near", "near"]
