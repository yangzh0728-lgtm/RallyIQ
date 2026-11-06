"""Hit-detection metrics matching the README acceptance rule:
a predicted hit counts when it is within ±``tol_frames`` of a true hit by the
same player (one-to-one matching)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

Hit = tuple[int, Optional[str]]  # (frame, player_id)


@dataclass(frozen=True)
class HitScore:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_hits(
    pred: Sequence[Hit], truth: Sequence[Hit], tol_frames: int = 3, same_player: bool = True
) -> HitScore:
    """Greedy one-to-one matching by frame distance."""
    pairs = []
    for i, (pf, pp) in enumerate(pred):
        for j, (tf, tp) in enumerate(truth):
            d = abs(pf - tf)
            if d <= tol_frames and (not same_player or pp == tp):
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    return HitScore(tp, len(pred) - tp, len(truth) - tp)
