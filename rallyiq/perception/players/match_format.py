"""Decide singles vs doubles from the first seconds of candidates.

Doubles shows four active players: two per half, most of the time. Singles
shows two, with the occasional extra person (umpire, line judge, neighbouring
court) that the filters already score down. The decision counts, per frame,
the confident candidates standing inside the play area and looks at how often
three or more of them are present with at least two on one half.

The result carries a confidence so a borderline clip can be flagged for review
instead of silently tracked with the wrong number of players.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .geometry import court_side
from .models import Candidate

FORMATS = ("singles", "doubles")


@dataclass(frozen=True)
class FormatConfig:
    min_score: float = 0.4  # candidates below this are not counted as players
    doubles_frac: float = 0.4  # share of frames with >= 3 players needed for doubles
    min_frames: int = 10  # fewer usable frames than this -> low confidence


@dataclass(frozen=True)
class FormatDecision:
    match_format: str
    confidence: float  # 0..1
    frames_used: int
    frac_three_plus: float


def _count(cands: Sequence[Candidate], min_score: float) -> tuple[int, bool]:
    """(confident players, whether some half has >= 2 of them)."""
    good = [c for c in cands if c.player_score >= min_score and c.on_court is not False]
    halves: dict[str, int] = {}
    for c in good:
        s = court_side(c.foot_court) if c.foot_court is not None else None
        if s is not None:
            halves[s] = halves.get(s, 0) + 1
    # Without calibration we cannot tell halves; the count alone has to do.
    paired = any(v >= 2 for v in halves.values()) or not halves
    return len(good), paired


def decide_format(frames: Sequence[Sequence[Candidate]], cfg: FormatConfig | None = None) -> FormatDecision:
    cfg = cfg or FormatConfig()
    used = 0
    three = 0
    for cands in frames:
        n, paired = _count(cands, cfg.min_score)
        if n == 0:
            continue
        used += 1
        if n >= 3 and paired:
            three += 1
    if used == 0:
        return FormatDecision("singles", 0.0, 0, 0.0)
    frac = three / used
    fmt = "doubles" if frac >= cfg.doubles_frac else "singles"
    # Distance from the threshold, scaled to each side's range.
    span = (1 - cfg.doubles_frac) if fmt == "doubles" else cfg.doubles_frac
    conf = min(abs(frac - cfg.doubles_frac) / max(span, 1e-6), 1.0)
    if used < cfg.min_frames:
        conf *= used / cfg.min_frames
    return FormatDecision(fmt, round(conf, 3), used, round(frac, 3))
