"""Rule-based stroke-type classifier (``rules_v1``).

Each stroke type is a fuzzy AND of physically motivated conditions on the
features in ``features.py`` (contact height, court depth of hitter and receiver,
speed, how high the flight rose). Memberships are 0..1 ramps; a feature that
could not be measured contributes 0.5 ("don't know") instead of a guess, and
lowers the evidence score. Scores are normalised into a distribution over the
12 types; the top three are reported as candidates.

Typical singles numbers behind the ramps (metres from the net, m/s mean
horizontal speed hitter -> receiver):

| type | contact | hitter | receiver | speed | flight |
| --- | --- | --- | --- | --- | --- |
| clear | overhead | back | back | 6-9 | high arc |
| smash | overhead | mid/back | mid | > 12 | flat/down |
| drop | overhead | back | front | 4-8 | flat |
| net kill | above tape | front | mid | fast | down |
| net shot / cross net | low-mid | front | front | < 3 | barely rises |
| push | mid | front/mid | mid/back | 7-12 | flat |
| drive | side | mid | mid/back | > 9 | flat |
| lift | low | front/mid | back | 6-9 | high arc |
| block | low | mid | front | < 6 | flat, after a smash |
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import SHOT_TYPES
from .features import StrokeFeatures


def ramp(x: float | None, a: float, b: float) -> float | None:
    """0 at x <= a, 1 at x >= b (a > b gives a falling ramp); None if x is unknown."""
    if x is None:
        return None
    if a < b:
        return max(0.0, min(1.0, (x - a) / (b - a)))
    return max(0.0, min(1.0, (a - x) / (a - b)))


def _u(m: float | None) -> float:
    return 0.5 if m is None else m


@dataclass(frozen=True)
class RulesConfig:
    overhead: tuple[float, float] = (1.05, 1.3)  # contact_h ramp
    low: tuple[float, float] = (0.75, 0.5)  # falling ramp
    front: tuple[float, float] = (3.0, 2.0)  # hitter/receiver depth (m from net), falling
    back: tuple[float, float] = (3.8, 4.8)
    fast_speed: tuple[float, float] = (11.0, 16.0)  # m/s
    slow_speed: tuple[float, float] = (6.0, 3.5)  # falling
    fast_flight: tuple[float, float] = (850.0, 550.0)  # ms, falling (fallback without court positions)
    slow_flight: tuple[float, float] = (800.0, 1100.0)
    high: tuple[float, float] = (0.9, 1.4)  # rise in player heights
    cross: tuple[float, float] = (1.8, 3.0)  # lateral metres for a cross-court net shot
    eps: float = 0.02


@dataclass
class Memberships:
    over: float
    low: float
    mid: float
    front: float
    back: float
    midc: float
    r_front: float
    r_back: float
    r_mid: float
    fast: float
    slow: float
    high: float
    cross: float
    known: dict[str, bool]


def memberships(f: StrokeFeatures, cfg: RulesConfig) -> Memberships:
    over = ramp(f.contact_h, *cfg.overhead)
    low = ramp(f.contact_h, *cfg.low)
    front = ramp(f.hitter_depth_m, *cfg.front)
    back = ramp(f.hitter_depth_m, *cfg.back)
    r_front = ramp(f.receiver_depth_m, *cfg.front)
    r_back = ramp(f.receiver_depth_m, *cfg.back)
    if f.court_speed_mps is not None:
        fast = ramp(f.court_speed_mps, *cfg.fast_speed)
        slow = ramp(f.court_speed_mps, *cfg.slow_speed)
    else:
        fast = ramp(f.flight_ms, *cfg.fast_flight)
        slow = ramp(f.flight_ms, *cfg.slow_flight)
    high = 1.0 if f.left_frame else ramp(f.rise, *cfg.high)
    cross = ramp(f.lateral_m, *cfg.cross)
    o, lo = _u(over), _u(low)
    fr, bk = _u(front), _u(back)
    rf, rb = _u(r_front), _u(r_back)
    return Memberships(
        over=o, low=lo, mid=max(0.0, 1.0 - o - lo),
        front=fr, back=bk, midc=max(0.0, 1.0 - fr - bk),
        r_front=rf, r_back=rb, r_mid=max(0.0, 1.0 - rf - rb),
        fast=_u(fast), slow=_u(slow), high=_u(high), cross=_u(cross),
        known={
            "contact": f.contact_h is not None,
            "hitter_court": f.hitter_depth_m is not None,
            "receiver_court": f.receiver_depth_m is not None,
            "speed": f.court_speed_mps is not None or f.flight_ms is not None,
            "flight_shape": f.rise is not None or f.left_frame,
        },
    )


def type_scores(f: StrokeFeatures, prev_type: str | None, cfg: RulesConfig) -> tuple[dict[str, float], Memberships]:
    m = memberships(f, cfg)
    first = 1.0 if f.first_in_rally else 0.0
    nf = 1.0 - first
    not_over = 1.0 - m.over
    flat = 1.0 - m.high
    after_smash = 1.0 if prev_type == "smash" else 0.0
    r_short = m.r_front + 0.5 * m.r_mid

    s = {
        # The first hit of a rally is the serve (underarm, from the service area).
        "short_serve": first * not_over * (0.5 * m.r_front + 0.5 * flat) * (1.0 - m.fast),
        "long_serve": first * not_over * (0.5 * m.r_back + 0.5 * m.high),
        "clear": nf * m.over * m.high * (0.4 + 0.6 * m.r_back) * (1.0 - m.fast),
        "smash": nf * m.over * m.fast * flat * (1.0 - m.front),
        "drop": nf * m.over * (1.0 - m.fast) * flat * r_short * (1.0 - m.front),
        "net_kill": nf * m.over * m.front * m.fast * flat,
        "net_shot": nf * not_over * m.front * m.r_front * (1.0 - m.fast) * (1.0 - 0.5 * m.high) * (1.0 - m.cross),
        "cross_net": nf * not_over * m.front * m.r_front * (1.0 - m.fast) * (1.0 - 0.5 * m.high) * m.cross,
        "push": nf * (1.0 - m.low) * (1.0 - m.over * 0.7) * (m.front + 0.6 * m.midc) * flat
        * (m.r_mid + m.r_back) * (1.0 - m.slow) * (1.0 - 0.6 * m.fast),
        "drive": nf * (m.mid + 0.3 * m.low) * (1.0 - m.front) * flat * (1.0 - m.r_front) * (0.3 + 0.7 * m.fast),
        "lift": nf * not_over * m.high * (0.1 + 0.9 * m.r_back),
        "block": nf * not_over * (1.0 - m.front) * r_short * flat * (1.0 - m.fast) * (0.6 + 0.4 * after_smash),
    }
    return s, m


def softmax_scores(scores: dict[str, float], eps: float) -> dict[str, float]:
    total = sum(v + eps for v in scores.values())
    return {k: (scores[k] + eps) / total for k in SHOT_TYPES}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0) / math.log(len(p))
