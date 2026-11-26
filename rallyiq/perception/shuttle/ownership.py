"""Is a shuttle trajectory ours or another court's? Judged from our players alone.

The air-space test in ``target_court`` cannot separate two courts that overlap along the
camera's line of sight: seen from behind our near corner, a rally on the next court hangs
in the same pixels as a high shuttle over ours. The players can:

* **Between the hitters.** A shuttle flies in the vertical plane through the two contact
  points, and a camera held upright shows vertical planes as vertical strips, so its image
  x stays between our outermost players (plus racket reach, plus where they were within
  ``SPAN_S``). A neighbour's rally spends most of its time outside that strip.
* **Who turns it round.** Our shuttle changes direction at a swinging racket of ours. A
  trajectory that keeps reversing where none of our wrists swings is somebody else's.

``CourtOwnership.logit`` combines both into one log-odds (> 0: another court's). The
weights are a logistic regression on hand-labelled tracklets of 17 own-footage videos and
the ShuttleSet training clips (all ours), see docs/train_v5. Without players on both halves there is no strip and the answer is 0
odds either way (``None``).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence

import numpy as np

from ...contracts import PlayerTracks

SPAN_S = 0.4  # players' reach of the strip around a frame
SWING_S = 0.15
CONTACT_BODIES = 0.65  # wrist to shuttle at a contact (arm's racket length, blur)
SWING_BODIES_PER_S = 1.5
MISSING_PAD = 0.06  # widen the strip (image widths) when a doubles player is untracked
_WRISTS = (9, 10)
_IDS = ("P1", "P2", "P3", "P4")

FEATURES = ("o0", "o4", "m0", "m4", "bad_turns", "good_turns", "contact_start", "contact_end")
# Filled by scripts/train_v5/fit_ownership.py.
WEIGHTS: dict[str, float] = {
    "o0": 4.345, "o4": 1.549, "m0": -14.454, "m4": 13.917, "bad_turns": 1.073, "good_turns": -0.061,
    "contact_start": -1.574, "contact_end": -1.22,
}
BIAS = 1.108
# Log-odds per unit of log(concurrency + 0.02): how much of the video has two shuttles in
# flight at once (``ShuttleTracker.concurrency``). One court alone never does.
CONCURRENCY_WEIGHT = 2.431


class CourtOwnership:
    def __init__(self, players: PlayerTracks | None, fps: float, width: int):
        self.fps, self.width = float(fps), float(width)
        self.expected = 4 if players is not None and players.match_format == "doubles" else 2
        rows: dict[int, list] = {}
        self.wrists: dict[str, dict[int, dict[int, tuple[float, float]]]] = {}
        for fr in (players.frames if players else []):
            out = []
            for p in fr.players:
                if p.player_id not in _IDS:
                    continue
                x1, y1, x2, y2 = p.bbox
                wd = {i: (k[0], k[1]) for i, k in enumerate(p.keypoints) if i in _WRISTS and len(k) > 2 and k[2] >= 0.2}
                out.append((p.player_id, p.side, x1, x2, max(y2 - y1, 1.0), wd))
                self.wrists.setdefault(p.player_id, {})[fr.frame] = wd
            if out:
                rows[fr.frame] = out
        self.rows = rows
        self.frames = sorted(rows)
        self._speed: dict[tuple[str, int], float] = {}
        self._strip: dict[int, tuple | None] = {}

    def __bool__(self) -> bool:
        return bool(self.frames)

    def strip(self, frame: int):
        """``(lo0, hi0, lo4, hi4)``: image x range of our players around ``frame`` with 0 and
        0.4 body heights of reach, or ``None`` when one half has no tracked player."""
        if frame not in self._strip:
            w = SPAN_S * self.fps
            i, j = bisect.bisect_left(self.frames, frame - w), bisect.bisect_right(self.frames, frame + w)
            step = max(1, (j - i) // 8)
            sides, best, n = set(), [math.inf, -math.inf, math.inf, -math.inf], 0
            for f in self.frames[i:j:step]:
                n = max(n, len(self.rows[f]))
                for _pid, side, x1, x2, h, _w in self.rows[f]:
                    sides.add(side)
                    best = [min(best[0], x1), max(best[1], x2), min(best[2], x1 - 0.4 * h), max(best[3], x2 + 0.4 * h)]
            if not {"near", "far"} <= sides:
                self._strip[frame] = None
            else:
                pad = MISSING_PAD * self.width if n < self.expected else 0.0
                self._strip[frame] = (best[0] - pad, best[1] + pad, best[2] - pad, best[3] + pad)
        return self._strip[frame]

    def wrist_speed(self, pid: str, frame: int) -> float:
        key = (pid, frame)
        if key not in self._speed:
            track, best = self.wrists.get(pid, {}), 0.0
            w = round(SWING_S * self.fps)
            for f in range(frame - w, frame + w + 1):
                b = track.get(f)
                if not b:
                    continue
                for s in (1, 2):
                    a = track.get(f - s)
                    if a:
                        for i in a.keys() & b.keys():
                            best = max(best, math.dist(a[i], b[i]) / s)
                        break
            self._speed[key] = best * self.fps
        return self._speed[key]

    def contact(self, frame: int, x: float, y: float) -> bool:
        """A swinging wrist of one of our players within racket reach of (x, y)."""
        for d in range(-3, 4):
            for pid, _side, _x1, _x2, h, wd in self.rows.get(frame + d, ()):
                h = max(h, 60.0)
                if wd and min(math.dist(q, (x, y)) for q in wd.values()) <= CONTACT_BODIES * h \
                        and self.wrist_speed(pid, frame + d) >= SWING_BODIES_PER_S * h:
                    return True
        return False

    def features(self, frames: Sequence[int], xy: Sequence[tuple[float, float]], turns: Sequence[int]) -> dict | None:
        """``turns``: indices of sharp turns. ``None`` without players on both halves for most of it."""
        o0, o4 = [], []
        for f, (x, _y) in zip(frames, xy):
            s = self.strip(f)
            if s is None:
                continue
            o0.append(max(s[0] - x, x - s[1], 0.0) / self.width)
            o4.append(max(s[2] - x, x - s[3], 0.0) / self.width)
        if not o0 or len(o0) < 0.5 * len(frames):
            return None
        good = sum(self.contact(frames[i], *xy[i]) for i in turns)
        return {
            "o0": float(np.mean([v > 0 for v in o0])), "o4": float(np.mean([v > 0 for v in o4])),
            "m0": float(np.mean(o0)), "m4": float(max(o4)),
            "bad_turns": float(min(len(turns) - good, 4)), "good_turns": float(min(good, 4)),
            "contact_start": float(self.contact(frames[0], *xy[0])),
            "contact_end": float(self.contact(frames[-1], *xy[-1])),
        }

    def logit(self, frames, xy, turns, concurrency: float = 0.3) -> float | None:
        """Log-odds that this trajectory belongs to another court."""
        ft = self.features(frames, xy, turns)
        prior = BIAS + CONCURRENCY_WEIGHT * math.log(concurrency + 0.02)
        return None if ft is None else prior + sum(WEIGHTS.get(k, 0.0) * v for k, v in ft.items())
