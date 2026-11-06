"""Check a provider's raw output against ``MatchStats`` before it reaches the user.

* Evidence keys that do not exist are removed; an insight left with no valid
  evidence is dropped (counted in ``CoachReport.dropped``).
* Evidence that is ``undeterminable`` / null does not count as support.
* An insight's confidence is capped at the weakest evidence it cites.
* Every number in the title/detail must match a value, sample size, count or
  share from the cited evidence (with rounding tolerance); otherwise the insight
  is kept but flagged ``grounded=False`` with a warning, so the UI can show it.
"""

from __future__ import annotations

import re

from ...contracts import CoachInsight, Distribution, MatchStats, Metric, TrainingItem
from ..stats import index

_RANK = {"low": 0, "medium": 1, "high": 2}
_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")
# Numbers that are labels rather than measurements: player names, zone names, list markers.
_LABEL_TOKENS = re.compile(r"P[12]|第\s*\d+|\d+\s*[\.、)）](?=\D)|[≤≥<>]\s*\d+")


def _usable(e: Metric | Distribution) -> bool:
    if e.status == "undeterminable":
        return False
    return e.value is not None if isinstance(e, Metric) else e.n > 0


def _allowed_numbers(evidence: list[Metric | Distribution]) -> list[float]:
    nums: list[float] = []
    for e in evidence:
        nums.append(float(e.n))
        nums.extend(float(x) for x in _NUM.findall(e.label))
        if isinstance(e, Metric):
            if e.value is not None:
                nums.append(e.value)
        else:
            for name, c in e.counts.items():
                nums.append(float(c))
                nums.extend(float(x) for x in _NUM.findall(name))
                if e.n:
                    nums.append(100.0 * c / e.n)
            nums.append(float(len(e.counts)))
    return nums


def _matches(x: float, allowed: list[float]) -> bool:
    for a in allowed:
        tol = max(0.6, 0.06 * abs(a))
        if abs(x - a) <= tol:
            return True
        # a percentage written as a fraction (0.63 vs 63) or vice versa
        if a <= 1.0 and abs(x - 100 * a) <= max(0.6, 6 * abs(a)):
            return True
    return False


def unverified_numbers(text: str, evidence: list[Metric | Distribution]) -> list[str]:
    allowed = _allowed_numbers(evidence)
    cleaned = _LABEL_TOKENS.sub(" ", text)
    bad = []
    for tok in _NUM.findall(cleaned):
        x = float(tok)
        if x <= 3 and "." not in tok:  # tiny counts ("两三拍", list markers) are not checked
            continue
        if not _matches(x, allowed):
            bad.append(tok)
    return bad


def ground_insights(raw: list[dict], stats: MatchStats) -> tuple[list[CoachInsight], int]:
    idx = index(stats)
    out: list[CoachInsight] = []
    dropped = 0
    for item in raw:
        keys = [k for k in item.get("evidence") or [] if k in idx]
        usable = [idx[k] for k in keys if _usable(idx[k])]
        warnings = []
        unknown = [k for k in item.get("evidence") or [] if k not in idx]
        if unknown:
            warnings.append(f"引用了不存在的指标：{', '.join(unknown)}")
        kind = item.get("kind", "pattern")
        if not usable and kind != "data_quality":
            dropped += 1
            continue
        conf = item.get("confidence", "low")
        if conf not in _RANK:
            conf = "low"
        if usable:
            weakest = min((e.confidence for e in usable), key=_RANK.__getitem__)
            if _RANK[weakest] < _RANK[conf]:
                conf = weakest
        bad = unverified_numbers(f"{item.get('title', '')} {item.get('detail', '')}", usable)
        if bad:
            warnings.append(f"以下数字在所引用的数据中找不到：{', '.join(bad)}")
        pid = item.get("player_id")
        out.append(CoachInsight(
            kind=kind if kind in ("strength", "weakness", "pattern", "tactic", "data_quality") else "pattern",
            player_id=pid if pid in ("P1", "P2") else None,
            title=str(item.get("title", "")).strip(), detail=str(item.get("detail", "")).strip(),
            evidence=keys, confidence=conf, grounded=not bad, warnings=warnings,
        ))
    return out, dropped


def ground_training(raw: list[dict], stats: MatchStats) -> list[TrainingItem]:
    idx = index(stats)
    out = []
    for item in raw:
        pid = item.get("player_id")
        retest = item.get("retest_metric")
        out.append(TrainingItem(
            player_id=pid if pid in ("P1", "P2") else None,
            drill=str(item.get("drill", "")).strip(), why=str(item.get("why", "")).strip(),
            evidence=[k for k in item.get("evidence") or [] if k in idx],
            retest_metric=retest if retest in idx else None,
        ))
    return [t for t in out if t.drill]
