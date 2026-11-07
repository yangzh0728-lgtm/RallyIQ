"""Turn shots + perception into ``StrokesResult`` (one ``StrokeInfo`` per shot).

The classifier backend is pluggable: ``rules`` (default, no weights), ``bst``
(optional skeleton + shuttle transformer, see ``bst.py``) or ``sklearn`` (gradient
boosting on the same evidence, see ``learned.py``). Whatever the backend,
the review policy here is the same:

* ``undeterminable`` with ``shot_type = null`` when too little evidence was measurable;
* ``ok`` only when the top type is confident, clearly ahead of the runner-up,
  backed by (almost) complete evidence and the hit itself is ``ok``;
* ``needs_review`` otherwise, with reason codes saying why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts import EventsResult, PlayerTracks, ShotStroke, ShotTypeScore, ShuttleTrack, StrokeInfo, StrokesResult
from .features import R_FIRST_NOT_SERVE, FeatureConfig, StrokeFeatures, shot_features
from .rules import RulesConfig, softmax_scores, type_scores

RULES_NAME = "rules_v1"

R_AMBIGUOUS = "ambiguous_types"
R_HIT_REVIEW = "hit_needs_review"
R_INCOMPLETE = "incomplete_evidence"
R_NO_RULE_FITS = "no_rule_fits"
R_INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class StrokesConfig:
    classifier: str = "rules"  # "rules" | "bst" | "sklearn"
    ok_threshold: float = 0.6
    ok_margin: float = 0.2
    min_evidence_ok: float = 0.8  # share of evidence groups measured for an "ok"
    min_evidence: float = 0.3  # below this: undeterminable
    min_rule_score: float = 0.05  # best raw rule score below this: no rule fits
    top_k: int = 3
    min_candidate_p: float = 0.05  # runner-up candidates below this are not listed
    features: FeatureConfig = field(default_factory=FeatureConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)


# A model backend: (events, shuttle, players, features) -> {shot_id: {type: p}} (missing = no prediction).
ModelFn = Callable[[EventsResult, ShuttleTrack | None, PlayerTracks | None, list[StrokeFeatures]], dict[str, dict[str, float]]]


def classify(
    events: EventsResult,
    shuttle: ShuttleTrack | None,
    players: PlayerTracks | None,
    fps: float,
    cfg: StrokesConfig | None = None,
    model: ModelFn | None = None,
    model_name: str | None = None,
) -> StrokesResult:
    cfg = cfg or StrokesConfig()
    feats = shot_features(events, shuttle, players, fps, cfg.features)
    shots = {s.shot_id: s for s in events.shots}
    model_probs = model(events, shuttle, players, feats) if model else {}
    name = model_name if model else RULES_NAME

    out: list[ShotStroke] = []
    prev_type: dict[str | None, str | None] = {}
    for f in feats:
        shot = shots[f.shot_id]
        scores, m = type_scores(f, prev_type.get(shot.rally_id) if not f.first_in_rally else None, cfg.rules)
        evidence = sum(m.known.values()) / len(m.known)
        reasons = list(f.reasons)
        probs = model_probs.get(f.shot_id)
        if probs is None:
            probs = softmax_scores(scores, cfg.rules.eps)
            if max(scores.values()) < cfg.min_rule_score:
                reasons.append(R_NO_RULE_FITS)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        (top, p1), p2 = ranked[0], ranked[1][1]
        if f.first_in_rally and not top.endswith("_serve") and R_FIRST_NOT_SERVE not in reasons:
            reasons.append(R_FIRST_NOT_SERVE)
        confidence = round(p1 * (0.5 + 0.5 * evidence), 3)

        if evidence < cfg.min_evidence:
            status, shot_type = "undeterminable", None
            reasons.append(R_INSUFFICIENT)
        else:
            shot_type = top
            if p1 - p2 < cfg.ok_margin:
                reasons.append(R_AMBIGUOUS)
            if evidence < cfg.min_evidence_ok:
                reasons.append(R_INCOMPLETE)
            if shot.status != "ok":
                reasons.append(R_HIT_REVIEW)
            blocking = [r for r in reasons if r not in ("last_shot_of_rally",)]
            ok = confidence >= cfg.ok_threshold and not blocking and shot.hitter is not None
            status = "ok" if ok else "needs_review"
        prev_type[shot.rally_id] = shot_type
        out.append(ShotStroke(
            shot_id=f.shot_id,
            frame=shot.frame,
            stroke=StrokeInfo(
                shot_type=shot_type,
                confidence=confidence,
                status=status,
                source="model",
                classifier=name,
                candidates=[ShotTypeScore(shot_type=k, p=round(v, 3))
                            for n, (k, v) in enumerate(ranked[: cfg.top_k]) if n == 0 or v >= cfg.min_candidate_p],
                reasons=reasons,
                evidence=f.evidence(),
            ),
        ))
    return StrokesResult(video_id=events.video_id, classifier=name, shots=out)


def merge_strokes(events: EventsResult, strokes: StrokesResult | None) -> EventsResult:
    """Attach stroke predictions to the raw events (by shot_id). Returns a copy."""
    if strokes is None:
        return events
    by_id = {s.shot_id: s.stroke for s in strokes.shots}
    shots = []
    for s in events.shots:
        st = by_id.get(s.shot_id)
        shots.append(s.model_copy(update={"stroke": st, "shot_type": st.shot_type if st else None}) if st else s)
    return events.model_copy(update={"shots": shots})
