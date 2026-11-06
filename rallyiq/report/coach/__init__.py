"""AI coach: pluggable providers turn ``MatchStats`` into grounded Chinese comments.

Provider selection: explicit ``provider`` argument, else ``RALLYIQ_COACH_PROVIDER``,
else ``anthropic``. If the default provider fails (not installed, no credentials,
API error) the rule-based ``offline`` provider is used and the reason is recorded in
``CoachReport.fallback_reason``. An explicitly requested provider that fails raises
``CoachUnavailable`` instead.
"""

from __future__ import annotations

import inspect
import os
from datetime import UTC, datetime

from ...contracts import CoachReport, MatchStats
from .advice import Knowledge, knowledge_context
from .anthropic_provider import AnthropicProvider
from .base import (
    CoachProvider,
    CoachProviderError,
    list_providers,
    make_provider,
    register_provider,
)
from .grounding import ground_insights, ground_training
from .offline import OfflineProvider

register_provider("anthropic", AnthropicProvider)
register_provider("offline", OfflineProvider)

DEFAULT_PROVIDER = "anthropic"


class CoachUnavailable(RuntimeError):
    pass


def default_provider_name() -> str:
    return os.environ.get("RALLYIQ_COACH_PROVIDER", DEFAULT_PROVIDER)


def build_report(raw: dict, stats: MatchStats, provider: CoachProvider, fallback_reason: str | None = None,
                 knowledge: Knowledge | None = None) -> CoachReport:
    insights, dropped = ground_insights(list(raw.get("insights") or []), stats)
    caveats = [str(c) for c in raw.get("caveats") or [] if str(c).strip()]
    for c in stats.caveats:  # data caveats are always shown, whatever the model wrote
        if c not in caveats:
            caveats.append(c)
    if dropped:
        caveats.append(f"有 {dropped} 条点评因为没有引用有效数据而被移除。")
    return CoachReport(
        video_id=stats.video_id, stats_fingerprint=stats.fingerprint, provider=provider.name,
        model=getattr(provider, "model", None), created_at=datetime.now(UTC),
        summary=str(raw.get("summary", "")).strip(), insights=insights,
        training=ground_training(list(raw.get("training") or []), stats, knowledge)[:5], caveats=caveats,
        fallback_reason=fallback_reason, dropped=dropped,
    )


def _generate(p: CoachProvider, stats: MatchStats, knowledge: Knowledge | None) -> dict:
    try:
        accepts = "knowledge" in inspect.signature(p.generate).parameters
    except (TypeError, ValueError):
        accepts = False
    return p.generate(stats, knowledge=knowledge) if accepts else p.generate(stats)


def generate_coach_report(stats: MatchStats, provider: str | None = None,
                          knowledge: Knowledge | None = None) -> CoachReport:
    """``knowledge`` defaults to the weaknesses found in ``stats`` plus the coaching notes
    retrieved for them (``advice.knowledge_context``)."""
    explicit = provider is not None
    name = provider or default_provider_name()
    try:
        p = make_provider(name)
    except KeyError as e:
        raise CoachUnavailable(f"未知的点评提供方：{name}（可用：{', '.join(list_providers())}）") from e
    if knowledge is None:
        knowledge = knowledge_context(stats)
    try:
        return build_report(_generate(p, stats, knowledge), stats, p, knowledge=knowledge)
    except CoachProviderError as e:
        if explicit or name == "offline":
            raise CoachUnavailable(str(e)) from e
        off = OfflineProvider()
        return build_report(_generate(off, stats, knowledge), stats, off, fallback_reason=str(e), knowledge=knowledge)


__all__ = [
    "AnthropicProvider",
    "CoachProvider",
    "CoachProviderError",
    "CoachUnavailable",
    "Knowledge",
    "OfflineProvider",
    "build_report",
    "default_provider_name",
    "generate_coach_report",
    "knowledge_context",
    "list_providers",
    "register_provider",
]
