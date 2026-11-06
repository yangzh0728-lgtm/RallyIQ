"""AI coach: pluggable providers turn ``MatchStats`` into grounded Chinese comments.

Provider selection: explicit ``provider`` argument, else ``RALLYIQ_COACH_PROVIDER``,
else ``anthropic``. If the default provider fails (not installed, no credentials,
API error) the rule-based ``offline`` provider is used and the reason is recorded in
``CoachReport.fallback_reason``. An explicitly requested provider that fails raises
``CoachUnavailable`` instead.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from ...contracts import CoachReport, MatchStats
from .anthropic_provider import AnthropicProvider
from .base import CoachProvider, CoachProviderError, list_providers, make_provider, register_provider
from .grounding import ground_insights, ground_training
from .offline import OfflineProvider

register_provider("anthropic", AnthropicProvider)
register_provider("offline", OfflineProvider)

DEFAULT_PROVIDER = "anthropic"


class CoachUnavailable(RuntimeError):
    pass


def default_provider_name() -> str:
    return os.environ.get("RALLYIQ_COACH_PROVIDER", DEFAULT_PROVIDER)


def build_report(raw: dict, stats: MatchStats, provider: CoachProvider, fallback_reason: str | None = None) -> CoachReport:
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
        training=ground_training(list(raw.get("training") or []), stats), caveats=caveats,
        fallback_reason=fallback_reason, dropped=dropped,
    )


def generate_coach_report(stats: MatchStats, provider: str | None = None) -> CoachReport:
    explicit = provider is not None
    name = provider or default_provider_name()
    try:
        p = make_provider(name)
    except KeyError as e:
        raise CoachUnavailable(f"未知的点评提供方：{name}（可用：{', '.join(list_providers())}）") from e
    try:
        return build_report(p.generate(stats), stats, p)
    except CoachProviderError as e:
        if explicit or name == "offline":
            raise CoachUnavailable(str(e)) from e
        off = OfflineProvider()
        return build_report(off.generate(stats), stats, off, fallback_reason=str(e))


__all__ = [
    "AnthropicProvider", "CoachProvider", "CoachProviderError", "CoachUnavailable", "OfflineProvider",
    "build_report", "default_provider_name", "generate_coach_report", "list_providers", "register_provider",
]
