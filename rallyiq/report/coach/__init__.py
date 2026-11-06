"""AI coach: pluggable providers turn ``MatchStats`` into grounded Chinese comments.

Provider selection: explicit ``provider`` argument, else ``RALLYIQ_COACH_PROVIDER``,
else ``auto``: the local, free ``ollama`` model first, then ``anthropic`` (Claude) only
when ``ANTHROPIC_API_KEY`` is set, then the rule-based ``offline`` provider. Whenever
the default choice falls back, the reasons are recorded in ``CoachReport.fallback_reason``.
An explicitly requested provider that fails raises ``CoachUnavailable`` instead.
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
from .ollama_provider import OllamaProvider

register_provider("anthropic", AnthropicProvider)
register_provider("offline", OfflineProvider)
register_provider("ollama", OllamaProvider)

DEFAULT_PROVIDER = "auto"
_NAMES = {"ollama": "本地模型", "anthropic": "Claude", "offline": "离线模板"}


def auto_chain() -> list[str]:
    """Providers ``auto`` tries, in order (``offline`` is always the last resort)."""
    chain = ["ollama"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.append("anthropic")
    return chain


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
    if knowledge is None:
        knowledge = knowledge_context(stats)
    chain = auto_chain() if name == "auto" else [name]
    errors: list[str] = []
    for n in chain:
        try:
            p = make_provider(n)
        except KeyError as e:
            raise CoachUnavailable(f"未知的点评提供方：{n}（可用：auto, {', '.join(list_providers())}）") from e
        try:
            return build_report(_generate(p, stats, knowledge), stats, p, knowledge=knowledge,
                                fallback_reason="；".join(errors) or None)
        except CoachProviderError as e:
            if explicit and name != "auto" or n == "offline":
                raise CoachUnavailable(str(e)) from e
            errors.append(f"{_NAMES.get(n, n)}不可用：{e}" if name == "auto" else str(e))
    off = OfflineProvider()
    return build_report(_generate(off, stats, knowledge), stats, off, fallback_reason="；".join(errors) or None,
                        knowledge=knowledge)


__all__ = [
    "AnthropicProvider",
    "CoachProvider",
    "CoachProviderError",
    "CoachUnavailable",
    "Knowledge",
    "OfflineProvider",
    "OllamaProvider",
    "auto_chain",
    "build_report",
    "default_provider_name",
    "generate_coach_report",
    "knowledge_context",
    "list_providers",
    "register_provider",
]
