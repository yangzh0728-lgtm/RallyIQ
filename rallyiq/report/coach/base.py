"""Provider protocol and registry. Add a provider with ``register_provider``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ...contracts import MatchStats


class CoachProviderError(RuntimeError):
    """The provider could not produce a report (not configured, API error, bad output)."""


class CoachProvider(Protocol):
    name: str
    model: str | None

    def generate(self, stats: MatchStats, knowledge=None) -> dict:
        """Return raw JSON matching ``prompt.OUTPUT_SCHEMA``; raise ``CoachProviderError`` on failure.

        ``knowledge`` (``advice.Knowledge``) holds the detected weaknesses and the coaching
        notes retrieved for them; training items cite those notes by id."""
        ...


_REGISTRY: dict[str, Callable[[], CoachProvider]] = {}


def register_provider(name: str, factory: Callable[[], CoachProvider]) -> None:
    _REGISTRY[name] = factory


def list_providers() -> list[str]:
    return sorted(_REGISTRY)


def make_provider(name: str) -> CoachProvider:
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]()
