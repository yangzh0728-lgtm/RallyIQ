"""Claude API provider (default). Needs ``pip install -e '.[coach]'`` and credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).

Env: ``RALLYIQ_COACH_MODEL`` (default ``claude-opus-5-5``), ``RALLYIQ_COACH_EFFORT``
(default ``high``).
"""

from __future__ import annotations

import json
import os

from ...contracts import MatchStats
from .base import CoachProviderError
from .prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, user_message

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        self.model = model or os.environ.get("RALLYIQ_COACH_MODEL", DEFAULT_MODEL)
        self.effort = effort or os.environ.get("RALLYIQ_COACH_EFFORT", "high")
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise CoachProviderError("未安装 anthropic SDK（pip install -e '.[coach]'）") from e
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, stats: MatchStats) -> dict:
        client = self._get_client()
        try:
            import anthropic

            api_errors: tuple[type[BaseException], ...] = (anthropic.APIError,)
        except ImportError:  # injected test client
            api_errors = ()
        try:
            # Streaming keeps long adaptive-thinking turns clear of HTTP timeouts.
            with client.beta.messages.stream(
                model=self.model,
                max_tokens=32000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_message(stats)}],
            ) as stream:
                msg = stream.get_final_message()
        except api_errors as e:  # auth, rate limit, network, server errors
            raise CoachProviderError(f"Claude API 调用失败：{type(e).__name__}: {e}") from e
        except TypeError as e:  # credentials could not be resolved by the SDK
            raise CoachProviderError("未配置 Claude API 凭证（设置 ANTHROPIC_API_KEY 或运行 ant auth login）") from e

        if msg.stop_reason == "refusal":
            raise CoachProviderError("Claude 拒绝了这次请求")
        if msg.stop_reason == "max_tokens":
            raise CoachProviderError("输出被截断（max_tokens）")
        self.model = getattr(msg, "model", None) or self.model
        text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
        if not text:
            raise CoachProviderError("Claude 没有返回文本")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CoachProviderError(f"返回的 JSON 无法解析：{e}") from e
        if not isinstance(data, dict):
            raise CoachProviderError("返回格式不正确")
        return data
