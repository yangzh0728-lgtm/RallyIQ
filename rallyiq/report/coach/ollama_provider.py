"""Local, free LLM via Ollama (https://ollama.com). Nothing leaves the machine.

Setup on the Mac: ``brew install ollama`` (or the app), ``ollama serve``, ``ollama pull qwen2.5:7b``.

Env: ``RALLYIQ_OLLAMA_URL`` (default ``http://localhost:11434``), ``RALLYIQ_OLLAMA_MODEL``
(default ``qwen2.5:7b``: good Chinese, reliable JSON, ~5 GB RAM), ``RALLYIQ_OLLAMA_CTX``
(context tokens, default 32768: the stats plus the coaching notes are ~15k tokens),
``RALLYIQ_OLLAMA_TIMEOUT`` (seconds, default 900), ``RALLYIQ_OLLAMA_KEEP_ALIVE`` (how long the
model stays in memory after a report, default ``5m``, so it does not hold RAM while training runs).

Uses only the standard library. The output schema goes in Ollama's ``format`` field
(structured outputs), and the same grounding checks as for Claude apply afterwards.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ...contracts import MatchStats
from .base import CoachProviderError
from .prompt import OUTPUT_SCHEMA, SYSTEM_PROMPT, user_message

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

_JSON_HINT = "\n\n只输出一个符合给定 JSON Schema 的 JSON 对象，不要输出其他文字。"


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None, url: str | None = None, timeout: float | None = None,
                 num_ctx: int | None = None, opener=None):
        self.model = model or os.environ.get("RALLYIQ_OLLAMA_MODEL", DEFAULT_MODEL)
        self.url = (url or os.environ.get("RALLYIQ_OLLAMA_URL", DEFAULT_URL)).rstrip("/")
        self.timeout = timeout or float(os.environ.get("RALLYIQ_OLLAMA_TIMEOUT", "900"))
        self.num_ctx = num_ctx or int(os.environ.get("RALLYIQ_OLLAMA_CTX", "32768"))
        self.keep_alive = os.environ.get("RALLYIQ_OLLAMA_KEEP_ALIVE", "5m")
        self._open = opener or urllib.request.urlopen

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else ""
            if e.code == 404 and "not found" in detail:
                raise CoachProviderError(f"本地模型 {self.model} 还没有下载（运行 ollama pull {self.model}）") from e
            raise CoachProviderError(f"Ollama 返回错误 {e.code}：{detail}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise CoachProviderError(f"连不上本地 Ollama（{self.url}）：{getattr(e, 'reason', e)}。"
                                     "请先安装并运行 Ollama") from e
        except json.JSONDecodeError as e:
            raise CoachProviderError(f"Ollama 返回的内容无法解析：{e}") from e

    def _check(self) -> None:
        """Fail in seconds, not after the long generation timeout, when Ollama is not running."""
        req = urllib.request.Request(self.url + "/api/tags", method="GET")
        try:
            with self._open(req, timeout=3) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise CoachProviderError(f"连不上本地 Ollama（{self.url}）：{getattr(e, 'reason', e)}。"
                                     "请先安装并运行 Ollama") from e
        names = {m.get("name") for m in tags.get("models") or []} | {m.get("model") for m in tags.get("models") or []}
        want = self.model if ":" in self.model else self.model + ":latest"
        if names and want not in names:
            raise CoachProviderError(f"本地模型 {self.model} 还没有下载（运行 ollama pull {self.model}）")

    def generate(self, stats: MatchStats, knowledge=None) -> dict:
        self._check()
        body = {
            "model": self.model,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.3, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + _JSON_HINT},
                {"role": "user", "content": user_message(stats, knowledge)},
            ],
        }
        res = self._post("/api/chat", body, self.timeout)
        text = (res.get("message") or {}).get("content") or ""
        if not text.strip():
            raise CoachProviderError("本地模型没有返回内容")
        if res.get("done_reason") == "length":
            raise CoachProviderError("本地模型输出被截断（可调大 RALLYIQ_OLLAMA_CTX）")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CoachProviderError(f"本地模型返回的 JSON 无法解析：{e}") from e
        if not isinstance(data, dict):
            raise CoachProviderError("本地模型返回格式不正确")
        for k in ("insights", "training", "caveats"):
            if not isinstance(data.get(k), list):
                data[k] = []
        data.setdefault("summary", "")
        return data
