"""OpenAI-compatible backend (official `openai` SDK, lazy client).

Uses Chat Completions so any OpenAI-compatible endpoint works, e.g. OpenRouter via
base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY".
"""
from __future__ import annotations

import os
import time
from typing import Any

from optimize_anything.llm.base import LLMError, LLMRefusalError, LLMResponse, check_effort
from optimize_anything.llm.models import compute_cost

# OpenAI reasoning_effort tops out at "high"; clamp our larger levels.
_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAILLM:
    backend = "openai"

    def __init__(self, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 600.0):
        self.base_url = base_url or os.environ.get("OA_OPENAI_BASE_URL") or None
        self.api_key_env = api_key_env
        self._timeout = timeout
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=os.environ.get(self.api_key_env),
                                         base_url=self.base_url, timeout=self._timeout)
        return self._client

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        check_effort(effort)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {"model": model, "messages": messages,
                                  "max_completion_tokens": max_tokens}
        if effort:
            if self.base_url and "openrouter" in self.base_url:
                kwargs["extra_body"] = {"reasoning": {"effort": _EFFORT_MAP[effort]}}
            else:
                kwargs["reasoning_effort"] = _EFFORT_MAP[effort]

        start = time.monotonic()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(f"openai request failed: {e}") from e
        latency = time.monotonic() - start

        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusalError(f"{model} refused: {choice.message.refusal}")
        text = choice.message.content or ""

        u = resp.usage
        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
        out = getattr(u, "completion_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        cost, warn = compute_cost(model, prompt_tokens - cached, out, cached)
        return LLMResponse(text=text, model=getattr(resp, "model", model) or model,
                           input_tokens=prompt_tokens, output_tokens=out, cache_read_tokens=cached,
                           cost_usd=cost, backend=self.backend, latency_s=latency,
                           warnings=[warn] if warn else [])
