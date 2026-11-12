"""Anthropic Messages API backend (official SDK, lazy client).

Current models reject `temperature` (or any non-default value), so it is never sent.
Effort goes through `output_config`; `thinking` is omitted (adaptive by default).
"""
from __future__ import annotations

import time
from typing import Any

from optimize_anything.llm.base import LLMError, LLMRefusalError, LLMResponse, check_effort
from optimize_anything.llm.models import compute_cost


class AnthropicLLM:
    backend = "anthropic"

    def __init__(self, api_key: str | None = None, timeout: float = 600.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        check_effort(effort)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if effort:
            kwargs["output_config"] = {"effort": effort}

        start = time.monotonic()
        try:
            msg = self.client.messages.create(**kwargs)
        except Exception as e:  # SDK raises many subclasses; normalise
            raise LLMError(f"anthropic request failed: {e}") from e
        latency = time.monotonic() - start

        if msg.stop_reason == "refusal":
            raise LLMRefusalError(f"{model} refused the request")

        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        u = msg.usage
        inp = u.input_tokens or 0
        out = u.output_tokens or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cost, warn = compute_cost(model, inp, out, cr, cw)
        return LLMResponse(text=text, model=getattr(msg, "model", model) or model,
                           input_tokens=inp, output_tokens=out, cache_read_tokens=cr,
                           cache_write_tokens=cw, cost_usd=cost, backend=self.backend,
                           latency_s=latency, warnings=[warn] if warn else [])
