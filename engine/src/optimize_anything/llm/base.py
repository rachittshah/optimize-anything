"""Backend-neutral LLM types: one response shape and one call protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class LLMError(RuntimeError):
    """An LLM backend call failed (transport, auth, CLI error, unparseable output)."""


class LLMRefusalError(LLMError):
    """The model declined to answer (Anthropic stop_reason == "refusal")."""


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0  # API-equivalent list price
    backend: str = ""
    latency_s: float = 0.0
    cache_write_tokens: int = 0
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str,
        max_tokens: int,
        effort: str | None = None,
    ) -> LLMResponse: ...


def check_effort(effort: str | None) -> None:
    if effort is not None and effort not in VALID_EFFORTS:
        raise ValueError(f"effort must be one of {VALID_EFFORTS}, got {effort!r}")
