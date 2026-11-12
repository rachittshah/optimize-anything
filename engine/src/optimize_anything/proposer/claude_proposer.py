"""Backward-compatible alias. Prefer ReflectiveProposer with an explicit LLMClient."""
from __future__ import annotations

from optimize_anything.llm.base import LLMClient
from optimize_anything.llm.models import resolve_model
from optimize_anything.proposer.reflective import ReflectiveProposer


class ClaudeProposer(ReflectiveProposer):
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 8192,
        temperature: float | None = None,  # deprecated, ignored
        llm: LLMClient | None = None,
        effort: str | None = "high",
    ):
        if llm is None:
            from optimize_anything.llm.anthropic_backend import AnthropicLLM
            llm = AnthropicLLM()  # lazy: no client until first call
        super().__init__(llm, model=model or resolve_model("proposer", "anthropic"),
                         effort=effort, max_tokens=max_tokens)
