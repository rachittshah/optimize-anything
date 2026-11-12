"""Deterministic, free backend for tests and `doctor` checks."""
from __future__ import annotations

import re
from typing import Callable

from optimize_anything.llm.base import LLMResponse, check_effort

_FENCE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


# Reflection prompts (proposer/prompts.py) put the artifact between these markers.
_CURRENT = re.compile(r"## Current Version\n```\n(.*?)\n```\n\n## Evaluation Results", re.DOTALL)


def _propose(prompt: str, n: int) -> str | None:
    """Proposer prompt -> the current version plus a deterministic tweak (a distinct revision
    line per call), so mock runs exercise the evaluate/accept path. None if not a proposal."""
    m = _CURRENT.search(prompt)
    if m is None:
        return None
    body = f"{m.group(1).rstrip()}\n(mock revision {n})"
    fence = "````" if "```" in body else "```"
    return f"{fence}\n{body}\n{fence}"


def _echo(prompt: str) -> str:
    # Return the last fenced block unchanged (a no-op proposal) plus a mid score line,
    # so both the proposer and the judge get parseable output.
    blocks = _FENCE.findall(prompt)
    body = blocks[-1].strip() if blocks else prompt[:200]
    return f"```\n{body}\n```\nSCORE: 5/10"


class MockLLM:
    def __init__(
        self,
        responses: list[str] | None = None,
        fn: Callable[[str], str] | None = None,
        cost_per_call: float = 0.0,
    ):
        self.responses = responses
        self.fn = fn
        self.cost_per_call = cost_per_call
        self.calls: list[dict] = []

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        check_effort(effort)
        self.calls.append({"prompt": prompt, "system": system, "model": model,
                           "max_tokens": max_tokens, "effort": effort})
        if self.fn is not None:
            text = self.fn(prompt)
        elif self.responses:
            text = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        else:
            text = _propose(prompt, len(self.calls)) or _echo(prompt)
        return LLMResponse(text=text, model=model, input_tokens=len(prompt) // 4,
                           output_tokens=len(text) // 4, cost_usd=self.cost_per_call, backend="mock")
