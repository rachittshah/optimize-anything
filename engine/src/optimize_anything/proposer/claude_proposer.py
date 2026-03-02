from __future__ import annotations
from typing import Any

import anthropic

from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate


class ClaudeProposer:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic()

    def propose(
        self,
        current_text: str,
        component_name: str,
        objective: str,
        asi_entries: list[dict[str, Any]],
        background: str | None = None,
    ) -> str:
        prompt = build_reflection_prompt(
            current_text=current_text,
            component_name=component_name,
            objective=objective,
            asi_entries=asi_entries,
            background=background,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return extract_candidate(response.content[0].text)
