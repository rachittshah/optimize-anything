from __future__ import annotations
from typing import Any

from optimize_anything.llm.base import LLMClient
from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate

SYSTEM_PROMPT = (
    "You improve text artifacts. Read the evaluation feedback, diagnose what is failing, "
    "and return a complete improved version inside one fenced code block."
)


class ReflectiveProposer:
    """GEPA-style reflective mutation: read ASI, diagnose, propose a full replacement."""

    def __init__(self, llm: LLMClient, model: str, effort: str | None = "high",
                 max_tokens: int = 8192):
        self.llm = llm
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def propose(
        self,
        current_text: str,
        component_name: str,
        objective: str,
        asi_entries: list[dict[str, Any]],
        background: str | None = None,
        operator: str = "rewrite",
        rejected_edits: list[dict[str, Any]] | None = None,
    ) -> str:
        prompt = build_reflection_prompt(
            current_text=current_text,
            component_name=component_name,
            objective=objective,
            asi_entries=asi_entries,
            background=background,
            operator=operator,
            rejected_edits=rejected_edits,
        )
        resp = self.llm.complete(prompt, system=SYSTEM_PROMPT, model=self.model,
                                 max_tokens=self.max_tokens, effort=self.effort)
        return extract_candidate(resp.text)
