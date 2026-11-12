from __future__ import annotations
from typing import Any, Protocol


class Proposer(Protocol):
    def propose(
        self,
        current_text: str,
        component_name: str,
        objective: str,
        asi_entries: list[dict[str, Any]],
        background: str | None = None,
    ) -> str:
        """Return the full replacement text for one component.

        Optional: accept `operator: str` ("rewrite" | "section_edit" | "compress") to support
        Config.mutation_operator / operator_bandit. Proposers without it still work (the engine
        only passes it when the proposer's signature accepts it)."""
        ...
