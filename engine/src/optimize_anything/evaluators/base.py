# engine/src/optimize_anything/evaluators/base.py
from __future__ import annotations
from typing import Any, Protocol


class Evaluator(Protocol):
    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate, return (score, ASI diagnostics)."""
        ...
