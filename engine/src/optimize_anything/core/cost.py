"""Per-run LLM cost accounting, shared by the proposer and the LLM judge."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

from optimize_anything.llm.base import LLMClient, LLMResponse


@dataclass
class RoleCost:
    usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class CostTracker:
    by_role: dict[str, RoleCost] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Class attribute (not a field): judge calls may run in threads (Config.eval_workers).
    # Distinct name + reentrant, so subclasses that wrap record() in their own lock are safe.
    _record_lock = threading.RLock()

    def record(self, role: str, response: LLMResponse) -> None:
        with self._record_lock:
            self._record(role, response)

    def _record(self, role: str, response: LLMResponse) -> None:
        rc = self.by_role.setdefault(role, RoleCost())
        rc.usd += response.cost_usd or 0.0
        rc.calls += 1
        rc.input_tokens += response.input_tokens
        rc.output_tokens += response.output_tokens
        rc.cache_read_tokens += response.cache_read_tokens
        for w in response.warnings:
            if w not in self.warnings:
                self.warnings.append(w)

    @property
    def total_usd(self) -> float:
        return sum(rc.usd for rc in self.by_role.values())

    @property
    def calls(self) -> int:
        return sum(rc.calls for rc in self.by_role.values())

    def to_dict(self) -> dict:
        d = {
            "total_usd": self.total_usd,
            "calls": self.calls,
            "by_role": {role: asdict(rc) for role, rc in self.by_role.items()},
        }
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d


class TrackedLLM:
    """LLMClient wrapper that records every response under a role."""

    def __init__(self, llm: LLMClient, tracker: CostTracker, role: str):
        self.llm = llm
        self.tracker = tracker
        self.role = role

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        resp = self.llm.complete(prompt, system=system, model=model,
                                 max_tokens=max_tokens, effort=effort)
        self.tracker.record(self.role, resp)
        return resp
