from __future__ import annotations
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from optimize_anything.core.cost import CostTracker


class Stopper(Protocol):
    def should_stop(self, **kwargs) -> bool: ...


class BudgetStopper:
    name = "budget"

    def __init__(self, max_evals: int):
        self.max_evals = max_evals

    def should_stop(self, total_evals: int = 0, **kwargs) -> bool:
        return total_evals >= self.max_evals


class IterationStopper:
    name = "iterations"

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def should_stop(self, iteration: int = 0, **kwargs) -> bool:
        return iteration >= self.max_iterations


class TimeoutStopper:
    name = "timeout"

    def __init__(self, seconds: int):
        self.seconds = seconds
        self._start = time.time()

    def should_stop(self, **kwargs) -> bool:
        return (time.time() - self._start) >= self.seconds


class NoImprovementStopper:
    """Stop after `patience` completed iterations without a new best score.

    Counting is keyed on the iteration number, so repeated should_stop calls within one
    iteration don't inflate the stale count.
    """

    name = "no_improvement"

    def __init__(self, patience: int, min_improvement: float = 0.0):
        self.patience = patience
        # Gains <= min_improvement over the last counted best don't reset patience (small gains
        # still accumulate: the reference best only moves on a counted improvement).
        self.min_improvement = min_improvement
        self._best_score = float("-inf")
        self._stale_count = 0
        self._last_iteration: int | None = None

    def should_stop(self, best_score: float = 0.0, iteration: int = 0, **kwargs) -> bool:
        if iteration != self._last_iteration:
            self._last_iteration = iteration
            if best_score > self._best_score + self.min_improvement:
                self._best_score = best_score
                self._stale_count = 0
            else:
                self._stale_count += 1
        return self._stale_count >= self.patience


class CostStopper:
    """Stop once cumulative LLM spend reaches max_usd."""

    name = "cost"

    def __init__(self, max_usd: float, tracker: CostTracker):
        self.max_usd = max_usd
        self.tracker = tracker

    def should_stop(self, **kwargs) -> bool:
        return self.tracker.total_usd >= self.max_usd


class CompositeStopper:
    def __init__(self, stoppers: list):
        self.stoppers = stoppers
        self.triggered: list[str] = []  # names of the stoppers that fired on the last check

    def should_stop(self, **kwargs) -> bool:
        # Evaluate every stopper (no short-circuit) so stateful ones see each iteration.
        results = [s.should_stop(**kwargs) for s in self.stoppers]
        self.triggered = [getattr(s, "name", type(s).__name__)
                          for s, hit in zip(self.stoppers, results) if hit]
        return any(results)

    @property
    def reason(self) -> str | None:
        return self.triggered[0] if self.triggered else None
