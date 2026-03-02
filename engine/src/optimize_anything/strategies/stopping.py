from __future__ import annotations
import time
from typing import Protocol


class Stopper(Protocol):
    def should_stop(self, **kwargs) -> bool: ...


class BudgetStopper:
    def __init__(self, max_evals: int):
        self.max_evals = max_evals

    def should_stop(self, total_evals: int = 0, **kwargs) -> bool:
        return total_evals >= self.max_evals


class IterationStopper:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def should_stop(self, iteration: int = 0, **kwargs) -> bool:
        return iteration >= self.max_iterations


class TimeoutStopper:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._start = time.time()

    def should_stop(self, **kwargs) -> bool:
        return (time.time() - self._start) >= self.seconds


class NoImprovementStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self._best_score = float("-inf")
        self._stale_count = 0

    def should_stop(self, best_score: float = 0.0, **kwargs) -> bool:
        if best_score > self._best_score:
            self._best_score = best_score
            self._stale_count = 0
        else:
            self._stale_count += 1
        return self._stale_count >= self.patience


class CompositeStopper:
    def __init__(self, stoppers: list):
        self.stoppers = stoppers

    def should_stop(self, **kwargs) -> bool:
        return any(s.should_stop(**kwargs) for s in self.stoppers)
