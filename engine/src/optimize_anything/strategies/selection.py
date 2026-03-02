from __future__ import annotations
import random
from typing import Protocol

from optimize_anything.core.state import State


class CandidateSelector(Protocol):
    def select(self, state: State) -> int: ...


class ParetoSelector:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def select(self, state: State) -> int:
        return state.frontier.select_candidate(state.agg_scores)


class BestSelector:
    def select(self, state: State) -> int:
        return max(range(len(state.agg_scores)), key=lambda i: state.agg_scores[i])


class EpsilonGreedySelector:
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select(self, state: State) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, len(state.candidates) - 1)
        return max(range(len(state.agg_scores)), key=lambda i: state.agg_scores[i])
