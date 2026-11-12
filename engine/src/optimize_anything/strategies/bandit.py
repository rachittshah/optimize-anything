"""Sliding-window UCB1 over mutation operators (Config.operator_bandit)."""
from __future__ import annotations
import math
from collections import deque

OPERATORS = ("rewrite", "section_edit", "compress")
COMPRESS_MIN_CHARS = 400  # compress is only offered for artifacts longer than this


class SlidingWindowUCB:
    """UCB1 computed over the last `window` plays (all arms share one window)."""

    def __init__(self, arms: tuple[str, ...] = OPERATORS, window: int = 20,
                 c: float = math.sqrt(2)):
        self.arms = arms
        self.window = window
        self.c = c
        self.history: deque[tuple[str, float]] = deque(maxlen=window)
        self.counts_total: dict[str, int] = {a: 0 for a in arms}

    def select(self, eligible: list[str] | None = None) -> str:
        eligible = [a for a in (eligible or list(self.arms)) if a in self.arms]
        if not eligible:
            raise ValueError("no eligible arms")
        plays = {a: 0 for a in eligible}
        sums = {a: 0.0 for a in eligible}
        for arm, r in self.history:
            if arm in plays:
                plays[arm] += 1
                sums[arm] += r
        for a in eligible:  # unplayed (in window) arms first, in declared order
            if plays[a] == 0:
                return a
        n = sum(plays.values())
        return max(eligible, key=lambda a: sums[a] / plays[a]
                   + self.c * math.sqrt(math.log(n) / plays[a]))

    def update(self, arm: str, reward: float) -> None:
        self.history.append((arm, reward))
        self.counts_total[arm] = self.counts_total.get(arm, 0) + 1
