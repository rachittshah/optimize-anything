"""Counterexample-guided minibatch sampling (Config.counterexample_mining), train-only.

Half the minibatch (ceil(k/2)) comes from a "hard pool" of train ids the current best fails on;
the rest comes from the usual epoch sampler, which still advances every iteration.
"""
from __future__ import annotations
import math
import random

from optimize_anything.strategies.sampling import EpochBatchSampler


def hard_ids(train_scores: dict[int, float]) -> set[int]:
    """Ids where the best scored below 1.0 and at or below its own mean on the seen examples."""
    if not train_scores:
        return set()
    mean = sum(train_scores.values()) / len(train_scores)
    return {i for i, s in train_scores.items() if s < 1.0 and s <= mean}


class HardPoolSampler:
    def __init__(self, inner: EpochBatchSampler, seed: int = 42):
        self.inner = inner
        self._rng = random.Random(seed + 1)  # own RNG: never perturbs the epoch sampler

    def next_batch(self, all_ids: list, iteration: int, hard_pool: set) -> list:
        base = self.inner.next_batch(all_ids, iteration)  # always advance the epoch
        k = self.inner.minibatch_size
        pool = sorted(i for i in hard_pool if i in set(all_ids))
        if not pool:
            return base
        n_hard = min(math.ceil(k / 2), len(pool))
        chosen = self._rng.sample(pool, n_hard)
        rest = [i for i in base if i not in chosen][: k - n_hard]
        return chosen + rest
