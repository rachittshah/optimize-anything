from __future__ import annotations
import random


class EpochBatchSampler:
    def __init__(self, minibatch_size: int = 3, seed: int = 42):
        self.minibatch_size = minibatch_size
        self._rng = random.Random(seed)
        self._shuffled: list = []
        self._epoch = -1

    def next_batch(self, all_ids: list, iteration: int) -> list:
        n = len(all_ids)
        base = iteration * self.minibatch_size
        epoch = base // n

        if epoch > self._epoch:
            self._epoch = epoch
            self._shuffled = list(all_ids)
            self._rng.shuffle(self._shuffled)
            # Pad to multiple of minibatch_size
            remainder = n % self.minibatch_size
            if remainder != 0:
                pad = self.minibatch_size - remainder
                self._shuffled.extend(self._shuffled[:pad])

        if not self._shuffled:
            self._shuffled = list(all_ids)
            self._rng.shuffle(self._shuffled)

        start = base % len(self._shuffled)
        end = start + self.minibatch_size
        return self._shuffled[start:end]
