# engine/src/optimize_anything/core/pareto.py
from __future__ import annotations
import random
from collections import defaultdict


class ParetoFrontier:
    def __init__(self, seed: int = 42):
        self._front: dict[str, float] = {}
        self._programs: dict[str, set[int]] = defaultdict(set)
        self._rng = random.Random(seed)

    def update(self, key: str, score: float, candidate_idx: int) -> bool:
        prev = self._front.get(key, float("-inf"))
        if score > prev:
            self._front[key] = score
            self._programs[key] = {candidate_idx}
            return True
        elif score == prev:
            self._programs[key].add(candidate_idx)
            return False
        return False

    def best_score(self, key: str) -> float:
        return self._front.get(key, float("-inf"))

    def programs_at(self, key: str) -> set[int]:
        return self._programs.get(key, set())

    def all_program_ids(self) -> set[int]:
        result = set()
        for programs in self._programs.values():
            result |= programs
        return result

    def aggregated_score(self, candidate_idx: int) -> float:
        scores = []
        for key, programs in self._programs.items():
            if candidate_idx in programs:
                scores.append(self._front[key])
        return sum(scores) / len(scores) if scores else 0.0

    def select_candidate(self, all_scores: list[float]) -> int:
        freq: dict[int, int] = defaultdict(int)
        for programs in self._programs.values():
            for pid in programs:
                freq[pid] += 1

        if not freq:
            return 0

        # Remove dominated: a program is dominated if for every key it appears in,
        # there's another program also in that key
        # Simplified: just do frequency-weighted sampling
        sampling_list = []
        for pid, count in freq.items():
            sampling_list.extend([pid] * count)

        return self._rng.choice(sampling_list)

    def to_dict(self) -> dict:
        return {
            "front": dict(self._front),
            "programs": {k: list(v) for k, v in self._programs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int = 42) -> ParetoFrontier:
        pf = cls(seed=seed)
        pf._front = data["front"]
        pf._programs = defaultdict(set, {k: set(v) for k, v in data["programs"].items()})
        return pf
