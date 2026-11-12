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

    def keys_of(self, candidate_idx: int) -> set[str]:
        return {k for k, progs in self._programs.items() if candidate_idx in progs}

    def pruned_fronts(self, scores: list[float] | dict[int, float] | None = None
                      ) -> dict[str, set[int]]:
        """Per-key fronts with dominated programs removed: exact port of GEPA's
        remove_dominated_programs (gepa-ai/gepa@ce5b0d6, src/gepa/gepa_utils.py#L54-L135).

        Programs are visited in ascending aggregate score (stable: ties keep first-seen order
        over the fronts). A program is dominated iff every front it is on still has another
        survivor; removing it decrements those fronts' survivor counts. Every non-empty front
        keeps at least one program. scores=None treats all programs as equal (GEPA default).
        """
        fronts = self._programs
        program_fronts: dict[int, list[str]] = {}
        remaining: dict[str, int] = {}
        for key, front in fronts.items():
            remaining[key] = len(front)
            for p in front:
                program_fronts.setdefault(p, []).append(key)
        programs = list(program_fronts)
        score_of = (lambda p: 1) if scores is None else (lambda p: scores[p])
        programs = sorted(programs, key=score_of)
        dominated: set[int] = set()
        for y in programs:
            keys = program_fronts[y]
            if all(remaining[k] > 1 for k in keys):
                dominated.add(y)
                for k in keys:
                    remaining[k] -= 1
        survivors = {p for p in programs if p not in dominated}
        pruned = {key: {p for p in front if p in survivors} for key, front in fronts.items()}
        assert all(pruned[k] for k, front in fronts.items() if front)
        return pruned

    def remove_dominated(self, scores: list[float] | dict[int, float] | None = None) -> set[int]:
        """Programs left on any front after GEPA dominance pruning (see pruned_fronts)."""
        out: set[int] = set()
        for front in self.pruned_fronts(scores).values():
            out |= front
        return out

    def select_candidate(self, all_scores: list[float]) -> int:
        """GEPA select_program_candidate_from_pareto_front: frequency-weighted sampling over
        the pruned fronts (list in first-seen order, each program repeated once per front)."""
        freq: dict[int, int] = {}
        for front in self.pruned_fronts(all_scores).values():
            for pid in front:
                freq[pid] = freq.get(pid, 0) + 1
        sampling_list = [pid for pid, n in freq.items() for _ in range(n)]
        if not sampling_list:
            return 0
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
