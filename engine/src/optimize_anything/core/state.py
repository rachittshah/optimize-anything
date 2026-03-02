# engine/src/optimize_anything/core/state.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from optimize_anything.core.candidate import Candidate
from optimize_anything.core.pareto import ParetoFrontier


@dataclass
class State:
    component_names: list[str]
    candidates: list[Candidate] = field(default_factory=list)
    scores: list[dict[str, float]] = field(default_factory=list)  # per-example scores
    agg_scores: list[float] = field(default_factory=list)  # mean score per candidate
    frontier: ParetoFrontier = field(default_factory=ParetoFrontier)
    iteration: int = -1
    total_evals: int = 0
    component_cursor: dict[int, int] = field(default_factory=dict)  # per-candidate round-robin

    def add_candidate(
        self,
        candidate: Candidate,
        scores: dict[str, float],
        parent_ids: list[int],
    ) -> int:
        idx = len(self.candidates)
        candidate.parent_ids = parent_ids
        self.candidates.append(candidate)
        self.scores.append(scores)
        agg = sum(scores.values()) / len(scores) if scores else 0.0
        self.agg_scores.append(agg)
        self.component_cursor[idx] = 0

        for key, score in scores.items():
            self.frontier.update(key, score, idx)

        return idx

    def best_candidate(self) -> Candidate:
        if not self.candidates:
            raise ValueError("No candidates in state")
        best_idx = max(range(len(self.agg_scores)), key=lambda i: self.agg_scores[i])
        return self.candidates[best_idx]

    def best_score(self) -> float:
        if not self.agg_scores:
            return 0.0
        return max(self.agg_scores)

    def record_evals(self, count: int):
        self.total_evals += count

    def next_component(self, candidate_idx: int) -> str:
        cursor = self.component_cursor.get(candidate_idx, 0)
        name = self.component_names[cursor % len(self.component_names)]
        self.component_cursor[candidate_idx] = cursor + 1
        return name

    def save(self, path: Path):
        data = {
            "component_names": self.component_names,
            "candidates": [
                {"components": c.components, "parent_ids": c.parent_ids, "tag": c.tag}
                for c in self.candidates
            ],
            "scores": self.scores,
            "agg_scores": self.agg_scores,
            "frontier": self.frontier.to_dict(),
            "iteration": self.iteration,
            "total_evals": self.total_evals,
            "component_cursor": {str(k): v for k, v in self.component_cursor.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> State:
        data = json.loads(path.read_text())
        state = cls(component_names=data["component_names"])
        for c_data in data["candidates"]:
            state.candidates.append(
                Candidate(c_data["components"], c_data["parent_ids"], c_data.get("tag", ""))
            )
        state.scores = data["scores"]
        state.agg_scores = data["agg_scores"]
        state.frontier = ParetoFrontier.from_dict(data["frontier"])
        state.iteration = data["iteration"]
        state.total_evals = data["total_evals"]
        state.component_cursor = {int(k): v for k, v in data["component_cursor"].items()}
        return state
