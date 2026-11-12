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
    failed_evals: int = 0
    cache_hits: int = 0
    # Flag-gated upgrade counters (stay 0 / empty under defaults)
    prescreen_skips: int = 0
    skipped_perfect: int = 0  # parent minibatches skipped because every score was perfect
    halving_rejects: int = 0
    halving_evals_saved: int = 0
    promotions_blocked: int = 0
    operator_counts: dict[str, int] = field(default_factory=dict)
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

    def save(self, path: Path, extra: dict | None = None):
        """Atomically write state; `extra` keys (e.g. cost) are added alongside."""
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
            "failed_evals": self.failed_evals,
            "cache_hits": self.cache_hits,
            "prescreen_skips": self.prescreen_skips,
            "skipped_perfect": self.skipped_perfect,
            "halving_rejects": self.halving_rejects,
            "halving_evals_saved": self.halving_evals_saved,
            "promotions_blocked": self.promotions_blocked,
            "operator_counts": dict(self.operator_counts),
            "component_cursor": {str(k): v for k, v in self.component_cursor.items()},
        }
        if extra:
            data.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

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
        state.failed_evals = data.get("failed_evals", 0)
        state.cache_hits = data.get("cache_hits", 0)
        for name in ("prescreen_skips", "skipped_perfect", "halving_rejects", "halving_evals_saved",
                     "promotions_blocked"):
            setattr(state, name, data.get(name, 0))
        state.operator_counts = dict(data.get("operator_counts", {}))
        state.component_cursor = {int(k): v for k, v in data["component_cursor"].items()}
        return state
