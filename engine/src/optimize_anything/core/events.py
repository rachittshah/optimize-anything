from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


def event(type: str, **kwargs) -> dict[str, Any]:
    return {"type": type, **kwargs}


def iteration_start(iteration: int, candidate_idx: int) -> dict:
    return event("iteration_start", iteration=iteration, candidate_idx=candidate_idx)


def evaluation_complete(candidate_idx: int, score: float, asi: dict) -> dict:
    return event("evaluation_complete", candidate_idx=candidate_idx, score=score, asi=asi)


def new_best_found(candidate_idx: int, score: float) -> dict:
    return event("new_best_found", candidate_idx=candidate_idx, score=score)


def frontier_updated(frontier_size: int) -> dict:
    return event("frontier_updated", frontier_size=frontier_size)


def optimization_complete(best_score: float, total_iterations: int, total_evals: int) -> dict:
    return event(
        "optimization_complete",
        best_score=best_score,
        total_iterations=total_iterations,
        total_evals=total_evals,
    )
