from __future__ import annotations
from typing import Any


def event(type: str, **kwargs) -> dict[str, Any]:
    return {"type": type, **kwargs}


def iteration_start(iteration: int, candidate_idx: int, cost_usd: float = 0.0) -> dict:
    return event("iteration_start", iteration=iteration, candidate_idx=candidate_idx,
                 cost_usd=cost_usd)


def evaluation_complete(candidate_idx: int, score: float, asi: dict) -> dict:
    return event("evaluation_complete", candidate_idx=candidate_idx, score=score, asi=asi)


def proposal_complete(iteration: int, component_name: str, cost_usd: float) -> dict:
    """cost_usd is cumulative run spend after the proposal call."""
    return event("proposal_complete", iteration=iteration, component_name=component_name,
                 cost_usd=cost_usd)


def new_best_found(candidate_idx: int, score: float, cost_usd: float = 0.0) -> dict:
    return event("new_best_found", candidate_idx=candidate_idx, score=score, cost_usd=cost_usd)


def frontier_updated(frontier_size: int) -> dict:
    return event("frontier_updated", frontier_size=frontier_size)


def optimization_complete(best_score: float, total_iterations: int, total_evals: int,
                          total_cost_usd: float = 0.0, seed_score: float | None = None,
                          stop_reason: str | None = None) -> dict:
    return event(
        "optimization_complete",
        best_score=best_score,
        total_iterations=total_iterations,
        total_evals=total_evals,
        total_cost_usd=total_cost_usd,
        seed_score=seed_score,
        stop_reason=stop_reason,
    )


def perfect_skip(iteration: int, candidate_idx: int, attempt: int) -> dict:
    """The selected parent scored perfect on its minibatch; a new parent/minibatch is drawn."""
    return event("perfect_skip", iteration=iteration, candidate_idx=candidate_idx, attempt=attempt)


def train_saturated(iteration: int, num_candidates: int) -> dict:
    """Every selectable candidate is perfect on every train example: the run stops.
    Grow (or make harder) the dataset to keep optimizing."""
    return event("train_saturated", iteration=iteration, num_candidates=num_candidates)


def candidate_rejected(iteration: int, reason: str, **extra: Any) -> dict:
    """A proposal was dropped. reason: minibatch | prescreen_duplicate |
    prescreen_near_duplicate | prescreen_signature | halving | val_failed | promotion |
    edit_size | budget (val sweep cut short by max_metric_calls with eval_workers > 1)."""
    return event("candidate_rejected", iteration=iteration, reason=reason, **extra)


def promotion_blocked(iteration: int, candidate_idx: int, incumbent_idx: int,
                      mean_gain: float, se: float) -> dict:
    """paired_se: a candidate with the top mean was not promoted to reported best."""
    return event("promotion_blocked", iteration=iteration, candidate_idx=candidate_idx,
                 incumbent_idx=incumbent_idx, mean_gain=mean_gain, se=se)
