from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from optimize_anything.config import Config, SelectionStrategy
from optimize_anything.core.candidate import Candidate
from optimize_anything.core.state import State
from optimize_anything.core import events
from optimize_anything.strategies.selection import ParetoSelector, BestSelector, EpsilonGreedySelector
from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import (
    BudgetStopper, IterationStopper, CompositeStopper, TimeoutStopper, NoImprovementStopper,
)


@dataclass
class OptimizationResult:
    best_candidate: dict[str, str]
    best_score: float
    all_candidates: list[dict[str, str]]
    all_scores: list[float]
    pareto_frontier: dict
    total_iterations: int
    total_evals: int


class Engine:
    def __init__(
        self,
        seed_candidate: dict[str, str],
        evaluator,
        proposer,
        objective: str = "",
        background: str | None = None,
        dataset: list | None = None,
        valset: list | None = None,
        config: Config | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.config = config or Config()
        self.evaluator = evaluator
        self.proposer = proposer
        self.objective = objective
        self.background = background
        self.dataset = dataset
        self.valset = valset
        self.on_event = on_event or (lambda e: None)

        self.state = State(component_names=list(seed_candidate.keys()))

        # Build selector
        if self.config.selection_strategy == SelectionStrategy.PARETO:
            self.selector = ParetoSelector()
        elif self.config.selection_strategy == SelectionStrategy.BEST:
            self.selector = BestSelector()
        else:
            self.selector = EpsilonGreedySelector(self.config.epsilon)

        self.sampler = EpochBatchSampler(self.config.reflection_minibatch_size)

        # Build stopper
        stoppers = [IterationStopper(self.config.max_iterations)]
        if self.config.max_metric_calls:
            stoppers.append(BudgetStopper(self.config.max_metric_calls))
        if self.config.timeout_seconds:
            stoppers.append(TimeoutStopper(self.config.timeout_seconds))
        if self.config.no_improvement_patience:
            stoppers.append(NoImprovementStopper(self.config.no_improvement_patience))
        self.stopper = CompositeStopper(stoppers)

        # Initial evaluation of seed
        self._seed_candidate = seed_candidate

    def run(self) -> OptimizationResult:
        # Evaluate seed candidate
        seed_scores = self._evaluate_candidate(self._seed_candidate)
        self.state.add_candidate(
            Candidate(components=dict(self._seed_candidate)),
            scores=seed_scores,
            parent_ids=[],
        )

        while not self._should_stop():
            self.state.iteration += 1
            iteration = self.state.iteration

            # Select parent candidate
            if len(self.state.candidates) == 1:
                parent_idx = 0
            else:
                parent_idx = self.selector.select(self.state)

            parent = self.state.candidates[parent_idx]
            self.on_event(events.iteration_start(iteration, parent_idx))

            # Select component to update (round-robin)
            component_name = self.state.next_component(parent_idx)

            # Get minibatch for evaluation
            if self.dataset:
                batch_ids = self.sampler.next_batch(
                    list(range(len(self.dataset))), iteration
                )
                batch = [self.dataset[i] for i in batch_ids]
            else:
                batch = [None]

            # Evaluate parent on minibatch
            parent_asi = []
            parent_scores = []
            for example in batch:
                score, asi = self.evaluator.evaluate(
                    parent.components.get(component_name, ""), example
                )
                parent_scores.append(score)
                parent_asi.append({"score": score, "feedback": asi})
                self.state.record_evals(1)

            self.on_event(events.evaluation_complete(
                parent_idx, sum(parent_scores) / len(parent_scores), {}
            ))

            # Skip if perfect
            if self.config.skip_perfect_score and all(s >= 1.0 for s in parent_scores):
                continue

            # Propose improvement
            new_text = self.proposer.propose(
                current_text=parent.components[component_name],
                component_name=component_name,
                objective=self.objective,
                asi_entries=parent_asi,
                background=self.background,
            )

            # Build new candidate
            new_components = dict(parent.components)
            new_components[component_name] = new_text
            new_candidate = Candidate(components=new_components, tag="reflective_mutation")

            # Evaluate new candidate on same minibatch
            new_scores = []
            for example in batch:
                score, asi = self.evaluator.evaluate(new_text, example)
                new_scores.append(score)
                self.state.record_evals(1)

            # Accept only if strictly improved on minibatch
            if sum(new_scores) > sum(parent_scores):
                # Full evaluation on valset (or dataset) for Pareto tracking
                full_scores = self._evaluate_candidate(new_components)
                new_idx = self.state.add_candidate(new_candidate, full_scores, [parent_idx])

                new_agg = self.state.agg_scores[new_idx]
                if new_agg >= self.state.best_score():
                    self.on_event(events.new_best_found(new_idx, new_agg))

                self.on_event(events.frontier_updated(len(self.state.frontier.all_program_ids())))

            # Checkpoint
            if (
                self.config.checkpoint_interval
                and iteration % self.config.checkpoint_interval == 0
            ):
                run_dir = Path(self.config.run_dir).expanduser()
                self.state.save(run_dir / "checkpoint.json")

        # Final result
        best = self.state.best_candidate()
        result = OptimizationResult(
            best_candidate=best.components,
            best_score=self.state.best_score(),
            all_candidates=[c.components for c in self.state.candidates],
            all_scores=self.state.agg_scores,
            pareto_frontier=self.state.frontier.to_dict(),
            total_iterations=self.state.iteration + 1,
            total_evals=self.state.total_evals,
        )

        self.on_event(events.optimization_complete(
            result.best_score, result.total_iterations, result.total_evals
        ))

        return result

    def _evaluate_candidate(self, components: dict[str, str]) -> dict[str, float]:
        examples = self.valset or self.dataset or [None]
        scores = {}
        for i, example in enumerate(examples):
            candidate_text = "\n".join(components.values())
            score, _ = self.evaluator.evaluate(candidate_text, example)
            scores[f"ex_{i}"] = score
            self.state.record_evals(1)
        return scores

    def _should_stop(self) -> bool:
        return self.stopper.should_stop(
            total_evals=self.state.total_evals,
            iteration=self.state.iteration + 1,
            best_score=self.state.best_score(),
        )
