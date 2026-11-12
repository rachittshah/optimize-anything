from __future__ import annotations
import dataclasses
from typing import Any, Callable

from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType, resolve_run_dir
from optimize_anything.core.cost import CostTracker, TrackedLLM
from optimize_anything.core.engine import Engine, OptimizationResult
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator
from optimize_anything.llm import LLMClient, make_llm, resolve_backend, resolve_effort, resolve_model
from optimize_anything.proposer.reflective import ReflectiveProposer


class _FunctionEvaluator:
    """Wraps a plain Python function as an evaluator."""
    def __init__(self, fn: Callable):
        self.fn = fn
        # A function can opt in to dict candidates with `fn.accepts_dict = True`.
        self.accepts_dict = bool(getattr(fn, "accepts_dict", False))

    def cache_identity(self) -> str:
        return f"fn:{getattr(self.fn, '__module__', '')}.{getattr(self.fn, '__qualname__', '')}:{id(self.fn)}"

    def evaluate(self, candidate: str | dict[str, str], example: dict | None = None) -> tuple[float, dict]:
        result = self.fn(candidate, example)
        if isinstance(result, tuple):
            return result
        return float(result), {}


def _build_evaluator(evaluator, judge_llm: LLMClient, judge_model: str, judge_effort: str | None):
    if callable(evaluator):
        return _FunctionEvaluator(evaluator)
    if isinstance(evaluator, EvaluatorConfig):
        if evaluator.type == EvaluatorType.PYTHON:
            return PythonEvaluator(code=evaluator.code, timeout=evaluator.timeout)
        elif evaluator.type == EvaluatorType.SHELL:
            return ShellEvaluator(
                command=evaluator.command,
                score_pattern=evaluator.score_pattern or r"([\d.]+)",
                timeout=evaluator.timeout,
            )
        elif evaluator.type == EvaluatorType.LLM_JUDGE:
            return LLMJudgeEvaluator(
                criteria=evaluator.criteria,
                model=evaluator.judge_model or judge_model,
                llm=judge_llm,
                effort=judge_effort,
                rubric=evaluator.rubric,
            )
    if hasattr(evaluator, "evaluate"):
        return evaluator
    raise ValueError(f"Invalid evaluator: {type(evaluator)}")


def optimize_anything(
    seed_candidate: dict[str, str] | None = None,
    evaluator=None,
    dataset: list | None = None,
    valset: list | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: Config | None = None,
    on_event: Callable[[dict], None] | None = None,
    resume_from: str | None = None,
    testset: list | None = None,
    cost_tracker: CostTracker | None = None,
) -> OptimizationResult:
    """Optimize any text artifact through iterative LLM-powered search.

    testset: optional held-out examples. Never used for selection or acceptance; the seed and
    final best are scored on it at the end (result.seed_test / best_test / val_test_gap).
    Those evals are counted in result.test_evals, not total_evals.
    cost_tracker: share a tracker so evaluator-side LLM spend counts toward max_cost_usd."""
    config = config or Config()

    if seed_candidate is None and objective is None:
        raise ValueError("Provide either seed_candidate or objective")
    if evaluator is None:
        raise ValueError("evaluator is required")

    # Generate seed from objective if not provided
    if seed_candidate is None:
        seed_candidate = {"artifact": ""}

    # Final run directory: explicit run_dir is used as-is (no extra nesting).
    config = dataclasses.replace(config, run_dir=str(resolve_run_dir(config)),
                                 run_id=None, temperature=None)

    # LLM backend + models (clients are lazy; nothing is called until the first request)
    backend = resolve_backend(config.backend)
    llm = make_llm(backend, base_url=config.openai_base_url, api_key_env=config.openai_api_key_env)
    tracker = cost_tracker if cost_tracker is not None else CostTracker()
    proposer_model = resolve_model("proposer", backend, config.resolved_proposer_model(), config.preset)
    judge_model_explicit = (
        evaluator.judge_model if isinstance(evaluator, EvaluatorConfig) and evaluator.judge_model
        else config.judge_model
    )
    needs_judge = isinstance(evaluator, EvaluatorConfig) and evaluator.type == EvaluatorType.LLM_JUDGE
    # Only resolve (and validate) the judge model when an LLM judge is actually used.
    judge_model = (
        resolve_model("judge", backend, judge_model_explicit, config.preset) if needs_judge else ""
    )
    eval_instance = _build_evaluator(
        evaluator, TrackedLLM(llm, tracker, "judge"), judge_model,
        resolve_effort("judge", config.judge_effort, config.preset),
    )
    proposer = ReflectiveProposer(
        TrackedLLM(llm, tracker, "propose"),
        model=proposer_model,
        effort=resolve_effort("proposer", config.proposer_effort, config.preset),
        max_tokens=config.max_tokens,
    )

    engine = Engine(
        seed_candidate=seed_candidate,
        evaluator=eval_instance,
        proposer=proposer,
        objective=objective or "",
        background=background,
        dataset=dataset,
        valset=valset,
        config=config,
        on_event=on_event,
        cost_tracker=tracker,
        testset=testset,
    )

    return engine.run()
