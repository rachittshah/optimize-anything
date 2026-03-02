from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Callable

from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import Engine, OptimizationResult
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator
from optimize_anything.proposer.claude_proposer import ClaudeProposer


class _FunctionEvaluator:
    """Wraps a plain Python function as an evaluator."""
    def __init__(self, fn: Callable):
        self.fn = fn

    def evaluate(self, candidate: str, example: dict | None = None) -> tuple[float, dict]:
        result = self.fn(candidate, example)
        if isinstance(result, tuple):
            return result
        return float(result), {}


def _build_evaluator(evaluator):
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
                model=evaluator.judge_model,
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
) -> OptimizationResult:
    """Optimize any text artifact through iterative LLM-powered search."""
    config = config or Config()

    if seed_candidate is None and objective is None:
        raise ValueError("Provide either seed_candidate or objective")
    if evaluator is None:
        raise ValueError("evaluator is required")

    # Generate seed from objective if not provided
    if seed_candidate is None:
        seed_candidate = {"artifact": ""}

    # Setup run directory
    run_id = str(uuid.uuid4())[:8]
    run_dir = Path(config.run_dir).expanduser() / run_id
    config = Config(**{**config.__dict__, "run_dir": str(run_dir)})

    # Build components
    eval_instance = _build_evaluator(evaluator)
    proposer = ClaudeProposer(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
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
    )

    return engine.run()
