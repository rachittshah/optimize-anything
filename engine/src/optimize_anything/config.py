from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class EvaluatorType(str, Enum):
    PYTHON = "python"
    SHELL = "shell"
    LLM_JUDGE = "llm_judge"


class SelectionStrategy(str, Enum):
    PARETO = "pareto"
    BEST = "best"
    EPSILON_GREEDY = "epsilon_greedy"


class FrontierType(str, Enum):
    INSTANCE = "instance"
    OBJECTIVE = "objective"
    HYBRID = "hybrid"


class ComponentSelector(str, Enum):
    ROUND_ROBIN = "round_robin"
    ALL = "all"


@dataclass
class EvaluatorConfig:
    type: EvaluatorType
    # For PYTHON: the code string defining evaluate(candidate, example) -> (score, asi)
    code: str | None = None
    # For SHELL: command template with {{candidate}} placeholder
    command: str | None = None
    score_pattern: str | None = None  # regex to extract score from stdout
    # For LLM_JUDGE: natural language criteria
    criteria: str | None = None
    judge_model: str = "claude-sonnet-4-6"
    # Common
    timeout: int = 30


@dataclass
class Config:
    # Proposer
    model: str = "claude-opus-4-6"
    max_tokens: int = 8192
    temperature: float = 1.0

    # Engine
    max_iterations: int = 100
    max_metric_calls: int | None = None
    selection_strategy: SelectionStrategy = SelectionStrategy.PARETO
    frontier_type: FrontierType = FrontierType.INSTANCE
    component_selector: ComponentSelector = ComponentSelector.ROUND_ROBIN
    reflection_minibatch_size: int = 3
    skip_perfect_score: bool = True
    use_merge: bool = False
    epsilon: float = 0.1  # for epsilon-greedy

    # Checkpointing
    checkpoint_interval: int = 5
    run_dir: str = "~/.optimize-anything/runs"

    # Stopping
    timeout_seconds: int | None = None
    no_improvement_patience: int | None = None
