from __future__ import annotations
import warnings
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
    # Optional rubric: list of criterion names or {"name", "description", "anchors": {"0","1","2"}}.
    # The judge returns per-criterion 0-2 scores as JSON; the code computes the total.
    rubric: list | None = None
    # None -> Config.judge_model -> OA_JUDGE_MODEL -> role default (llm/models.py)
    judge_model: str | None = None
    # Common
    timeout: int = 30


DEFAULT_RUN_DIR = "~/.optimize-anything/runs"


@dataclass
class Config:
    # Model preset: default | cheap | quality (llm/models.py PRESETS). Explicit fields override.
    preset: str = "default"
    # LLM backend: auto | anthropic | openai | claude-cli | codex-cli | mock (env: OA_BACKEND)
    backend: str = "auto"
    # Proposer model. None -> OA_PROPOSER_MODEL -> preset.
    proposer_model: str | None = None
    # Backward-compatible alias for proposer_model (proposer_model wins if both are set).
    model: str | None = None
    # LLM-judge model (same model for minibatch gate and full eval). None -> OA_JUDGE_MODEL -> preset.
    judge_model: str | None = None
    proposer_effort: str | None = None  # low|medium|high|xhigh|max; None -> preset
    judge_effort: str | None = None  # None -> preset
    max_tokens: int = 8192
    # OpenAI-compatible endpoint (e.g. OpenRouter); env: OA_OPENAI_BASE_URL
    openai_base_url: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    # DEPRECATED: accepted for backward compatibility but ignored. Current Claude models
    # reject temperature; use proposer_effort / judge_effort instead.
    temperature: float | None = None

    # Engine
    max_iterations: int = 100
    max_metric_calls: int | None = None
    selection_strategy: SelectionStrategy = SelectionStrategy.PARETO
    frontier_type: FrontierType = FrontierType.INSTANCE
    component_selector: ComponentSelector = ComponentSelector.ROUND_ROBIN
    reflection_minibatch_size: int = 3
    skip_perfect_score: bool = True
    # Perfect parent minibatch -> re-select parent + minibatch up to this many times within one
    # iteration (re-selections don't count toward max_iterations; counted in skipped_perfect).
    max_perfect_resamples: int = 5
    use_merge: bool = False
    epsilon: float = 0.1  # for epsilon-greedy
    # Cache (evaluator, candidate, example) -> (score, asi). Disable for stochastic evaluators.
    cache_evals: bool = True
    # RNG seed for sampler / selector / frontier. None -> 42 (original behaviour).
    seed: int | None = None

    # ---- Flag-gated upgrades (defaults preserve original behaviour) ----
    # Which candidate is returned as best: "max" (argmax mean val) or "paired_se"
    # (challenger must beat incumbent by > promotion_z * SE of paired per-example diffs).
    # Default paired_se: +0.015 held-out test vs the original engine at 0.73x cost on the bench (RESULTS.md).
    promotion: str = "paired_se"
    promotion_z: float = 1.0
    promotion_max_drop: float | None = 0.25  # paired_se: reject if any example drops more; None off
    # Full val evaluation of accepted children: "full" or "halving" (paired successive halving).
    val_eval: str = "full"
    halving_margin: float = 0.0
    # Skip duplicate / near-duplicate / no-change proposals before spending evals.
    prescreen: bool = True  # exact duplicates only by default: free, can't change acceptance
    # Near-duplicate filter (difflib ratio vs parent >= threshold). None = exact/whitespace-
    # normalized duplicates only (a ratio filter also blocks 1-2 char fixes on long artifacts).
    dedupe_threshold: float | None = None
    # Draw half of each train minibatch from examples the current best fails on.
    counterexample_mining: bool = False
    # Mutation operator: fixed ("rewrite" | "section_edit"), or picked by a sliding-window UCB.
    mutation_operator: str = "rewrite"
    operator_bandit: bool = False
    # Show the proposer the last N rejected edits for this parent+component ("don't repeat").
    proposal_memory: bool = False
    proposal_memory_size: int = 5
    # Evaluate a batch's examples concurrently. >1 truncates batches to the remaining
    # max_metric_calls budget (no overshoot); 1 = original sequential behaviour.
    eval_workers: int = 1
    # Reject (before any eval) a child whose difflib change ratio vs the parent exceeds this.
    max_edit_ratio: float | None = None

    # Checkpointing. An explicit run_dir (anything but the default) is used as-is;
    # otherwise the run lands in <default or OA_RUNS_DIR>/<run_id or random id>.
    checkpoint_interval: int = 5
    run_dir: str = DEFAULT_RUN_DIR
    run_id: str | None = None

    # Stopping
    timeout_seconds: int | None = None
    no_improvement_patience: int | None = None
    # An iteration counts as an improvement only if best gains more than this (patience stopper).
    min_improvement: float = 0.0
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            warnings.warn(
                "Config.temperature is deprecated and ignored (current Claude models reject it); "
                "use proposer_effort / judge_effort.",
                DeprecationWarning,
                stacklevel=3,
            )
        # String flags come straight from JSON (bench/CLI): fail loudly on typos.
        for name, allowed in (("promotion", ("max", "paired_se")),
                              ("val_eval", ("full", "halving")),
                              ("mutation_operator", ("rewrite", "section_edit", "compress"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"Config.{name}={getattr(self, name)!r}; expected one of {allowed}")

    def resolved_seed(self) -> int:
        return 42 if self.seed is None else self.seed

    def resolved_proposer_model(self) -> str | None:
        return self.proposer_model or self.model


def resolve_run_dir(config: Config) -> Path:
    """Final directory for a run's checkpoint.json / result.json.

    - run_id set: <base>/<run_id>
    - explicit run_dir (not the default): used as-is (the MCP server passes its own dir)
    - otherwise: <base>/<random 8-char id>
    where <base> is run_dir, or OA_RUNS_DIR when run_dir is the default.
    """
    import os
    import uuid

    is_default = config.run_dir == DEFAULT_RUN_DIR
    base = Path(os.environ.get("OA_RUNS_DIR") or config.run_dir) if is_default else Path(config.run_dir)
    base = base.expanduser()
    if config.run_id:
        return base / config.run_id
    if not is_default:
        return base
    return base / uuid.uuid4().hex[:8]
