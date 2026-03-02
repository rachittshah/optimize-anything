# optimize_anything Design Document

**Date:** 2026-03-02
**Author:** Claude Opus 4.6 + rshah
**Status:** Approved

## Overview

A standalone universal optimization system inspired by GEPA's `optimize_anything` API. Optimizes any text artifact (prompts, code, configs, agent architectures, SVGs) through iterative LLM-powered search with evaluation feedback. Exposes as a Python CLI/library, TypeScript MCP server, and Claude Code skill.

## Architecture

```
~/optimize-anything/
├── engine/                    # Python (uv) - optimization logic
│   ├── src/optimize_anything/
│   │   ├── __init__.py
│   │   ├── api.py             # Public API: optimize_anything()
│   │   ├── cli.py             # CLI entry point
│   │   ├── core/
│   │   │   ├── engine.py      # Main optimization loop
│   │   │   ├── state.py       # Candidate pool, score history, budget
│   │   │   ├── pareto.py      # Pareto frontier tracking
│   │   │   ├── candidate.py   # Candidate representation
│   │   │   └── checkpoint.py  # Serialization/resume
│   │   ├── proposer/
│   │   │   ├── claude.py      # Claude API proposer (reflection + proposal)
│   │   │   └── prompts.py     # Prompt templates for reflection
│   │   ├── evaluators/
│   │   │   ├── base.py        # Evaluator protocol
│   │   │   ├── python_eval.py # Python code string evaluator
│   │   │   ├── shell_eval.py  # Shell command evaluator
│   │   │   └── llm_judge.py   # LLM-as-judge evaluator
│   │   ├── strategies/
│   │   │   ├── selection.py   # Candidate selection (Pareto, best, epsilon-greedy)
│   │   │   ├── sampling.py    # Batch sampling from dataset
│   │   │   └── stopping.py    # Budget, time, no-improvement stoppers
│   │   └── config.py          # Configuration dataclasses
│   ├── tests/
│   └── pyproject.toml
├── mcp-server/                # TypeScript - thin MCP wrapper
│   ├── src/
│   │   ├── index.ts           # MCP server entry point
│   │   ├── tools.ts           # Tool definitions
│   │   └── process.ts         # Python subprocess management
│   ├── package.json
│   └── tsconfig.json
└── skill/                     # Claude Code skill
    └── SKILL.md
```

## Component 1: Python Optimization Engine

### Core Loop

```
1. Initialize: seed_candidate → candidate_pool[0]
2. Loop until budget exhausted or stopped:
   a. SELECT candidate from Pareto frontier (or best)
   b. SAMPLE minibatch from dataset (if provided)
   c. EVALUATE candidate → scores + ASI (diagnostics)
   d. REFLECT: build reflection context from ASI + scores
   e. PROPOSE: Claude generates improved candidate via reflection
   f. EVALUATE new candidate on same minibatch
   g. UPDATE Pareto frontier if new candidate is non-dominated
   h. Optional MERGE: combine two frontier candidates
   i. CHECK stopping conditions
   j. EMIT progress event + checkpoint if interval reached
3. Return best candidate + full history + Pareto frontier
```

### Candidate Representation

`dict[str, str]` — keys are component names, values are text:

```python
# Single component
{"system_prompt": "You are a helpful assistant..."}

# Multi-component
{"system_prompt": "...", "few_shot_examples": "...", "output_format": "..."}

# Code artifact
{"algorithm": "def solve(input):\n    ..."}
```

### ASI (Actionable Side Information)

Evaluators return `(score: float, asi: dict)`:

```python
score, {
    "error": "TypeError at line 42...",
    "stdout": "Test 3/5 passed",
    "runtime_ms": 142.5,
}
```

The proposer sees ASI during reflection — this is what makes it intelligent search, not blind evolution.

### Three Optimization Modes

1. **Single-task**: No dataset. `optimize_anything(seed, evaluator)`
2. **Multi-task**: Dataset of related problems. `optimize_anything(seed, evaluator, dataset=tasks)`
3. **Generalization**: Train + val. `optimize_anything(seed, evaluator, dataset=train, valset=val)`

### Streaming & Progress

Engine emits events via callback:
- `iteration_start(iteration, candidate_id)`
- `evaluation_complete(candidate_id, score, asi)`
- `new_best_found(candidate, score)`
- `frontier_updated(frontier_size)`
- `optimization_complete(result)`

### Checkpointing & Resume

State serialized to `{run_dir}/checkpoint.json` every N iterations:
- Full candidate pool with scores
- Pareto frontier
- Budget consumed
- Configuration

Resume: `optimize_anything(resume_from="path/to/run")`

### Public API

```python
from optimize_anything import optimize_anything, Config

result = optimize_anything(
    seed_candidate={"system_prompt": "..."},
    evaluator=evaluate_fn,          # or evaluator config dict
    dataset=None,                   # optional: list of examples
    valset=None,                    # optional: validation set
    objective="Optimize for...",    # natural language objective
    background="Domain context...", # domain knowledge
    config=Config(
        max_iterations=100,
        model="claude-opus-4-6",    # proposer model
        checkpoint_interval=5,
        run_dir="~/.optimize-anything/runs/",
    ),
    on_event=callback,              # optional: progress callback
)

print(result.best_candidate)
print(result.best_score)
print(result.pareto_frontier)
print(result.total_iterations)
```

## Component 2: Evaluator Types

### Python Evaluator

```python
evaluator_code = """
def evaluate(candidate: str, example: dict | None = None) -> tuple[float, dict]:
    import subprocess
    result = subprocess.run(["python", "-c", candidate], capture_output=True, timeout=10)
    passed = "OK" in result.stdout.decode()
    return (1.0 if passed else 0.0), {"stdout": result.stdout.decode(), "stderr": result.stderr.decode()}
"""
```

Executed in a subprocess with configurable timeout. Access to standard library + project venv.

### Shell Evaluator

```python
shell_config = {
    "type": "shell",
    "command": "echo '{{candidate}}' | python test_runner.py",
    "score_pattern": r"Score: ([\d.]+)",
    "timeout": 30
}
```

### LLM-as-Judge Evaluator

```python
judge_config = {
    "type": "llm_judge",
    "criteria": "Rate clarity, specificity, completeness. Score 0-10.",
    "model": "claude-sonnet-4-6",
}
```

Returns normalized score [0,1] + judge's reasoning as ASI.

## Component 3: TypeScript MCP Server

### Tools

| Tool | Params | Returns |
|------|--------|---------|
| `optimize_anything` | seed_candidate, evaluator, objective, dataset, valset, config | run_id |
| `check_optimization` | run_id | status, iteration, best_score, frontier_size |
| `get_best_candidate` | run_id | candidate text, score |
| `get_pareto_frontier` | run_id | list of (candidate, scores) |
| `stop_optimization` | run_id | confirmation |
| `resume_optimization` | run_id | new run_id |
| `list_runs` | (none) | list of (run_id, status, best_score) |

### Process Management

```
MCP Client → TS Server (stdio) → spawn `uv run optimize-anything run --config <json>` → Python Engine → Claude API
```

- Runs are async: `optimize_anything` returns immediately with run_id
- Progress via `check_optimization` polling
- State on disk at `~/.optimize-anything/runs/{run_id}/`

## Component 4: Claude Code Skill

### Purpose

Teaches Claude Code how to formulate optimization problems and invoke the MCP tools.

### Key Teachings

1. **Problem formulation**: Decompose task into (candidate, evaluator, objective)
2. **Evaluator selection**: Python for complex logic, shell for CLI tools, LLM-judge for zero-code
3. **Mode selection**: Single-task for one problem, multi-task for related batch, generalization for transfer
4. **ASI design**: Craft diagnostic feedback that helps the proposer improve

### Example Recipes

**Recipe 1: Optimize a system prompt**
```
Candidate: {"system_prompt": "<current prompt>"}
Evaluator: LLM-judge with criteria "Rate helpfulness, accuracy, conciseness"
Mode: Generalization (train examples + held-out val)
```

**Recipe 2: Optimize code for performance**
```
Candidate: {"algorithm": "<current code>"}
Evaluator: Shell — run benchmarks, extract timing
Mode: Single-task (one algorithm to optimize)
```

**Recipe 3: Discover agent architecture**
```
Candidate: {"agent_code": "<initial harness>"}
Evaluator: Python — run agent on task, score output
Mode: Generalization (train tasks + val tasks)
```

**Recipe 4: Optimize config/policy**
```
Candidate: {"config": "<yaml or json config>"}
Evaluator: Shell — deploy config, run simulation, extract metrics
Mode: Multi-task (multiple scenarios)
```

## Tech Stack

- **Python engine**: uv, anthropic SDK, pydantic for config
- **TypeScript MCP**: @modelcontextprotocol/sdk, child_process for subprocess
- **Claude models**: Opus 4.6 for proposer, Sonnet 4.6 for LLM-judge evaluator
- **Storage**: JSON files on disk for state/checkpoints

## Non-Goals (v1)

- No web UI or dashboard
- No multi-provider LLM support (Claude only)
- No distributed/concurrent optimization
- No image ASI support (text-only v1)
