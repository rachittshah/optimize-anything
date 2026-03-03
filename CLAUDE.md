# optimize-anything

Universal text artifact optimizer using LLM-powered iterative search with diagnostic feedback (ASI).

## Project Structure

- `engine/` — Python optimization engine (uv, Python 3.12+)
  - `src/optimize_anything/config.py` — Config dataclasses, enums (EvaluatorType, SelectionStrategy, FrontierType)
  - `src/optimize_anything/core/candidate.py` — Candidate dataclass (components dict, parent_ids, tag)
  - `src/optimize_anything/core/state.py` — Run state: candidate pool, scores, Pareto frontier, budget tracking
  - `src/optimize_anything/core/pareto.py` — Pareto frontier: per-key tracking, frequency-weighted selection
  - `src/optimize_anything/evaluators/` — Python, shell, LLM-judge evaluators (all return `(float, dict)`)
  - `src/optimize_anything/proposer/` — Claude reflection proposer + prompt templates
  - `src/optimize_anything/strategies/` — Selection (Pareto/Best/EpsilonGreedy), sampling (epoch-based), stopping
  - `tests/` — pytest unit tests with mocking for LLM calls
- `mcp-server/` — TypeScript MCP server (5 tools: optimize_anything, check_optimization, get_best_candidate, stop_optimization, list_optimization_runs)
- `skill/` — Claude Code skill for problem formulation

## Commands

- `cd engine && uv sync` — install Python deps
- `cd engine && uv run pytest` — run all tests
- `cd engine && uv run pytest tests/test_pareto.py -v` — run specific test file
- `cd engine && uv run optimize-anything --help` — CLI (run, status subcommands)
- `cd mcp-server && npm install && npm run build` — build MCP server

## Architecture

Python engine does all optimization logic. TS MCP server spawns Python subprocess.

**Core loop**: SELECT candidate → SAMPLE minibatch → EVALUATE → REFLECT on ASI → PROPOSE improvement → EVALUATE new → UPDATE Pareto frontier → CHECKPOINT

**Key concepts**:
- **ASI (Actionable Side Information)**: Evaluators return `(score: float, asi: dict)` — diagnostic feedback (errors, stdout, timing) that the proposer reads to make targeted improvements
- **Pareto frontier**: Per-example tracking of non-dominated candidates with frequency-weighted selection
- **Reflective mutation**: Claude reads ASI, diagnoses failures, proposes fixes (not blind evolution)
- **Candidates**: `dict[str, str]` — component names to text. Supports single or multi-component optimization

**Three evaluator types**: Python code, shell command, LLM-as-judge (Claude Sonnet).
**Three modes**: single-task, multi-task (dataset), generalization (dataset + valset).

## Conventions

- Python 3.12+, type hints everywhere, `from __future__ import annotations`
- Pydantic for config validation, dataclasses for internal state
- Evaluators follow `Evaluator` protocol: `evaluate(candidate, example) -> (float, dict)`
- Tests use pytest with `unittest.mock` for LLM API calls
- All scores normalized to [0, 1] range
- Proposer model: claude-opus-4-6, Judge model: claude-sonnet-4-6

## Implementation Status

All components implemented. 28 tests passing.

| Component | Status |
|-----------|--------|
| Config, Candidate, State, Pareto | Done |
| Evaluators (Python, Shell, LLM-Judge) | Done |
| Proposer + prompt templates | Done |
| Selection, sampling, stopping strategies | Done |
| Checkpoint save/load | Done |
| Main engine loop (`core/engine.py`) | Done |
| CLI entry point (`cli.py`) | Done |
| Public API (`api.py`, `__init__.py`) | Done |
| MCP server (process-manager, tools, index) | Done |
| Integration tests | Done |
| Claude Code skill | Done |
