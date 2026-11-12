# optimize-anything

Universal text artifact optimizer using LLM-powered iterative search with diagnostic feedback (ASI).

## Project Structure

- `engine/` — Python optimization engine (uv, Python 3.12+)
  - `src/optimize_anything/config.py` — Config dataclasses, enums (EvaluatorType, SelectionStrategy, FrontierType)
  - `src/optimize_anything/core/candidate.py` — Candidate dataclass (components dict, parent_ids, tag)
  - `src/optimize_anything/core/state.py` — Run state: candidate pool, scores, Pareto frontier, budget tracking
  - `src/optimize_anything/core/pareto.py` — Pareto frontier: per-key tracking, frequency-weighted selection
  - `src/optimize_anything/evaluators/` — Python, shell, LLM-judge evaluators (all return `(float, dict)`)
  - `src/optimize_anything/llm/` — LLM backend layer: `LLMClient` protocol, `LLMResponse`, backends (anthropic, openai, claude-cli, codex-cli, mock), `models.py` (defaults + pricing; the only place model IDs live)
  - `src/optimize_anything/core/cost.py` — `CostTracker` / `TrackedLLM`: per-role ($, tokens) accounting
  - `src/optimize_anything/proposer/` — `Proposer` protocol, `ReflectiveProposer` (`ClaudeProposer` = compat alias) + prompt templates
  - `src/optimize_anything/strategies/` — Selection (Pareto/Best/EpsilonGreedy), sampling (epoch-based), stopping
  - `tests/` — pytest unit tests with mocking for LLM calls
- `mcp-server/` — TypeScript MCP server (5 tools: optimize_anything, check_optimization, get_best_candidate, stop_optimization, list_optimization_runs)
- `skill/` — agent-neutral skill for problem formulation (hill-climbed with this engine; see RESULTS.md)
- `scripts/bench/` — engine benchmark (3 deterministic tasks, A/B rounds); `scripts/optimize_skill.py` + `scripts/skill_eval/` — skill benchmark + hill-climb

## Commands

- `cd engine && uv sync --extra dev` — install Python deps (incl. pytest)
- `cd engine && uv run pytest` — run all tests
- `cd engine && uv run pytest tests/test_pareto.py -v` — run specific test file
- `cd engine && uv run optimize-anything --help` — CLI (run, status subcommands)
- `cd mcp-server && npm install && npm run build` — build MCP server; `npm test` — node:test (line splitter, status derivation, guardrails)
- `scripts/install.sh [--dry-run|--uninstall|--doctor [--live <agents>]] [--agents ...]` — register MCP server + skill for Claude/Codex/OpenCode/Gemini (`scripts/oa_install.py` does the config edits, `scripts/doctor.py` the checks)

## Architecture

Python engine does all optimization logic. TS MCP server spawns Python subprocess.

**Core loop**: SELECT candidate → SAMPLE minibatch → EVALUATE → REFLECT on ASI → PROPOSE improvement → EVALUATE new → UPDATE Pareto frontier → CHECKPOINT

**Key concepts**:
- **ASI (Actionable Side Information)**: Evaluators return `(score: float, asi: dict)` — diagnostic feedback (errors, stdout, timing) that the proposer reads to make targeted improvements
- **Pareto frontier**: Per-example tracking of non-dominated candidates with frequency-weighted selection
- **Reflective mutation**: Claude reads ASI, diagnoses failures, proposes fixes (not blind evolution)
- **Candidates**: `dict[str, str]` — component names to text. Supports single or multi-component optimization

**Three evaluator types**: Python code, shell command, LLM-as-judge.
**Evaluator input**: single-component → the text; multi-component → texts joined with "\n", or the `dict` if the evaluator sets `accepts_dict = True` (Python and LLM-judge evaluators do).
**Run dir**: explicit `run_dir` is final; default → `~/.optimize-anything/runs/<id>` (`OA_RUNS_DIR` overrides base). `checkpoint.json` after seed, each accepted candidate, and at end; `result.json` at end.
**Three modes**: single-task, multi-task (dataset), generalization (dataset + valset).

## Conventions

- Python 3.12+, type hints everywhere, `from __future__ import annotations`
- Pydantic for config validation, dataclasses for internal state
- Evaluators follow `Evaluator` protocol: `evaluate(candidate, example) -> (float, dict)`
- Tests never call real LLMs: `tests/conftest.py` forces `OA_BACKEND=mock` and a temp `OA_RUNS_DIR`
- All scores normalized to [0, 1] range
- Model presets (`Config.preset`, `llm/models.py`): `default` = proposer `claude-opus-5-5` (medium) + judge `claude-sonnet-5-5` (low); `cheap` = sonnet-5-5 + haiku-5-5; `quality` = proposer `claude-fable-5-2` (high). Explicit model/effort fields and env override. Never send `temperature` to Claude; use effort
- Evaluators raise `EvaluationError` for infrastructure failures (judge 429/5xx/refusal/unparseable); the engine excludes those evals from acceptance + frontier and counts `failed_evals`. Never score a broken judge as 0
- Eval cache (`cache_evals=True`): key = sha256(evaluator identity, candidate, example); hits don't count toward `total_evals`, reported as `cache_hits`
- Upgrade flags (Config). Defaults set from the bench A/B in RESULTS.md: `promotion="paired_se"` and `prescreen=True` (exact duplicates) are ON; everything else is OFF because it didn't win. Flags: `seed` (None→42), `promotion` max|paired_se (+`promotion_z`, `promotion_max_drop`), `val_eval` full|halving (+`halving_margin`), `prescreen` (+`dedupe_threshold`), `counterexample_mining`, `mutation_operator` rewrite|section_edit|compress, `operator_bandit`, `proposal_memory` (+`proposal_memory_size`; rejected diffs shown to the proposer), `eval_workers` (>1 = threaded evals, batches truncated to the remaining budget), `max_edit_ratio` (difflib change cap, `edit_size_rejects`). `Engine(testset=...)` / `optimize_anything(testset=...)` reports `seed_test`/`best_test`/`val_test_gap`; test evals go in `test_evals`, not `total_evals`. Shell evaluator placeholders: `{{candidate}}`, `{{candidate_file}}`, `{{example_file}}`, `{{example_json}}` (shell-quoted)
- Dominance pruning = exact GEPA rule (`ParetoFrontier.pruned_fronts`, gepa-ai/gepa@ce5b0d6 `remove_dominated_programs`): ascending aggregate score, a program is dropped iff every front it's on keeps another survivor. Replaced the old strict-subset rule (identical key sets now drop the lower-scored program)
- Perfect parent minibatch → re-draw parent + minibatch (≤ `max_perfect_resamples`, not counted as iterations; `skipped_perfect`). If every selectable candidate is perfect on all train examples the run stops with `stop_reason="train_saturated"`: grow or harden the dataset. `stop_reason` (budget | iterations | cost | timeout | no_improvement | train_saturated) is in the result and result.json
- Env: `OA_BACKEND`, `OA_PROPOSER_MODEL`, `OA_JUDGE_MODEL`, `OA_OPENAI_BASE_URL`, `OA_RUNS_DIR`

## Implementation Status

All components implemented. Run `uv run pytest -q` for the current count.

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
| LLM backend layer + cost tracking | Done |
| Claude Code skill | Done |
