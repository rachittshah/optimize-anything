---
name: llm-evals
description: Use when building LLM evaluations, testing prompts, comparing prompt versions, optimizing prompts, setting up CI gates for LLM outputs, or when the user mentions eval, benchmark, prompt testing, regression detection, or scoring LLM responses. Also use when someone says "test my prompt", "is this prompt better", "eval suite", or "prompt optimization".
---

# LLM Evals

## Overview

The `evals` framework at `/Users/rshah/evals` is a unified LLM eval + optimization engine. Use it to measure prompt quality with statistical rigor, detect regressions, auto-optimize prompts, and gate deployments in CI.

**Core principle:** Eval with N trials and statistical tests, not vibes.

## When to Use

- Setting up evaluation for any LLM prompt or pipeline
- Comparing two prompt versions to see which is better
- Optimizing a prompt automatically
- Adding CI gates that block bad prompt changes
- Scoring LLM outputs with custom or built-in metrics
- Testing RAG pipelines, agents, or any async LLM workflow

## Quick Reference

### Installation

```bash
cd /Users/rshah/evals
pip install -e .   # or: uv sync
```

### CLI Commands

| Command | Purpose |
|---------|---------|
| `evals run suite.yaml` | Run evaluation suite |
| `evals run suite.yaml --trials 5 --baseline main` | Run with baseline comparison |
| `evals run suite.yaml --ci junit --output report.xml` | Generate JUnit for CI |
| `evals compare ID_A ID_B` | Statistical comparison of two experiments |
| `evals optimize prompt.yaml --dataset data.yaml --metric exact_match --strategy opro` | Auto-optimize |
| `evals server start --port 3000` | Start REST API |

### Built-in Metrics

| Metric | What |
|--------|------|
| `exact_match` | output == expected |
| `contains` | expected in output |
| `json_schema` | Valid JSON matching schema |
| `regex_match` | re.fullmatch(expected, output) |
| `cosine_similarity` | Token-overlap cosine [0,1] |
| `levenshtein_similarity` | 1 - edit_distance/max_len |

### Python API — Fastest Path

```python
from evals.engine import Eval
from evals.models.core import Dataset, Prompt, TestCase
from evals.metrics.builtin import exact_match, contains

experiment = Eval(
    name="my_eval",
    prompt=Prompt(
        name="qa",
        template="Answer: {{ question }}",
        model="claude-sonnet-4-20250514",
    ),
    dataset=Dataset(name="test", cases=[
        TestCase(input={"question": "Capital of France?"}, expected="Paris"),
    ]),
    metrics=[exact_match, contains],
    trials=3,
).run()

for m in experiment.summary.metrics:
    print(f"{m.metric}: {m.mean:.4f} CI=[{m.ci.lower:.4f}, {m.ci.upper:.4f}]")
```

### Task-Based Eval (no prompt — for RAG, agents, pipelines)

```python
async def my_pipeline(input_dict: dict) -> str:
    # Your RAG/agent/pipeline logic here
    return answer

experiment = Eval(
    name="pipeline_eval",
    task=my_pipeline,  # replaces prompt=
    dataset=dataset,
    metrics=[exact_match, relevancy],
).run()
```

### Custom Metric

```python
from evals.models.core import Metric

@Metric.code("my_metric", threshold=0.8)
def my_metric(output: str, expected: str) -> float:
    return 1.0 if some_condition(output, expected) else 0.0
```

### Suite YAML Format

```yaml
name: suite_name
prompt: path/to/prompt.yaml  # or inline Prompt object
dataset:
  name: test_data
  cases:
    - input: { question: "..." }
      expected: "..."
metrics:
  - exact_match
  - contains
trials: 3
concurrency: 10
```

### Prompt YAML Format

```yaml
name: my_prompt
template: "Answer: {{ question }}"
system: "Be concise."
model: claude-sonnet-4-20250514
parameters:
  temperature: 0.3
  max_tokens: 256
```

### Statistical Comparison

```python
from evals.engine.statistical import compare_experiments

comparison = compare_experiments(baseline_exp, current_exp, "exact_match")
# Returns: delta_mean, p_value (Wilcoxon), significant, effect_size (Cohen's d)
# Plus per-case regressions and improvements
```

### Optimization Strategies

| Strategy | How |
|----------|-----|
| `opro` | Meta-prompt with history, LLM proposes candidates |
| `textgrad` | Per-failure critiques, aggregated into edits |
| `bootstrap` | Mines passing examples as few-shot demos |

### MCP Server

```bash
# Add to Claude Code:
claude mcp add --transport stdio evals-mcp -- uv run --directory /Users/rshah/evals evals-mcp

# Or run standalone:
uv run evals-mcp
```

Tools exposed: `run_eval`, `compare_experiments`, `list_experiments`, `get_experiment`, `list_metrics`, `score_output`, `create_test_cases`, `optimize_prompt`, `generate_junit`, `generate_pr_comment`.

## Key Architecture Decisions

- **Content-hash versioning**: Prompt.version = SHA-256(template+system+model+params). Changes when content changes.
- **N-trial statistical rigor**: Bootstrap CI (10K resamples), Wilcoxon signed-rank (non-parametric), Cohen's d effect size, pass@k.
- **Async-first**: All I/O is async (httpx, aiosqlite). EvalEngine uses asyncio.Semaphore for concurrency.
- **Provider auto-detect**: `claude-*` → Anthropic, `gpt-*` → OpenAI, `llama*` → Ollama.
- **Cache**: Content-addressed file cache. Only caches temperature=0 calls.

## Project Structure

```
/Users/rshah/evals/src/evals/
├── models/core.py      # 20+ Pydantic models
├── gateway/             # OpenAI, Anthropic, Ollama providers
├── metrics/             # Builtin + LLM judge + registry
├── engine/              # EvalEngine, statistical, cache
├── optimize/            # OPRO, TextGrad, BootstrapFewShot
├── storage/             # SQLite + Postgres backends
├── ci/                  # JUnit XML + GitHub PR comments
├── cli/                 # Click CLI (run, compare, optimize, server)
├── server/              # FastAPI REST API (15 endpoints)
└── mcp/                 # MCP server (10 tools)
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Running 1 trial and trusting the result | Use trials=3+ for statistical validity |
| Comparing prompts by eyeballing | Use `evals compare` with Wilcoxon test |
| Optimizing without a val split | The optimizer auto-splits 80/20 if no splits defined |
| Using temperature>0 and expecting cache hits | Cache only stores temperature=0 calls |
| Forgetting to set API key env vars | CLI auto-detects ANTHROPIC_API_KEY, OPENAI_API_KEY |


## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: evals` | Run `cd /Users/rshah/evals && pip install -e .` |
| API key not found | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` env var |
| Eval hangs | Check concurrency setting, reduce if rate-limited |
| Cache not working | Only `temperature=0` calls are cached |
| Low statistical power | Increase `trials` (minimum 5 for Wilcoxon) |

## Step-by-Step: Your First Eval

1. Install: `cd /Users/rshah/evals && pip install -e .`
2. Create a prompt YAML:
```yaml
name: my_prompt
template: "Answer: {{ question }}"
model: claude-sonnet-4-20250514
```
3. Create a dataset YAML:
```yaml
name: test_data
cases:
  - input: { question: "Capital of France?" }
    expected: "Paris"
```
4. Create a suite YAML:
```yaml
name: my_eval
prompt: prompt.yaml
dataset: dataset.yaml
metrics: [exact_match, contains]
trials: 5
```
5. Run: `evals run suite.yaml --trials 5`
6. Compare: `evals compare <baseline_id> <current_id>`

## When NOT to Use

- Simple string matching that doesn't need statistical rigor
- One-off manual prompt testing (just test in the Claude UI)
- Evaluating non-text outputs (images, audio)
- When you don't have test cases or expected outputs yet (write those first)
