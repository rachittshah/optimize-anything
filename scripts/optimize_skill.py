#!/usr/bin/env python3
"""Optimize a Claude Code skill using optimize-anything with a mock proposer.

This script demonstrates the optimization flow. In production, the proposer
would be backed by Claude API calls. Here we use pre-generated improvements
to show the pipeline end-to-end.
"""
import sys
sys.path.insert(0, "/Users/rshah/optimize-anything/engine/src")

from optimize_anything import optimize_anything, Config
from pathlib import Path


def skill_evaluator(candidate: str, example: dict | None = None) -> tuple[float, dict]:
    """Evaluate a Claude Code skill on multiple quality dimensions.

    Scores 0-1 based on presence of key skill components.
    """
    criteria = example or {}
    dimension = criteria.get("dimension", "completeness")

    checks = {
        "completeness": [
            ("frontmatter", "---\nname:" in candidate and "description:" in candidate),
            ("overview", "## overview" in candidate.lower() or "## what" in candidate.lower()),
            ("when_to_use", "when to use" in candidate.lower() or "## when" in candidate.lower()),
            ("quick_reference", "quick" in candidate.lower() or "## reference" in candidate.lower()),
            ("examples", "```" in candidate),
            ("common_mistakes", "mistake" in candidate.lower() or "pitfall" in candidate.lower() or "avoid" in candidate.lower()),
        ],
        "actionability": [
            ("has_code_examples", candidate.count("```") >= 4),
            ("has_cli_commands", "```bash" in candidate or "```sh" in candidate),
            ("has_copy_paste_ready", "```python" in candidate or "```yaml" in candidate),
            ("has_concrete_values", any(x in candidate for x in ["=", ":", "import"])),
            ("step_by_step", any(x in candidate for x in ["step 1", "1.", "first,", "## step"])),
        ],
        "clarity": [
            ("short_sentences", len(candidate.split("\n")) > 20),
            ("uses_tables", "|" in candidate and "---" in candidate),
            ("uses_headers", candidate.count("##") >= 3),
            ("not_too_long", len(candidate) < 15000),
            ("has_structure", candidate.count("##") >= 5),
        ],
        "trigger_coverage": [
            ("has_trigger_words", "description:" in candidate and len(candidate.split("description:")[1].split("\n")[0]) > 50),
            ("covers_synonyms", sum(1 for w in ["eval", "test", "benchmark", "score", "measure", "assess"] if w in candidate.lower()) >= 3),
            ("covers_use_cases", sum(1 for w in ["prompt", "pipeline", "agent", "rag", "ci", "regression"] if w in candidate.lower()) >= 4),
            ("negative_triggers", "when not" in candidate.lower() or "don't use" in candidate.lower()),
        ],
        "developer_experience": [
            ("install_instructions", "install" in candidate.lower() or "pip" in candidate.lower() or "uv" in candidate.lower()),
            ("import_examples", "import" in candidate and "from" in candidate),
            ("error_handling", "error" in candidate.lower() or "troubleshoot" in candidate.lower()),
            ("architecture_overview", "architecture" in candidate.lower() or "structure" in candidate.lower()),
            ("api_reference", "api" in candidate.lower() and ("parameter" in candidate.lower() or "argument" in candidate.lower() or "returns" in candidate.lower())),
        ],
    }

    dim_checks = checks.get(dimension, checks["completeness"])
    passed = sum(1 for _, check in dim_checks if check)
    total = len(dim_checks)
    score = passed / total

    details = {name: "PASS" if check else "FAIL" for name, check in dim_checks}
    details["dimension"] = dimension
    details["passed"] = passed
    details["total"] = total

    return score, details


def main():
    # Read current skill
    skill_path = Path("/Users/rshah/.claude/skills/llm-evals/SKILL.md")
    current_skill = skill_path.read_text()

    # Evaluation dataset — test across all quality dimensions
    dataset = [
        {"dimension": "completeness"},
        {"dimension": "actionability"},
        {"dimension": "clarity"},
        {"dimension": "trigger_coverage"},
        {"dimension": "developer_experience"},
    ]

    # Score current skill
    print("=" * 60)
    print("EVALUATING CURRENT llm-evals SKILL")
    print("=" * 60)

    total_score = 0
    for example in dataset:
        score, details = skill_evaluator(current_skill, example)
        total_score += score
        dim = details["dimension"]
        print(f"\n  {dim}: {score:.0%} ({details['passed']}/{details['total']})")
        for k, v in details.items():
            if k not in ("dimension", "passed", "total") and v == "FAIL":
                print(f"    - {k}: FAIL")

    avg_score = total_score / len(dataset)
    print(f"\n  OVERALL: {avg_score:.0%}")
    print()

    # Now run optimize-anything with a mock proposer that returns
    # pre-generated improvements
    from unittest.mock import patch, MagicMock

    # Generate improved versions targeting the failures
    improvements = []

    # V1: Add troubleshooting section + step-by-step
    v1 = current_skill + """

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
"""
    improvements.append(v1)

    # V2: Expand trigger description + add more synonyms coverage
    v2_desc = "description: Use when building LLM evaluations, testing prompts, comparing prompt versions, optimizing prompts, setting up CI gates for LLM outputs, benchmarking model performance, measuring prompt quality, detecting regressions in LLM behavior, scoring LLM responses, assessing output quality, or running A/B tests on prompts. Also use when someone says 'test my prompt', 'is this prompt better', 'eval suite', 'prompt optimization', 'measure accuracy', 'benchmark this', 'regression test', or 'score the output'."
    v2 = current_skill.replace(
        "description: Use when building LLM evaluations, testing prompts, comparing prompt versions, optimizing prompts, setting up CI gates for LLM outputs, or when the user mentions eval, benchmark, prompt testing, regression detection, or scoring LLM responses. Also use when someone says \"test my prompt\", \"is this prompt better\", \"eval suite\", or \"prompt optimization\".",
        v2_desc
    )
    # Also add the troubleshooting and step-by-step
    v2 = v2 + """

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: evals` | Run `cd /Users/rshah/evals && pip install -e .` |
| API key not found | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` env var |
| Eval hangs | Check concurrency setting, reduce if rate-limited |
| Cache not working | Only `temperature=0` calls are cached |
| Low statistical power | Increase `trials` (minimum 5 for Wilcoxon) |
| Metric returns NaN | Check that expected values are set in test cases |

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

## API Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Experiment name |
| `prompt` | Prompt | None | Prompt template to evaluate |
| `task` | Callable | None | Async function for pipeline eval |
| `dataset` | Dataset | required | Test cases with input/expected |
| `metrics` | list[Metric] | required | Scoring functions |
| `trials` | int | 1 | Number of repeated evaluations |
| `concurrency` | int | 10 | Max parallel API calls |

### Metric Protocol

```python
def my_metric(output: str, expected: str) -> float:
    \"\"\"Returns score in [0, 1]. Higher is better.\"\"\"
    ...
```

### Returns: Experiment

```python
experiment.summary.metrics  # list of MetricSummary
experiment.summary.metrics[0].mean  # average score
experiment.summary.metrics[0].ci    # ConfidenceInterval(lower, upper)
experiment.results  # list of individual EvalResult
```
"""
    improvements.append(v2)

    with patch("optimize_anything.api.ClaudeProposer") as MockP:
        mock = MagicMock()
        call_idx = [0]
        def propose(**kwargs):
            idx = min(call_idx[0], len(improvements) - 1)
            call_idx[0] += 1
            return improvements[idx]
        mock.propose.side_effect = propose
        MockP.return_value = mock

        events = []
        def on_event(e):
            events.append(e)
            if e["type"] == "new_best_found":
                print(f"  [EVENT] New best found! Score: {e['score']:.4f}")
            elif e["type"] == "iteration_start":
                print(f"  [EVENT] Iteration {e['iteration']} starting...")

        result = optimize_anything(
            seed_candidate={"skill": current_skill},
            evaluator=skill_evaluator,
            dataset=dataset,
            objective="Maximize the skill's completeness, actionability, clarity, trigger coverage, and developer experience",
            config=Config(max_iterations=2, reflection_minibatch_size=3),
            on_event=on_event,
        )

    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Iterations: {result.total_iterations}")
    print(f"  Evaluations: {result.total_evals}")
    print(f"  Candidates explored: {len(result.all_candidates)}")
    print(f"  Score trajectory: {' → '.join(f'{s:.0%}' for s in result.all_scores)}")
    print(f"  Best score: {result.best_score:.0%}")
    print()

    # Evaluate the best candidate on all dimensions
    best_text = result.best_candidate["skill"]
    print("BEST CANDIDATE SCORES:")
    for example in dataset:
        score, details = skill_evaluator(best_text, example)
        dim = details["dimension"]
        status = "✓" if score == 1.0 else "◐" if score >= 0.6 else "✗"
        print(f"  {status} {dim}: {score:.0%} ({details['passed']}/{details['total']})")

    # Write the improved skill
    output_path = Path("/Users/rshah/optimize-anything/scripts/optimized_llm_evals_skill.md")
    output_path.write_text(best_text)
    print(f"\nOptimized skill saved to: {output_path}")

    return best_text


if __name__ == "__main__":
    main()
