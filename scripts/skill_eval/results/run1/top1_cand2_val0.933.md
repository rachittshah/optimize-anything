---
name: optimize-anything
description: Run the optimize_anything tool (GEPA-style search) to iteratively improve a text artifact (prompt, code, regex, config, policy, agent harness, multi-file text) against an evaluator. Use when the request has (1) an artifact that is text, (2) a success signal that can score candidates: metric, tests, labeled or expected outputs, benchmark script, or a rubric, and (3) a wish for repeated propose-evaluate cycles ("optimize/evolve/search/keep improving until it passes the tests or benchmark"). Also use when the user names GEPA or optimize_anything. Do NOT use for one-shot edits, rewrites or critiques with no evaluator; building an eval set or judge; writing tests; profiling or speeding up one query by hand; fine-tuning model weights; numeric hyperparameter tuning (Optuna, grid search); or questions about how the method works.
---

# optimize_anything

The tool proposes edits to a text candidate, scores each edit with your evaluator, reflects on the evaluator's diagnostics (ASI), and keeps a Pareto frontier until the budget runs out. Your job is to emit one sound tool call, or to ask for clarification when you cannot build one.

## When to clarify instead of calling

Reply with `CLARIFY:` and numbered questions only for blocking gaps:
- no artifact (nothing to seed);
- no success signal (no tests, labels, benchmark or criteria);
- no data when the goal depends on inputs.

For each question, say what the answer unlocks, e.g. "Share 10 labeled examples → lets me score accuracy with a python evaluator."

If the gap is non-blocking, make a reasonable assumption, state it, and call the tool.

## Tool arguments (one JSON object)

| Field | Rule |
|---|---|
| `seed_candidate` | Dict of component name → full current text. One key per file or part being changed, with the actual contents copied in, not a placeholder. |
| `evaluator` | `python`, `shell`, or `llm_judge` (see below), plus a `timeout` in seconds of at least 2-3× one evaluation's runtime. |
| `objective` | Specific and measurable. Name the metric, any numeric target, and the hard constraints (signature, correctness, length, no deps). |
| `background` | Facts the proposer needs: how scoring works, what the data looks like, constraints. Do not dictate the solution. |
| `dataset` | List of example dicts. Omit it for single-task runs. |
| `valset` | Held-out examples, disjoint from `dataset`. |
| `config` | Always set both `max_metric_calls` and `max_iterations`. |

## Choosing the evaluator

- **Verifiable signal** (labels, expected outputs, tests, benchmark) → `python` or `shell`, never `llm_judge`.
- **Existing script, test runner or benchmark** → `shell`.
- **Labeled examples, or logic you write yourself** → `python`.
- **Subjective target only** (tone, clarity, readability) → `llm_judge`.

The score must measure exactly what the user asked for. Do not add extra terms, such as order-sensitive exact match when the user only cares about set membership, or length penalties nobody requested. Gate on correctness: a wrong output scores 0 or near it, however fast it is.

## llm_judge rules

- Write a few concrete, checkable criteria, such as "names at least 2 next steps" or "no jargon". Avoid vague goals like "is good".
- Turn each hard constraint into an explicit score cap, e.g. "over 90 words → max 4" or "missing a link home → max 5". Put the same constraints in `objective`.
- Anchor the scale: "10 = …, 7 = …, 4 = …, 0 = …".
- For ASI, require the judge to output, before the score, one line per criterion: a 0-10 sub-score, the quoted offending text, and one specific fix.
- End the criteria text with: "Finish with `SCORE: X/10`."

## Python evaluator rules

- Stdlib only. It runs in a subprocess, so put test data in `dataset`/`valset` rather than reading files.
- Define `evaluate(candidate, example)`.
- Accept `candidate` as a str or a dict.
- Handle `example=None` by scoring a small inline sample of train cases, never val cases.
- Return `(score, info)`, where `score` is a float in [0,1] and `info` is a JSON-serializable dict.
- Catch exceptions (bad regex, syntax errors, exec failures) and return 0 with the error message in the ASI. Never crash.
- Rich ASI is what drives improvement. Include the input, expected vs got, missing and extra items, failing checks, and parse or exec errors. With several components, make it clear which component likely caused each failure.

Template:

```python
import json, re

SAMPLE = [{"input": "...", "expected": "..."}]  # a few train cases, for example=None

def as_dict(candidate):
    if isinstance(candidate, dict):
        return candidate
    return {"main": str(candidate)}

def score_one(c, ex):
    try:
        got = run(c, ex["input"])           # apply candidate to the input
    except Exception as err:
        return 0.0, {"input": ex["input"], "error": repr(err)}
    exp = ex["expected"]
    s = metric(got, exp)                    # matches the user's notion of success, in [0,1]
    return s, {"input": ex["input"], "expected": exp, "got": got,
               "failing_checks": diff(got, exp)}

def evaluate(candidate, example=None):
    c = as_dict(candidate)
    if example is None:
        results = [score_one(c, ex) for ex in SAMPLE]
        mean = sum(s for s, _ in results) / len(results)
        return mean, {"cases": [info for _, info in results]}
    return score_one(c, example)
```

## Shell evaluator rules

- Pass the candidate via `{{candidate_file}}`. Never inline `{{candidate}}`, which is unquoted injection. For a dict candidate, the file holds the component text, so use a single component when a script expects one file.
- The command must print a score already in [0,1], matched by `score_pattern` (e.g. `"Score: ([\\d.]+)"`). If the script reports raw time or counts, normalize them in a small wrapper.
- Examples reach the command only via `{{example_file}}` or `{{example_json}}`. Without one of these, the run is single-task.
- Stdout and stderr become the ASI. Append `2>&1`, and prefer a verbose mode or wrapper that prints per-case pass/fail, inputs, expected vs got, and timings, not just one number.

## Modes and splits

- **Single-task:** one problem, no `dataset`. Typical for a benchmark script that fixes its own inputs, or for polishing one document with a judge.
- **Multi-task:** `dataset` of examples, so insights transfer across cases.
- **Generalization:** `dataset` plus a disjoint `valset`. Use it when the user cares about unseen inputs or has enough data, roughly 20+ examples, holding out about 20-30%. Never copy val examples into `dataset`, `background`, the seed or the evaluator's sample.

## Budget

- Respect any budget the user states, translated into `max_metric_calls`.
- Cheap, deterministic evaluations (under 1s): `max_metric_calls` 150-400, `max_iterations` 20-40.
- Expensive evaluations (LLM calls, builds over 10s): `max_metric_calls` 30-100, `max_iterations` 10-20.
- llm_judge on a single artifact: keep it small, with `max_metric_calls` 15-30 and `max_iterations` 6-10. Each call costs money and the gains plateau quickly.
- Each iteration costs about minibatch size plus validation calls, so scale `max_metric_calls` with dataset size.

## Example call (multi-component, labeled data)

```json
{
  "seed_candidate": {"split.txt": "<file text>", "stopwords.txt": "<file text>"},
  "evaluator": {"type": "python", "timeout": 30, "code": "<follows template; set-F1 vs expected; ASI lists missing/extra tokens>"},
  "objective": "Maximize mean keyword F1 vs expected keywords; split.txt must remain a valid Python regex.",
  "background": "Tokens come from re.split(split, title.lower()) minus stopwords. Punctuation is handled by the split pattern, function words by stopwords.",
  "dataset": [{"text": "...", "keywords": ["..."]}],
  "config": {"max_metric_calls": 200, "max_iterations": 25}
}
```

For a subjective single document, use `{"type": "llm_judge", "timeout": 120, "criteria": "<checkable criteria; caps for hard limits; per-criterion sub-score + fix; SCORE: X/10>"}`, no `dataset`, and `config` `{"max_metric_calls": 16, "max_iterations": 8}`.

## After the call

The call returns a `run_id`. Monitor progress with `check_optimization`, fetch the winner with `get_best_candidate`, then apply it and verify it against the user's original tests.