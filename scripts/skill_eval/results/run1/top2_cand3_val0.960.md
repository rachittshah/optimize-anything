---
name: optimize-anything
description: Run the optimize_anything tool (GEPA-style search) to iteratively improve a text artifact (prompt, agent instructions, code, regex, config, policy, agent harness, multi-file text) against an evaluator. Use when the request has (1) an artifact that is text, (2) a success signal that can score candidates: metric, tests, labeled or expected outputs, benchmark script, checkable constraints, or a rubric, and (3) a wish for repeated propose-evaluate cycles ("optimize/tune/evolve/keep improving until it passes the tests or benchmark, make it generalize"). Also use when the user names GEPA or optimize_anything. Do NOT use for one-shot edits, rewrites or critiques with no evaluator; building an eval set or judge; writing tests; profiling or speeding up one query by hand; fine-tuning model weights; numeric hyperparameter tuning (Optuna, grid search); or questions about how the method works.
---

# optimize_anything

The tool proposes edits to a text candidate, scores each edit with your evaluator, reflects on the evaluator's diagnostics (ASI), and keeps a Pareto frontier until the budget runs out. Your job is to emit one sound tool call, or to ask for clarification only when no sound call is possible.

## Call or clarify

**Default to calling.** Reply with `CLARIFY:` only for blocking gaps:
- no artifact (nothing to seed);
- no success signal (no tests, labels, expected outputs, benchmark, checkable constraints or criteria);
- no data when the goal depends on inputs.

These are NOT blocking. Assume, state the assumption in one line, and call:
- **The artifact is a prompt or instructions that a model must run.** The python evaluator calls an LLM API with stdlib `urllib`. It reads the key and model name from environment variables with a default, and returns 0 with the error in the ASI if the key is missing.
- **The dataset is small.** With 5 or more examples you can still split (e.g. 3/2, or 7/3).
- **A detail is ambiguous** (basename or path, exact format). Pick the reading that best fits the files and say so.

In a clarify reply, give numbered questions. For each one, say what the answer unlocks, e.g. "Share 10 labeled examples → lets me score accuracy with a python evaluator." Then add a **default plan** you will run once the answers arrive: the artifact as seed, the evaluator type and metric, the ASI contents, the mode and split, the objective, and a budget. A typical default for "improve our agent": optimize the system prompt against a task suite with pass/fail checks, a python evaluator, a generalization split, and 100 metric calls.

## Tool arguments (one JSON object)

| Field | Rule |
|---|---|
| `seed_candidate` | Dict of component name → full current text. One key per file or part being changed, with the actual contents copied in, not a placeholder. |
| `evaluator` | `python`, `shell`, or `llm_judge` (see below), plus a `timeout` in seconds of at least 2-3× one evaluation's runtime. |
| `objective` | Specific and measurable. Name the metric, any numeric target, and the hard constraints (format, length, signature, no deps). |
| `background` | Facts the proposer needs: how scoring works, what the data looks like, constraints. Do not dictate the solution. |
| `dataset` | List of example dicts. Omit it for single-task runs. |
| `valset` | Held-out examples, disjoint from `dataset`. |
| `config` | Always set both `max_metric_calls` and `max_iterations`. |

## Choosing the evaluator

- **Verifiable signal** (labels, expected outputs, tests, benchmark, constraints checkable in code such as regex format or length limits) → `python` or `shell`, never `llm_judge`.
- **Existing script, test runner or benchmark** → `shell`.
- **Labeled examples, checkable rules, or logic you write yourself** → `python`.
- **Subjective target only** (tone, clarity, readability) → `llm_judge`.

The score must measure exactly what the user asked for. Do not add extra terms, such as order-sensitive exact match when only set membership matters, or length penalties nobody requested. With several hard requirements, check each one in code and score their mean or product. Gate on correctness: a wrong output scores 0 or near it, however fast it is.

## llm_judge rules

- Write a few concrete, checkable criteria, such as "names at least 2 next steps" or "no jargon". Avoid vague goals like "is good".
- Turn each hard constraint into an explicit score cap, e.g. "over 90 words → max 4". Put the same constraints in `objective`.
- Anchor the scale: "10 = …, 7 = …, 4 = …, 0 = …".
- For ASI, require the judge to output, before the score, one line per criterion: a 0-10 sub-score, the quoted offending text, and one specific fix.
- End the criteria text with: "Finish with `SCORE: X/10`."

## Python evaluator rules

- Stdlib only. It runs in a subprocess, so put test data in `dataset`/`valset` rather than reading files.
- Define `evaluate(candidate, example)`, accept `candidate` as a str or a dict, and handle `example=None`.
- For `example=None`, score a few cases copied verbatim from `dataset`. Never use val cases or invented ones.
- Return `(score, info)`, where `score` is a float in [0,1] and `info` is a JSON-serializable dict.
- Catch every exception (bad regex, syntax errors, API or network errors, missing key) and return 0 with the error in the ASI. Never crash.
- Rich, case-specific ASI drives improvement. Include:
  - the input, expected vs got, and missing and extra items;
  - a per-check boolean dict (e.g. `{"format_ok": true, "type_ok": false, "len_ok": true}`);
  - the raw model output and any parse errors.

  With several components, say which component likely caused each failure.

Template:

```python
import json, os, re, urllib.request

SAMPLE = [...]  # 2-3 examples copied from dataset (train only), for example=None

def as_dict(candidate):
    return candidate if isinstance(candidate, dict) else {"main": str(candidate)}

def call_llm(prompt):  # only when the candidate is a prompt a model must run
    key = os.environ.get("ANTHROPIC_API_KEY")  # or the provider the user names
    if not key:
        raise RuntimeError("API key env var not set")
    model = os.environ.get("EVAL_MODEL", "<small default model>")
    ...  # urllib POST, temperature 0, return text

def score_one(c, ex):
    try:
        got = run(c, ex)                    # apply candidate (maybe via call_llm)
    except Exception as err:
        return 0.0, {"input": ex, "error": repr(err)}
    checks = {"correct": ..., "format_ok": ..., "len_ok": ...}
    s = sum(checks.values()) / len(checks)  # or gate: 0 if not checks["correct"]
    return s, {"input": ex, "expected": ex.get("expected"), "got": got,
               "checks": checks, "failed": [k for k, v in checks.items() if not v]}

def evaluate(candidate, example=None):
    c = as_dict(candidate)
    if example is None:
        res = [score_one(c, ex) for ex in SAMPLE]
        return sum(s for s, _ in res) / len(res), {"cases": [i for _, i in res]}
    return score_one(c, example)
```

## Shell evaluator rules

- Pass the candidate via `{{candidate_file}}`. Never inline `{{candidate}}`, which is unquoted injection. For a dict candidate, use a single component when the script expects one file.
- The command must print a score already in [0,1], matched by `score_pattern` (e.g. `"Score: ([\\d.]+)"`). If the script reports raw time or counts, normalize them in a small wrapper.
- Examples reach the command only via `{{example_file}}` or `{{example_json}}`. Without one of these, the run is single-task.
- Stdout and stderr become the ASI. Append `2>&1`, and prefer a verbose mode that prints per-case pass/fail, inputs, expected vs got, and timings.

## Modes and splits

- **Single-task:** one problem, no `dataset`. Typical for a benchmark script with fixed inputs, or for polishing one document with a judge.
- **Multi-task:** a `dataset` of examples, so insights transfer across cases.
- **Generalization:** `dataset` plus a disjoint `valset`. Use it whenever the user says "generalize", "unseen" or "new inputs", even with only 5-10 examples. Hold out about 30-40%, and cover each label in both sets where possible. Never copy val examples into `dataset`, `background`, the seed or the evaluator's sample.

## Budget

- Respect any budget the user states, translated into `max_metric_calls`.
- Cheap, deterministic evaluations (under 1s): `max_metric_calls` 150-400, `max_iterations` 20-40.
- LLM-backed evaluations or builds over 10s: `max_metric_calls` 40-100, `max_iterations` 10-20.
- llm_judge on a single artifact: `max_metric_calls` 15-30, `max_iterations` 6-10.
- Each iteration costs about minibatch size plus validation calls, so scale `max_metric_calls` with dataset size.

## Example call (multi-component, labeled data)

```json
{
  "seed_candidate": {"split.txt": "<file text>", "stopwords.txt": "<file text>"},
  "evaluator": {"type": "python", "timeout": 30, "code": "<follows template; set-F1 vs expected; ASI lists missing/extra tokens>"},
  "objective": "Maximize mean keyword F1 vs expected keywords; split.txt must remain a valid Python regex.",
  "background": "Tokens come from re.split(split, title.lower()) minus stopwords.",
  "dataset": [{"text": "...", "keywords": ["..."]}],
  "config": {"max_metric_calls": 200, "max_iterations": 25}
}
```

For a subjective single document, use `{"type": "llm_judge", "timeout": 120, "criteria": "<checkable criteria; caps; per-criterion sub-score + fix; SCORE: X/10>"}`, no `dataset`, and `config` `{"max_metric_calls": 16, "max_iterations": 8}`.

## After the call

The call returns a `run_id`. Monitor progress with `check_optimization`, fetch the winner with `get_best_candidate`, then apply it and verify it against the user's original tests.