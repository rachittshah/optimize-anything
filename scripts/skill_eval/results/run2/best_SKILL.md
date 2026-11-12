---
name: optimize-anything
description: Use when a text artifact (prompt, code, regex, config, policy, agent harness, page copy) should be improved through repeated propose-evaluate cycles with the optimize_anything tool, scored by a metric, tests, labeled examples, a benchmark, or a rubric/LLM judge. Also use when the user mentions GEPA, evolving, or optimizing via search. Not for one-shot edits or critiques with no evaluation loop, building evals or writing tests, fine-tuning model weights, numeric hyperparameter tuning (e.g. Optuna), profiling or speeding up a single query, or questions about how the method works.
---

# optimize_anything

The tool iteratively improves any text artifact. Each cycle it proposes a candidate, scores it with your evaluator, reads the diagnostics (ASI) and retries until the budget runs out, keeping a Pareto frontier of the best candidates.

## Output contract

Emit ONE JSON object holding the tool arguments. Emit `CLARIFY:` plus questions instead only if no sound call is possible (see section 5).

```json
{
  "seed_candidate": {"<component>": "<full current text>"},
  "evaluator": {"type": "python|shell|llm_judge", "...": "..."},
  "objective": "specific, measurable goal + hard constraints",
  "background": "domain facts, constraints, assumptions you made, what must not change",
  "dataset": [{"input": "...", "expected": "..."}],
  "valset": [{"input": "...", "expected": "..."}],
  "config": {"max_metric_calls": 60, "max_iterations": 15}
}
```

Field rules:
- **seed_candidate:** use the real current text from the user's files, never a summary. Use one key per independently editable part.
- **Omit unneeded fields:** leave out `dataset` and `valset` in single-task mode.

## 1. Pick the evaluator from the signal

| Signal available | Evaluator |
|---|---|
| Labels, expected outputs, tests, a benchmark script, a measurable property | `python` (stdlib) or `shell` |
| Purely subjective target (tone, readability) with no verifiable signal | `llm_judge` |

If any verifiable signal exists, do not use `llm_judge`.

### Python evaluator
```python
import json, re
def evaluate(candidate, example):
    text = candidate if isinstance(candidate, str) else next(iter(candidate.values()), "")
    if example is None:              # single-task / seed sanity call
        example = {"input": "", "expected": ""}
    try:
        got = run(text, example["input"])  # your stdlib-only logic
    except Exception as e:
        return 0.0, {"input": example["input"], "error": repr(e)}
    ok = got == example["expected"]
    return (1.0 if ok else 0.0), {
        "input": example["input"], "expected": example["expected"],
        "got": got, "failing_checks": [] if ok else ["exact_match"]}
```
Rules:
- **Stdlib only.** It runs in a subprocess.
- **Signature:** define `evaluate(candidate, example)`.
- **Inputs:** accept the candidate as a `str` OR a `dict` (multi-component), and handle `example=None`.
- **Return value:** `(score, info)`, with the score a float in [0,1] and `info` JSON-serializable.
- **No file reads:** put test data in `dataset`/`valset`.
- **Never raise:** wrap exec, compile, parse and network calls in try/except, and return 0 with the error in the ASI.
- **Rich ASI:** include input, expected vs got, failed checks, errors and partial-credit parts. Thin ASI gives the proposer nothing to fix.
- **Partial credit:** prefer a fraction of checks passed over a bare 0/1.

### Prompts and agent instructions (model in the loop)
When the artifact is text a model consumes (system prompt, AGENTS.md/GEMINI.md/CLAUDE.md, classifier or extractor prompt) and labels exist, run the model inside the evaluator. Do not clarify about how to invoke it.

Write the python evaluator as follows:
- **Request:** send the candidate as the system message and `example["input"]` as the user message to an OpenAI-compatible chat endpoint via `urllib.request`.
- **Config:** read the key, base URL and model from environment variables, with defaults (e.g. `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL`).
- **Scoring:** score the reply deterministically against the labels: keyword or line hit, exact match, JSON fields.
- **Anti-gaming:** if a candidate could win by flagging or listing everything, include a precision or false-positive penalty, e.g. credit only if the target is found and the extra findings are few.
- **Timeouts:** set a request timeout, and catch HTTP and timeout errors, returning 0 plus the error.
- **State assumptions:** record the endpoint and env-var assumptions in `background`.

A shell evaluator calling the user's agent CLI also works when the user names the command.

### Shell evaluator
```json
{"type": "shell",
 "command": "python3 bench.py --candidate {{candidate_file}} --example {{example_file}}",
 "score_pattern": "SCORE: ([0-9.]+)", "timeout": 120}
```
Rules:
- **Candidate from a file:** use `{{candidate_file}}`, never inline `{{candidate}}`, which is an unquoted injection risk.
- **Score range:** print a score already in [0,1] that `score_pattern` matches. Convert raw metrics, e.g. with a `python3 -c` wrapper computing `baseline/time` capped at 1.
- **Examples:** they reach the command only via `{{example_file}}` or `{{example_json}}`. Without those placeholders the run is single-task.
- **Diagnostics:** print failures, stderr and timings as ASI.
- **Files:** context files sit in the working directory, so use relative paths.

### LLM judge (subjective only)
- **Anchored criteria:** make them concrete, with hard limits (word cap, required elements, forbidden content).
- **Score anchors:** e.g. "10 = all met; 7 = one minor miss; 4 = hard constraint violated; 1 = off-task".
- **Structured feedback:** ask for violations as a list (criterion -> issue -> fix).
- **Ending:** the criteria must end with `SCORE: X/10`.

## 2. Choose mode and split

| Mode | Use when | Fields |
|---|---|---|
| Single-task | One artifact, one goal (a solver, page copy) | no dataset |
| Multi-task | Several cases, all of which must improve | `dataset` |
| Generalization | Must work on unseen inputs (prompts, classifiers, extractors, agent instructions) | `dataset` + disjoint `valset` |

Split rules:
- **Unseen inputs:** if the user says "unseen", "held-out", "new inputs" or "not tuned on", use generalization even with little data. Hold out about 20–30%, at least 1–2 examples (e.g. 6 examples gives a 4/2 split), and note the noise in `background`.
- **Small data otherwise:** with fewer than about 6 examples and no generalization request, use multi-task.
- **No leakage:** val examples must never appear in `dataset`, the seed or `background`.

## 3. Write the objective

Name the metric, the target and every hard constraint, including any anti-gaming penalty. For example: "Maximize the planted-bug hit rate on held-out snippets, penalizing false positives; keep the file under 400 words." Avoid "make it better".

## 4. Size the budget

Set `config.max_metric_calls` (it bounds cost) and optionally `max_iterations`:
- **Respect the user:** honor any budget or time limit they state.
- **Cheap python evaluators:** 100–300 calls.
- **Slow benchmarks, test suites or model-in-the-loop evaluators:** 30–80 calls, with a `timeout` 2–3x one evaluation's runtime.
- **llm_judge:** 20–40 calls.
- **Dataset size:** budget roughly (dataset size + valset size) × the number of iterations you want.

## 5. Clarify only when blocked

Output `CLARIFY:` only if a sound call is impossible:
- **No artifact:** there is nothing to optimize and no seed can be written.
- **No success signal:** a goal needs one, but there are no labels, tests or metric.
- **No data:** generalization is needed, but there are no examples at all.

For each question, say what the answer unlocks, e.g. "Share 10–20 labeled tickets → enables an exact-match evaluator with a held-out valset."

Do NOT clarify in these cases; assume a default and state it in `background`:
- **Subjective signal:** use an anchored judge.
- **Unknown invocation:** you don't know exactly how the user runs their model or agent. Use the model-in-the-loop pattern.
- **Small but present data:** split it anyway.
- **Missing defaults:** budget, split ratio, component names, length limits.

## Safety checklist before emitting
- **Valid JSON:** evaluator code is a valid string with escaped quotes and newlines.
- **Python evaluator:** stdlib only, handles `str`/`dict`/`None`, never raises, returns a [0,1] score plus a dict.
- **Shell evaluator:** uses `{{candidate_file}}`, prints a normalized score, the pattern matches and a timeout is set.
- **No destructive actions:** no deploys or destructive commands. Network use only for a needed model call.
- **Seed:** it runs through the evaluator without errors.

## After the call
The tool returns a run_id. Monitor progress, fetch the best candidate, show the diff and score gain, and apply it only with the user's agreement.