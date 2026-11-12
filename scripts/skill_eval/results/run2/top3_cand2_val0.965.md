---
name: optimize-anything
description: Use when a text artifact (prompt, code, regex, config, policy, agent instructions/harness, page copy) should be improved through repeated propose-evaluate cycles with the optimize_anything tool, scored by a metric, tests, labeled examples, a benchmark, or a rubric/LLM judge. Also use when the user mentions GEPA, evolving, or optimizing via search, even if details are missing. Not for one-shot edits or critiques with no evaluation loop, building evals or writing tests, fine-tuning model weights, numeric hyperparameter tuning (e.g. Optuna), profiling or speeding up a single query, or questions about how the method works.
---

# optimize_anything

The tool iteratively improves any text artifact. On each cycle it proposes a candidate, scores it with your evaluator, reads the diagnostics (ASI) and tries again until the budget runs out. It keeps a Pareto frontier of the best candidates.

## Output contract

Emit ONE JSON object holding the tool arguments. If no sound call is possible, emit `CLARIFY:` followed by questions instead.

```json
{
  "seed_candidate": {"<component>": "<full current text>"},
  "evaluator": {"type": "python|shell|llm_judge", "...": "..."},
  "objective": "specific, measurable goal + hard constraints",
  "background": "domain facts, assumptions, what must not change",
  "dataset": [{"input": "...", "expected": "..."}],
  "valset": [{"input": "...", "expected": "..."}],
  "config": {"max_metric_calls": 60, "max_iterations": 15}
}
```

Field rules:
- **seed_candidate:** use the real current text from the user's files, never a summary. Use one key per independently editable part.
- **Examples:** copy them from the user's files, keeping the files' real field names. The evaluator must read exactly those keys.
- **Omit unused fields:** leave out `dataset` and `valset` in single-task mode.

## 1. Pick the evaluator from the signal

| Signal available | Evaluator |
|---|---|
| Labels, expected outputs, tests, a benchmark, checkable rules (format, length, required content) | `python` (stdlib) or `shell` |
| Purely subjective target (tone, readability) with no verifiable signal | `llm_judge` |

If any verifiable signal exists, do not use `llm_judge`.

### Artifacts that drive a model (prompts, agent instructions)
Score the real behavior: run the candidate with a model on each example's input, then check the output against the label or rules in code.
- **No proxies:** never replace the model with a keyword or regex simulator.
- **Preserve the format:** never force the artifact into a format your evaluator happens to parse.
- **Calling the model:** use stdlib `urllib` against an OpenAI-compatible endpoint. Read the URL, key and model name from environment variables, such as `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `MODEL`.
- **No CLI assumptions:** do not assume any particular agent CLI is installed.
- **Model errors:** wrap the call in try/except. If it fails, return 0 with the error in the ASI, so the seed still runs.
- **State your assumptions:** name the endpoint and env vars in `background`.

### Python evaluator
```python
import json, os, urllib.request
def evaluate(candidate, example):
    text = candidate if isinstance(candidate, str) else next(iter(candidate.values()), "")
    if example is None:                      # single-task / seed sanity call
        example = {"input": "", "expected": ""}
    try:
        got = run(text, example["input"])    # e.g. model call or stdlib logic
    except Exception as e:
        return 0.0, {"input": example["input"], "error": repr(e)}
    checks = {"exact_label": got.strip() == example["expected"]}
    score = sum(checks.values()) / len(checks)
    return score, {"input": example["input"], "expected": example["expected"],
                   "got": got, "checks": checks,
                   "failing_checks": [k for k, v in checks.items() if not v]}
```
Rules:
- **Stdlib only:** it runs in a subprocess.
- **Signature:** define `evaluate(candidate, example)`.
- **Candidate type:** accept the candidate as a `str` OR a `dict` (multi-component).
- **No example:** handle `example=None`.
- **Return value:** return `(score, info)`, with the score a float in [0,1] and `info` JSON-serializable.
- **Never raise:** catch errors and return 0 with the error text in the ASI.
- **No file reads:** put test data in `dataset`/`valset`.
- **Rich ASI:** include the input, expected vs got, a per-check boolean dict, failing checks, and parse or compile errors. Thin ASI gives the proposer nothing to fix.
- **Partial credit:** prefer the fraction of checks passed over a bare 0/1.

### Shell evaluator
```json
{"type": "shell",
 "command": "python3 bench.py --candidate {{candidate_file}} --example {{example_file}}",
 "score_pattern": "SCORE: ([0-9.]+)", "timeout": 120}
```
Rules:
- **Read the candidate from a file:** use `{{candidate_file}}`. Never inline `{{candidate}}`, which is an unquoted injection risk.
- **Score range:** print a score already normalized to [0,1], matched by `score_pattern`. Convert raw metrics, e.g. with a `python3 -c` wrapper computing `baseline/time` capped at 1.
- **Examples:** they reach the command only via `{{example_file}}` or `{{example_json}}`. Without them the run is single-task.
- **Diagnostics:** print failures, stderr and timings as ASI.
- **Paths:** context files are in the working directory, so use relative paths.

### LLM judge (subjective only)
- **Anchored criteria:** make them concrete, with hard limits (word cap, required elements, forbidden content).
- **Score anchors:** define them, e.g. "10 = all met; 7 = one minor miss; 4 = a hard constraint violated; 1 = off-task".
- **Structured feedback:** ask for violations as a list (criterion -> issue -> fix).
- **Ending:** the criteria must end with `SCORE: X/10`.

## 2. Choose mode and split

| Mode | Use when | Fields |
|---|---|---|
| Single-task | One artifact, one goal (a solver, page copy) | no dataset |
| Multi-task | Several cases that must all improve, and unseen inputs don't matter | `dataset` |
| Generalization | The user says "generalize", "unseen", or "new inputs", or the artifact is a prompt or classifier meant for live traffic | `dataset` + disjoint `valset` |

Split rules:
- **Always split when generalization is needed:** this holds even with few examples. Hold out about 20–30%, at least 1–2 examples, and cover the classes where possible. Mention in `background` that more data would strengthen validation.
- **No leakage:** val examples never appear in `dataset`, the seed, the objective or the background.

## 3. Write the objective

Name the metric, the target and every hard constraint. For example: "Maximize exact-match routing accuracy on held-out tickets; output must be one label from {A,B,C}." Avoid vague goals like "make it better".

## 4. Size the budget

Set `config.max_metric_calls`, which bounds cost, and optionally `max_iterations`:
- **User limits:** honor any budget or time limit the user states.
- **Cheap python evaluators** (ms–s): 100–300 calls.
- **Model calls, slow benchmarks or test suites:** 30–80 calls. Set the evaluator `timeout` to about 2–3x one evaluation's runtime.
- **llm_judge:** 20–40 calls.
- **Dataset size:** budget roughly (dataset size + valset size) × the number of iterations you want.

## 5. Clarify only when blocked

Output `CLARIFY:` only if a sound call is impossible:
- **No artifact:** nothing to optimize, and no seed can be written.
- **No success signal:** the goal needs one (e.g. "smarter", "more accurate"), but there are no labels, tests or metric.
- **No data:** generalization is needed, but no examples were given.

How to ask:
- **Only blocking questions:** budget, split ratio and component names are not blocking.
- **Say what each answer unlocks.** For example: "Share 10–20 labeled tickets → exact-match python evaluator with a held-out valset."
- **Propose a default plan.** For example: "Once I have the system prompt and a suite of tasks with pass/fail checks, I'll optimize the prompt in generalization mode with a python evaluator reporting per-task pass/fail as ASI, using about 50 calls."

Do not clarify when the signal is subjective. Use an anchored judge instead, and state any assumed defaults in `background`.

## Safety checklist before emitting
- **Valid JSON:** the evaluator code is a valid string, with escaped quotes and newlines.
- **Python evaluator:** stdlib only, handles `str`/`dict`/`None`, never raises, uses the data's real keys, and returns a [0,1] score plus a dict.
- **Shell evaluator:** uses `{{candidate_file}}`, prints a normalized score that the pattern matches, and sets a timeout.
- **No destructive actions:** no network calls beyond the model endpoint, and no deploys unless the user asked.
- **Seed:** it runs through the evaluator without errors.

## After the call
The tool returns a run_id. Monitor progress, fetch the best candidate, show the diff and the score gain, and apply it only with the user's agreement.