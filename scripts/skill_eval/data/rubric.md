# Formulation-quality rubric

Score each criterion 0–2. Total out of 14, normalized to [0,1]. Judge the agent's emitted
`optimize_anything` call (or its clarifying reply) against the request, `context_files`, the
item's `gold` and its `rubric_notes`. A gold is one correct answer, not the only one: give full
credit for a different formulation that is equally sound.

| # | Criterion | 0 | 1 | 2 |
|---|-----------|---|---|---|
| 1 | **Evaluator fits the signal** | Uses `llm_judge` when a verifiable signal exists (labels, expected outputs, tests, a benchmark), or an evaluator that can't measure the objective | Right family but a weak proxy (e.g. string similarity to a reference SQL instead of execution match) | Verifiable signal → `python`/`shell` that checks it; `llm_judge` only for subjective targets, with concrete anchored criteria that end in `SCORE: X/10` |
| 2 | **ASI richness** | Returns a bare number | Some diagnostics (e.g. only `{"error": ...}` on failure) | Per-example dict the proposer can act on: input, expected vs got, failing check names, compile/parse errors, constraint values |
| 3 | **Mode and split** | Wrong mode (single-task when examples were given; dataset with a shell evaluator that ignores examples), or val examples leak into `dataset` or few-shots | Mode right, but no valset when the user asked about unseen data, or a valset carved from too few examples | Single-task / multi-task / generalization matches the request; `dataset` ∩ `valset` = ∅; held-out items never appear in the candidate |
| 4 | **Objective** | Vague ("make it better") | Names the goal but not the metric or constraint | Specific and measurable, naming the metric and any hard constraint (e.g. "maximize legit pass rate s.t. scraper ≤ 5%") |
| 5 | **Budget** | Missing, or unbounded for an expensive evaluator; ignores a budget the user stated | Present but mismatched (e.g. 100 iterations of a 2-minute benchmark) | `max_metric_calls` / `max_iterations` / `timeout_seconds` fit the cost per eval and any user-stated limit; evaluator `timeout` exceeds the eval's runtime |
| 6 | **Clarify when needed** | Invents an artifact, data or metric and calls the tool anyway; or asks needless questions when everything needed was given | Asks, but misses the blocking gap (artifact, success signal or data) | Calls the tool when an evaluator can be built; otherwise asks targeted questions and says what each answer unlocks, surfacing proxy trade-offs (e.g. judge vs real conversions) |
| 7 | **Safety and runnability** | Destructive or unsafe commands (`rm -rf`, network writes, prod deploys), or `{{candidate}}` substituted inline into a shell command (unquoted injection) | Runs, but is fragile (references files that don't exist, scores outside [0,1], non-stdlib imports) | Only `{{candidate_file}}`; references only provided files; stdlib-only python defining `evaluate(candidate, example)` that returns `(score ∈ [0,1], dict)` and handles `example=None` and str-or-dict candidates |

For clarify items (`gold.clarify == true`), score criterion 6 and give criteria 1–5 and 7 a 2
only if the reply sketches a sound plan conditional on the answers. Otherwise give them 1.
