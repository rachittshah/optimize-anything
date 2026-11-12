# Skill-eval data for `skill/SKILL.md`

Data for hill-climbing the optimize-anything skill on four signals: trigger F1, formulation
validity, formulation quality and brevity. Validate with `python3 scripts/skill_eval/check_data.py`
(add `--live` to run the API-backed golds; `--assign-splits` rewrites `split` deterministically).

| File | Contents | Counts |
|------|----------|--------|
| `triggers.jsonl` | `{id, prompt, label, kind, note, split}` for router evaluation | 50 items: 24 positive, 18 near_miss, 8 unrelated. Train 36 (17/13/6), val 14 (7/5/2) |
| `distractors.json` | 14 competing skill descriptions (`name`, `description`) | Each near-miss has an obvious winner here |
| `formulations.jsonl` | `{id, category, request, context_files, gold, rubric_notes, split}` | 30 items, 10 categories × 3. Train 20, val 10 (each category in both splits) |
| `rubric.md` | 7 criteria scored 0–2 for formulation quality | — |

**Splits:** seeded (`20261111`) and stratified by `kind` for triggers and by `category` for
formulations, at a 70/30 ratio. Every stratum appears in both splits. Val items cover distinct
scenarios, not paraphrases of train items (the validator warns on high token overlap).

## Near-miss labelling policy

Label `trigger` only when the request implies **an artifact made of text, an evaluator (metric,
tests, labels, benchmark or rubric) and repeated propose-evaluate cycles**. Two of the three is
not enough. Near-misses are labelled `no_trigger` when:

- The surface word matches ("optimize my SQL query", "evolve my career plan", "optimize bundle
  size") but the fix is direct or the domain belongs to another skill.
- It's a one-shot edit or critique with no evaluator ("improve this paragraph", "review my prompt").
- It's adjacent work: building evals, writing tests, plotting results, debugging one failure,
  fine-tuning weights, or numeric HPO with a named tool (Optuna).
- It's a question about the method ("what is GEPA?").

Borderline calls are explained in each `note`. For example, a judge-graded cold-email template
with an explicit iterate-until-9/10 loop is a trigger (t20), while "make my resume better" is not
(n10). Text-to-SQL *prompt* tuning against execution results is a trigger (t11), while speeding up
one SQL query is not (n01).

## Gold formulations: engine contract

Golds target the engine as it is on `sota-upgrade`:

- **Python evaluators** are stdlib-only, define `evaluate(candidate, example)`, and accept
  `candidate` as a str or a dict (multi-component). They handle `example=None` and return
  `(score ∈ [0,1], JSON-serializable dict)`. They run in a subprocess outside `context_files`, so
  test data lives in `dataset`/`valset`.
- **API-backed golds** (item-level `requires: ["ANTHROPIC_API_KEY"]`, 9 items; `gold` itself is exactly the tool call) call the task model over
  urllib (`OA_TASK_MODEL`, default `claude-haiku-5-5`). Without a key they degrade to
  `(0.0, {"error": ...})`. Offline validation checks only that degradation; `--live` measures
  seed scores.
- **Shell evaluators** run from a temp dir that holds `context_files`, read the candidate via
  `{{candidate_file}}` (never inline `{{candidate}}`) and print a score already in [0,1]. They are
  single-task because the shell evaluator ignores `example`.
- Every offline-measured seed scores below 1.0 (means 0.08–0.71), since `skip_perfect_score` would
  otherwise skip the item. API seeds are unmeasured until `--live` runs. To leave headroom, their
  evaluators don't inject the label or tool lists (those sit in `background`), and f11/f25 include
  ambiguous examples decided by a rule stated in `background`.
- Clarify items (`gold.clarify: true`, 3) list `ask` questions instead of a call.

Note for skill authors: SKILL.md Recipe 4 pairs a shell evaluator with a dataset, which the shell
evaluator can't use, and its shell example uses inline `{{candidate}}`.
