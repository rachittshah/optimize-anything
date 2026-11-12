# OSS architecture scout — optimize-anything design space (2026-11-11)

**Method.** I used `gh api` for stars and last push (pulled 2026-11-11), cloned 16 repos and read source for the 12 pinned below, and read issues and PRs with `gh issue/pr view`. Links are pinned to the commit I read, using these bases:
`[G]`=github.com/gepa-ai/gepa/blob/ce5b0d6e3d · `[OE]`=github.com/algorithmicsuperintelligence/openevolve/blob/3e72b5bcce · `[SH]`=github.com/SakanaAI/ShinkaEvolve/blob/14dd50515f · `[SO]`=github.com/microsoft/SkillOpt/blob/45f65ca04f · `[DS]`=github.com/stanfordnlp/dspy/blob/3983960123 · `[ACE]`=github.com/ace-agent/ace/blob/82709de050 · `[KA]`=github.com/kayba-ai/agentic-context-engine/blob/3a31983178 · `[VF]`=github.com/PrimeIntellect-ai/verifiers/blob/ece57c1030 · `[ART]`=github.com/OpenPipe/ART/blob/534620a41b · `[IN]`=github.com/UKGovernmentBEIS/inspect_ai/blob/0dac400e88 · `[LV]`=github.com/ttanv/levi/blob/341bd17d78 · `[DGM]`=github.com/jennyzzt/dgm/blob/a565fd2d1d.
Issue refs: `gepa#N` = github.com/gepa-ai/gepa/issues/N, and likewise for other repos.

**Our engine today** (re-read 2026-11-11 from the uncommitted working tree, `core/engine.py` at 484 lines; teammates are still editing it). The loop is still serial: one parent, one component, one proposal per iteration, with a strict minibatch-sum gate.

**Already landed or in flight. These are marked LANDED in §3 and not re-proposed:**
- exact-key eval cache (`cache_evals`) and `resume_from`
- `testset` that is never used for selection (`_evaluate_test`)
- `val_eval="halving"`: paired successive halving that is **frontier-aware**, so a child that wins any example is kept (`core/halving.py` `should_stop_early`)
- `promotion="paired_se"` with `promotion_z` and `promotion_max_drop`
- `prescreen`: exact and near-duplicate skip at `dedupe_threshold=0.98`
- `counterexample_mining`, `mutation_operator=rewrite|section_edit`, `operator_bandit`
- `EvaluationError`, which excludes a failed eval instead of scoring it 0 (`failed_evals`)

**Still absent:**
- parallel eval and N proposals per iteration
- rejection memory passed to the proposer
- repeated evals for stochastic evaluators (the config comment says to turn the cache off instead)
- edit-size budget
- pairwise judging
- map-reduce reflection
- epoch-boundary consolidation

`use_merge` is still unused. Team grid `docs/research/sota-2026.md` already lists tabu memory (A-7) and a feasibility gate (A-6) as candidates. It *rejects* the cheap reflector, QD archives and best-of-N; rows 11, 12 and 18 below conflict with that and are flagged.

## 1. Repo table (stars and last push from gh, 2026-11-11)

| Repo | ★ | Last push | Cat | Why relevant / status |
|---|---|---|---|---|
| gepa-ai/gepa | 7,386 | 11-09 | a | Our direct ancestor. It now has an `oa/` package with an eval server, agentic engines, ComBEE and ensembles. **Studied deeply.** |
| stanfordnlp/dspy | 39,011 | 11-11 | a | SIMBA, MIPROv2 and the new ReAnchor (fold-checked acceptance). Studied in `teleprompt/`. |
| microsoft/SkillOpt | 18,992 | 11-10 | a/d | **2026 newcomer, NeurIPS'26 spotlight.** Treats a skill doc like weights: edit budget ("textual LR"), rejected-edit buffer, validation gate. Studied. |
| alchaincyf/darwin-skill | 6,888 | 09-18 | d | A ratchet loop for SKILL.md with multiple judges and human checkpoints. **README only.** |
| NousResearch/hermes-agent-self-evolution | 5,691 | 06-17 | d | Uses DSPy+GEPA on skills, with constraint gates (tests, size ≤15KB, semantic drift). README only; near the 6-month cutoff. |
| algorithmicsuperintelligence/openevolve | 7,763 | 11-08 | b | AlphaEvolve clone: islands, MAP-Elites, cascade eval, novelty gate, coevolution. Studied. |
| SakanaAI/ShinkaEvolve | 1,506 | 11-10 | b | Async proposal/eval pools, cost-aware UCB model bandit, meta-summarizer, prompt co-evolution. Studied. |
| ttanv/levi | 51 | 06-22 | b | Small repo but on-target: a cheap mutator plus a strong "paradigm-shift" model, and a CVT-MAP-Elites archive. Read README, `behavior/`, `pool/`. |
| jennyzzt/dgm | 2,500 | 2025-08 | b | **Stale (>6 mo).** Kept only for its archive selection and staged eval, which are small and verified in code. |
| PrimeIntellect-ai/verifiers | 4,778 | 11-11 | c | RL environments with a rubric judge that has a `drop` verdict; ships a GEPA adapter. Studied `v1/judges`, `v1/gepa`. |
| OpenPipe/ART | 10,853 | 11-10 | c | RULER: a relative, group-wise LLM judge. Studied `rewards/ruler.py`. |
| huggingface/trl | 19,721 | 11-11 | c | GRPO config: multi-objective normalisation, adaptive entropy. Read `grpo_config.py` via gh api. |
| kayba-ai/agentic-context-engine | 2,611 | 09-24 | d | ACE implementation: a skillbook of delta ops, a generalisation gate, a changelog of failures. Studied. |
| ace-agent/ace | 1,447 | 08-24 | d | Reference ACE implementation (helpful/harmful counters, embedding dedup). Studied `core/`. |
| UKGovernmentBEIS/inspect_ai | 3,228 | 11-11 | e | Epochs and reducers, clustered stderr, cascade scorer, model cache. Studied `scorer/`. |
| promptfoo/promptfoo | 27,064 | 11-11 | e | Studied only its cache-key issues, which are useful pain points. Source beyond `src/cache.ts` paths not read. |
| **Considered, not studied** | | | | volcengine/verl (24.2k), OpenRLHF (10.1k), prime-rl (2.2k), deepeval (19.4k), lm-eval-harness (14.5k), autoevals (1.1k), langgraph (44.8k), autogen (61.8k), openai-agents (30.6k), smolagents (30.3k), letta (25.4k; I found no "sleeptime" code, so I make no claim). Not opened; no claims made about them. |
| **Excluded as stale** | | | | zou-group/textgrad (last push 2025-07), microsoft/sammo (2025-06), PromptWizard (2025-10), openai/evals (2026-04), karpathy/autoresearch (99.8k★, 2026-03; its *pattern* lives on in GEPA's `oa/engines/autoresearch.py`). |

## 2. Per-repo architecture notes

**GEPA** (it has grown well past our design)
- The loop is pluggable. `SamplingStrategy` returns N (parent, minibatch) tasks per iteration ([G]/src/gepa/strategies/proposal_sampling.py). `SelectionStrategy` (`AllImprovements`, `BestImprovement`, `TopKImprovements`) filters a batch of proposals ([G]/src/gepa/strategies/proposal_selection.py). The engine drops identical proposals before admitting them ([G]/src/gepa/core/engine.py#L624-L636).
- Acceptance is a protocol: `StrictImprovement` (the default) or `ImprovementOrEqual` for lateral moves, plus `reject_reason()` for logs ([G]/src/gepa/strategies/acceptance.py). The val `EvaluationPolicy` decides which val IDs to score ([G]/src/gepa/strategies/eval_policy.py).
- The eval cache is keyed by (candidate, example, **split**). `valset_cache_split` stops train and val results being shared wrongly. It is batched: hits are fetched, then one adapter call covers all misses ([G]/src/gepa/core/engine.py#L116-L123,#L317-L359). Resume needs `gepa_state.bin` in `run_dir` (#L807-L833). PR #547 rejects checkpoints with a newer schema.
- **Eval server** = the single choke point ([G]/src/gepa/oa/eval_server.py). It exposes an HTTP `/evaluate`, returns 429 when the budget runs out, and **seals the test set** at the HTTP layer so agentic engines never see it. Budget has two ledgers ([G]/src/gepa/oa/budget.py): eval count is enforced centrally, and proposer USD is enforced by each engine.
- Engines include `gepa`, `best_of_n` (baseline), `autoresearch` (one `claude --print` session editing `candidate.txt` and calling `eval.sh`) and `meta_harness` (Claude reads the frontier and history files and writes `pending_eval.json`) ([G]/src/gepa/oa/engines/). Ensembles offer sequential, adaptive-sequential-on-plateau, parallel, best-of and vote ([G]/src/gepa/oa/ensemble.py).
- **ComBEE** reflection is map-reduce: shuffle, split into k=⌊√n⌋ groups, make k proposals, then reduce them to one. It only switches on when n≥4, at a cost of k+1 calls ([G]/src/gepa/proposer/reflective_mutation/combee.py, arXiv 2604.04247).
- The frontier type can be `instance | objective | hybrid | cartesian` ([G]/src/gepa/core/state.py#L24). `reflective_dataset_enricher` (PR #447) enables AdaMAST learned failure taxonomies, reported as up to +9.7pp held-out (PR #491, author-reported).
- Open direction: rejection memory (gepa#379, open PR #384), token budget (#272), mutation rate (#277), minibatch-as-LR schedule (#132), per-instance weights (#111), skipping redundant full val evals (#103), pairwise comparison (#128), pluggable coding-agent backend (#459).

**OpenEvolve**
- Config is a single YAML schema ([OE]/configs/default_config.yaml). It covers diff vs full rewrite with `full_rewrite_probability` (#L13-L17; PR #580 mixes rewrites into diff mode), `max_code_length`, and `enforce_evolve_blocks`.
- The population is islands plus MAP-Elites feature bins, with migration interval and rate and elite/explore/exploit ratios (#L98-L151). issue #289 reported islands whose generations never advanced, and PR #200 in Shinka fixed a similar migration-order bug.
- **Cascade eval**: `evaluate_stage1/2/3` with thresholds [0.5, 0.75, 0.9]. A wrong config silently falls back to direct eval; the code now warns ([OE]/openevolve/evaluator.py#L204-L227).
- **Novelty gate**: embedding cosine above `similarity_threshold` sends the candidate to an LLM novelty judge, which can reject it before it enters the archive ([OE]/openevolve/database.py#L1165-L1240).
- `coevolution.py` evolves against frozen cohorts of opponents. `evolution_trace` exports JSONL/HDF5 traces intended for RL training.
- Process-pool parallelism (`process_parallel.py`). Users report pool crashes (#246) and CUDA leaks (#330).

**ShinkaEvolve**
- The async runner has separate pools: `max_proposal_jobs` (deliberately larger than the eval pool), `max_evaluation_jobs` and `max_db_workers` ([SH]/docs/async_evolution.md). PR #198 measures timeouts from eval start, not from proposal start.
- **Cost-aware model bandit** (`AsymmetricUCB`): `final = (1-c)·(reward + UCB) + c·cheapness` ([SH]/docs/bandit_selection.md, `shinka/llm/prioritization.py#L295`).
- A `MetaSummarizer` writes cross-generation recommendations in 3 steps ([SH]/shinka/core/summarizer.py). `SystemPromptEvolver` co-evolves the mutation system prompt ([SH]/shinka/core/prompt_evolver.py). It also has a novelty judge with an async variant.
- Maintainers state the design **assumes low-variance fitness**; noisy copied elites poison islands (Shinka#119). The agent skills (`shinka-setup/run/inspect`) are the user surface ([SH]/docs/agentic_usage.md).

**SkillOpt** (the closest analogue to our skill use-case)
- Each step runs rollout → reflect (failure minibatches, parallel analysts) → aggregate similar patches → **select top-k edits = learning rate** → apply → **gate on the selection split** ([SO]/docs/guide/training-loop.md).
- Edits are bounded add, delete and replace operations. The LR schedule can be cosine, linear or constant (`optimizer.learning_rate→edit_budget`, `lr_scheduler`, [SO]/skillopt/config.py#L154-L164).
- The **rejected-edit buffer** feeds the last steps' failure patterns plus rejected edits and their score before/after back into reflection ([SO]/skillopt/engine/trainer.py#L640-L670,#L1683-L1708).
- At epoch boundaries, a **slow update** compares the previous and current skill on the same samples (improved, regressed, persistent-fail, stable) and writes guidance. A **meta-skill** memory carries across epochs. Gating is `slow_update_gate_with_selection`, **default `false`** ([SO]/configs/_base_/default.yaml#L100), so by default it is force-injected. SkillOpt#22 (closed) flagged the mismatch with the paper.
- Issues: gate drift between paper and code (#22), split leak 125 vs 124 (#158), optimizer-strength fairness (#12), and one-off preferences being Goodharted into rules (#154).

**DSPy**
- SIMBA buckets examples by **max-to-min score gap across sampled programs**, so it focuses on high-variance examples, then appends a rule or a demo ([DS]/dspy/teleprompt/simba.py#L213-L249).
- ReAnchor (experimental) only accepts a parameter change if a **k-fold held-out check** (FOLDS=5) beats the current setting. This discourages gains confined to small slices ([DS]/dspy/teleprompt/reanchor/calibrate.py docstring).
- MIPROv2 scores minibatches with a full eval every `minibatch_full_eval_steps=5` ([DS]/dspy/teleprompt/mipro_optimizer_v2.py#L120-L122). Pain points: context explosion when the minibatch param was ignored (dspy#9059), and GEPA multimodal memory blow-up (#8848).

**ACE implementations**
- Bullets carry `helpful=X harmful=Y` counters. Dedup uses sentence-embedding plus FAISS similarity, then an LLM merge ([ACE]/ace/core/bulletpoint_analyzer.py#L25-L46,#L157-L167).
- Kayba v0.12 added a cross-trace **generalisation gate** (≥3 instances across ≥2 domains before writing a broad skill), an atomicity rule, and evidence-only tagging. It **removed auto-delete at harmful≥3** because heavily used skills pile up harmful tags without being net-negative ([KA]/CHANGELOG.md). v0.13 counts cache reads at 0.1× in the budget.

**verifiers / ART / TRL (RL)**
- The verifiers rubric `Criterion` has worst→best `choices`, weights, and a **`drop`** list. A drop verdict (for example, a task defect) removes the trace rather than scoring 0 ([VF]/verifiers/v1/judges/rubric.py#L57-L100). verifiers also ships a GEPA adapter that bridges sync to async with one event loop and a semaphore ([VF]/verifiers/v1/gepa/adapter.py).
- ART RULER has a judge rank N trajectories *relative* to each other ("relative scoring is easier than absolute"), and retries a malformed ranking once ([ART]/src/art/rewards/ruler.py).
- TRL `multi_objective_aggregation=normalize_then_sum` (GDPO) and `use_adaptive_entropy` (target-tracking exploration) ([trl grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py), main@2026-11-11, L784-L923).

**inspect_ai / promptfoo (eval)**
- Epochs plus reducers (`mean/median/mode/pass_at/at_least/max`, [IN]/src/inspect_ai/scorer/_reducer/reducer.py). Bootstrap and **clustered stderr** ([IN]/src/inspect_ai/scorer/_metrics/std.py#L18-L111). A `cascade(threshold, **scorers)` scorer runs the cheapest stage first and stamps `decided_by` ([IN]/src/inspect_ai/scorer/_cascade.py).
- promptfoo cache-key bugs: a volatile `__evalId` in the key meant no cross-run hits (promptfoo#12132), the MCP config was missing from the key (#7467), and caching collided with `repeat` (#360, #1431).

**LEVI / DGM**
- LEVI: cheap models do most mutations. A **stronger model periodically proposes "paradigm shifts"**. A CVT-MAP-Elites archive over AST features keeps strategies diverse, and there are $/eval/time caps ([LV]/README.md, `levi/pool/cvt_map_elites.py`, `levi/behavior/extractor.py`). It claims to beat GEPA on prompts at under half the rollouts (author-reported).
- DGM parent probability ∝ score × 1/(1+children_count) ([DGM]/DGM_outer.py#L91-L98). Staged small→medium→full eval, with full eval only for top nodes (#L192-L209). `--num_swe_evals` for repeats (#L234).

## 3. Idea list as candidate grid rows
Each M column is the expected effect: Val = valset score, SE = sample efficiency, $ = cost, Gap = train/val gap, Eff = implementation effort (1–5), Ev = evidence. Evidence levels: **S** = shipped in code, **P** = proposed in an issue or PR only, **A** = author-reported benchmark, **T** = thin. NEW = no fit among the 11 current axes. **Sign convention:** each sign is the direction the quantity moves, not good or bad. Val/SE `+` = better. `$` `−` = cheaper. Gap `−` = smaller train/val gap (good). Eff 1 = easiest. **LANDED** = already in our working tree.

| # | Idea | Axis | Val | SE | $ | Gap | Eff | Ev | Sources |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Rejected-proposal memory: per parent+component, feed the last k rejected texts/diffs with before→after scores into reflection | feedback signal | + | ++ | − | 0 | 1 | S (SkillOpt), P (gepa#384) | [SO]/trainer.py#L1683; gepa#379 |
| 2 | **LANDED** (`prescreen`). Exact dedup: hash the candidate; skip eval if seen or identical to its parent | eval hygiene | 0 | + | −− | 0 | 1 | S | [G]/core/engine.py#L624; gepa PR #512 |
| 3 | N proposals per iteration + parallel (candidate×example) eval + TopK/Best selection | **NEW: concurrency / throughput** | 0/+ | 0 | 0 ($), −− wall-clock | 0 | 3 | S | [G]/strategies/proposal_sampling.py; [SH]/docs/async_evolution.md |
| 4 | **LANDED** (`val_eval=halving`, frontier-aware). Staged (cascade) val eval: score a val subset first, abort if below parent − δ *and* the child wins no instance | budget control | 0 | ++ | −− | 0 | 2 | S (OE, DGM, inspect); P (gepa#103) | [OE]/configs/default_config.yaml#L165; [DGM]/DGM_outer.py#L192 |
| 5 | **Partly LANDED** (`paired_se` promotion); repeats still missing. Noise-aware gate: repeat evals for stochastic evaluators (reducer = mean/median), accept only if Δ > k·stderr; re-confirm a new best before crowning it | acceptance rule + eval hygiene | + | − | + | − | 2 | S (inspect reducers), T (Shinka#119 report) | [IN]/scorer/_metrics/std.py; Shinka#119 |
| 6 | **LANDED** (`testset`); an eval-server boundary is only needed for agentic engines. Sealed test split behind an eval choke point; report seed/best test score and train−val−test gap | split discipline | 0 | 0 | + (small) | measured | 2 | S | [G]/oa/eval_server.py; [G]/oa/engines/autoresearch.py docstring |
| 7 | Textual learning rate: cap edit size (ops or diff ratio) with a cosine schedule; re-ask if over cap | edit granularity (+ NEW sub-axis: step-size schedule) | + | + | 0 | − | 3 | S (SkillOpt), P (gepa#277, #132) | [SO]/config.py#L154-L159 |
| 8 | Hard constraint gates separate from score: token/char budget, required sections, "semantic preservation" judge | acceptance rule | 0 | 0 | 0 | − | 1 | S (OE `max_code_length`), P (gepa#272) | [hermes README](https://github.com/NousResearch/hermes-agent-self-evolution#guardrails) |
| 9 | Map-reduce reflection over large minibatches (ComBEE, k=√n groups) | feedback signal | +? | + | + (k+1 calls) | 0 | 2 | S + A (paper) | [G]/proposer/reflective_mutation/combee.py |
| 10 | Learned failure taxonomy enricher: a judge labels each failure with a mined taxonomy and attaches the label to the evaluator feedback | feedback signal | + (≤9.7pp A) | + | + | 0 | 3 | A | gepa PR #491, #447 |
| 11 | ⚠ conflicts with team grid (cheap reflector rejected). Tiered routing: cheap mutator for most steps, strong model every K steps or on plateau ("paradigm shift"); optional cost-aware UCB over models | model routing | 0/+ | 0 | −− | 0 | 3 | S (Shinka UCB), A (LEVI) | [SH]/docs/bandit_selection.md; [LV]/README.md |
| 12 | ⚠ team grid rejects QD archives; only the cheap pre-eval novelty check is proposed. Novelty gate: embedding similarity to the archive above τ sends the candidate to an LLM "is it novel?" check before eval | **NEW: diversity maintenance** | +? | + | − | 0 | 2 | S | [OE]/database.py#L1188; [SH]/core/novelty_judge.py |
| 13 | **≈LANDED** (`counterexample_mining` draws failures of the best; the SIMBA variance criterion differs). High-variance minibatch sampling (SIMBA max-to-min gap) or per-example weights | **NEW: example curriculum** | + | + | 0 | +? (overfit risk) | 2 | S (DSPy), P (gepa#111) | [DS]/teleprompt/simba.py#L213 |
| 14 | Pairwise/relative LLM judging of parent vs child outputs on the same example, for the accept gate | feedback signal / acceptance | + | + | + | 0 | 3 | S (ART RULER), P (gepa#128) | [ART]/rewards/ruler.py |
| 15 | **Half LANDED** (`EvaluationError` excludes failed evals). New part: an evaluator `drop` verdict for *defective examples*, excluded rather than scored 0 | eval hygiene | + | + | 0 | 0 | 1 | S | [VF]/v1/judges/rubric.py#L57 |
| 16 | Epoch-boundary slow update: compare previous vs current best on the same items (improved/regressed) and write guidance, **gated** | **NEW: memory / consolidation** | + | 0 | + | − | 3 | S (configurable; default ungated, [SO]/configs/_base_/default.yaml#L100) | [SO]/docs/guide/training-loop.md |
| 17 | Delta-op edits (ADD/UPDATE/REMOVE bullets with helpful/harmful counters) instead of full rewrites for list-shaped artifacts | edit granularity | + (long docs) | + | − (tokens) | 0 | 4 | S (ACE) | [ACE]/core/bulletpoint_analyzer.py; [KA]/CHANGELOG.md |
| 18 | ⚠ team grid rejects best-of-N as a *method*; keep it as a *benchmark baseline* only. Engine portfolio: best-of-N baseline, plus adaptive-sequential engines on plateau sharing one budget | search strategy (meta) | +? | 0 | 0 | 0 | 3 | S (GEPA), T (OE#100: best-of-N ≈ evolve on one saturated task) | [G]/oa/ensemble.py; OE#100 |
| 19 | Agentic proposer: a coding agent reads frontier/history files and writes candidates (pluggable backend) | **NEW: proposer agency** | +? | ? | + | ? | 4 | S (GEPA meta_harness), P (gepa#459 decoupling) | [G]/oa/engines/meta_harness.py |
| 20 | Child-count-penalised parent selection (score × 1/(1+children)) | selection | 0/+ | + | 0 | 0 | 1 | S | [DGM]/DGM_outer.py#L91 |

## 4. Top 5 for our codebase (ranked by value ÷ effort; excludes everything LANDED)

**1. Rejected-proposal memory (row 1; team grid A-7 "tabu"). Effort 1.**
- This is an upgrade on `prescreen`. Prescreen catches the *same text*; this catches the *same failed idea* in new words.
- `State.rejected: dict[str, list]` keyed by string `f"{parent_idx}:{component}"`, because tuple keys don't serialise to `checkpoint.json`. Each list is capped at 3 entries of `{diff_summary (difflib unified, ≤40 lines), parent_mb, child_mb, reason}`, where `reason` is one of `minibatch | halving | promotion | prescreen_*`.
- Push an entry on every `candidate_rejected` path. Pass it to `proposer.propose(rejected=...)` only if `_accepts_kwarg` says the proposer takes it (same pattern as `operator`). `build_reflection_prompt` renders "Tried from this parent; did NOT help: …", the same shape as [SO]/trainer.py#L640-L670.
- Evidence: shipped in SkillOpt; open GEPA PR #384 for gepa#379. Measure the rejection rate per parent before vs after.

**2. Parallel evaluation for wall-clock (row 3). Effort 3.**
- `Config.eval_workers=1` by default. A `ThreadPoolExecutor` maps `(payload, example)` pairs inside the minibatch loop, `_evaluate_candidate`, the halving rungs and `_evaluate_test`.
- Lock the `_cache` dict and the `state` counters (`record_evals`, `cache_hits`, `failed_evals`), or write them from the main thread after `as_completed`.
- Keep N=1 proposal per iteration by default; the team grid flags the winner's curse for N>1 (O-C5).
- Truncate each dispatched batch to the remaining `max_metric_calls`, so the budget can't overshoot.
- Add invariant tests: `total_evals + cache_hits` equals the number of requested pairs, and cost totals match serial mode. Parallel bookkeeping bugs are common (§5).

**3. Stochastic-evaluator hygiene: repeats instead of cache-off (row 5 remainder). Effort 2.**
- Today the config says to *disable* the cache for stochastic judges, which throws away all reuse.
- Add `Config.eval_repeats=1` (suggest 3 for `llm_judge`). The cache key gains `repeat_idx`, so repeats are cached but not collapsed (promptfoo#360/#1431). Reduce by mean and pass per-repeat spread into ASI.
- `paired_se` can then use within-example variance as well.
- Make `llm_judge.cache_identity` cover model, effort, criteria and the prompt-template hash (promptfoo#7467/#12132).
- Before `new_best_found`, re-evaluate a stochastic new best with repeats (Shinka#119).

**4. Textual learning rate + size gate (rows 7, 8; complements `section_edit` and team A-6 feasibility). Effort 2.**
- `Config.max_edit_ratio=None` (e.g. 0.35) with `edit_schedule: constant|cosine` down to `min_edit_ratio`. The budget goes into the proposer prompt.
- Post-check `1 − difflib.SequenceMatcher(None, old, new).ratio()`. Over the cap means one re-ask, then a free rejection with reason `edit_too_large`, which goes into memory #1.
- `max_chars` and `required_substrings` are checked before any eval.
- Evidence: shipped in SkillOpt (`learning_rate→edit_budget`, `lr_scheduler`); proposed in gepa#277/#272/#132. The OpenEvolve diff-vs-rewrite history (#241, PR #580) argues for a cap, not for forbidding rewrites.

**5. Pairwise judge mode for the accept gate (row 14). Effort 3.**
- With `llm_judge`, judge parent and child outputs on the same example *in one call*: "which better meets the criteria, by how much", randomising A/B order. That gives Δ directly.
- Use it only at the minibatch gate, where Δ is what's needed. Keep absolute scores for the val frontier.
- Expected effect: less judge noise at the gate. Costs about 1 call per example instead of 2, if outputs are cached.
- Evidence: ART RULER ships relative judging ("relative scoring is easier than absolute"); requested in gepa#128 and inspect#3994. No head-to-head result versus absolute gating found, so A/B it.

Runners-up: ComBEE map-reduce reflection when `counterexample_mining` or a larger minibatch makes n≥9 (row 9, effort 2), the evaluator `drop` verdict (row 15, effort 1), and a gated epoch-boundary slow update (row 16, effort 3).

## 5. Pain points from issues to avoid
- **Cache-key correctness.** Volatile fields in the key mean zero hits (promptfoo#12132). Missing config in the key means wrong hits (promptfoo#7467; GEPA's split-keyed cache exists for this reason). Caching stochastic evals defeats repeats (promptfoo#360). Our `llm_judge.cache_identity` must cover model, criteria, effort and prompt template.
- **Noisy false elites.** A lucky score copied without re-evaluation locks search for 50+ generations (Shinka#119). The maintainers' answer was "reduce variance in the evaluator", so document this assumption.
- **Reward hacking through weak checkers.** Shinka#206: two "formal" leaderboard winners passed a bounded 16-cycle check but were not equivalent. 5 of 27 winners silently fell back to a weaker testbench on timeout. Fallbacks must be flagged in diagnostics, never scored the same. Related: the target-value leak debate (openevolve#117) and proxy Goodharting (SkillOpt#154).
- **Edit-format brittleness.** Weaker models fail SEARCH/REPLACE diffs, so "no valid diffs" stalls evolution (openevolve#241). Mitigation: fall back to a full rewrite (openevolve PR #580 mixes both). Also normalise CRLF (openevolve PR #594, GEPA PR #524).
- **Gate semantics drift and leakage.** Paper says the slow update is gated; code made it configurable, but the default is still ungated (`slow_update_gate_with_selection: false`, SkillOpt#22, closed). A test item leaked into val (SkillOpt#158). Unfair optimizer strength in baselines (SkillOpt#12). Lesson: put the gate mode in `result.json` and hash split manifests.
- **Agent-backend coupling.** Engines hardwired to the `claude` CLI and `~/.claude` mounts (gepa#459). We already have `llm/claude_cli.py` + `codex_cli.py`; keep any agentic proposer behind that interface.
- **Context and memory blow-ups.** An ignored minibatch param exploded context (dspy#9059); multimodal reflection exhausted RAM (dspy#8848); process pools crashed or leaked GPU memory (openevolve#246, #330). Cap the feedback bytes per entry (openevolve has `max_artifact_bytes: 20480`).
- **Bookkeeping drift in parallel or island code.** Islands never incremented (openevolve#289); migrants were moved before selection (Shinka PR #200); island copies double-counted cost (Shinka PR #215); `best_program_info` disagreed with the top score (openevolve#135). Parallelism (Top-5 #2) needs invariant tests on counts and costs.
- **Evolution vs best-of-N.** One user ablation (openevolve#100) found the difference negligible on a saturated task. Always ship a best-of-N baseline in benchmarks, and report median ± spread rather than best-of-N alone.
