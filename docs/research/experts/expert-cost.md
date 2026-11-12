# Expert panel: Infra / Cost engineer — optimize-anything

Metric: **$ per run** and **$ per valset point gained** (`$/pt = run_cost / (best_val − seed_val)`, val score in 0–100 points).
Date of verification: 2026-11-11. Repo read: `engine/src/optimize_anything/{core/engine.py, evaluators/llm_judge.py, proposer/claude_proposer.py}`.

## 0. Findings in the current code that cost money or break (fix first)

| # | Where | Problem | $ / correctness effect |
|---|---|---|---|
| F1 | `claude_proposer.py` `temperature=1.0` | Opus 5.5 rejects `temperature` with 400; Sonnet 5.5 / Haiku 5.5 reject non-default values. Swapping the model ID alone breaks every proposal. | Run fails |
| F2 | `llm_judge.py` `except Exception: return 0.0` | 429/5xx/timeouts become score 0. A rejected child or a zeroed valset score corrupts the Pareto frontier, and the spent calls are wasted. Gets worse once calls run in parallel. | Silent waste + wrong selection |
| F3 | `llm_judge.py` `max_tokens=2048`, regex `SCORE:` | On 5.5 models thinking is adaptive, on by default, and counts against `max_tokens`. Sonnet 5.5 defaults to effort `high`, so long thinking can truncate before `SCORE:` and the parse falls back to 0.0. | Silent zeros |
| F4 | `engine.py` gate vs `_evaluate_candidate` | The gate scores one component's text, but the valset scores `"\n".join(components.values())`. These are different strings, so a cache keyed on text can't share them, and gate and val measure different things. | Lower cache hits, gate/val mismatch |
| F5 | `engine.py` child gate loop | Feedback from the child's gate evals is thrown away, but full ~800-token rationales are still generated and paid for. | ~30% of judge output tokens wasted |
| F6 | no `usage` read anywhere | No token or cost accounting, so you can't see $/run or set a $ budget. | Can't measure the metric |

## 1. Verified model table (official sources; per 1M tokens)

| Provider | API model ID | $ in | $ out | Cache read | Cache write 5m / 1h | Batch | Notes | Source |
|---|---|---|---|---|---|---|---|---|
| Anthropic | `claude-fable-5-2` | 8.00 | 40.00 | 0.20 (0.025×) | 10 / 16 | −50% | Thinking always on; temperature/top_p/top_k → 400; default effort high | [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [models](https://platform.claude.com/docs/en/about-claude/models/overview) |
| Anthropic | `claude-opus-5-5` | 4.00 | 20.00 | 0.20 (0.05×) | 5 / 8 | −50% ($2/$10) | Thinking can't be disabled; **default effort `medium`**; sampling params → 400; no prefill; forced tool_choice → 400 | same |
| Anthropic | `claude-sonnet-5-5` | 2.00 | 10.00 | 0.10 (0.05×) | 2.50 / 4 | −50% ($1/$5) | Adaptive thinking, default effort `high`; non-default temperature → 400; `thinking:{type:"between_tools"}` turns up-front thinking off | same |
| Anthropic | `claude-haiku-5-5` | 0.10 (≤100K prompt) | 0.50 | 0.01 | 0.125 / 0.20 | −50% ($0.05/$0.25) | Prompts >100K tokens: $0.50/$2.50; thinking on by default, default effort `medium` | same |
| Anthropic | `claude-opus-4-6` (current proposer default) | 5.00 | 25.00 | 0.50 | 6.25 / 10 | −50% | Legacy; old tokenizer (newer tokenizer gives ~30% more tokens for the same text) | same |
| Anthropic | `claude-sonnet-4-6` (current judge default) | 3.00 | 15.00 | 0.30 | 3.75 / 6 | −50% | Legacy; temperature allowed only on the old API version (400 under `anthropic-version: 2026-10-21`) | same |
| OpenAI | `gpt-6.1-sol` | 2.00 | 10.00 | 0.10 | n/a (automatic) | −50% (Batch or Flex) | Same price as Sonnet 5.5 | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI | `gpt-6.1-terra` | 1.00 | 5.00 | 0.05 | n/a | −50% | | same |
| OpenAI | `gpt-6-luna` | 0.10 | 0.50 | 0.01 | n/a | −50% | Same price as Haiku 5.5 | same |
| OpenAI | `gpt-6-astra` | 10.00 | 50.00 | 1.00 | n/a | −50% | | same |

Cross-cutting facts, all verified:
- **Minimum cacheable prefix: 512 tokens** on Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.2. It was 4096 on Opus 4.6. A shorter prefix silently doesn't cache. Top-level automatic `cache_control` exists. ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))
- **Batch discount stacks with the cache multipliers.** Fast mode is not available with Batch; Opus 5.5 fast mode costs $8/$40. ([pricing](https://platform.claude.com/docs/en/about-claude/pricing))
- **Temperature:** API version `2026-10-21` rejects `temperature`/`top_p`/`top_k` on every model; Opus 5.5 and Fable 5.2 reject them on every version. A deterministic temperature-0 judge is no longer an option on Claude. (Anthropic API-version migration guide: https://platform.claude.com/docs/en/api/migration-2026-10-21)
- **Structured outputs:** GA through `output_config.format` (json_schema) on all of the 5.5 / 5.2 models, with no beta header. Not supported: numeric `minimum`/`maximum` constraints and recursive schemas. The old `output_format` parameter is deprecated. ([structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs))
- **OpenRouter list prices quoted in the brief match the official prices** for opus-5.5, sonnet-5.5, haiku-5.5, gpt-6.1-sol and gpt-6-luna. I didn't check gemini-3.8-flash. OpenAI temperature support on GPT-6: **?** (pricing page only).

## 2. Source digest: routing, caching, parallelism, real costs

| Source | Finding (numbers) | Bearing |
|---|---|---|
| [GEPA paper, arXiv 2507.19457](https://arxiv.org/html/2507.19457v2) App. E.2/E.3 | All modules use the same model (Qwen3-8B or GPT-4.1-mini). The text implies the reflector is the task model but doesn't say so explicitly. All Table 2 GEPA runs on GPT-4.1-mini cost **$86 in total**. Rollouts per benchmark: 1.8k–7k vs GRPO's 24k. "The majority of GEPA's rollout budget is spent on validation." | A mid-tier reflector can work. **Valset evals dominate**, so attack those first. |
| [Decagon, GEPA in production](https://decagon.ai/blog/optimizing-gepa-for-production) | GPT-4o-mini as reflector: prompt left essentially unchanged. GPT-4.1/5.2/Sonnet/Opus: +5–6%. Reflector is **5–10% of total cost**, ~10–20 reflection calls per run. A 20-example train set beat 50 at **2.5× cheaper**; 10 examples was −5% at 6× cheaper; 500 was −2% at 10× the cost. | Don't downgrade the reflector to small-model tier. Small data plus a frontier reflector is the cheap sweet spot. |
| [gepa-ai/gepa `api.py`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py) | `cache_evaluation` caches (candidate, split, example) and keeps valset in a separate namespace. Has `max_reflection_cost` (a USD stopper over a `TrackingLM` that accumulates `total_cost` and token usage) and `sampling_strategy` (`IndependentSampling(n)`, `PxNSampling`) for parallel proposals. `cache_evaluation` defaults to False. | Reference design for cache key, cost tracker and $ stop. |
| [LEVI](https://ttanv.github.io/levi/) | More than 90% of mutations go to small models (Qwen3-30B-A3B, MiMo-Flash); a larger model handles "paradigm shift" calls on a fixed interval or on stagnation. Cost **$4.50 vs $15–30** per problem for GEPA/OpenEvolve/Shinka on GPT-5/Gemini-3-Pro, winning 6 of 7. On prompt optimization vs GEPA: 62.0 vs 61.3 with **44% of the rollouts**. Async producer/evaluator loop. | Cascades pay off **for the mutator in code search**. Caveat: the comparison is against best published baselines, not matched reruns. |
| [ShinkaEvolve, arXiv 2509.19349](https://arxiv.org/abs/2509.19349) + [review](https://liner.com/review/shinkaevolve-towards-openended-and-sampleefficient-program-evolution) | SOTA circle packing in ~150 samples. A UCB1 bandit over an LLM ensemble gives only **slight** gains; parent sampling and novelty rejection matter more. | Skip bandit routing; selection logic beats model mixing. |
| [OpenEvolve README](https://github.com/algorithmicsuperintelligence/openevolve) | Per iteration: o3 $0.15–0.60, Gemini-2.5-Flash $0.01–0.05. Recommends cascade evaluation (cheap checks first). One user reports a $9 full run ([blog](https://dmytro.bearblog.dev/anthropic-performance-takehome-openevolve/)). The [ADRS paper](https://arxiv.org/html/2510.06189v1) reports "a few to tens of dollars" per case study. | Real runs cost $5–30. Our 30-iteration run should land well under $10. |
| [Dria, open evolutionary agents](https://dria.co/blog/towards-open-evolutionary-agents) | Parallel evaluation is "essential". Uses a staged evaluator (quick validation → performance → full). | Parallelism buys wall clock; staging buys $. |

## 3. Candidate designs on the axes

Fixed for all: 6. artifact scope = single text artifact (~2k tokens), components round-robin.

| ID | 1 Search | 2 Selection | 3 Feedback | 4 Routing (propose / gate-judge / val-judge) | 5 Budget | Extra axis: caching / accounting |
|---|---|---|---|---|---|---|
| A0 Baseline | 1 mutation per iteration | Pareto | Full rationale on every eval | Opus 4.6 / Sonnet 4.6 / Sonnet 4.6 | Iterations + metric calls | None |
| A1 Refresh | same | same | same | Opus 5.5 (effort medium) / Sonnet 5.5 (low) / Sonnet 5.5 (low) | same | Usage ledger only |
| B Lean judge | same | same | Rationale only for **parent** gate evals; score-only JSON for child gate, val and seed | as A1 | + `max_cost_usd`, early stop | Memo cache + prompt-cache layout + async parallel |
| C Cheap judge | same | same | as B | Opus 5.5 / Haiku 5.5 / Haiku 5.5, behind a calibration gate | as B | as B |
| D Cheap preset | same | same | as B | Sonnet 5.5 / Haiku 5.5 / Haiku 5.5 | as B | as B |
| E Batch val | as B, but val evals go through the async Batch API | same | as B | as A1 | as B | Batch −50% on val |
| F Quality preset | as B | same | as B | Fable 5.2 (effort high) / Sonnet 5.5 / Sonnet 5.5 | as B | as B |

## 4. Per-candidate estimates

**Worked-estimate assumptions** (30 iterations, minibatch 3, valset 10, 30% of children accepted):
- Judge calls = seed 10 + 30×(3 parent + 3 child) + 9 accepted × 10 val = **280**: 90 parent-gate, 90 child-gate, 100 val/seed.
- Judge input = criteria 0.5k + candidate 2.0k + instructions 0.1k + example 0.5k = **3.1k tokens**.
- Judge output = rationale **0.8k** + thinking ~0.6k at effort low, or score-only **0.15k** + 0.6k thinking.
- Proposer: 30 calls, **6k in** (artifact plus 3 rationales plus objective), **5k out** (2k artifact + ~3k thinking).
- Legacy 4.6 models get ÷1.3 tokens to account for the older tokenizer.
- Prompt caching (B and later): per candidate, the 2.6k prefix (criteria + candidate + instructions) is written once and read by that candidate's remaining examples. Memo hit rate assumed 10%.

| ID | Judge $ | Proposer $ | **$/run** | Valset effect vs A1 | Sample efficiency | Train/val gap risk | Effort (1–5) | Evidence |
|---|---|---|---|---|---|---|---|---|
| A0 | 280×(2.4k×$3 + 0.8k×$15) = **5.38** | 30×(4.6k×$5 + 1.6k×$25) = **1.89** | **≈ $7.3** | Baseline | Baseline | Med (3-example gate is noisy) | 0 | Code |
| A1 | 280×(3.1k×$2 + 1.4k×$10) = **5.66** | 30×(6k×$4 + 5k×$20) = **3.72** | **≈ $9.4** | + (newer models; size of gain **?**, no eval exists) | = | = | 1 (IDs plus F1/F3) | Pricing pages |
| B | Parent-gate 90×$0.0173 + child-gate 90×$0.0108 + val 100×$0.0094 = 3.48 ×0.9 memo = **3.13** | **3.72** | **≈ $6.9** (−27% vs A1) | ≈ 0, possibly + (no silent zeros, consistent cached scores) | + (failed calls retried, not scored 0) | Lower: the memo removes judge-noise double counting | 3 | Pricing, GEPA cache design |
| C | ≈ 3.48/20 ≈ **0.18** | **3.72** | **≈ $3.9** (−59%) | **?** Depends on judge fidelity. A weak judge invites Goodhart. | = if calibrated | **Higher**: a cheap judge is easier to game | 3 | No direct study for a GEPA judge. Industry pattern: cheap eval, frontier reflection. |
| D | **0.18** | 30×(6k×$2 + 5k×$10) = **1.86** | **≈ $2.0** (−79%) | **?** Decagon: Sonnet-class reflector ≈ Opus-class; small models fail | ≈ | Higher (C plus weaker proposals) | 3 (presets only) | Decagon, LEVI (mutator) |
| E | B minus 50% of val ≈ **2.71** | **3.72** | **≈ $6.4** (−7% vs B) | = | = | = | 4 (async loop, polling) | Pricing; latency up to hours |
| F | **3.13** | 30×(6k×$8 + 5k×$40) = **7.44** | **≈ $10.6** | Likely + on hard artifacts; **?** at a 2k-token artifact | + | = | 1 | Model docs |

**$/pt:** can't be filled without a measured gain. The ledger in §5 records it per run. At ~$7 per run, a +5-point gain costs ~$1.4/pt (B) vs ~$0.8/pt (D), *if* D reaches the same gain. Decagon reports small-model reflectors gaining roughly 0 points, which makes $/pt infinite.

What dominates the cost: after B, the **proposer is 54% of the run**, and ~80% of that is output (thinking plus a full rewrite). Next levers, in order: proposer effort (`low`/`medium` sweep), then diff/patch output instead of full rewrite (out of scope here; the search-strategy expert owns it).

## 5. Top 3 upgrades by $/run and $/pt

### #1 Cost ledger, USD budget, and the fail-closed fixes (F1, F2, F3, F6). Effort 2. Quality risk ~0. Required before anything else.
```python
@dataclass(frozen=True)
class Price:                       # per 1M tokens, dated, with source URL
    inp: float; out: float; cache_read: float; cache_write_5m: float
PRICES = {  # verified 2026-11-11, platform.claude.com/docs/en/about-claude/pricing
  "claude-opus-5-5":  Price(4.0, 20.0, 0.20, 5.0),
  "claude-sonnet-5-5":Price(2.0, 10.0, 0.10, 2.5),
  "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),   # <=100K-token prompts
  "claude-fable-5-2": Price(8.0, 40.0, 0.20, 10.0),
}
@dataclass
class CallRecord:   # one line of run_dir/costs.jsonl
    ts: float; iteration: int; role: Literal["propose","judge_gate_parent","judge_gate_child","judge_val","judge_seed"]
    model: str; candidate_hash: str; example_id: str | None
    input_tokens: int; output_tokens: int; cache_read_tokens: int; cache_write_tokens: int
    batch: bool; memo_hit: bool; latency_s: float; cost_usd: float; error: str | None
```
- Compute `cost_usd` from `response.usage` (`input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens`); apply ×0.5 for batch. Unknown model: log a warning and record `cost_usd=None`; never guess a price.
- Emit a `cost` field on every event. `OptimizationResult` gains `total_cost_usd`, `cost_by_role`, `cost_by_model`, `seed_val`, `best_val`, `usd_per_point`.
- Add a `BudgetStopper(max_cost_usd)`, checked before each proposal using a projected next-iteration cost (moving average).
- Add a `--estimate` dry run: `count_tokens` on the seed prompt × the call plan, printed before spending anything.
- **Errors:** retry 429/5xx/connection errors through SDK `max_retries` (raise to 4). A call that still fails, or whose score doesn't parse, returns `None`, not 0. The engine then re-runs it once; if it fails again, the candidate is **marked unevaluated and skipped**, never added to the frontier with zeros.
- Drop `temperature` from the proposer. Judge: set `max_tokens` ≥ 8000 and `output_config.effort="low"`.

### #2 Lean judge: memo cache, prompt-cache layout, score-only mode, structured output, async. Effort 3. −35–45% judge $.
- **Memo key:** `sha256(canon_json({"ev": evaluator.fingerprint, "cand": sha256(text), "ex": sha256(canon_json(example)), "split": "train"|"val", "k": sample_idx}))`.
  - `evaluator.fingerprint` = class + model + effort + sha256(criteria) + max_score + `PROMPT_VERSION` + mode (feedback or score). Changing any of these busts the cache, as it should.
  - Store in SQLite at `run_dir/evals.sqlite`, with an optional shared `~/.cache/optimize-anything/`. Value = (score, feedback, usage).
  - Fix F4 first: hash exactly the string the judge sees.
  - Like GEPA, give val its own namespace.
  - Bonus: resuming a run after a crash re-pays $0 for finished evals.
- **Prompt layout:** put `system`/`user` prefix = criteria + instructions + candidate, then `cache_control` (≥512 tokens: met at 2.6k), then the example last. Fire the first example's call, await its usage, then fan out the rest. Concurrent calls before the first write completes miss the cache.
  - Verify with `cache_read_input_tokens > 0`.
  - 5-minute TTL, not 1 hour. All examples for one candidate finish within seconds.
- **Two modes, same model and effort:** `feedback` (structured `{analysis, actionable_feedback, score}`) only for the **parent** gate evals that feed reflection. `score` (`{score: int}`, ≤ 200 visible tokens) for child gate, val and seed. Thinking stays on in both, so the score distribution shouldn't shift. **Validate this on the seed (same scores ±0.5 across 10 examples) before shipping.**
  - Structured output removes regex parse failures (F3). Range-check 0…max in code, since the schema can't express `minimum`/`maximum`.
- **Async:** `AsyncAnthropic` + `asyncio.Semaphore(8)`. Run all minibatch and valset calls concurrently. No $ change, but roughly 6–10× less wall clock.

### #3 Default routing table, presets, and a calibration gate for cheap judges. Effort 2.

| Role | `default` | `cheap` | `quality` |
|---|---|---|---|
| Proposer / reflector | `claude-opus-5-5`, effort `medium`, adaptive (no temperature) | `claude-sonnet-5-5`, effort `medium` | `claude-fable-5-2`, effort `high` |
| Judge (gate and val; **always the same model for both**) | `claude-sonnet-5-5`, effort `low` | `claude-haiku-5-5`, effort `low` | `claude-sonnet-5-5`, effort `medium` |
| Est. $/run (§4 assumptions) | ≈ $6.9 | ≈ $2.0 | ≈ $10.6 |

- Before a run whose judge differs from `default`, run a **calibration gate**: score the seed plus 2 cheap perturbations (truncated, criteria-violating) on 10 examples with both the chosen judge and Sonnet 5.5. Require Spearman ≥ 0.8 and correct ordering of the perturbations, otherwise fall back to the default judge with a warning. Costs ≈ $0.35 once.
- Read model IDs and prices from one `models.toml`, so the next model refresh is a config change, not a code change (the 4.6 IDs that went stale are why this is needed).

## 6. Advise AGAINST

1. **Haiku/luna-class reflector as a default.** It's 5–10% of cost, and Decagon saw small reflectors produce ~0 gain, so $/pt goes to infinity. Save on the judge, not the reflector.
2. **Different judges for gate and val** (e.g. a Haiku gate with a Sonnet val). Scores aren't comparable, so the gate accepts children the val judge disagrees with and pays for val evals that go nowhere. Use one judge model; vary only output length.
3. **Batch API inside the loop.** It saves ~7% of a $7 run, and each iteration can block for up to hours. Offer it only as an `--overnight` mode for valset re-scoring or final multi-sample evals.
4. **A multi-provider SDK abstraction justified by price.** gpt-6-luna equals Haiku 5.5 and gpt-6.1-sol equals Sonnet 5.5 at list price. The only good reason is cross-family judging to reduce self-preference bias, and that should be a later, measured choice.
5. **Bandit/ensemble model routing** (ShinkaEvolve's ablation shows only slight gains), **fast mode** (2× price, and no Batch), **a 1-hour cache TTL** (calls are seconds apart, so the 2× write cost buys nothing), and **context editing** (it fights prompt caching).
6. **Temperature-0 "deterministic" judges.** They return 400 on the new models. Get consistency from the memo cache, and from averaging k samples only where the gate margin is small.
7. **Treating LEVI's 3–7× savings as transferable to prompt artifacts.** That's a code-search result against unmatched baselines; for text the reflector-quality evidence (Decagon) points the other way.
