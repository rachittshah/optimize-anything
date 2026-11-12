# Optimization Researcher: optimize-anything upgrade (metric = held-out valset score)

Date 2026-11-11. Every number below links to its source. "unverified" means I couldn't confirm it from a primary source.
**Lens:** the reward is *held-out* score, so I weight generalization evidence (val→test gap, prompt length, noise-aware selection) above raw search power.

## 0. Codebase facts that change the ranking (read from `engine/src/optimize_anything/core/`)
1. **The two evaluations score different things (bug).** The minibatch gate scores `evaluator.evaluate(parent.components[component_name], ex)`, which is one component on its own (engine.py:113, 145). Full-valset eval scores `"\n".join(components.values())` (engine.py:191). For a multi-component artifact, acceptance is decided on artifact A while the frontier and the best pick track artifact B.
2. **Dominated programs are never pruned.** `pareto.py:select_candidate` samples in proportion to raw frequency (its comment admits this is "simplified"). GEPA prunes them first ([gepa_utils.remove_dominated_programs](https://github.com/gepa-ai/gepa/blob/main/src/gepa/gepa_utils.py)).
3. **The valset does triple duty:** frontier, parent selection, and the final `best_candidate()` argmax. Whatever is reported as "best score" is the maximum of N noisy valset estimates, which is optimistically biased. Unless a separate test split exists, there is no held-out number.
4. **Strict `>` on a 3-example sum.** With binary or 0-10 judge scores, ties are frequent and get rejected. Only one proposal per iteration, one proposer model, and a judge that makes one call parsed by a regex (`SCORE: X/10`).

## 1. Source digest
| # | Source | Mechanism / numbers that matter |
|---|---|---|
| S1 | GEPA, [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) (v2 Feb 2026, ICLR'26 oral); [HTML v2](https://arxiv.org/html/2507.19457v2) | Reflective mutation + Pareto selection. **Pareto sampling vs SelectBest: +12.44 vs +6.05** aggregate gain (Qwen3-8B, Table 3); BeamSearch(N=4) +5.11. **Merge is mixed:** Qwen3-8B 54.85 → 52.40 (IFBench 38.61 → 28.23, PUPA 91.85 → 86.26); GPT-4.1-mini 65.22 → 66.36. Prompts are up to 9.2x shorter than MIPROv2's; higher-scoring optimizers produced shorter prompts. Evolved instructions show a smaller val→test gap than few-shot (Fig 16, no number given). A full valset eval runs only if the minibatch improves (Alg. 1). Most of the rollout budget goes to validation: 79–737 train rollouts to reach best. Headline "35x fewer rollouts" vs "up to 19%/20%" is internally inconsistent. |
| S2 | gepa-ai/gepa source, [v0.1.4 2026-07-15](https://github.com/gepa-ai/gepa/releases) | Options shipped: `ParetoCandidateSelector`, `CurrentBest`, `EpsilonGreedy`, **`TopKParetoCandidateSelector`** ([candidate_selector.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/candidate_selector.py)). `StrictImprovementAcceptance` / **`ImprovementOrEqualAcceptance`** ([acceptance.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/acceptance.py)). Proposal sampling `SingleMutation` / `SameParent(n)` / `Independent(n)` / **`PxNSampling`** ([proposal_sampling.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/proposal_sampling.py)) plus `BestImprovement` / `TopKImprovements` filters. Component selectors: RoundRobin / All. Merge = ancestor-diff crossover with `val_overlap_floor=5` ([merge.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/merge.py)). Frontier types instance / objective / hybrid / cartesian ([DSPy GEPA docs](https://dspy.ai/current/api/optimizers/GEPA/overview/)). |
| S3 | optimize_anything paper, [arXiv 2605.19633](https://arxiv.org/html/2605.19633) (CAIS'26) | Same API, three modes. **ASI ablation:** reached val 0.80 in ~100 rollouts with SI vs ~600 without; **test 86.32 vs 82.5**. Refiner fixes malformed output before eval. Content-addressed eval cache. Proposer: GPT-5-nano is 90% cheaper but reaches 2.512 vs 2.636 (circle packing). **Evaluator cost dominates; reflection is cheap.** ARC-AGI val 93.5% vs test 89.5% (shows the gap). Multi-task with unrelated tasks hurts: 2.6360 → 2.5973. |
| S4 | RLMOpt, [arXiv 2608.10471](https://arxiv.org/html/2608.10471v1) (Aug 2026) | Held-out mean **0.610 vs GEPA-light 0.589**; won 9 of 11 seed-benchmark pairs; never finished below its seed prompt (GEPA did, twice). Used **135–500 rollouts vs 640–840**. Harness gates: **per-field regression floor 0.05, significance gate 1.65 SE**. GEPA overfit on Chia (val 0.636 > 0.593 but test 0.562 < 0.568). **A larger budget gave lower held-out accuracy** in its synthetic diagnostic. |
| S5 | SEPO, [arXiv 2608.28067](https://arxiv.org/abs/2608.28067) (Aug 2026) | **Localized typed edits instead of full rewrites**, plus a per-edit log of which examples were fixed or broken, fed to later proposals. **+3.1pp (Llama-3.1-8B), +2.2pp (Qwen3-8B) over GEPA** on a 14-task held-out suite; 2.9M vs 4.1M tokens; prompts more than 5x shorter. Abstract only; ablations unverified. |
| S6 | p1, [arXiv 2604.08801](https://arxiv.org/html/2604.08801) (Apr 2026) | Optimizes on the K=2–4 training prompts with the **highest between-candidate variance**. AIME25 54.01 vs GEPA 46.87 (base 47.03). GEPA's prompt held problem-specific guidance ("memorization"). **Loses on homogeneous IFBench** (37.41 vs GEPA 39.12). K=1 overfits. |
| S7 | ACE, [arXiv 2510.04618](https://arxiv.org/abs/2510.04618) (v3 Mar 2026, ICLR'26) | Playbook built by Generator → Reflector → Curator with incremental updates against "context collapse" from repeated rewriting. +10.6% agents, +8.6% finance. Delta-update details and cost figures: unverified from the abstract. |
| S8 | ShinkaEvolve, [arXiv 2509.19349](https://arxiv.org/html/2509.19349) (Sep 2025, background) | Embedding novelty rejection at **cos > 0.95**, then an LLM judge. UCB1 bandit over proposer LLMs with reward `exp(max(r−r_b,0))−1`. Parent weight `σ(λ(F−median))·1/(1+offspring)`, λ=10. SOTA circle packing in ~150 samples. Ablation values not in the text; the paper says "significantly" in one place and "slightly" in another. Single-task code search only, no held-out test. |
| S9 | OpenEvolve ([repo](https://github.com/algorithmicsuperintelligence/openevolve)), AlphaEvolve ([2506.13131](https://arxiv.org/abs/2506.13131), background) | MAP-Elites islands and cascade eval (cheap stage, then full). Under matched GPT-5.1, OA beat OpenEvolve: 2.63598 in 63 evals (~$3.18) vs 2.6307 at 200 iterations (~$6.85) (S3). |
| S10 | DSPy SIMBA ([docs](https://dspy.ai/current/api/optimizers/SIMBA/)); MIPRO ([2406.11695](https://arxiv.org/abs/2406.11695), background); TextGrad ([2406.07496](https://arxiv.org/abs/2406.07496), background) | SIMBA: bsize 32, 6 candidates per step, picks examples with the largest max−min score across sampled trajectories, then appends a rule or demo. MIPRO: surrogate over instruction+demo combinations, up to +13%. GEPA beats MIPROv2 by >10% (S1). TextGrad: GPQA 51 → 55%. |

## 2. Candidate designs (points on the axes)
Axis 7 (new) = **acceptance/promotion rule**. Axis 8 (new) = **edit granularity**.

| ID | Candidate | 1 search strategy | 2 selection | 3 feedback signal | 4 model routing | 5 budget control | 6 artifact scope | 7 accept/promote (new) | 8 edit granularity (new) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | Current | reflective mutation | Pareto freq., no pruning | ASI + 1-call judge | 1 Claude for all roles | fixed iterations/evals | round-robin component (eval mismatch) | strict > on 3; argmax val | full rewrite |
| C1 | **GEPA-faithful core** | reflective mutation | Pareto + dominance pruning (opt. TopK-Pareto) | ASI | same | fixed + eval cache | multi-comp, **consistent eval** | strict > (≥ as opt-in flag) | full rewrite |
| C2 | **Noise-aware promotion + split val** | C1 | C1 for parents; SE-gated best on select-split | ASI + judge averaged over k samples | same | early stop on stall | C1 | promote only if Δ > 1.65·SE, per-ex floor −0.05 | full rewrite |
| C3 | **Edit-based proposer + edit ledger** (SEPO/ACE-style) | reflective *localized* edits | C1 | ASI + per-edit fixed/broke ids | strong proposer | fixed | C1 | C2 | typed diff edits |
| C4 | GEPA merge on | reflective + crossover | Pareto pruned | ASI | same | ≤5 merges | **needs ≥2 components** | merge ≥ max(parents) on overlap | swap components |
| C5 | Parallel PxN + best-of-N | reflective | Pareto | ASI | same | P×N per iteration, threads | C1 | BestImprovement | full rewrite |
| C6 | Shinka-style novelty + model bandit | evolution | weighted sigmoid + islands | ASI | UCB1 over {Haiku, Sonnet, Opus} | fixed | whole program | novelty reject cos > 0.95 | full rewrite |
| C7 | Agentic RLM optimizer (RLMOpt) | agent writes search code | Pareto polish | compressed failure evidence | strong agent + sub-LMs | agent-chosen, ≤B | multi-comp | 1.65 SE + 0.05 floor | agent-chosen |
| C8 | High-variance minibatch sampler (p1/SIMBA) | C1 | C1 | ASI | same | same | C1 | C1 | full rewrite |

## 3. Scores per candidate (vs C0)
Effort: 1 = hours, 5 = weeks. Sample efficiency = held-out gain per evaluator call.

| ID | Valset (held-out) effect | Sample efficiency | $/run | Train/val-gap risk | Effort | Evidence |
|---|---|---|---|---|---|---|
| C1 | **+ (likely the biggest per hour of work).** The eval-mismatch fix is a correctness fix; its size depends on the artifact (?). Pareto vs SelectBest +6.4pp in S1 T3, but **pruning alone is "?"** because no source ablates pruned vs unpruned | + from pruning (unpruned frequency over-samples redundant programs; magnitude ?); ± if ≥ flag on (each tie costs a full-val eval) | ≈ same; cache lowers it (S3) | neutral | **1** | strong for Pareto (S1 T3); none for pruning alone; ≥ unverified (GEPA default is strict, [api.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py)) |
| C2 | **+ on held-out, possibly − on in-search val.** RLMOpt's gates beat GEPA held-out 0.610 vs 0.589 and never regressed below the seed (S4). GEPA val > test on Chia (S4); ARC val 93.5 vs test 89.5 (S3) | + (fewer spurious promotions; a smaller budget can be better, S4) | +5–15% for judge k=3 on promotion only (my estimate) | **lowers it** (the point of this design) | **2** | medium (S4 is one paper, single seed for the mean; gates not ablated separately) |
| C3 | **+2–3pp held-out** over GEPA (S5); shorter prompts correlate with better scores (S1, S5) | + (2.9M vs 4.1M tokens, S5) | − (fewer tokens) | lowers it (localized edits generalize; GEPA memorizes, S6) | **3** | medium (S5 abstract only; ACE S7 supports anti-collapse, numbers unverified) |
| C4 | **±.** −2.45 aggregate on Qwen3-8B, +1.14 on GPT-4.1-mini (S1). **Zero effect for single-component artifacts** | ± | + (extra subsample evals) | ? (no data) | 2 | medium, mixed (S1 T1/T2) |
| C5 | ≈0 per eval; it buys wall-clock time, not held-out score. Best-of-N on a 3-example minibatch amplifies selection noise | − or ≈ (N× evals per step) | +N× per iteration | **raises it** (winner's curse on tiny minibatches) | 2 | weak (S2 code exists, no published ablation) |
| C6 | ? for held-out. Gains are shown only on single-task code search (S8); ablation numbers are not in the text | + on code search (150 samples, S8) | + embeddings + judge calls | ? | 4 | weak for this metric |
| C7 | + (0.610 vs 0.589, 9/11, S4) | **++** (135–500 vs 640–840 rollouts) | ? (agent tokens not totaled in the text I read) | lowers it | 5 | medium (one paper, Aug 2026) |
| C8 | + on heterogeneous tasks (AIME +7pp vs GEPA, S6); **− on homogeneous** (IFBench, S6) | + | + (variance estimation is costly: ~1,400 s/step, S6) | K=1 overfits; K=2–4 OK (S6) | 3 | medium, task-dependent |

## 4. Ranked top 3 upgrades (by held-out valset score)

### #1: GEPA-faithful core (C1). Effort 1, do it first.
- **Consistent evaluation.** Define `render(components) -> str` once and use it everywhere. The minibatch gate must evaluate `render(new_components)` vs `render(parent.components)`, not the one component on its own. Keep `component_name` only as a hint to the proposer.
- **Dominance pruning before sampling** (port of GEPA `remove_dominated_programs`):
  ```
  fronts = {key: set(progs)}                      # existing _programs
  P = all progs on any front, sorted by agg_score ASC
  remaining[key] = len(fronts[key])
  for y in P:
      if all(remaining[k] > 1 for k in keys_of(y)):
          dominated.add(y)
          for k in keys_of(y): remaining[k] -= 1
  freq[p] = #keys where p in front and p not dominated
  parent ~ Categorical(freq / sum(freq))
  ```
  Optional `TopKPareto(k)`: intersect each front with the top-k programs by agg_score and fall back to argmax (S2).
- **Acceptance:** keep strict `>` as the default (GEPA's default is `"strict_improvement"`, [api.py](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py)). Add an opt-in `>=` flag that also requires `new_text != parent_text` (GEPA `ImprovementOrEqualAcceptance`). Cost: with binary scores, ties may be the majority, and each one triggers a full-val eval, the dominant cost (S3). The effect is unverified; A/B test it.
- **Eval cache** keyed on `sha256(render(c)) + example_id` (S3). This cuts cost; it doesn't change score.

### #2: Noise-aware promotion with split validation (C2). Effort 2.
- **Size rule (my heuristic; S4 does not report its gate's valset sizes):** if |valset| ≥ 45, split into **V_pareto** (≈2/3; frontier and parents) and **V_select** (≈1/3; picks the final best only). Cost: a smaller V_pareto weakens the frontier's signal. If 15 ≤ |valset| < 45, don't split; apply the gate below to the full valset. If |valset| < 15, use argmax plus the −0.05 per-example floor, because the SE gate on so few coarse scores almost never promotes and would return the seed. If the user supplies a test set, report it untouched.
- Promote candidate c to "best" only if, on V_select (or the full valset per the size rule):
  `mean_c − mean_best > 1.65 · SE(d)`, where `d_i = s_c,i − s_best,i` (paired) and `SE(d) = std(d)/√n`,
  **and** no example regresses by more than 0.05 (from the S4 harness, adapted from per-field to per-example; my adaptation).
  Otherwise keep the incumbent. Ties go to the shorter artifact (weak support: a cross-optimizer correlation between length and score in S1 and S5, not within-run evidence).
- **Judge noise:** on promotion evals only, average the judge over k=3 samples (temperature > 0) and parse a numeric field strictly, with a retry on parse failure. This step is unverified, from first principles. The parent-selection stage can stay single-call.
- **Stopping:** stop after `patience` promotions fail the SE gate. S4 found that a larger budget lowered held-out accuracy.

### #3: Localized-edit proposer with an edit ledger (C3). Effort 3.
- The proposer returns a list of edits `{op: replace|insert|delete, anchor: <exact substring or section id>, text}` instead of a full rewrite. Apply them deterministically. If an anchor is missing, fall back to a full rewrite, tagged as such.
- After the minibatch eval, write a ledger row on the lineage: `{edit_id, fixed: [ex ids 0→1], broke: [ex ids 1→0], Δsum}`. On later proposals for any descendant, pass the last ~10 ledger rows and tell the proposer to (a) never re-introduce an edit whose `broke` is non-empty without addressing it and (b) prefer generalizable rules over example-specific facts. This counters the memorization seen in S6.
- Length guard: reject an edit that grows the component by more than 30% unless Δsum > 0 on ≥2 examples (my heuristic, unverified).

## 5. Advise against (or defer)
- **Merge as a default (C4).** It hurt Qwen3-8B by 2.45 aggregate (IFBench −10.4) and gave only +1.14 on GPT-4.1-mini (S1). It does nothing for single-component artifacts, since GEPA's merge swaps whole components (S2 merge.py). If added, keep it off by default, require `len(components) ≥ 2` and `val_overlap_floor=5`, and accept only if the subsample score ≥ max(parents).
- **Best-of-N / PxN as a score lever (C5).** The evaluator dominates cost (S3), and argmax over N proposals on 3 examples is the winner's curse. Use parallelism for wall-clock time only, with the C2 gate behind it.
- **Novelty rejection + model bandit as priority work (C6).** The evidence is single-task code search with no held-out test, ablation values are not reported in the text, and the paper's wording is inconsistent ("significantly" vs "slightly"; S8). Low expected held-out return for effort 4.
- **A weaker or cheaper proposer to save money.** Reflection cost is small next to eval cost, and GPT-5-nano lost 0.12 on circle packing (S3). Keep the strongest proposer. Route cheap models to the *task* or judge only if they are validated.
- **Optimizing on a small valset for long.** S4's synthetic diagnostic showed held-out accuracy falling as budget grew, and a ~10-example valset overfit (val ↑, test −0.03). If |valset| < ~30, cap the budget and use C2's no-split / small-set rules.
- **p1-style variance subsetting by default (C8).** It loses on homogeneous tasks (IFBench, S6). Offer it only as an opt-in sampler.

## 6. Open / unverified
- DSPy GEPA defaults (`reflection_minibatch_size`, `auto` budgets): not on the fetched page.
- ACE delta-update and grow-and-refine specifics, and its cost reductions: not verified beyond the abstract.
- C2's size thresholds (45 / 15) are my heuristics.
- No source isolates dominance pruning or ≥ acceptance. Treat both as low-risk, unmeasured fixes; measure them with an A/B on a held-out split.
