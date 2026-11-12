# Expert panel: RL researcher — reward = sample efficiency

**Metric definition.** Sample efficiency = area under the curve of best full-valset score vs. **evaluator calls** (`state.total_evals`) at a matched `max_metric_calls`, or equivalently calls needed to reach seed + X. Proposer (LLM reflection) calls are counted separately and go into $/run, not this metric.

**One-line thesis.** In GEPA-style loops, most of the evaluator budget goes to *validation* and not to learning. The cheapest wins come from (a) not wasting validation calls on gate false positives and redundant evals, and (b) racing candidates on the valset rather than giving every gate-passer a full eval. Weight-level RL does not belong in this codebase.

## 0. Codebase facts that drive the ranking (engine.py, state.py, pareto.py)

| # | Fact | Location | Sample-efficiency impact |
|---|---|---|---|
| F1 | **Gate and valset score different artifacts.** The minibatch gate evaluates `parent.components[component_name]`, a single component. `_evaluate_candidate` evaluates `"\n".join(components.values())`. | engine.py `run()` vs `_evaluate_candidate` | For multi-component artifacts the gate is a noisy proxy for the wrong thing. Gate passes then burn \|V\| calls each. |
| F2 | **The parent is re-evaluated on the minibatch every iteration**, even though per-example scores may already sit in `state.scores[parent_idx]`. The full eval discards ASI (`score, _ =`). | engine.py | Wastes up to 3 of the 6 minibatch calls per iteration. |
| F3 | **The child's minibatch scores are thrown away.** They are recomputed inside the full eval when the dataset and valset overlap (always, when `valset=None`). | engine.py | Wastes 3 calls per accepted candidate. |
| F4 | **No dominance pruning.** `select_candidate` samples by tie frequency, and the code comment admits this is "simplified". Programs tied on easy examples get oversampled. | pareto.py | Hurts score per call. GEPA's Pareto gain assumes real frontier selection with dominated programs removed. |
| F5 | **There is only one operator** (`reflective_mutation`) and components are chosen round-robin. | state.py `next_component` | "Bandit over operators" has no arms today. The available arms are components (and, later, proposer models). |

## 1. Source digest

| # | Source | Key numbers / finding | Relevance |
|---|---|---|---|
| S1 | GEPA, Agrawal et al., ICLR 2026 oral, v2 Feb 2026 — https://arxiv.org/abs/2507.19457 , https://arxiv.org/html/2507.19457v2 | Six tasks on Qwen3-8B. GRPO uses **24,000 rollouts** (LoRA); GEPA uses **1,839–7,051**. Aggregate **54.85 vs 48.91** (v2: "+6% avg, up to 20%"; v1 said +10% over 4 tasks, so cite v2). GEPA loses on AIME-25 (32 vs 38). GEPA needed **243–1,179 total rollouts** to match GRPO's best val, and only **6–179 train rollouts**. Quote: *"The majority of GEPA's rollout budget is spent on validation, where scores are utilized solely for candidate selection."* Train-only, it needed **79–737** rollouts to reach its best. | Core evidence for both "reflective beats RL" and "validation is the budget sink". |
| S2 | GEPA Table 3 (same URL) | Selection ablation: **Pareto 61.28** vs **SelectBest 54.89** vs **Beam(N=4) 53.95** aggregate (4 tasks). | Keep Pareto, and fix it (F4). |
| S3 | GEPA Table 1/2 (same URL) | Merge: Qwen3-8B aggregate drops to **52.40** (IFBench **38.61→28.23**). GPT-4.1-mini gains slightly (**66.36 vs 65.22**). | Merge is model-dependent. Defer it. |
| S4 | TRIPLE, Shi et al., NeurIPS 2024 — https://arxiv.org/html/2402.09723v3 | Prompt selection framed as fixed-budget best-arm identification. With 30 prompts and **150 LLM evals**, TRIPLE-SH beats uniform allocation by **~15%** (GPT-3.5, Llama2). Plugged into APE and APO, it beats their uniform/UCB selectors (APE avg rank 4.00→2.92). | Direct evidence that successive halving over val examples raises quality per call. |
| S5 | HbBoPs, ICML 2025 — https://arxiv.org/abs/2412.07820 | Hyperband with **#validation instances as the fidelity**. Claims better performance and efficiency on 10 benchmarks × 3 LLMs. *Abstract only; no savings figure verified.* Also notes that the val-optimal prompt may not be test-optimal with small valsets. | Multi-fidelity valset evaluation. |
| S6 | ShinkaEvolve, Lange et al., ICLR 2026 — https://arxiv.org/abs/2509.19349 | SOTA circle packing in **150 samples**. Ablations (secondary summaries): **novelty rejection** (embedding cosine **0.95** cutoff plus an LLM "meaningfully different?" check) gives *substantial* gains. The **UCB1 LLM-ensemble bandit** gives only a *slight* gain. Self-reported, no replication. | Novelty rejection is a cheap filter before evaluation. Bandit evidence is weak. |
| S7 | AdaEvolve, Feb 2026 — https://arxiv.org/html/2602.20133v1 | UCB over islands with reward r = (f′ − f_k*)/f*_global, decayed sums R←ρR+r, V←ρV+1, score R/V + √2·√(ln N/n_k). Exploration intensity I = 0.1 + 0.6/(1+√(G+ε)). Frontier-CS (GPT-5, 50 calls): **AdaEvolve 61.33 vs GEPA 43.04**, OpenEvolve 50.75. Circle packing roughly ties GEPA (2.629 vs 2.613). | The best-specified bandit formula for LLM evolution. Its gains bundle three mechanisms. |
| S8 | OpenEvolve / AlphaEvolve evaluation cascades — https://www.everydev.ai/tools/openevolve/llms.txt , GigaEvo https://arxiv.org/pdf/2511.17592v1 | Staged evaluation: cheap tests first, the expensive eval only for survivors. *No thresholds verified.* | Same idea as racing. Medium evidence (widely used, rarely ablated). |
| S9 | Yue et al., NeurIPS 2025 oral — https://neurips.cc/virtual/2025/poster/119944 (arXiv 2504.13837) | RLVR raises pass@1 but **base models match or beat RL models at large pass@k**. RLVR reweights paths the base model already has. | Evidence against weight-RL buying anything a search loop can't. Irrelevant anyway: we can't train weights. |
| S10 | DAPO / GSPO / Dr.GRPO — *secondary only*: https://primeintellect.ai/rl-glossary/rl-algorithms , https://huggingface.co/docs/trl/v0.22.1/en/paper_index | DAPO: clip-higher (0.28/0.2), **dynamic sampling** drops groups with zero reward variance (AIME 30→36→38 in its ablation, unverified primary). GSPO: sequence-level ratio. Dr.GRPO: drops std and length normalization. | Only the dynamic-sampling idea transfers: don't spend proposer calls on minibatches with no signal. `skip_perfect_score` covers part of this. |
| S11 | BetterTogether / mmGRPO — https://www.emergentmind.com/papers/2508.04660 | Prompt optimization followed by weight RL beats either alone, by **up to 11 pts**. | Only relevant if weight training ever becomes possible. |
| S12 | Judge hacking: "One Token to Fool LLM-as-a-Judge" https://arxiv.org/abs/2507.08794 ; EST invariance test https://arxiv.org/pdf/2507.05619v2 | A single token or a phrase like "Solution" inflates generative-RM scores. EST flags gaming when scores change under superficial edits. | Pick feedback signals carefully. Detecting hacking costs extra calls. |
| S13 | OPRO — https://arxiv.org/pdf/2309.03409 | Train accuracy runs **5–20 pts above test** without a held-out set. | Partial evaluation increases winner's-curse risk, so confirm before promoting. |
| S14 | Bandit operator selection in LLM evolution — RIDER (windowed Thompson sampling over 9 operators) https://www.fruct.org/files/publications/volume-39/fruct39/Dra.pdf ; Deliberate Evolution https://arxiv.org/pdf/2606.04360 ; TIDE https://arxiv.org/pdf/2601.21239 | Posteriors concentrate on productive operators, with a probability floor to keep exploring. *No clean ablation numbers verified.* | Weak evidence. Cheap to implement. |
| S15 | GEPA library `val_evaluation_policy` — https://gepa-ai.github.io/gepa/api/core/optimize/ | Upstream exposes a pluggable policy (default `full_eval`). Docs disagree on whether the incremental policy has shipped. RoboPhD's claim that "a 200-ex valset → ~7 candidates" is a **search snippet only (unverified)**: https://arxiv.org/pdf/2604.04347 | Upstream sees the same bottleneck. |

## 2. Candidate designs (points on the axes)

Axes: (1) search, (2) selection, (3) feedback, (4) routing, (5) budget control, (6) artifact scope. A seventh axis, **(7) eval reuse** (none / memoized / paired), is added because it dominates the metric here.

| ID | (1) search | (2) selection | (3) feedback | (4) routing | (5) budget | (6) scope | (7) eval reuse |
|---|---|---|---|---|---|---|---|
| A. "Hygiene" | reflective mutation | Pareto **with dominance pruning** | ASI | 1 proposer | fixed | multi-comp, **gate evaluates full assembled artifact** | memoized (text, ex) → (score, ASI) |
| B. "Raced valset" | reflective mutation | Pareto + **paired successive halving** vs frontier | ASI | 1 proposer | **adaptive: SH on valset**, full eval before promotion | multi-comp | paired against stored frontier scores |
| C. "Bandit-steered mutation" | reflective mutation | Pareto (parent) + **UCB1 over components** (+ later proposer models) | ASI | bandit across models | fixed | multi-comp | memoized |
| D. "Novelty-filtered, group proposals" | reflective, **K=2–4 proposals per step** (GRPO-style group) | best-in-group on minibatch, then B | ASI + text-similarity filter | cheap model proposes K, strong model reflects | adaptive | multi-comp | memoized |
| E. "Weight RL (GRPO/DAPO)" | RL on policy weights | n/a | verifier reward | trained small model | fixed 10k+ rollouts | whole program | none |
| F. "AdaEvolve-style islands" | evolution + islands | UCB over islands, adaptive explore | ASI | multi | adaptive | whole program | memoized |

## 3. Scores

Scales: valset score and sample efficiency are relative to current; cost is $/run vs current; effort 1 = hours … 5 = weeks.

| ID | valset score | sample eff. | $/run | train/val gap risk | effort | evidence |
|---|---|---|---|---|---|---|
| A | ≈ to + (better gate → fewer bad accepts; pruning → better parents) | **++** (saves ~3 calls/iter from F2, 3/accept from F3, plus false-positive full evals from F1) | − (fewer calls) | = | **1** | strong for F1–F3 (arithmetic from code); medium for pruning (S2) |
| B | ≈ (risk: wrongly eliminates late bloomers) | **+++**: validation is "the majority" of the budget (S1). Expect each rejected candidate to cost ~\|V\|/8 to \|V\|/4 calls instead of \|V\|. | −− | ↑ slightly; mitigated by full eval before promotion | **2** | strong-medium: S1 (where budget goes), S4 (+~15% at fixed budget), S5 |
| C | + small | + | ≈ | = | **1–2** | weak-medium: S6 "slight", S7, S14 |
| D | + (more proposals per gate) | + (novelty filter skips evals); best-of-K costs K×minibatch | + (K proposer calls) | = | 2 | medium for novelty filter (S6 ablation, self-reported); ? for best-of-K: no prompt-level ablation found |
| E | ? (GEPA beats it in 5/6 tasks, S1) | −−− (24k vs ≤7k rollouts, S1) | +++ (GPUs) | ? | 5 | strong against (S1, S9) |
| F | + on code tasks (S7) | ? (bundled mechanisms, no per-call ablation) | ≈ | = | 4 | medium, single paper (S7) |

## 4. Ranked top 3 upgrades (by sample efficiency)

### #1 — Evaluation hygiene (design A). Pure gain, ~hours of work.
1. **Gate consistency (F1).** Score the minibatch on the *assembled candidate*, the same text `_evaluate_candidate` builds: `assemble(components) = "\n".join(components.values())`, used in both places. Better still, pass the dict through and let the evaluator assemble it. This stops gate false positives from triggering \|V\|-call evals.
2. **Memoize** `cache[(sha256(assemble(c)), example_id)] = (score, asi)` and store ASI during the full eval (stop discarding it).
   - Parent minibatch evals become cache hits whenever the parent has a full eval on the same pool. That is always true when `valset is None`, and holds whenever the minibatch is drawn from the valset.
   - Child minibatch scores get reused inside its full eval.
   - **Caveat:** only valid for deterministic evaluators. Add `config.cache_evals: bool = True` and document it. For stochastic evaluators, keep the cache but set a `max_age` in iterations.
3. **Dominance pruning (F4).** Before frequency sampling, drop program p if some q has `scores[q][k] ≥ scores[p][k]` for every k with strict inequality on at least one, as GEPA's real `remove_dominated_programs` does. Then sample with weight ∝ the number of keys where p sits at the front.
- **Expected effect.** With batch b = 3, per-iteration evaluator calls drop from 2b = 6 to b = 3 (only the child is new). Accepted candidates also save b calls in the full eval. So at a fixed budget you get roughly 1.5–2× more iterations before any valset change. (This follows from the code; no paper needed.)

### #2 — Paired successive halving for the valset eval (design B)
Replaces "every gate-passer gets the full \|V\|".
```
V = valset ids, fixed random permutation π drawn once per run (same order for all candidates → pairing)
r = 0; n_r = ceil(|V| / 8)      # rounds use prefixes π[:n_r], doubling: |V|/8, /4, /2, 1
ref = frontier reference = per-example max over frontier programs: m_k = max_p scores[p][k]  (already stored, 0 calls)
loop:
    evaluate child on π[n_{r-1}:n_r]   (incremental; cached)
    D_r = mean_{k in π[:n_r]} ( s_child[k] - s_parent[k] )         # paired vs parent
    W_r = #{k in π[:n_r] : s_child[k] >= m_k}                         # keys where child would join the front
    stop & REJECT if  D_r < -eps_r  and  W_r == 0
          where eps_r = c * sigma_D_r / sqrt(n_r)  (c = 1.0; sigma = std of paired diffs, floor 0.05)
    if n_r == |V|: ACCEPT -> state.add_candidate(child, full scores)
    n_r = min(2 n_r, |V|)
```
- **Frontier rule (decided).** Rejected candidates *never* enter `State` or the frontier, and partial scores live only in the cache. Only full-coverage candidates get `add_candidate`, so per-key Pareto stays well-defined, and no candidate becomes `best` without a full eval (guards against S13's winner's curse).
- **Why the W_r term matters.** It keeps Pareto specialists alive. A candidate that is worse on average but wins on some example is exactly what GEPA's Pareto selection exploits (S2).
- **Expected effect.** Most gate-passers that would *not* improve get cut at \|V\|/8 or \|V\|/4. Combined with S1's finding that validation dominates GEPA's budget, this plausibly doubles the number of candidates explored per budget. That figure is an estimate, not measured. Log `evals_saved` and check it against a full-eval A/B on one task.
- **Effort.** About 2: a new `strategies/eval_policy.py` plus an incremental `_evaluate_candidate(components, ids)`.

### #3 — Novelty rejection, then a UCB1 component bandit (designs D and C, cheap parts only)
- **Novelty filter (pre-evaluation, 0 calls).** Reject a proposal if `difflib.SequenceMatcher(None, new, x).ratio() ≥ 0.95` for x ∈ {parent text, the last 10 proposals on this component}. Re-prompt once with "your proposal was near-identical; make a substantive change", otherwise skip. ShinkaEvolve uses the same 0.95 threshold on embeddings (S6). difflib avoids an embedding dependency.
- **UCB1 over components** replaces round-robin `next_component`:
  `reward_t = (Σ new_scores − Σ parent_scores) / b` clipped to [−1, 1], then mapped to [0, 1] via (r+1)/2. Rejected proposals also count, with negative reward.
  `pick j = argmax_j  μ̂_j + sqrt(2 ln N / n_j)`, try unplayed arms first; use AdaEvolve-style decay (μ̂ from R←ρR+r, V←ρV+1, ρ = 0.9) because rewards are non-stationary as components saturate (S7).
  - Keep it **global** (per component), not per candidate. Per-candidate counts are too sparse.
  - Extend the arms to (component × proposer model) when a second proposer model exists, as in ShinkaEvolve's LLM ensemble.
- **Evidence.** Novelty filter: medium. Bandit: weak ("slight" gain in S6). Rank it below #1 and #2 for that reason.

## 5. Advise AGAINST

| Don't | Why (evidence) |
|---|---|
| Weight-level GRPO/DAPO/GSPO, or "train a prompt-writer with RL" | 24k rollouts vs GEPA's 1.8–7k at lower quality on 5/6 tasks (S1). RLVR gains stay within base-model pass@k (S9). No hosted-API path to weight updates anyway. |
| Best-only or beam selection to "save" frontier bookkeeping | −6.4 / −7.3 pts vs Pareto (S2). |
| Turning on Merge by default | Hurts Qwen3-8B aggregate (54.85→52.40; IFBench −10.4 pts), helps GPT-4.1-mini by only ~1 pt (S3). Make it opt-in. |
| An LLM judge as the *only* reward signal | Single-token exploits (S12). If a judge is used, add an EST-style invariance check: re-score the best candidate under a trivial reformat and flag any \|Δ\| > 0.1 (S12). This costs extra calls, so run it only on promotion. |
| Promoting on partial valset scores, or shrinking the valset to explore more | Winner's curse: a 5–20 pt train/test gap without a held-out set (S13). Small-valset tradeoff (S5, S15). Race, but confirm with a full eval. |
| Over-engineering credit assignment (contextual bandits, learned PRMs over mutation steps) | Only one operator exists (F5). Bandit gains in this literature are small or unablated (S6, S14). A process reward model needs training data we don't have. |
| Porting GRPO advantage normalization as a "selection signal" | Group-relative z-scoring adds nothing over raw paired deltas at K ≤ 4. The only transferable idea is DAPO's zero-variance filter, which `skip_perfect_score` already half-covers. Extend it to also skip when all parent minibatch scores are equal *and* the ASI is empty (S10, secondary). |

## 6. Open items / unverified
- TRIPLE gains are at a 150-call budget with 30 *pre-generated* prompts. Whether they transfer to sequentially generated candidates is unverified. (UNCONFIRMED)
- HbBoPs savings figure: not verified (abstract only).
- ShinkaEvolve ablation magnitudes: secondary summaries only.
- The DAPO/GSPO/Dr.GRPO numbers: secondary sources only.
- No independent replication of GEPA vs GRPO was found as of 2026-11-11.
