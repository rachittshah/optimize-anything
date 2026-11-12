from __future__ import annotations
import dataclasses
import difflib
import hashlib
import inspect
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from optimize_anything.config import Config, SelectionStrategy, resolve_run_dir
from optimize_anything.core.candidate import Candidate
from optimize_anything.core.cost import CostTracker
from optimize_anything.core.state import State
from optimize_anything.core import events
from optimize_anything.core.halving import rung_sizes, should_stop_early, shuffled_order
from optimize_anything.core.pareto import ParetoFrontier
from optimize_anything.core.prescreen import (
    components_key, is_near_duplicate, normalize, output_signature,
)
from optimize_anything.core.promotion import paired_gain, should_promote
from optimize_anything.strategies.bandit import COMPRESS_MIN_CHARS, SlidingWindowUCB
from optimize_anything.strategies.hard_pool import HardPoolSampler, hard_ids
from optimize_anything.evaluators.base import EvaluationError
from optimize_anything.strategies.selection import ParetoSelector, BestSelector, EpsilonGreedySelector
from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import (
    BudgetStopper, IterationStopper, CompositeStopper, TimeoutStopper, NoImprovementStopper,
    CostStopper,
)


@dataclass
class OptimizationResult:
    best_candidate: dict[str, str]
    best_score: float
    all_candidates: list[dict[str, str]]
    all_scores: list[float]
    pareto_frontier: dict
    total_iterations: int
    total_evals: int
    seed_score: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_point: float | None = None  # USD per +1 score point (scores x100)
    cost: dict = field(default_factory=dict)
    run_dir: str | None = None
    failed_evals: int = 0  # evals that raised EvaluationError (excluded, not scored 0)
    cache_hits: int = 0  # evals served from the cache (not counted in total_evals)
    # Held-out test split (Engine(testset=...)); never used for selection. None if no testset.
    seed_test: float | None = None
    best_test: float | None = None
    val_test_gap: float | None = None  # best_score (val) - best_test
    test_evals: int = 0  # counted separately from total_evals / max_metric_calls
    # Flag-gated upgrade counters
    prescreen_skips: int = 0
    halving_rejects: int = 0
    halving_evals_saved: int = 0
    promotions_blocked: int = 0
    operator_counts: dict = field(default_factory=dict)
    operator_unsupported: int = 0  # non-rewrite operator requested, proposer can't take it
    skipped_perfect: int = 0  # perfect parent minibatches re-drawn (not counted as iterations)
    edit_size_rejects: int = 0  # children over Config.max_edit_ratio (rejected before any eval)
    # budget | iterations | cost | timeout | no_improvement | train_saturated
    stop_reason: str | None = None


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _cost_per_point(total_cost: float, seed: float, best: float) -> float | None:
    gain = best - seed
    if gain <= 0:
        return None
    return total_cost / max(1e-9, gain * 100)


class Engine:
    def __init__(
        self,
        seed_candidate: dict[str, str],
        evaluator,
        proposer,
        objective: str = "",
        background: str | None = None,
        dataset: list | None = None,
        valset: list | None = None,
        config: Config | None = None,
        on_event: Callable[[dict], None] | None = None,
        cost_tracker: CostTracker | None = None,
        testset: list | None = None,
    ):
        self.config = config or Config()
        self.run_dir = resolve_run_dir(self.config)
        self.evaluator = evaluator
        self.proposer = proposer
        self.objective = objective
        self.background = background
        self.dataset = dataset
        self.valset = valset
        self.on_event = on_event or (lambda e: None)
        self.cost = cost_tracker or CostTracker()

        self.testset = testset
        seed = self.config.resolved_seed()  # None -> 42, the original hard-coded seed
        self.state = State(component_names=list(seed_candidate.keys()),
                           frontier=ParetoFrontier(seed=seed))

        # Build selector
        if self.config.selection_strategy == SelectionStrategy.PARETO:
            self.selector = ParetoSelector(seed=seed)
        elif self.config.selection_strategy == SelectionStrategy.BEST:
            self.selector = BestSelector()
        else:
            self.selector = EpsilonGreedySelector(self.config.epsilon, seed=seed)

        self.sampler = EpochBatchSampler(self.config.reflection_minibatch_size, seed=seed)

        # Flag-gated upgrades. Each has its own RNG, so defaults draw nothing extra.
        self._seed = seed
        self._best_idx = 0  # promotion="paired_se" incumbent
        self._hard_sampler = (HardPoolSampler(self.sampler, seed)
                              if self.config.counterexample_mining else None)
        self._train_scores: dict[int, dict[int, float]] = {}  # cand idx -> train id -> score
        self._bandit = SlidingWindowUCB() if self.config.operator_bandit else None
        self._proposer_takes_operator = _accepts_kwarg(self.proposer.propose, "operator")
        self._operator_unsupported = 0
        self._val_order: list[int] | None = None
        self._test_evals = 0
        self._edit_size_rejects = 0
        self._memory: dict[tuple[int, str], deque[dict]] = {}  # (parent, component) -> rejects
        self._proposer_takes_memory = _accepts_kwarg(self.proposer.propose, "rejected_edits")
        self._lock = threading.Lock()  # cache + counters when eval_workers > 1
        self._truncated = 0  # examples the last _eval_many skipped for budget (workers > 1)
        self._val_truncated = False

        # Build stopper
        stoppers: list = [IterationStopper(self.config.max_iterations)]
        if self.config.max_metric_calls:
            stoppers.append(BudgetStopper(self.config.max_metric_calls))
        if self.config.timeout_seconds:
            stoppers.append(TimeoutStopper(self.config.timeout_seconds))
        if self.config.no_improvement_patience:
            stoppers.append(NoImprovementStopper(self.config.no_improvement_patience,
                                                 self.config.min_improvement))
        if self.config.max_cost_usd is not None:
            stoppers.append(CostStopper(self.config.max_cost_usd, self.cost))
        self.stopper = CompositeStopper(stoppers)
        # Resource-only stoppers (shared instances), checked between perfect-skip re-draws.
        self._resource_stopper = CompositeStopper(
            [s for s in stoppers if not isinstance(s, (IterationStopper, NoImprovementStopper))])
        self._stop_reason: str | None = None
        self._sample_step = 0  # sampler position; == iteration unless perfect re-draws happen

        self._seed_candidate = seed_candidate
        self._seed_score = 0.0
        self._cache: dict[str, tuple[float, dict]] = {}
        ident = getattr(self.evaluator, "cache_identity", None)
        self._evaluator_identity = (
            ident() if callable(ident) else f"{type(self.evaluator).__qualname__}:{id(self.evaluator)}"
        )

    # ---- single evaluation: cache + failure handling ----
    def _eval(self, payload: str | dict[str, str], example: Any) -> tuple[float, dict] | None:
        """Evaluate once. Returns None if the evaluator raised EvaluationError.

        Cache hits (same evaluator, candidate, example) don't count toward total_evals or
        max_metric_calls. Failed evals are counted in total_evals and failed_evals.
        """
        key = self._cache_key(payload, example)
        if key is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.state.cache_hits += 1
                return hit[0], dict(hit[1])
        return self._eval_uncached(payload, example, key)

    def _cache_key(self, payload: str | dict[str, str], example: Any) -> str | None:
        if not self.config.cache_evals:
            return None
        raw = json.dumps([self._evaluator_identity, payload, example], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _eval_uncached(self, payload: str | dict[str, str], example: Any,
                       key: str | None) -> tuple[float, dict] | None:
        with self._lock:
            self.state.record_evals(1)
        try:
            score, asi = self.evaluator.evaluate(payload, example)
        except EvaluationError:
            with self._lock:
                self.state.failed_evals += 1
            return None
        if key is not None:
            with self._lock:
                self._cache[key] = (score, dict(asi))
        return score, asi

    def _eval_many(self, payload: str | dict[str, str], examples: list) -> list:
        """Evaluate examples in order. eval_workers > 1: concurrently, de-duplicated by cache
        key, and truncated to the remaining max_metric_calls (truncated -> None, uncounted)."""
        workers = self.config.eval_workers
        self._truncated = 0
        if workers <= 1:
            return [self._eval(payload, ex) for ex in examples]
        results: list[tuple[float, dict] | None] = [None] * len(examples)
        keys = [self._cache_key(payload, ex) for ex in examples]
        cap = self.config.max_metric_calls
        remaining = None if not cap else max(0, cap - self.state.total_evals)
        todo: list[int] = []
        first: dict[str, int] = {}
        dups: dict[int, int] = {}
        for i, key in enumerate(keys):
            if key is not None and key in self._cache:
                hit = self._cache[key]
                self.state.cache_hits += 1
                results[i] = (hit[0], dict(hit[1]))
            elif key is not None and key in first:
                dups[i] = first[key]
            elif remaining is None or len(todo) < remaining:
                todo.append(i)
                if key is not None:
                    first[key] = i
            else:
                self._truncated += 1
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(todo)))) as pool:
            outs = list(pool.map(lambda i: self._eval_uncached(payload, examples[i], keys[i]),
                                 todo))
        for i, out in zip(todo, outs):
            results[i] = out
        for i, j in dups.items():
            if results[j] is not None:
                self.state.cache_hits += 1
                results[i] = (results[j][0], dict(results[j][1]))
        return results

    # ---- evaluator payload: one rule for every evaluation ----
    def _payload(self, components: dict[str, str]) -> str | dict[str, str]:
        """Single component -> its text. Multi-component -> dict if the evaluator declares
        `accepts_dict = True`, else the texts joined with newlines."""
        if len(components) == 1:
            return next(iter(components.values()))
        if getattr(self.evaluator, "accepts_dict", False):
            return dict(components)
        return "\n".join(components.values())

    def run(self) -> OptimizationResult:
        seed_scores = self._evaluate_candidate(self._seed_candidate)
        self.state.add_candidate(
            Candidate(components=dict(self._seed_candidate)),
            scores=seed_scores,
            parent_ids=[],
        )
        self._seed_score = self.state.agg_scores[0]
        self._checkpoint()

        while not self._should_stop():
            self.state.iteration += 1
            iteration = self.state.iteration

            # SELECT + SAMPLE + parent eval. A perfect parent minibatch has nothing to reflect
            # on: re-draw parent and minibatch (up to max_perfect_resamples) instead of burning
            # the iteration; stop if the whole selectable pool is perfect on all of train.
            perfect = False
            for attempt in range(self.config.max_perfect_resamples + 1):
                parent_idx, batch_ids, batch, parent_results = self._select_and_eval_parent()
                if attempt == 0:
                    self.on_event(events.iteration_start(iteration, parent_idx,
                                                         self.cost.total_usd))
                parent_scores = [r[0] for r in parent_results if r is not None]
                if not parent_scores:
                    break
                self.on_event(events.evaluation_complete(
                    parent_idx, sum(parent_scores) / len(parent_scores), {}
                ))
                perfect = self.config.skip_perfect_score and all(s >= 1.0 for s in parent_scores)
                if not perfect:
                    break
                self.state.skipped_perfect += 1
                self.on_event(events.perfect_skip(iteration, parent_idx, attempt))
                if self._train_saturated():
                    self._stop_reason = "train_saturated"
                    self.on_event(events.train_saturated(iteration, len(self.state.candidates)))
                    break
                if self._resource_stopper.should_stop(total_evals=self.state.total_evals,
                                                      total_cost_usd=self.cost.total_usd):
                    break
            if self._stop_reason is not None:
                self.state.iteration -= 1  # no proposal happened: don't count it
                break
            if not parent_scores or perfect:
                # All parent evals failed (judge down), or every re-draw was perfect.
                continue

            parent = self.state.candidates[parent_idx]
            component_name = self.state.next_component(parent_idx)
            parent_asi = [{"score": r[0], "feedback": r[1]} for r in parent_results if r is not None]

            operator = self._choose_operator(parent.components[component_name])
            new_text = self._propose(parent.components[component_name], component_name,
                                     parent_asi, operator,
                                     self._memory.get((parent_idx, component_name)))
            self.on_event(events.proposal_complete(iteration, component_name, self.cost.total_usd))

            new_components = dict(parent.components)
            new_components[component_name] = new_text
            new_candidate = Candidate(components=new_components, tag="reflective_mutation")

            parent_text = parent.components[component_name]
            cap = self.config.max_edit_ratio
            if cap is not None and self._change_ratio(parent_text, new_text) > cap:
                self._edit_size_rejects += 1
                self.on_event(events.candidate_rejected(iteration, "edit_size", operator=operator))
                self._remember(parent_idx, component_name, parent_text, new_text,
                               note=f"edit too large: changed more than {cap:.0%} of the text")
                if self._bandit is not None:
                    self._bandit.update(operator, 0.0)
                continue

            skip = self._prescreen(parent.components, new_components, component_name)
            if skip is not None:  # the proposer call is spent; the evals are not
                self.state.prescreen_skips += 1
                self.on_event(events.candidate_rejected(iteration, skip, operator=operator))
                if self._bandit is not None:
                    self._bandit.update(operator, 0.0)
                continue

            # Evaluate the full new candidate on the same minibatch
            new_payload = self._payload(new_components)
            new_results = self._eval_many(new_payload, batch)
            # Compare only on examples where both evals succeeded.
            pairs = [(p[0], n[0]) for p, n in zip(parent_results, new_results)
                     if p is not None and n is not None]
            improved = bool(pairs) and sum(n for _, n in pairs) > sum(p for p, _ in pairs)
            if self._bandit is not None:
                gain = (sum(n for _, n in pairs) - sum(p for p, _ in pairs)) / len(pairs) if pairs else 0.0
                self._bandit.update(operator, max(0.0, gain))
            if not improved:
                reason = "minibatch"
                if (self.config.prescreen
                        and output_signature(new_results) == output_signature(parent_results)):
                    # Inert for acceptance (strict > already rejects equal scores); observability.
                    self.state.prescreen_skips += 1
                    reason = "prescreen_signature"
                self.on_event(events.candidate_rejected(iteration, reason, operator=operator))
                if pairs:
                    self._remember(parent_idx, component_name, parent_text, new_text,
                                   sum(p for p, _ in pairs) / len(pairs),
                                   sum(n for _, n in pairs) / len(pairs))

            self._val_truncated = False
            if not improved:
                full_scores: dict[str, float] | None = {}
            elif self.config.val_eval == "halving":
                full_scores = self._evaluate_halving(new_components, parent_idx, iteration)
                if full_scores is None and not self._val_truncated:
                    self._remember(parent_idx, component_name, parent_text, new_text,
                                   note="passed the minibatch but was worse on validation")
            else:
                full_scores = self._evaluate_candidate(new_components)
            if self._val_truncated:
                # Budget ran out mid val sweep (eval_workers > 1): never add a partly scored child.
                self.on_event(events.candidate_rejected(iteration, "budget"))
                full_scores = None
            if improved and not full_scores and full_scores is not None:
                # every val eval failed (EvaluationError): nothing to add
                self.on_event(events.candidate_rejected(iteration, "val_failed"))
            if improved and full_scores:
                prev_best = self.state.best_score()
                new_idx = self.state.add_candidate(new_candidate, full_scores, [parent_idx])
                self._record_train(new_idx, batch_ids, new_results)

                new_agg = self.state.agg_scores[new_idx]
                if self.config.promotion == "paired_se":
                    self._update_incumbent(new_idx, iteration)
                elif new_agg >= prev_best:
                    self.on_event(events.new_best_found(new_idx, new_agg, self.cost.total_usd))
                self.on_event(events.frontier_updated(len(self.state.frontier.all_program_ids())))
                self._checkpoint()
            elif self.config.checkpoint_interval and iteration % self.config.checkpoint_interval == 0:
                self._checkpoint()

        if self._stop_reason is None:
            self._stop_reason = self.stopper.reason
        result = self._build_result()
        self._checkpoint()
        self._write_result(result)
        self.on_event(events.optimization_complete(
            result.best_score, result.total_iterations, result.total_evals,
            result.total_cost_usd, result.seed_score, stop_reason=result.stop_reason,
        ))
        return result

    def _select_and_eval_parent(self) -> tuple[int, list[int], list, list]:
        """Pick a parent, draw the next minibatch, evaluate the parent on it."""
        if len(self.state.candidates) == 1:
            parent_idx = 0
        else:
            parent_idx = self.selector.select(self.state)
        step = self._sample_step
        self._sample_step += 1
        if self.dataset:
            all_ids = list(range(len(self.dataset)))
            if self._hard_sampler is not None:
                batch_ids = self._hard_sampler.next_batch(all_ids, step, self._hard_pool())
            else:
                batch_ids = self.sampler.next_batch(all_ids, step)
            batch = [self.dataset[i] for i in batch_ids]
        else:
            batch_ids = [0]  # single-task mode: one pseudo-example
            batch = [None]
        payload = self._payload(self.state.candidates[parent_idx].components)
        results = self._eval_many(payload, batch)
        self._record_train(parent_idx, batch_ids, results)
        return parent_idx, batch_ids, batch, results

    def _train_saturated(self) -> bool:
        """Every candidate the selector can pick is known perfect on every train example."""
        n_train = len(self.dataset) if self.dataset else 1
        if self.config.selection_strategy == SelectionStrategy.PARETO:
            pool = self.state.frontier.remove_dominated(self.state.agg_scores) or {0}
        else:
            pool = set(range(len(self.state.candidates)))
        for idx in pool:
            known = self._train_scores.get(idx, {})
            if len(known) < n_train or any(s < 1.0 for s in known.values()):
                return False
        return True

    def _build_result(self) -> OptimizationResult:
        best_idx = self._best_index()
        best = self.state.candidates[best_idx]
        best_score = self.state.agg_scores[best_idx]
        seed_test = best_test = gap = None
        if self.testset:  # before reading cost, so total_cost_usd and cost agree
            seed_test = self._evaluate_test(self.state.candidates[0].components)
            best_test = (seed_test if best_idx == 0
                         else self._evaluate_test(best.components))
            if best_test is not None:
                gap = best_score - best_test
        total_cost = self.cost.total_usd
        return OptimizationResult(
            best_candidate=best.components,
            best_score=best_score,
            all_candidates=[c.components for c in self.state.candidates],
            all_scores=self.state.agg_scores,
            pareto_frontier=self.state.frontier.to_dict(),
            total_iterations=self.state.iteration + 1,
            total_evals=self.state.total_evals,
            seed_score=self._seed_score,
            total_cost_usd=total_cost,
            cost_per_point=_cost_per_point(total_cost, self._seed_score, best_score),
            cost=self.cost.to_dict(),
            run_dir=str(self.run_dir),
            failed_evals=self.state.failed_evals,
            cache_hits=self.state.cache_hits,
            seed_test=seed_test,
            best_test=best_test,
            val_test_gap=gap,
            test_evals=self._test_evals,
            prescreen_skips=self.state.prescreen_skips,
            halving_rejects=self.state.halving_rejects,
            halving_evals_saved=self.state.halving_evals_saved,
            promotions_blocked=self.state.promotions_blocked,
            operator_counts=dict(self.state.operator_counts),
            operator_unsupported=self._operator_unsupported,
            skipped_perfect=self.state.skipped_perfect,
            edit_size_rejects=self._edit_size_rejects,
            stop_reason=self._stop_reason,
        )

    # ---- flag-gated upgrades ----
    def _best_index(self) -> int:
        """Reported best: argmax mean val ("max", first on ties) or the paired_se incumbent."""
        if self.config.promotion == "paired_se":
            return self._best_idx
        return max(range(len(self.state.agg_scores)), key=lambda i: self.state.agg_scores[i])

    def _update_incumbent(self, new_idx: int, iteration: int) -> None:
        inc = self._best_idx
        new_scores, inc_scores = self.state.scores[new_idx], self.state.scores[inc]
        if should_promote(new_scores, inc_scores, self.config.promotion_z,
                          self.config.promotion_max_drop):
            self._best_idx = new_idx
            self.on_event(events.new_best_found(new_idx, self.state.agg_scores[new_idx],
                                                self.cost.total_usd))
        elif self.state.agg_scores[new_idx] > self.state.agg_scores[inc]:
            self.state.promotions_blocked += 1
            mean, se, _ = paired_gain(new_scores, inc_scores)
            self.on_event(events.promotion_blocked(iteration, new_idx, inc, mean, se))
            self.on_event(events.candidate_rejected(iteration, "promotion", candidate_idx=new_idx))

    def _choose_operator(self, current_text: str) -> str:
        if self._bandit is None:
            return self.config.mutation_operator
        eligible = ["rewrite", "section_edit"]
        if len(current_text) > COMPRESS_MIN_CHARS:
            eligible.append("compress")
        return self._bandit.select(eligible)

    def _propose(self, current_text: str, component_name: str, asi_entries: list[dict],
                 operator: str, memory: deque[dict] | None = None) -> str:
        kwargs: dict[str, Any] = dict(
            current_text=current_text, component_name=component_name,
            objective=self.objective, asi_entries=asi_entries, background=self.background,
        )
        self.state.operator_counts[operator] = self.state.operator_counts.get(operator, 0) + 1
        if operator != "rewrite":  # default path calls propose() exactly as before
            if self._proposer_takes_operator:
                kwargs["operator"] = operator
            else:
                self._operator_unsupported += 1
        if memory and self._proposer_takes_memory:  # only non-empty when proposal_memory on
            kwargs["rejected_edits"] = list(memory)
        return self.proposer.propose(**kwargs)

    def _remember(self, parent_idx: int, component_name: str, parent_text: str,
                  child_text: str, parent_score: float | None = None,
                  child_score: float | None = None, note: str | None = None) -> None:
        """proposal_memory: keep the last N rejected edits per (parent, component)."""
        if not self.config.proposal_memory:
            return
        diff = "\n".join(difflib.unified_diff(parent_text.splitlines(), child_text.splitlines(),
                                              "parent", "rejected", n=1, lineterm=""))
        if len(diff) > 600:
            diff = diff[:600] + "\n... (truncated)"
        buf = self._memory.setdefault((parent_idx, component_name),
                                      deque(maxlen=self.config.proposal_memory_size))
        buf.append({"diff": diff, "parent_score": parent_score, "child_score": child_score,
                    "note": note})

    @staticmethod
    def _change_ratio(parent_text: str, child_text: str) -> float:
        """1 - difflib similarity: 0 = identical, 1 = nothing in common."""
        return 1.0 - difflib.SequenceMatcher(None, parent_text, child_text,
                                             autojunk=False).ratio()

    def _prescreen(self, parent: dict[str, str], child: dict[str, str],
                   component_name: str) -> str | None:
        """Reason to skip the child before any eval, or None."""
        if not self.config.prescreen:
            return None
        key = components_key(child)
        if key == components_key(parent) or any(
                key == components_key(c.components) for c in self.state.candidates):
            return "prescreen_duplicate"
        threshold = self.config.dedupe_threshold
        if threshold is not None and is_near_duplicate(
                normalize(parent[component_name]), normalize(child[component_name]), threshold):
            return "prescreen_near_duplicate"
        return None

    def _record_train(self, cand_idx: int, batch_ids: list[int],
                      results: list[tuple[float, dict] | None]) -> None:
        """Remember train-minibatch scores already paid for (train saturation + mining)."""
        if not batch_ids:
            return
        known = self._train_scores.setdefault(cand_idx, {})
        for tid, r in zip(batch_ids, results):
            if r is not None:
                known[tid] = r[0]

    def _hard_pool(self) -> set[int]:
        return hard_ids(self._train_scores.get(self._best_index(), {}))

    def _evaluate_halving(self, components: dict[str, str], parent_idx: int,
                          iteration: int) -> dict[str, float] | None:
        """Paired successive halving on val. None = stopped early (child is dropped)."""
        examples = self.valset or self.dataset or [None]
        n = len(examples)
        if self._val_order is None:
            self._val_order = shuffled_order(n, self._seed)
        payload = self._payload(components)
        parent_scores = self.state.scores[parent_idx]
        sizes = rung_sizes(n)
        scores: dict[str, float] = {}
        done = 0
        for rung, size in enumerate(sizes):
            rung_ids = self._val_order[done:size]
            for i, r in zip(rung_ids, self._eval_many(payload, [examples[i] for i in rung_ids])):
                if r is not None:
                    scores[f"ex_{i}"] = r[0]
            if self._truncated:
                self._val_truncated = True
                return None
            done = size
            if rung < len(sizes) - 1 and should_stop_early(
                    scores, parent_scores, self.state.frontier.best_score,
                    self.config.halving_margin):
                self.state.halving_rejects += 1
                self.state.halving_evals_saved += n - done
                self.on_event(events.candidate_rejected(iteration, "halving", evaluated=done,
                                                        saved=n - done))
                return None
        return dict(sorted(scores.items(), key=lambda kv: int(kv[0][3:])))

    def _evaluate_test(self, components: dict[str, str]) -> float | None:
        """Mean score on the test split. Bypasses the cache and total_evals; failures skipped."""
        payload = self._payload(components)
        scores = []
        for example in self.testset or []:
            self._test_evals += 1
            try:
                scores.append(self.evaluator.evaluate(payload, example)[0])
            except EvaluationError:
                continue
        return sum(scores) / len(scores) if scores else None

    def _checkpoint(self) -> None:
        self.state.save(self.run_dir / "checkpoint.json",
                        extra={"cost": self.cost.to_dict(), "seed_score": self._seed_score,
                               "best_idx": self._best_index(), "test_evals": self._test_evals,
                               "operator_unsupported": self._operator_unsupported,
                               "edit_size_rejects": self._edit_size_rejects})

    def _write_result(self, result: OptimizationResult) -> None:
        data = {k: v for k, v in dataclasses.asdict(result).items()
                if k not in ("all_candidates", "all_scores", "pareto_frontier")}
        path = self.run_dir / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def _evaluate_candidate(self, components: dict[str, str]) -> dict[str, float]:
        examples = self.valset or self.dataset or [None]
        payload = self._payload(components)
        scores = {}
        for i, r in enumerate(self._eval_many(payload, list(examples))):
            if r is not None:  # failed examples are left out of scores and the frontier
                scores[f"ex_{i}"] = r[0]
        self._val_truncated = self._truncated > 0
        return scores

    def _should_stop(self) -> bool:
        return self.stopper.should_stop(
            total_evals=self.state.total_evals,
            iteration=self.state.iteration + 1,
            best_score=self.state.best_score(),
            total_cost_usd=self.cost.total_usd,
        )
