"""SkillScorer: official score_skill() and the engine evaluator for SKILL.md candidates.

Item-level results (one router decision, one formulation result) are memoized per
(candidate hash, item) as futures on a shared thread pool, so the official score and the
engine's per-example evals reuse the same work. LLM sub-calls are also disk-cached.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from optimize_anything.evaluators.base import EvaluationError
from optimize_anything.llm import LLMClient

from .common import (CACHE_DIR, SKILL_NAME, EvalData, brevity_score, check_skill_text, f1_score,
                     make_shards, shard_score, subsample, weighted_score)
from .formulation import evaluate_formulation
from .llm_cache import CachedLLM, LockedCostTracker
from .router import route


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


class SkillScorer:
    def __init__(self, data: EvalData, llm: LLMClient, tracker: LockedCostTracker, *,
                 env_fn: Callable[[], dict[str, str]], cache_dir: Path | None = CACHE_DIR / "llm",
                 workers: int = 8, limit: int | None = None, effort: str = "low"):
        self.data = data
        self.tracker = tracker
        self.env_fn = env_fn
        self.effort = effort
        self.limit = limit
        self.router_llm = CachedLLM(llm, tracker, "router", cache_dir)
        self.agent_llm = CachedLLM(llm, tracker, "agent", cache_dir)
        self.judge_llm = CachedLLM(llm, tracker, "judge", cache_dir)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skill-eval")
        self._futures: dict[tuple, Future] = {}
        self._lock = threading.Lock()
        self._prefetched: set[str] = set()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ---------------- items ----------------
    def items(self, kind: str, split: str) -> list[dict]:
        rows = self.data.items(kind, split)
        return subsample(rows, self.limit, "kind" if kind == "trigger" else "category")

    def _item_by_id(self, kind: str, iid: str) -> dict:
        rows = self.data.triggers if kind == "trigger" else self.data.formulations
        return next(r for r in rows if r["id"] == iid)

    def _run_item(self, text: str, fm: dict, kind: str, item: dict) -> dict:
        if kind == "trigger":
            pick = route(self.router_llm, fm, self.data.distractors, item, self.effort)
            return {"id": item["id"], "pick": pick, "gold": item["label"] == "trigger",
                    "pred": pick == SKILL_NAME, "kind": item["kind"], "prompt": item["prompt"]}
        return evaluate_formulation(text, item, agent_llm=self.agent_llm, judge_llm=self.judge_llm,
                                    rubric=self.data.rubric, env=self.env_fn(),
                                    effort=self.effort).to_dict()

    def submit(self, text: str, fm: dict, kind: str, item: dict) -> Future:
        key = (text_hash(text), kind, item["id"])
        with self._lock:
            fut = self._futures.get(key)
            if fut is not None and not (fut.done() and fut.exception() is not None):
                return fut
            fut = self.pool.submit(self._run_item, text, fm, kind, item)
            self._futures[key] = fut  # failed futures are replaced on the next submit
            return fut

    # ---------------- official score ----------------
    def score_skill(self, text: str, split: str) -> dict:
        """Official weighted score per split: 0.3 F1 + 0.3 validity + 0.3 quality + 0.1 brevity."""
        splits = ["train", "val"] if split == "both" else [split]
        fm, errors, warnings = check_skill_text(text)
        brev = brevity_score(text, (fm or {}).get("description"))
        if fm is None:
            zero = {"trigger_f1": 0.0, "formulation_validity": 0.0, "formulation_quality": 0.0,
                    "brevity": brev}
            return {s: {"components": zero, "weighted": 0.0, "flags": errors + warnings,
                        "malformed": True} for s in splits}
        futs = {s: ([(it, self.submit(text, fm, "trigger", it)) for it in self.items("trigger", s)],
                    [(it, self.submit(text, fm, "formulation", it)) for it in self.items("formulation", s)])
                for s in splits}
        out = {}
        for s in splits:
            trig, form = [], []
            failed: list[dict] = []
            for it, f in futs[s][0]:
                try:
                    trig.append(f.result())
                except EvaluationError as e:
                    failed.append({"id": it["id"], "error": str(e)[:300]})
            for it, f in futs[s][1]:
                try:
                    form.append(f.result())
                except EvaluationError as e:
                    failed.append({"id": it["id"], "error": str(e)[:300]})
            f1 = f1_score([r["gold"] for r in trig], [r["pred"] for r in trig]) if trig else None
            val = sum(r["validity"] for r in form) / len(form) if form else None
            qual = sum(r["quality"] for r in form) / len(form) if form else None
            comps = {"trigger_f1": f1, "formulation_validity": val, "formulation_quality": qual,
                     "brevity": brev}
            out[s] = {
                "components": comps, "weighted": weighted_score(comps),
                "n_trigger": len(trig), "n_formulation": len(form), "failed": failed,
                "flags": warnings,
                "trigger_confusion": {
                    "tp": sum(r["gold"] and r["pred"] for r in trig),
                    "fp": sum(r["pred"] and not r["gold"] for r in trig),
                    "fn": sum(r["gold"] and not r["pred"] for r in trig),
                    "tn": sum(not r["gold"] and not r["pred"] for r in trig)},
                "misrouted": [{"id": r["id"], "kind": r["kind"], "pick": r["pick"]}
                              for r in trig if r["gold"] != r["pred"]],
                "valid_low_quality": [r["id"] for r in form if r["validity"] and r["quality"] < 0.5],
                "invalid_high_quality": [r["id"] for r in form if not r["validity"] and r["quality"] >= 0.7],
                "formulations": [{k: r[k] for k in ("id", "validity", "quality", "mode",
                                                    "validity_reason", "criteria", "critique")}
                                 for r in form],
                "agent_outputs": {r["id"]: r["agent_output"] for r in form},
            }
        return out

    # ---------------- engine examples ----------------
    def build_examples(self, split: str) -> list[dict]:
        """Trigger shards (~5 items, stratified) + one example per formulation item.
        Examples carry only ids; item content is looked up here and never sent to the engine."""
        ex = [{"kind": "trigger_shard", "split": split, "id": f"shard-{split}-{i}",
               "item_ids": [r["id"] for r in shard]}
              for i, shard in enumerate(make_shards(self.items("trigger", split)))]
        ex += [{"kind": "formulation", "split": split, "id": r["id"]}
               for r in self.items("formulation", split)]
        return ex

    def _prefetch(self, text: str, fm: dict, split: str) -> None:
        key = f"{text_hash(text)}:{split}"
        if key in self._prefetched:
            return
        self._prefetched.add(key)
        for it in self.items("trigger", split):
            self.submit(text, fm, "trigger", it)
        for it in self.items("formulation", split):
            self.submit(text, fm, "formulation", it)

    def evaluate(self, candidate: str | dict, example: dict | None) -> tuple[float, dict[str, Any]]:
        text = candidate if isinstance(candidate, str) else next(iter(candidate.values()))
        fm, errors, warnings = check_skill_text(text)
        if fm is None:
            return 0.0, {"error": "malformed SKILL.md, nothing can route to it: " + "; ".join(errors)}
        if example is None:
            raise EvaluationError("skill evaluator needs an example")
        brev = brevity_score(text, fm.get("description"))
        lines = len(text.splitlines())
        base = {"brevity": f"{brev:.2f} ({lines} lines; full credit at <=150, zero at 300; "
                           f"description {len(fm['description'])}/1024 chars)"}
        if warnings:
            base["flags"] = warnings
        if example["split"] == "val":  # the engine evaluates the whole valset: run it in parallel
            self._prefetch(text, fm, "val")

        if example["kind"] == "trigger_shard":
            items = [self._item_by_id("trigger", i) for i in example["item_ids"]]
            futs = [self.submit(text, fm, "trigger", it) for it in items]
            rows = [f.result() for f in futs]  # EvaluationError propagates -> eval excluded
            s = shard_score([r["gold"] for r in rows], [r["pred"] for r in rows])
            miss = []
            for r in rows:
                if r["gold"] != r["pred"]:
                    want = (f"route to {SKILL_NAME}" if r["gold"]
                            else f"NOT route to {SKILL_NAME} ({r['kind'].replace('_', '-')})")
                    miss.append(f"should {want}; router picked {r['pick']!r}: \"{_trunc(r['prompt'], 220)}\"")
            asi = {"example": f"trigger routing, {len(rows)} prompts (router sees only name + description)",
                   "routing_score": f"{s:.2f} ({'F1' if any(r['gold'] for r in rows) else 'accuracy'}); "
                                    f"{sum(r['gold'] == r['pred'] for r in rows)}/{len(rows)} correct",
                   "misrouted": miss or "none", **base}
            return 0.9 * s + 0.1 * brev, asi

        item = self._item_by_id("formulation", example["id"])
        r = self.submit(text, fm, "formulation", item).result()
        comp = 0.5 * r["validity"] + 0.5 * r["quality"]
        asi = {"example": f"formulation {item['id']} ({item['category']})",
               "request": _trunc(item["request"], 400),
               "validity": f"{'VALID' if r['validity'] else 'INVALID'}: {_trunc(r['validity_reason'], 500)}",
               "quality": f"{round(r['quality'] * 14)}/14 {r['criteria']}",
               "judge_critique": r["critique"],
               "what_a_good_answer_does": item.get("rubric_notes", ""),
               "agent_reply": _trunc(r["agent_output"], 1500), **base}
        if r.get("warnings"):
            asi["call_warnings"] = r["warnings"]
        return 0.9 * comp + 0.1 * brev, asi


class EngineEvaluator:
    """Adapter the engine calls: evaluate(candidate, example) + a stable cache identity."""
    accepts_dict = False

    def __init__(self, scorer: SkillScorer):
        self.scorer = scorer

    def cache_identity(self) -> str:
        return "skill-eval:v1"

    def evaluate(self, candidate, example=None):
        return self.scorer.evaluate(candidate, example)
