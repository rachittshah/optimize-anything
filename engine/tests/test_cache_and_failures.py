"""Eval cache, failed-eval handling, and model presets."""
from __future__ import annotations

import json

import pytest

from optimize_anything.api import optimize_anything
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import Engine
from optimize_anything.evaluators.base import EvaluationError
from optimize_anything.llm.models import PRESETS, resolve_effort, resolve_model


class AppendProposer:
    def __init__(self):
        self.calls = 0

    def propose(self, current_text, component_name, objective, asi_entries, background=None):
        self.calls += 1
        return current_text + f" v{self.calls}"


class CountingEvaluator:
    def __init__(self):
        self.calls: list = []

    def evaluate(self, candidate, example=None):
        self.calls.append((candidate, json.dumps(example)))
        return min(1.0, len(candidate) / 200), {"len": len(candidate)}


# ---------- eval cache ----------

def test_cache_hits_do_not_count_toward_total_evals():
    ev = CountingEvaluator()
    engine = Engine(seed_candidate={"p": "x"}, evaluator=ev, proposer=AppendProposer(),
                    config=Config(max_iterations=3, checkpoint_interval=0))
    result = engine.run()
    # every evaluator call is a unique (candidate, example) pair
    assert len(ev.calls) == len(set(ev.calls))
    assert result.total_evals == len(ev.calls)
    assert result.cache_hits > 0


def test_parent_minibatch_hits_cache_after_full_eval():
    ev = CountingEvaluator()
    dataset = [{"i": 0}, {"i": 1}]
    engine = Engine(seed_candidate={"p": "seed"}, evaluator=ev, proposer=AppendProposer(),
                    dataset=dataset,
                    config=Config(max_iterations=1, reflection_minibatch_size=2, checkpoint_interval=0))
    engine.run()
    seed_calls = [c for c in ev.calls if c[0] == "seed"]
    assert len(seed_calls) == 2  # full eval only; minibatch re-eval came from cache


def test_cache_can_be_disabled():
    ev = CountingEvaluator()
    result = Engine(seed_candidate={"p": "x"}, evaluator=ev, proposer=AppendProposer(),
                    config=Config(max_iterations=3, checkpoint_interval=0, cache_evals=False)).run()
    assert result.cache_hits == 0
    assert len(ev.calls) > len(set(ev.calls))


def test_cache_respects_max_metric_calls():
    ev = CountingEvaluator()
    result = Engine(seed_candidate={"p": "x"}, evaluator=ev, proposer=AppendProposer(),
                    config=Config(max_iterations=100, max_metric_calls=6, checkpoint_interval=0)).run()
    assert result.total_evals == len(ev.calls)
    assert result.total_evals <= 8


# ---------- failed evals ----------

class FailingOnEvaluator:
    """Raises EvaluationError for candidates containing `bad`."""

    def __init__(self, bad: str):
        self.bad = bad

    def evaluate(self, candidate, example=None):
        if self.bad in candidate:
            raise EvaluationError("judge unavailable")
        return 0.1, {}


class FixedProposer:
    def __init__(self, text):
        self.text = text

    def propose(self, *a, **kw):
        return self.text


def test_failed_eval_is_not_accepted_or_scored_zero():
    engine = Engine(seed_candidate={"p": "fine"}, evaluator=FailingOnEvaluator("BAD"),
                    proposer=FixedProposer("BAD text"),
                    config=Config(max_iterations=2, checkpoint_interval=0))
    result = engine.run()
    assert len(result.all_candidates) == 1  # failed candidate never enters the pool
    assert result.failed_evals >= 2
    assert result.best_score == pytest.approx(0.1)


def test_failed_parent_eval_skips_proposal():
    proposer = AppendProposer()
    engine = Engine(seed_candidate={"p": "BAD seed"}, evaluator=FailingOnEvaluator("BAD"),
                    proposer=proposer, config=Config(max_iterations=2, checkpoint_interval=0))
    result = engine.run()
    assert proposer.calls == 0
    assert result.failed_evals >= 3
    assert result.total_iterations == 2


def test_failed_examples_excluded_from_frontier():
    class PartFail:
        def evaluate(self, candidate, example=None):
            if example["i"] == 1:
                raise EvaluationError("boom")
            return 0.5, {}

    engine = Engine(seed_candidate={"p": "s"}, evaluator=PartFail(), proposer=AppendProposer(),
                    dataset=[{"i": 0}, {"i": 1}],
                    config=Config(max_iterations=1, checkpoint_interval=0))
    result = engine.run()
    assert set(result.pareto_frontier["front"]) == {"ex_0"}
    assert result.all_scores[0] == pytest.approx(0.5)


def test_failed_evals_persist_in_checkpoint(tmp_path):
    engine = Engine(seed_candidate={"p": "fine"}, evaluator=FailingOnEvaluator("BAD"),
                    proposer=FixedProposer("BAD"),
                    config=Config(max_iterations=1, run_dir=str(tmp_path / "r")))
    engine.run()
    res = json.loads((tmp_path / "r" / "result.json").read_text())
    ckpt = json.loads((tmp_path / "r" / "checkpoint.json").read_text())
    assert res["failed_evals"] >= 1 and ckpt["failed_evals"] >= 1
    assert "cache_hits" in res


# ---------- presets ----------

def test_presets_defined():
    assert PRESETS["default"]["proposer"] == "claude-opus-5-5"
    assert PRESETS["default"]["judge"] == "claude-sonnet-5-5"
    assert PRESETS["cheap"]["proposer"] == "claude-sonnet-5-5"
    assert PRESETS["cheap"]["judge"] == "claude-haiku-5-5"
    assert PRESETS["quality"]["proposer"] == "claude-fable-5-2"


def test_preset_resolution_and_explicit_override():
    assert resolve_model("judge", "anthropic") == "claude-sonnet-5-5"
    assert resolve_model("judge", "anthropic", preset="cheap") == "claude-haiku-5-5"
    assert resolve_model("proposer", "anthropic", preset="quality") == "claude-fable-5-2"
    assert resolve_model("proposer", "anthropic", "claude-haiku-5-5", preset="quality") == "claude-haiku-5-5"
    assert resolve_effort("proposer", None, "default") == "medium"
    assert resolve_effort("proposer", None, "quality") == "high"
    assert resolve_effort("judge", None, "default") == "low"
    assert resolve_effort("proposer", "xhigh", "default") == "xhigh"
    with pytest.raises(ValueError):
        resolve_model("proposer", "anthropic", preset="nope")


def test_api_applies_preset_to_proposer_and_judge(monkeypatch):
    from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator
    from optimize_anything.proposer.reflective import ReflectiveProposer
    seen: dict = {}
    real_j, real_p = LLMJudgeEvaluator.__init__, ReflectiveProposer.__init__

    def spy_j(self, *a, **kw):
        seen["judge"] = kw
        real_j(self, *a, **kw)

    def spy_p(self, *a, **kw):
        seen["proposer"] = kw
        real_p(self, *a, **kw)

    monkeypatch.setattr(LLMJudgeEvaluator, "__init__", spy_j)
    monkeypatch.setattr(ReflectiveProposer, "__init__", spy_p)
    optimize_anything(seed_candidate={"t": "x"},
                      evaluator=EvaluatorConfig(type=EvaluatorType.LLM_JUDGE, criteria="c"),
                      objective="o", config=Config(max_iterations=1, preset="cheap"))
    assert seen["proposer"]["model"] == "claude-sonnet-5-5"
    assert seen["judge"]["model"] == "claude-haiku-5-5"
    assert seen["proposer"]["effort"] == "medium"
    assert seen["judge"]["effort"] == "low"
