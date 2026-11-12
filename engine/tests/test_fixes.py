"""Regression tests for the engine foundation bug fixes."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from optimize_anything.api import optimize_anything
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import Engine
from optimize_anything.core.pareto import ParetoFrontier
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.llm import MockLLM
from optimize_anything.proposer.claude_proposer import ClaudeProposer
from optimize_anything.proposer.reflective import ReflectiveProposer
from optimize_anything.strategies.stopping import CompositeStopper, NoImprovementStopper


class AppendProposer:
    def propose(self, current_text, component_name, objective, asi_entries, background=None):
        return current_text + " more"


def _len_eval(candidate, example=None):
    return min(1.0, len(candidate) / 100), {"len": len(candidate)}


# 1. run-dir double nesting
def test_explicit_run_dir_is_final(tmp_path):
    run_dir = tmp_path / "runs" / "mcp123"
    result = optimize_anything(
        seed_candidate={"t": "a"}, evaluator=_len_eval, objective="longer",
        config=Config(max_iterations=2, run_dir=str(run_dir)),
    )
    assert (run_dir / "checkpoint.json").exists()
    assert not any(p.is_dir() for p in run_dir.iterdir())  # no nested uuid dir
    assert result.run_dir == str(run_dir)


def test_default_run_dir_gets_run_id(tmp_path, monkeypatch):
    result = optimize_anything(
        seed_candidate={"t": "a"}, evaluator=_len_eval, objective="longer",
        config=Config(max_iterations=1, run_id="abc"),
    )
    assert Path(result.run_dir).name == "abc"
    assert (Path(result.run_dir) / "checkpoint.json").exists()


# 2. checkpoint + result.json always written
def test_short_run_writes_checkpoint_and_result(tmp_path):
    run_dir = tmp_path / "r"
    engine = Engine(
        seed_candidate={"p": "hi"}, evaluator=type("E", (), {"evaluate": staticmethod(_len_eval)})(),
        proposer=AppendProposer(), config=Config(max_iterations=2, checkpoint_interval=5,
                                                 run_dir=str(run_dir)),
    )
    engine.run()
    ckpt = json.loads((run_dir / "checkpoint.json").read_text())
    assert ckpt["iteration"] == 1
    assert "cost" in ckpt
    res = json.loads((run_dir / "result.json").read_text())
    assert res["best_score"] > 0
    assert res["best_candidate"]["p"].startswith("hi")
    assert {"total_iterations", "total_evals", "cost", "seed_score"} <= res.keys()


# 3. multi-component consistency
class RecordingEvaluator:
    def __init__(self, accepts_dict: bool = False):
        self.accepts_dict = accepts_dict
        self.seen: list = []

    def evaluate(self, candidate, example=None):
        self.seen.append(candidate)
        return 0.1, {}


def test_multi_component_evaluator_gets_joined_text_everywhere():
    ev = RecordingEvaluator()
    Engine(seed_candidate={"a": "AAA", "b": "BBB"}, evaluator=ev, proposer=AppendProposer(),
           config=Config(max_iterations=2, checkpoint_interval=0)).run()
    assert all(isinstance(c, str) and "AAA" in c and "BBB" in c for c in ev.seen)


def test_multi_component_accepts_dict_gets_dict():
    ev = RecordingEvaluator(accepts_dict=True)
    Engine(seed_candidate={"a": "AAA", "b": "BBB"}, evaluator=ev, proposer=AppendProposer(),
           config=Config(max_iterations=2, checkpoint_interval=0)).run()
    assert all(isinstance(c, dict) and set(c) == {"a", "b"} for c in ev.seen)


def test_single_component_gets_string():
    ev = RecordingEvaluator(accepts_dict=True)
    Engine(seed_candidate={"a": "AAA"}, evaluator=ev, proposer=AppendProposer(),
           config=Config(max_iterations=1, checkpoint_interval=0)).run()
    assert all(isinstance(c, str) for c in ev.seen)


def test_python_evaluator_passes_dict_to_user_code():
    code = '''
def evaluate(candidate, example=None):
    return (1.0 if isinstance(candidate, dict) and candidate["b"] == "y" else 0.0), {}
'''
    ev = PythonEvaluator(code=code, timeout=10)
    assert ev.accepts_dict is True
    assert ev.evaluate({"a": "x", "b": "y"})[0] == 1.0


# 4. no eager clients
def test_proposer_and_judge_construct_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ClaudeProposer()
    LLMJudgeEvaluator(criteria="good")


# 5. temperature accepted but ignored
def test_temperature_config_is_deprecated_and_ignored():
    with pytest.warns(DeprecationWarning):
        Config(temperature=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config()  # default must not warn


def test_reflective_proposer_sends_effort_not_temperature():
    llm = MockLLM(responses=["```\nnew text\n```"])
    p = ReflectiveProposer(llm, model="claude-opus-5-5", effort="high", max_tokens=100)
    out = p.propose("old", "c", "obj", [{"score": 0.1, "feedback": {}}])
    assert out == "new text"
    call = llm.calls[0]
    assert call["effort"] == "high" and call["model"] == "claude-opus-5-5"
    assert "temperature" not in call


def test_llm_judge_uses_client_and_reports_cost():
    llm = MockLLM(responses=["fine work\nSCORE: 7/10"], cost_per_call=0.003)
    judge = LLMJudgeEvaluator(criteria="c", llm=llm, model="claude-haiku-5-5", effort="low")
    score, asi = judge.evaluate("cand")
    assert score == pytest.approx(0.7)
    assert asi["judge_cost_usd"] == pytest.approx(0.003)
    assert llm.calls[0]["model"] == "claude-haiku-5-5"


def test_llm_judge_clamps_score():
    judge = LLMJudgeEvaluator(criteria="c", llm=MockLLM(responses=["SCORE: 14/10"]), model="m")
    assert judge.evaluate("x")[0] == 1.0


# 6. Pareto: dominated programs removed before sampling
def test_remove_dominated():
    f = ParetoFrontier()
    # program 0 best on ex1 only; program 1 best (tied) on ex1 and also on ex2 -> 0 dominated
    f.update("ex1", 0.9, 0)
    f.update("ex1", 0.9, 1)
    f.update("ex2", 0.8, 1)
    # program 2 best uniquely on ex3 -> not dominated
    f.update("ex3", 0.7, 2)
    assert f.remove_dominated() == {1, 2}
    picks = {f.select_candidate([0.9, 0.85, 0.7]) for _ in range(200)}
    assert 0 not in picks
    assert picks == {1, 2}


def test_remove_dominated_identical_key_sets_drops_lower_scored():
    # GEPA rule: identical key sets -> the lower aggregate score is removed (one survives).
    f = ParetoFrontier()
    f.update("ex1", 0.5, 0)
    f.update("ex1", 0.5, 1)
    assert f.remove_dominated([0.9, 0.4]) == {0}
    assert f.remove_dominated([0.4, 0.9]) == {1}


# 7. NoImprovementStopper counts iterations, not calls
def test_no_improvement_counts_iterations():
    s = NoImprovementStopper(patience=2)
    for _ in range(5):
        assert s.should_stop(iteration=0, best_score=0.5) is False
    assert s.should_stop(iteration=1, best_score=0.5) is False
    assert s.should_stop(iteration=1, best_score=0.5) is False
    assert s.should_stop(iteration=2, best_score=0.5) is True


def test_no_improvement_resets_on_new_best():
    s = NoImprovementStopper(patience=2)
    s.should_stop(iteration=0, best_score=0.1)
    s.should_stop(iteration=1, best_score=0.1)
    assert s.should_stop(iteration=2, best_score=0.2) is False
    assert s.should_stop(iteration=3, best_score=0.2) is False
    assert s.should_stop(iteration=4, best_score=0.2) is True


def test_composite_stopper_does_not_short_circuit():
    class Always:
        def should_stop(self, **kw):
            return True
    s = NoImprovementStopper(patience=1)
    c = CompositeStopper([Always(), s])
    c.should_stop(iteration=0, best_score=0.5)
    c.should_stop(iteration=1, best_score=0.5)
    assert s.should_stop(iteration=1, best_score=0.5) is True  # it saw both iterations


# Cost in results
def test_result_has_cost_fields():
    result = optimize_anything(
        seed_candidate={"t": "x"}, evaluator=_len_eval, objective="longer",
        config=Config(max_iterations=2, backend="mock"),
    )
    assert result.total_cost_usd == 0.0
    assert result.seed_score == pytest.approx(0.01)
    assert result.cost is not None and "by_role" in result.cost


def test_cost_per_point():
    from optimize_anything.core.engine import _cost_per_point
    assert _cost_per_point(1.0, 0.5, 0.6) == pytest.approx(0.1)  # $1 for 10 points
    assert _cost_per_point(1.0, 0.5, 0.5) is None


def test_max_cost_stops_run():
    proposer_llm = MockLLM(cost_per_call=0.05)
    from optimize_anything.core.cost import CostTracker, TrackedLLM
    tracker = CostTracker()
    proposer = ReflectiveProposer(TrackedLLM(proposer_llm, tracker, "propose"), model="m")
    engine = Engine(seed_candidate={"p": "x"},
                    evaluator=type("E", (), {"evaluate": staticmethod(lambda c, e=None: (0.1, {}))})(),
                    proposer=proposer, cost_tracker=tracker,
                    config=Config(max_iterations=50, max_cost_usd=0.12, checkpoint_interval=0))
    result = engine.run()
    assert len(proposer_llm.calls) == 3  # stops once cumulative cost >= 0.12
    assert result.total_cost_usd == pytest.approx(0.15)


def test_api_uses_judge_model_for_llm_judge(monkeypatch):
    seen = {}
    real = LLMJudgeEvaluator.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        real(self, *a, **kw)

    monkeypatch.setattr(LLMJudgeEvaluator, "__init__", spy)
    optimize_anything(
        seed_candidate={"t": "x"},
        evaluator=EvaluatorConfig(type=EvaluatorType.LLM_JUDGE, criteria="good"),
        objective="o",
        config=Config(max_iterations=1, judge_model="claude-haiku-5-5", judge_effort="low"),
    )
    assert seen["model"] == "claude-haiku-5-5"
    assert seen["effort"] == "low"


def test_model_alias_maps_to_proposer_model():
    assert Config(model="claude-sonnet-5-5").resolved_proposer_model() == "claude-sonnet-5-5"
    assert Config(proposer_model="a", model="b").resolved_proposer_model() == "a"


def test_no_legacy_model_ids_in_source():
    src = Path(__file__).parent.parent / "src"
    for f in src.rglob("*.py"):
        text = f.read_text()
        assert "claude-opus-4-6" not in text and "claude-sonnet-4-6" not in text, f
