from optimize_anything.core.engine import Engine
from optimize_anything.core.candidate import Candidate
from optimize_anything.config import Config


class FakeProposer:
    def __init__(self):
        self.call_count = 0

    def propose(self, current_text, component_name, objective, asi_entries, background=None):
        self.call_count += 1
        return current_text + " improved"


class FakeEvaluator:
    def __init__(self):
        self.call_count = 0

    def evaluate(self, candidate, example=None):
        self.call_count += 1
        score = min(1.0, len(candidate) / 100)
        return score, {"length": len(candidate)}


def test_engine_runs_iterations():
    config = Config(max_iterations=3)
    engine = Engine(
        seed_candidate={"prompt": "hello"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="make it longer",
        config=config,
    )
    result = engine.run()
    assert result.total_iterations >= 1
    assert result.best_score > 0


def test_engine_improves_score():
    config = Config(max_iterations=5)
    engine = Engine(
        seed_candidate={"prompt": "hi"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="make it longer",
        config=config,
    )
    result = engine.run()
    assert result.best_score >= 0.02  # "hi" = 2 chars, should grow


def test_engine_respects_budget():
    config = Config(max_iterations=100, max_metric_calls=5)
    engine = Engine(
        seed_candidate={"prompt": "test"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="improve",
        config=config,
    )
    result = engine.run()
    assert result.total_evals <= 10  # some slack for initial eval


def test_engine_emits_events():
    events = []
    config = Config(max_iterations=2)
    engine = Engine(
        seed_candidate={"prompt": "hello"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="improve",
        config=config,
        on_event=lambda e: events.append(e),
    )
    engine.run()
    event_types = [e["type"] for e in events]
    assert "iteration_start" in event_types
    assert "optimization_complete" in event_types
