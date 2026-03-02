# engine/tests/test_integration.py
"""End-to-end integration test using mock proposer (no API calls)."""
from unittest.mock import patch, MagicMock
from optimize_anything.api import optimize_anything
from optimize_anything.config import Config


def test_e2e_single_task():
    """Single-task mode: optimize a string to contain 'hello world'."""
    call_count = 0

    def evaluator(candidate, example=None):
        words = {"hello", "world", "foo", "bar"}
        found = sum(1 for w in words if w in candidate)
        return found / len(words), {"found": found, "total": len(words)}

    with patch("optimize_anything.api.ClaudeProposer") as MockP:
        mock = MagicMock()
        responses = ["hello there", "hello world", "hello world foo", "hello world foo bar"]
        mock.propose.side_effect = lambda **kw: responses[min(mock.propose.call_count - 1, len(responses) - 1)]
        MockP.return_value = mock

        result = optimize_anything(
            seed_candidate={"text": "nothing"},
            evaluator=evaluator,
            objective="include all target words",
            config=Config(max_iterations=4),
        )

    assert result.best_score > 0
    assert result.total_iterations >= 1


def test_e2e_with_dataset():
    """Multi-task mode: optimize across multiple examples."""
    def evaluator(candidate, example=None):
        if example and "target" in example:
            match = example["target"] in candidate
            return (1.0 if match else 0.0), {"target": example["target"], "matched": match}
        return 0.5, {}

    with patch("optimize_anything.api.ClaudeProposer") as MockP:
        mock = MagicMock()
        mock.propose.return_value = "I contain apple and banana"
        MockP.return_value = mock

        result = optimize_anything(
            seed_candidate={"text": "empty"},
            evaluator=evaluator,
            dataset=[{"target": "apple"}, {"target": "banana"}, {"target": "cherry"}],
            objective="contain all targets",
            config=Config(max_iterations=3, reflection_minibatch_size=2),
        )

    assert result.total_iterations >= 1
    assert len(result.all_candidates) >= 1
