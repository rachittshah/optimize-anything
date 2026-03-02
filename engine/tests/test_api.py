from unittest.mock import patch, MagicMock

from optimize_anything.api import optimize_anything
from optimize_anything.config import Config


def test_api_single_task_mode():
    def simple_eval(candidate, example=None):
        return min(1.0, len(candidate) / 50), {"length": len(candidate)}

    with patch("optimize_anything.api.ClaudeProposer") as MockProposer:
        mock_instance = MagicMock()
        mock_instance.propose.return_value = "hello world this is a longer text now"
        MockProposer.return_value = mock_instance

        result = optimize_anything(
            seed_candidate={"text": "hello"},
            evaluator=simple_eval,
            objective="make it longer",
            config=Config(max_iterations=3),
        )
        assert result.best_score > 0
        assert "text" in result.best_candidate


def test_api_with_dataset():
    def eval_fn(candidate, example=None):
        target = example.get("target", "") if example else ""
        overlap = len(set(candidate.split()) & set(target.split()))
        return overlap / max(len(target.split()), 1), {}

    with patch("optimize_anything.api.ClaudeProposer") as MockProposer:
        mock_instance = MagicMock()
        mock_instance.propose.return_value = "hello world foo bar"
        MockProposer.return_value = mock_instance

        result = optimize_anything(
            seed_candidate={"answer": "hello world"},
            evaluator=eval_fn,
            dataset=[{"target": "hello world foo"}, {"target": "hello bar"}],
            objective="match targets",
            config=Config(max_iterations=2),
        )
        assert result.total_iterations >= 1
