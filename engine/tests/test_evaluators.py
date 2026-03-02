# engine/tests/test_evaluators.py
import pytest
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator


def test_python_evaluator_basic():
    code = '''
def evaluate(candidate, example=None):
    score = 1.0 if "hello" in candidate else 0.0
    return score, {"matched": "hello" in candidate}
'''
    ev = PythonEvaluator(code=code, timeout=10)
    score, asi = ev.evaluate("hello world")
    assert score == 1.0
    assert asi["matched"] is True


def test_python_evaluator_with_example():
    code = '''
def evaluate(candidate, example=None):
    expected = example.get("expected", "") if example else ""
    score = 1.0 if candidate.strip() == expected else 0.0
    return score, {"expected": expected, "got": candidate.strip()}
'''
    ev = PythonEvaluator(code=code, timeout=10)
    score, asi = ev.evaluate("foo", example={"expected": "foo"})
    assert score == 1.0


def test_python_evaluator_timeout():
    code = '''
import time
def evaluate(candidate, example=None):
    time.sleep(60)
    return 1.0, {}
'''
    ev = PythonEvaluator(code=code, timeout=2)
    score, asi = ev.evaluate("test")
    assert score == 0.0
    assert "timeout" in asi.get("error", "").lower()


def test_shell_evaluator_basic():
    ev = ShellEvaluator(
        command='echo "Score: 0.85"',
        score_pattern=r"Score: ([\d.]+)",
        timeout=10,
    )
    score, asi = ev.evaluate("anything")
    assert score == 0.85


def test_shell_evaluator_with_candidate():
    ev = ShellEvaluator(
        command='echo "Score: $(echo "{{candidate}}" | wc -c | tr -d " ")"',
        score_pattern=r"Score: (\d+)",
        timeout=10,
    )
    score, asi = ev.evaluate("hello")
    assert score > 0


# LLM judge tests need mocking — tested in integration
