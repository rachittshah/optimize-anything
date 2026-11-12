# engine/src/optimize_anything/evaluators/base.py
from __future__ import annotations
from typing import Any, Protocol


class Evaluator(Protocol):
    """Evaluate a full candidate, return (score in [0, 1], ASI diagnostics).

    What `candidate` is:
    - single-component candidate: the component's text (str)
    - multi-component candidate: the component texts joined with "\n" (str), unless the
      evaluator sets the class attribute `accepts_dict = True`, in which case it gets the
      dict[str, str] of component name -> text.
    """

    def evaluate(
        self, candidate: str | dict[str, str], example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        ...


class EvaluationError(RuntimeError):
    """The evaluation itself failed (e.g. judge API error, unparseable judge output).

    The engine records this as a failed eval: it is excluded from acceptance decisions and
    the Pareto frontier and counted in `failed_evals`, instead of being scored 0. Plain
    candidate failures (code that crashes, wrong output) should still return a low score.
    """
