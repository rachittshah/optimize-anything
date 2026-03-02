from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType

__version__ = "0.1.0"

__all__ = [
    "optimize_anything",
    "Config",
    "EvaluatorConfig",
    "EvaluatorType",
]


def optimize_anything(
    seed_candidate: dict[str, str] | None = None,
    evaluator=None,
    dataset: list | None = None,
    valset: list | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: Config | None = None,
    on_event=None,
):
    """Optimize any text artifact through iterative LLM-powered search."""
    raise NotImplementedError("Engine not yet implemented")
