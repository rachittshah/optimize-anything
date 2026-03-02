from optimize_anything.api import optimize_anything
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import OptimizationResult

__version__ = "0.1.0"

__all__ = [
    "optimize_anything",
    "Config",
    "EvaluatorConfig",
    "EvaluatorType",
    "OptimizationResult",
]
