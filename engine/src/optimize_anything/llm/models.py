"""Single source of truth for default model IDs and per-token pricing.

Nothing outside this module should hardcode a model ID.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens
    cache_read: float  # USD per 1M cache-read tokens
    source: str = "Anthropic list price"
    verified: bool = True


PRICING: dict[str, ModelPrice] = {
    # Anthropic
    "claude-fable-5-2": ModelPrice(8.0, 40.0, 0.20),
    "claude-opus-5-5": ModelPrice(4.0, 20.0, 0.20),
    "claude-sonnet-5-5": ModelPrice(2.0, 10.0, 0.10),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01, source="cache-read price assumed", verified=False),
    # OpenAI: source = OpenRouter list price 2026-11-11, VERIFY against official pricing.
    "gpt-6.1-sol": ModelPrice(2.0, 10.0, 0.20, source="OpenRouter list price 2026-11-11, verify", verified=False),
    "gpt-6-luna": ModelPrice(0.10, 0.50, 0.01, source="OpenRouter list price 2026-11-11, verify", verified=False),
    "gpt-6-astra": ModelPrice(10.0, 50.0, 1.0, source="OpenRouter list price 2026-11-11, verify", verified=False),
}

# Anthropic 5-minute cache writes bill at 1.25x the input price.
CACHE_WRITE_MULTIPLIER = 1.25

# Presets: role -> model and role_effort -> effort. Explicit config fields and OA_*_MODEL env
# override a preset. The same judge model is used for the minibatch gate and the full eval.
PRESETS: dict[str, dict[str, str]] = {
    "default": {"proposer": "claude-opus-5-5", "proposer_effort": "medium",
                "judge": "claude-sonnet-5-5", "judge_effort": "low", "cheap": "claude-haiku-5-5"},
    "cheap": {"proposer": "claude-sonnet-5-5", "proposer_effort": "medium",
              "judge": "claude-haiku-5-5", "judge_effort": "low", "cheap": "claude-haiku-5-5"},
    "quality": {"proposer": "claude-fable-5-2", "proposer_effort": "high",
                "judge": "claude-sonnet-5-5", "judge_effort": "low", "cheap": "claude-haiku-5-5"},
}
# Same presets on OpenAI-only backends (openai, codex-cli). Prices unverified, see PRICING.
OPENAI_PRESETS: dict[str, dict[str, str]] = {
    "default": {"proposer": "gpt-6.1-sol", "judge": "gpt-6-luna", "cheap": "gpt-6-luna"},
    "cheap": {"proposer": "gpt-6-luna", "judge": "gpt-6-luna", "cheap": "gpt-6-luna"},
    "quality": {"proposer": "gpt-6-astra", "judge": "gpt-6.1-sol", "cheap": "gpt-6-luna"},
}

OPENAI_BACKENDS = ("openai", "codex-cli")
ROLE_ENV = {"proposer": "OA_PROPOSER_MODEL", "judge": "OA_JUDGE_MODEL"}


def _preset(name: str, backend: str) -> dict[str, str]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {tuple(PRESETS)}")
    table = OPENAI_PRESETS if backend in OPENAI_BACKENDS else PRESETS
    return table[name]


def _strip_provider(model: str) -> str:
    # OpenRouter IDs look like "openai/gpt-6-luna" or "anthropic/claude-haiku-5-5".
    return model.split("/", 1)[1] if "/" in model else model


def default_model(role: str, backend: str, preset: str = "default") -> str:
    return _preset(preset, backend)[role]


def resolve_effort(role: str, explicit: str | None = None, preset: str = "default") -> str:
    """Effort for a role: explicit config > preset > "low"."""
    return explicit or PRESETS[preset if preset in PRESETS else "default"].get(f"{role}_effort", "low")


def resolve_model(role: str, backend: str, explicit: str | None = None,
                  preset: str = "default") -> str:
    """Pick a model for a role. Precedence: explicit config > OA_*_MODEL env > preset.

    Raises ValueError if a claude-* model is requested on an OpenAI-only backend, rather
    than silently sending a request that will fail.
    """
    fallback = default_model(role, backend, preset)  # validates the preset name
    model = explicit or os.environ.get(ROLE_ENV.get(role, ""), "") or fallback
    if backend in OPENAI_BACKENDS and model.startswith("claude-"):
        raise ValueError(
            f"Model {model!r} is a claude- model but backend is {backend!r}. "
            f"Set {role}_model to an OpenAI model (default {fallback!r}) "
            "or use backend 'anthropic'/'claude-cli'."
        )
    return model


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> tuple[float, str | None]:
    """Return (usd, warning). input_tokens excludes cache reads/writes. Unknown model -> (0.0, warning)."""
    price = PRICING.get(_strip_provider(model))
    if price is None:
        return 0.0, f"unknown model {model!r}: cost not tracked (add it to llm/models.py PRICING)"
    usd = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_tokens * price.cache_read
        + cache_write_tokens * price.input * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000
    return usd, None
