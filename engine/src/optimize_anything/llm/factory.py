"""Backend selection and construction."""
from __future__ import annotations

import os
import shutil

from optimize_anything.llm.base import LLMClient, LLMError

BACKENDS = ("anthropic", "openai", "claude-cli", "codex-cli", "mock")


def resolve_backend(name: str | None = None) -> str:
    """Resolve a backend name. An explicit non-auto name wins; then OA_BACKEND; then auto-detect:
    ANTHROPIC_API_KEY -> anthropic, OPENAI_API_KEY -> openai, `claude` on PATH -> claude-cli,
    `codex` on PATH -> codex-cli."""
    if name and name != "auto":
        if name not in BACKENDS:
            raise ValueError(f"Unknown backend {name!r}; choose from {('auto',) + BACKENDS}")
        return name
    env = os.environ.get("OA_BACKEND", "").strip()
    if env and env != "auto":
        return resolve_backend(env)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if shutil.which("claude"):
        return "claude-cli"
    if shutil.which("codex"):
        return "codex-cli"
    raise LLMError(
        "No LLM backend available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, install the "
        "`claude` CLI (Claude subscription) or `codex` CLI, or set OA_BACKEND / config.backend."
    )


def make_llm(backend: str | None = None, *, base_url: str | None = None,
             api_key_env: str | None = None, timeout: float | None = None) -> LLMClient:
    name = resolve_backend(backend)
    kw = {"timeout": timeout} if timeout else {}
    if name == "anthropic":
        from optimize_anything.llm.anthropic_backend import AnthropicLLM
        return AnthropicLLM(**kw)
    if name == "openai":
        from optimize_anything.llm.openai_backend import OpenAILLM
        return OpenAILLM(base_url=base_url, api_key_env=api_key_env or "OPENAI_API_KEY", **kw)
    if name == "claude-cli":
        from optimize_anything.llm.claude_cli import ClaudeCLI
        return ClaudeCLI(**kw)
    if name == "codex-cli":
        from optimize_anything.llm.codex_cli import CodexCLI
        return CodexCLI(**kw)
    from optimize_anything.llm.mock import MockLLM
    return MockLLM()
