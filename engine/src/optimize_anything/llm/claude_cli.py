"""Claude Code CLI backend: runs `claude -p` fully isolated, on subscription auth.

Isolation: no settings sources (so no user CLAUDE.md, hooks, plugins), no MCP, no tools,
no slash commands, no session file, a fixed system prompt, and an empty temp cwd so no
project CLAUDE.md is discovered. The prompt goes in on stdin (no argv length limit).

Do NOT add `--bare`: it requires ANTHROPIC_API_KEY. If ANTHROPIC_API_KEY is set in the
environment the CLI will bill that key instead of the subscription.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time

from optimize_anything.llm.base import LLMError, LLMResponse, check_effort

DEFAULT_SYSTEM = "You are a precise assistant."


class ClaudeCLI:
    backend = "claude-cli"

    def __init__(self, executable: str = "claude", timeout: float = 900.0):
        self.executable = executable
        self.timeout = timeout

    def build_args(self, model: str, system: str | None, effort: str | None) -> list[str]:
        args = [
            self.executable, "-p",
            "--model", model,
            "--setting-sources", "",
            "--strict-mcp-config",
            "--tools", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt", system or DEFAULT_SYSTEM,
            "--output-format", "json",
        ]
        if effort:
            args += ["--effort", effort]
        return args

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        # max_tokens has no CLI flag; the CLI applies the model's own output cap.
        check_effort(effort)
        args = self.build_args(model, system, effort)
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="oa-claude-") as cwd:
            try:
                proc = subprocess.run(args, input=prompt, capture_output=True, text=True,
                                      cwd=cwd, timeout=self.timeout)
            except FileNotFoundError as e:
                raise LLMError(f"`{self.executable}` not found on PATH") from e
            except subprocess.TimeoutExpired as e:
                raise LLMError(f"claude -p timed out after {self.timeout}s") from e
        latency = time.monotonic() - start

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"claude -p returned non-JSON (exit {proc.returncode}): "
                           f"{(proc.stderr or proc.stdout)[:500]}") from e
        if data.get("is_error") or proc.returncode != 0:
            raise LLMError(f"claude -p error (exit {proc.returncode}): {data.get('result') or proc.stderr[:500]}")

        u = data.get("usage") or {}
        cw = u.get("cache_creation_input_tokens", 0) or 0
        return LLMResponse(
            text=data.get("result") or "",
            model=model,
            input_tokens=(u.get("input_tokens", 0) or 0) + cw,
            output_tokens=u.get("output_tokens", 0) or 0,
            cache_read_tokens=u.get("cache_read_input_tokens", 0) or 0,
            cache_write_tokens=cw,
            # The CLI reports API-equivalent list price even on subscription auth.
            cost_usd=float(data.get("total_cost_usd") or 0.0),
            backend=self.backend,
            latency_s=latency,
        )
