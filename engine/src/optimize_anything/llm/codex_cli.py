"""Codex CLI backend: `codex exec --json`, isolated from user config, read-only sandbox.

Cost note: codex adds ~21.5k input tokens of harness prompt to every call, so it is only
economical with a cheap model (e.g. the "judge"/"cheap" OpenAI default).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time

from optimize_anything.llm.base import LLMError, LLMResponse, check_effort
from optimize_anything.llm.models import compute_cost

_EFFORT_MAP = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh"}
_ARGV_LIMIT = 100_000  # longer prompts go via stdin ("-")


class CodexCLI:
    backend = "codex-cli"

    def __init__(self, executable: str = "codex", timeout: float = 900.0):
        self.executable = executable
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        check_effort(effort)
        full = f"{system}\n\n{prompt}" if system else prompt
        args = [self.executable, "exec", "--json", "--ephemeral", "--ignore-user-config",
                "--skip-git-repo-check", "-s", "read-only", "-m", model]
        if effort:
            args += ["-c", f"model_reasoning_effort={_EFFORT_MAP[effort]}"]
        use_stdin = len(full) > _ARGV_LIMIT
        args.append("-" if use_stdin else full)

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="oa-codex-") as cwd:
            try:
                proc = subprocess.run(
                    args, input=full if use_stdin else None,
                    stdin=None if use_stdin else subprocess.DEVNULL,
                    capture_output=True, text=True, cwd=cwd, timeout=self.timeout,
                )
            except FileNotFoundError as e:
                raise LLMError(f"`{self.executable}` not found on PATH") from e
            except subprocess.TimeoutExpired as e:
                raise LLMError(f"codex exec timed out after {self.timeout}s") from e
        latency = time.monotonic() - start

        text, usage, error = None, {}, None
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            item = ev.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                text = item.get("text", "")
            elif kind == "turn.completed":
                usage = ev.get("usage") or {}
            elif kind in ("turn.failed", "error"):
                err = ev.get("error") or ev
                error = err.get("message") if isinstance(err, dict) else str(err)
        if error or text is None:
            raise LLMError(f"codex exec failed (exit {proc.returncode}): "
                           f"{error or (proc.stderr or 'no agent_message')[:500]}")

        inp = usage.get("input_tokens", 0) or 0
        cached = usage.get("cached_input_tokens", 0) or 0
        out = usage.get("output_tokens", 0) or 0
        cost, warn = compute_cost(model, inp - cached, out, cached)
        return LLMResponse(text=text, model=model, input_tokens=inp, output_tokens=out,
                           cache_read_tokens=cached, cost_usd=cost, backend=self.backend,
                           latency_s=latency, warnings=[warn] if warn else [])
