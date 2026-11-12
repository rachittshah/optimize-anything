"""Proposers injected into Engine by the bench (independent of engine/src proposer code).

ClaudeCLIProposer calls an isolated `claude -p` (subscription auth, no user config, no tools).
MockProposer is deterministic and LLM-free, for tests.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from typing import Any

from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate

SYSTEM_PROMPT = "You are an expert optimizer. Output the improved artifact in one fenced code block."
_HEADER_RE = re.compile(r"^###\s*(\w+)\s*$")


def _restore_header(current_text: str, new_text: str) -> str:
    """Multi-component bench candidates keep a '### <name>' header line; put it back if the LLM dropped it."""
    first = current_text.strip().splitlines()[0] if current_text.strip() else ""
    if _HEADER_RE.match(first) and not new_text.lstrip().startswith(first):
        return f"{first}\n{new_text.strip()}"
    return new_text


class _Base:
    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self.total_cost_usd = 0.0
        self.total_latency_s = 0.0


class ClaudeCLIProposer(_Base):
    def __init__(self, model: str = "claude-haiku-5-5", timeout_s: float = 120.0) -> None:
        super().__init__()
        self.model = model
        self.timeout_s = timeout_s

    def propose(self, current_text: str, component_name: str, objective: str,
                asi_entries: list[dict[str, Any]], background: str | None = None, **_kw) -> str:
        # Forward a non-default mutation operator (Config.mutation_operator / operator_bandit)
        # and rejected-edit memory (Config.proposal_memory); omitted when unset so older engine
        # checkouts (--engine-src) keep working.
        op_kw: dict[str, Any] = {}
        if _kw.get("operator", "rewrite") != "rewrite":
            op_kw["operator"] = _kw["operator"]
        if _kw.get("rejected_edits"):
            op_kw["rejected_edits"] = _kw["rejected_edits"]
        prompt = build_reflection_prompt(
            current_text=current_text, component_name=component_name, objective=objective,
            asi_entries=asi_entries, background=background, **op_kw,
        )
        cmd = [
            "claude", "-p", prompt, "--model", self.model,
            "--setting-sources", "", "--strict-mcp-config", "--tools", "",
            "--disable-slash-commands", "--no-session-persistence",
            "--system-prompt", SYSTEM_PROMPT, "--output-format", "json",
        ]
        self.calls += 1
        t0 = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="oa-bench-proposer-") as cwd:
                proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                      cwd=cwd, timeout=self.timeout_s)
            data = json.loads(proc.stdout)
            self.total_cost_usd += float(data.get("total_cost_usd") or 0.0)
            if proc.returncode != 0 or data.get("is_error"):
                raise RuntimeError(f"claude exit {proc.returncode}: {str(data.get('result'))[:200]}")
            text = extract_candidate(data["result"])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, RuntimeError, OSError) as e:
            self.failures += 1
            self.last_error = repr(e)
            return current_text
        finally:
            self.total_latency_s += time.monotonic() - t0
        return _restore_header(current_text, text)


class MockProposer(_Base):
    """Deterministic: the first proposal per component returns the task's reference text for that
    component (so the accept + full-eval path runs); later proposals return the current text
    unchanged, which the engine rejects."""

    def __init__(self, reference: dict[str, str]) -> None:
        super().__init__()
        self.reference = reference
        self._used: set[str] = set()

    def propose(self, current_text: str, component_name: str, objective: str,
                asi_entries: list[dict[str, Any]], background: str | None = None, **_kw) -> str:
        self.calls += 1
        if component_name not in self._used and component_name in self.reference:
            self._used.add(component_name)
            return self.reference[component_name]
        return current_text


def make_proposer(kind: str, model: str, reference: dict[str, str]):
    if kind == "mock":
        return MockProposer(reference)
    if kind == "claude":
        return ClaudeCLIProposer(model=model)
    raise ValueError(f"unknown proposer {kind!r}")
