from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from optimize_anything.llm import (
    LLMError,
    LLMRefusalError,
    LLMResponse,
    MockLLM,
    make_llm,
    resolve_backend,
)
from optimize_anything.llm.anthropic_backend import AnthropicLLM
from optimize_anything.llm.claude_cli import ClaudeCLI
from optimize_anything.llm.codex_cli import CodexCLI
from optimize_anything.llm.models import compute_cost, resolve_model
from optimize_anything.llm.openai_backend import OpenAILLM


# ---------- models / pricing ----------

def test_compute_cost_known_model():
    # opus 5.5: $4 in / $20 out per 1M, cache read $0.20
    cost, warn = compute_cost("claude-opus-5-5", input_tokens=1_000_000, output_tokens=100_000,
                              cache_read_tokens=1_000_000)
    assert cost == pytest.approx(4 + 2 + 0.20)
    assert warn is None


def test_compute_cost_unknown_model_is_zero_with_warning():
    cost, warn = compute_cost("some-unknown-model", input_tokens=10, output_tokens=10)
    assert cost == 0.0
    assert "unknown" in warn.lower()


def test_resolve_model_role_defaults():
    assert resolve_model("proposer", "anthropic") == "claude-opus-5-5"
    assert resolve_model("judge", "claude-cli") == "claude-sonnet-5-5"  # "default" preset
    assert not resolve_model("proposer", "openai").startswith("claude-")


def test_resolve_model_env_override(monkeypatch):
    monkeypatch.setenv("OA_PROPOSER_MODEL", "claude-sonnet-5-5")
    assert resolve_model("proposer", "anthropic") == "claude-sonnet-5-5"
    # explicit beats env
    assert resolve_model("proposer", "anthropic", "claude-fable-5-2") == "claude-fable-5-2"


def test_resolve_model_rejects_claude_on_openai_backends():
    with pytest.raises(ValueError, match="claude-"):
        resolve_model("proposer", "openai", "claude-opus-5-5")
    with pytest.raises(ValueError, match="claude-"):
        resolve_model("judge", "codex-cli", "claude-haiku-5-5")


# ---------- backend resolution ----------

def test_resolve_backend_explicit_and_env(monkeypatch):
    assert resolve_backend("anthropic") == "anthropic"
    monkeypatch.setenv("OA_BACKEND", "claude-cli")
    assert resolve_backend(None) == "claude-cli"
    assert resolve_backend("auto") == "claude-cli"  # env decides when config says auto


def test_resolve_backend_auto_order(monkeypatch):
    monkeypatch.delenv("OA_BACKEND")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("optimize_anything.llm.factory.shutil.which", return_value=None):
        with pytest.raises(LLMError, match="No LLM backend"):
            resolve_backend("auto")
    with patch("optimize_anything.llm.factory.shutil.which",
               side_effect=lambda n: "/x/codex" if n == "codex" else None):
        assert resolve_backend("auto") == "codex-cli"
    with patch("optimize_anything.llm.factory.shutil.which", return_value="/x/bin"):
        assert resolve_backend("auto") == "claude-cli"
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert resolve_backend("auto") == "openai"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert resolve_backend("auto") == "anthropic"


def test_make_llm_returns_backend_types():
    assert isinstance(make_llm("mock"), MockLLM)
    assert isinstance(make_llm("anthropic"), AnthropicLLM)
    assert isinstance(make_llm("claude-cli"), ClaudeCLI)
    assert isinstance(make_llm("codex-cli"), CodexCLI)
    assert isinstance(make_llm("openai", base_url="https://openrouter.ai/api/v1"), OpenAILLM)


def test_backends_construct_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    AnthropicLLM()  # must not raise: client is lazy
    OpenAILLM()


# ---------- mock ----------

def test_mock_echoes_last_fenced_block_and_is_free():
    llm = MockLLM()
    r = llm.complete("intro\n```\nold text\n```\nmore", model="m", max_tokens=10)
    assert "```\nold text\n```" in r.text
    assert "SCORE:" in r.text
    assert r.cost_usd == 0.0 and r.backend == "mock"


def test_mock_canned_responses_cycle():
    llm = MockLLM(responses=["a", "b"])
    assert [llm.complete("p", model="m", max_tokens=1).text for _ in range(3)] == ["a", "b", "b"]
    assert len(llm.calls) == 3


# ---------- anthropic ----------

def _anthropic_message(text="hi", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def test_anthropic_request_has_effort_and_no_temperature():
    llm = AnthropicLLM()
    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_message("answer")
    llm._client = fake
    r = llm.complete("p", system="sys", model="claude-haiku-5-5", max_tokens=50, effort="low")
    kwargs = fake.messages.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert "thinking" not in kwargs
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["system"] == "sys"
    assert r.text == "answer"
    assert r.input_tokens == 1000 and r.output_tokens == 200
    assert r.cost_usd == pytest.approx(1000 * 0.10e-6 + 200 * 0.50e-6)
    assert r.backend == "anthropic"


def test_anthropic_refusal_raises():
    llm = AnthropicLLM()
    fake = MagicMock()
    fake.messages.create.return_value = _anthropic_message("", stop_reason="refusal")
    llm._client = fake
    with pytest.raises(LLMRefusalError):
        llm.complete("p", model="claude-haiku-5-5", max_tokens=5)


def test_anthropic_rejects_bad_effort():
    with pytest.raises(ValueError):
        AnthropicLLM().complete("p", model="claude-haiku-5-5", max_tokens=5, effort="extreme")


# ---------- openai ----------

def test_openai_backend_parses_usage_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    llm = OpenAILLM(base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY")
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(
        model="gpt-6-luna",
        choices=[SimpleNamespace(message=SimpleNamespace(content="out", refusal=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
    )
    llm._client = fake
    r = llm.complete("p", system="s", model="gpt-6-luna", max_tokens=10, effort="low")
    assert r.text == "out" and r.backend == "openai"
    assert r.cost_usd == pytest.approx(1000 * 0.10e-6 + 100 * 0.50e-6)
    msgs = fake.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "s"}


def test_openai_unknown_model_cost_zero_with_warning():
    llm = OpenAILLM()
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(
        model="mystery",
        choices=[SimpleNamespace(message=SimpleNamespace(content="x", refusal=None), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, prompt_tokens_details=None),
    )
    llm._client = fake
    r = llm.complete("p", model="mystery", max_tokens=10)
    assert r.cost_usd == 0.0
    assert r.warnings


# ---------- claude-cli ----------

def _completed(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_claude_cli_isolated_invocation_and_parse():
    payload = {
        "result": "PONG", "is_error": False, "total_cost_usd": 0.00026,
        "usage": {"input_tokens": 2, "output_tokens": 5, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 659},
    }
    with patch("optimize_anything.llm.claude_cli.subprocess.run",
               return_value=_completed(json.dumps(payload))) as run:
        r = ClaudeCLI().complete("hello", model="claude-haiku-5-5", max_tokens=100, effort="low")
    args = run.call_args.args[0]
    kwargs = run.call_args.kwargs
    assert args[:2] == ["claude", "-p"]
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in args
    assert args[args.index("--setting-sources") + 1] == ""
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "claude-haiku-5-5"
    assert args[args.index("--effort") + 1] == "low"
    assert args[args.index("--output-format") + 1] == "json"
    assert "--system-prompt" in args
    assert "--bare" not in args
    assert kwargs["input"] == "hello"  # prompt via stdin, never argv
    assert "hello" not in args
    assert kwargs["cwd"]  # isolated temp dir
    assert r.text == "PONG"
    assert r.cost_usd == pytest.approx(0.00026)
    assert r.output_tokens == 5
    assert r.backend == "claude-cli"


def test_claude_cli_error_raises():
    payload = {"result": "Credit balance too low", "is_error": True, "total_cost_usd": 0}
    with patch("optimize_anything.llm.claude_cli.subprocess.run",
               return_value=_completed(json.dumps(payload), returncode=1)):
        with pytest.raises(LLMError, match="Credit balance"):
            ClaudeCLI().complete("x", model="claude-haiku-5-5", max_tokens=10)


# ---------- codex-cli ----------

def test_codex_cli_parses_jsonl():
    lines = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "final answer"}},
        {"type": "turn.completed", "usage": {"input_tokens": 21500, "cached_input_tokens": 20000,
                                             "output_tokens": 50}},
    ]
    stdout = "\n".join(json.dumps(x) for x in lines)
    with patch("optimize_anything.llm.codex_cli.subprocess.run", return_value=_completed(stdout)) as run:
        r = CodexCLI().complete("hi", system="sys", model="gpt-6-luna", max_tokens=10)
    args = run.call_args.args[0]
    for flag in ("exec", "--json", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check"):
        assert flag in args
    assert args[args.index("-s") + 1] == "read-only"
    assert args[args.index("-m") + 1] == "gpt-6-luna"
    assert r.text == "final answer"
    assert r.input_tokens == 21500 and r.cache_read_tokens == 20000 and r.output_tokens == 50
    # 1500 uncached * 0.10 + 20000 cached * cache_read + 50 * 0.50 (per 1M)
    assert r.cost_usd > 0
    assert r.backend == "codex-cli"


def test_codex_cli_failure_raises():
    stdout = json.dumps({"type": "turn.failed", "error": {"message": "model not found"}})
    with patch("optimize_anything.llm.codex_cli.subprocess.run", return_value=_completed(stdout, 1)):
        with pytest.raises(LLMError, match="model not found"):
            CodexCLI().complete("hi", model="gpt-6-luna", max_tokens=10)
