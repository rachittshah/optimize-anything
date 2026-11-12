from __future__ import annotations

import pytest

_OA_ENV = ("OA_BACKEND", "OA_PROPOSER_MODEL", "OA_JUDGE_MODEL", "OA_OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch, tmp_path):
    """Force the mock backend and a temp run dir so no test ever calls a real LLM
    or writes into ~/.optimize-anything."""
    for name in _OA_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OA_BACKEND", "mock")
    monkeypatch.setenv("OA_RUNS_DIR", str(tmp_path / "runs"))
    yield
