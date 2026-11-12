from __future__ import annotations

import pytest

from optimize_anything.core.cost import CostTracker, TrackedLLM
from optimize_anything.llm import LLMResponse, MockLLM
from optimize_anything.strategies.stopping import CostStopper


def _resp(cost: float, inp: int = 10, out: int = 5) -> LLMResponse:
    return LLMResponse(text="x", model="m", input_tokens=inp, output_tokens=out,
                       cache_read_tokens=0, cost_usd=cost, backend="mock", latency_s=0.0)


def test_cost_tracker_totals_by_role():
    t = CostTracker()
    t.record("propose", _resp(0.10))
    t.record("propose", _resp(0.05))
    t.record("judge", _resp(0.01))
    assert t.total_usd == pytest.approx(0.16)
    assert t.calls == 3
    d = t.to_dict()
    assert d["by_role"]["propose"]["usd"] == pytest.approx(0.15)
    assert d["by_role"]["propose"]["calls"] == 2
    assert d["by_role"]["judge"]["input_tokens"] == 10
    assert d["total_usd"] == pytest.approx(0.16)


def test_tracked_llm_records_each_call():
    t = CostTracker()
    llm = TrackedLLM(MockLLM(responses=["ok"], cost_per_call=0.02), t, role="judge")
    llm.complete("p", model="m", max_tokens=5)
    llm.complete("p", model="m", max_tokens=5)
    assert t.by_role["judge"].calls == 2
    assert t.total_usd == pytest.approx(0.04)


def test_cost_stopper():
    t = CostTracker()
    s = CostStopper(0.10, t)
    assert s.should_stop() is False
    t.record("propose", _resp(0.11))
    assert s.should_stop() is True
