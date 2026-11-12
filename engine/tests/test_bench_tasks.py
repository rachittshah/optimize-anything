"""Tests for the engine benchmark harness in scripts/bench/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(BENCH_DIR))

import bench_tasks as bt  # noqa: E402
import run_bench  # noqa: E402
import summarize  # noqa: E402
from bench_proposer import _restore_header  # noqa: E402


@pytest.fixture(scope="module", params=bt.TASK_NAMES)
def task(request):
    return bt.get_task(request.param)


def test_splits_sizes_and_disjoint(task):
    s = task.splits
    assert [len(s[k]) for k in ("train", "val", "test")] == [24, 18, 18]
    ids = [{ex["id"] for ex in s[k]} for k in ("train", "val", "test")]
    inputs = [{ex["input"] for ex in s[k]} for k in ("train", "val", "test")]
    for group in (ids, inputs):
        assert not (group[0] & group[1]) and not (group[0] & group[2]) and not (group[1] & group[2])


def test_reference_candidate_is_near_perfect(task):
    # Guards against mislabeled gold data, which would silently cap every variant.
    ev = task.evaluator()
    for name, exs in task.splits.items():
        assert bt.score_split(ev, task.reference_candidate, exs) >= 0.95, name


def test_seed_candidate_is_weak(task):
    ev = task.evaluator()
    for name, exs in task.splits.items():
        assert 0.2 <= bt.score_split(ev, task.seed_candidate, exs) <= 0.55, name


def test_evaluator_accepts_joined_string(task):
    ev = task.evaluator()
    joined = "\n".join(task.reference_candidate.values())
    assert bt.score_split(ev, joined, task.splits["val"]) >= 0.95


def test_regex_bad_candidates():
    t = bt.get_task("regex_emails")
    ev = t.evaluator()
    ex = t.splits["train"][0]
    score, asi = ev.evaluate("([a-z", ex)
    assert score == 0.0 and "does not compile" in asi["error"]
    # catastrophic backtracking must time out, not hang
    score, asi = ev.evaluate(r"(a+)+$|(.*a){30}x", {**ex, "input": "a" * 40 + "!"})
    assert score in (0.0, 1.0)
    evil = bt.RegexEmailEvaluator([{"id": "x", "input": "a" * 40 + "!"}])
    score, asi = evil.evaluate(r"(a+)+b", {"id": "x", "input": "a" * 40 + "!", "label": True, "hint": ""})
    assert score == 0.0 and "timeout" in asi["error"]


def test_parse_dates_bad_candidates():
    t = bt.get_task("parse_dates")
    ev = t.evaluator()
    ex = t.splits["train"][0]
    score, asi = ev.evaluate("def parse(s):\n    raise ValueError('boom')\n", ex)
    assert score == 0.0 and "boom" in asi["exception"]
    score, asi = ev.evaluate("this is not python", ex)
    assert score == 0.0 and "failed to load" in asi["error"]
    score, asi = ev.evaluate("def parse(s):\n    while True: pass\n", ex)
    assert score == 0.0 and "timeout" in (asi.get("exception", "") + asi.get("error", ""))


def test_ticket_router_parsing_and_asi():
    t = bt.get_task("ticket_router")
    ev = t.evaluator()
    ex = next(e for e in t.splits["train"] + t.splits["val"] + t.splits["test"] if e["priority"] == "P1")
    bad = {"keyword_rules": "### keyword_rules\n{not json", "priority_rules": "### priority_rules\nif bogus -> P9"}
    score, asi = ev.evaluate(bad, ex)
    assert "keyword_rules_error" in asi and "priority_rules_errors" in asi
    assert score <= 0.5
    # single component text (what `main` passes during minibatch evals) must not crash
    score, asi = ev.evaluate(t.reference_candidate["priority_rules"], ex)
    assert "note" in asi and score >= 0.5
    rules, errs = bt.parse_priority_rules('if contains "refund" and queue == billing -> P2\ndefault -> P3')
    assert not errs
    r = bt.route_ticket("please refund me", {"refund": "billing"}, rules)
    assert (r["queue"], r["priority"]) == ("billing", "P2")


def test_restore_header():
    assert _restore_header("### keyword_rules\n{}", '{"a": "billing"}').startswith("### keyword_rules\n")
    assert _restore_header("### keyword_rules\n{}", "### keyword_rules\n{}") == "### keyword_rules\n{}"
    assert _restore_header("plain", "new") == "new"


def test_mock_end_to_end_all_tasks(tmp_path):
    out = tmp_path / "res.jsonl"
    records = run_bench.main([
        "--tasks", "all", "--seeds", "0", "--max-iterations", "2", "--proposer", "mock",
        "--variant-name", "test", "--out", str(out), "--config-json", '{"not_a_real_field": 1}',
    ])
    assert len(records) == 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["task"] for r in lines} == set(bt.TASK_NAMES)
    for r in lines:
        assert r["best_val"] >= r["seed_val"]
        assert r["proposer_calls"] >= 1 and r["total_evals"] > 0
        assert "not_a_real_field" in r["config_dropped"]
        assert r["val_trajectory"]
    table = summarize.summarize(lines)
    assert "| test | regex_emails |" in table


def test_claude_cli_proposer_forwards_operator(monkeypatch):
    import bench_proposer

    prompts: list[str] = []

    class Proc:
        returncode = 0
        stdout = json.dumps({"result": "```\nnew\n```", "total_cost_usd": 0})

    def fake_run(cmd, **kw):
        prompts.append(cmd[cmd.index("-p") + 1])
        return Proc()

    monkeypatch.setattr(bench_proposer.subprocess, "run", fake_run)
    prop = bench_proposer.ClaudeCLIProposer()
    assert prop.propose("t", "p", "o", [], operator="section_edit") == "new"
    prop.propose("t", "p", "o", [])
    from optimize_anything.proposer.prompts import build_reflection_prompt
    assert "verbatim" in prompts[0]
    assert prompts[1] == build_reflection_prompt("t", "p", "o", [])


def test_claude_cli_proposer_forwards_rejected_edits(monkeypatch):
    import bench_proposer

    prompts: list[str] = []

    class Proc:
        returncode = 0
        stdout = json.dumps({"result": "```\nnew\n```", "total_cost_usd": 0})

    monkeypatch.setattr(bench_proposer.subprocess, "run",
                        lambda cmd, **kw: prompts.append(cmd[cmd.index("-p") + 1]) or Proc())
    bench_proposer.ClaudeCLIProposer().propose(
        "t", "p", "o", [], rejected_edits=[{"diff": "+tried", "note": "x"}])
    assert "don't repeat them" in prompts[0] and "+tried" in prompts[0]
