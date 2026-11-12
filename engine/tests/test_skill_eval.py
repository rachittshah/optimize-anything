"""Tests for the skill hill-climb harness (scripts/skill_eval + scripts/optimize_skill.py).

Mock LLMs only. Formulation runs use real subprocesses against the local stub server.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from optimize_anything.config import Config  # noqa: E402
from optimize_anything.core.engine import Engine  # noqa: E402
from optimize_anything.evaluators.base import EvaluationError  # noqa: E402
from optimize_anything.llm import LLMResponse, MockLLM  # noqa: E402
from optimize_anything.proposer.reflective import ReflectiveProposer  # noqa: E402

from skill_eval import common  # noqa: E402
from skill_eval.common import (brevity_score, check_skill_text, f1_score, leakage_report,  # noqa: E402
                               load_data, make_shards, parse_frontmatter, shard_score,
                               weighted_score, FrontmatterError)
from skill_eval.formulation import (TOOL_SCHEMA, check_validity, parse_agent_output,  # noqa: E402
                                    parse_judge, validate_call)
from skill_eval.llm_cache import CachedLLM, LockedCostTracker  # noqa: E402
from skill_eval.router import build_menu, parse_choice  # noqa: E402
from skill_eval.scorer import EngineEvaluator, SkillScorer  # noqa: E402
from skill_eval.stub_server import CANNED_TEXT, StubServer  # noqa: E402

REPO = SCRIPTS.parent
SKILL = (REPO / "skill" / "SKILL.md").read_text()


def _load_cli():
    spec = importlib.util.spec_from_file_location("optimize_skill", SCRIPTS / "optimize_skill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    s = StubServer(tmp_path_factory.mktemp("site")).start()
    yield s
    s.stop()


# ---------------- frontmatter ----------------
def test_frontmatter_plain_and_quoted():
    fm = parse_frontmatter("---\nname: optimize-anything\ndescription: \"Use when: x, y\"\n---\nbody")
    assert fm == {"name": "optimize-anything", "description": "Use when: x, y"}


def test_frontmatter_block_scalar_and_continuation():
    text = "---\nname: a\ndescription: >-\n  Use when one\n  and two.\nother: x\n  more\n---\n"
    fm = parse_frontmatter(text)
    assert fm["description"] == "Use when one and two."
    assert fm["other"] == "x more"


def test_frontmatter_errors():
    with pytest.raises(FrontmatterError):
        parse_frontmatter("# no frontmatter")
    with pytest.raises(FrontmatterError):
        parse_frontmatter("---\nname: a\n")


def test_seed_skill_parses():
    fm, errors, warnings = check_skill_text(SKILL)
    assert not errors and fm["name"] == "optimize-anything" and fm["description"].startswith("Use when")
    assert not warnings


def test_check_skill_text_flags():
    fm, errors, _ = check_skill_text("---\nname: other\ndescription: d\n---\n")
    assert fm is None and "name" in errors[0]
    long = "---\nname: optimize-anything\ndescription: " + "x" * 1100 + "\nallowed-tools: Bash\n---\n"
    fm, errors, warnings = check_skill_text(long)
    assert fm and not errors
    assert any("1024" in w for w in warnings) and any("agent-neutral" in w for w in warnings)


# ---------------- metrics ----------------
def test_f1_math():
    assert f1_score([True, True, False, False], [True, False, True, False]) == pytest.approx(0.5)
    assert f1_score([True, True], [True, True]) == 1.0
    assert f1_score([False, False], [False, False]) == 1.0  # nothing to find, nothing claimed
    assert f1_score([True], [False]) == 0.0


def test_shard_score_falls_back_to_accuracy():
    assert shard_score([False, False, False], [False, True, False]) == pytest.approx(2 / 3)
    assert shard_score([True, False], [True, True]) == pytest.approx(2 / 3)  # F1


def test_brevity_curve():
    mk = lambda n: "\n".join(["x"] * n)  # noqa: E731
    assert brevity_score(mk(150)) == 1.0
    assert brevity_score(mk(225)) == pytest.approx(0.5)
    assert brevity_score(mk(300)) == 0.0
    assert brevity_score(mk(400)) == 0.0
    assert brevity_score(mk(10), description="d" * 1025) == 0.0


def test_weighted_score():
    c = {"trigger_f1": 1.0, "formulation_validity": 0.5, "formulation_quality": 0.5, "brevity": 1.0}
    assert weighted_score(c) == pytest.approx(0.3 + 0.15 + 0.15 + 0.1)
    assert weighted_score({**c, "brevity": None}) is None


def test_shards_stratified(data):
    for split in ("train", "val"):
        items = data.items("trigger", split)
        shards = make_shards(items)
        assert sorted(r["id"] for s in shards for r in s) == sorted(r["id"] for r in items)
        assert all(any(r["kind"] == "positive" for r in s) for s in shards)
        assert all(4 <= len(s) <= 6 for s in shards)


# ---------------- router ----------------
def test_menu_shuffle_deterministic(data):
    cand = {"name": "optimize-anything", "description": "d"}
    a = build_menu(cand, data.distractors, "t01")
    assert a == build_menu(cand, data.distractors, "t01")
    assert len(a) == len(data.distractors) + 1
    assert [m["name"] for m in a] != [m["name"] for m in build_menu(cand, data.distractors, "t02")]


def test_parse_choice():
    names = ["optimize-anything", "prompt-writing", "code-review"]
    assert parse_choice("optimize-anything", names) == "optimize-anything"
    assert parse_choice("`prompt-writing`.", names) == "prompt-writing"
    assert parse_choice("None", names) == "none"
    assert parse_choice("I'd load code-review here", names) == "code-review"
    assert parse_choice("hmm", names).startswith("?unparsed")


# ---------------- schema ----------------
def _py_call(**kw):
    call = {"seed_candidate": {"p": "x"}, "evaluator": {"type": "python",
            "code": "def evaluate(candidate, example):\n    return 0.5, {'ok': 1}\n"},
            "objective": "o"}
    call.update(kw)
    return call


def test_schema_valid_python():
    errors, warnings = validate_call(_py_call())
    assert errors == [] and warnings == []


def test_schema_rejects_inline_candidate():
    call = _py_call(evaluator={"type": "shell", "command": "python b.py '{{candidate}}'"})
    errors, _ = validate_call(call)
    assert any("{{candidate}}" in e and "inline" in e for e in errors)
    assert any("{{candidate_file}}" in e for e in errors)
    errors, _ = validate_call(_py_call(evaluator={"type": "shell", "command": "python b.py {{candidate_file}}"}))
    assert errors == []


def test_schema_type_specific_fields():
    assert validate_call({"evaluator": {}})[0]
    assert any("criteria" in e for e in validate_call(_py_call(evaluator={"type": "llm_judge"}))[0])
    assert any("define evaluate" in e for e in validate_call(_py_call(evaluator={"type": "python", "code": "x=1"}))[0])
    assert any("seed_candidate or objective" in e
               for e in validate_call({"evaluator": {"type": "llm_judge", "criteria": "c"}})[0])
    _, warnings = validate_call(_py_call(config={"max_metric_calls": 5, "bogus": 1}))
    assert any("bogus" in w for w in warnings)


def test_tool_schema_matches_tools_ts():
    ts = (REPO / "mcp-server" / "src" / "tools.ts").read_text()
    block = ts.split("name: 'check_optimization'")[0]
    for key in TOOL_SCHEMA["input_schema"]["properties"]:
        assert f"{key}:" in block, f"tools.ts lost property {key}"
    for key in TOOL_SCHEMA["input_schema"]["properties"]["evaluator"]["properties"]:
        assert f"{key}:" in block


# ---------------- agent output parsing ----------------
def test_parse_agent_output_modes():
    p = parse_agent_output('Notes.\n```json\n{"evaluator": {"type": "shell"}}\n```\n')
    assert p["mode"] == "call" and p["call"]["evaluator"]["type"] == "shell"
    p = parse_agent_output('```json\n{"name": "optimize_anything", "arguments": {"evaluator": {"type": "python"}}}\n```')
    assert p["call"]["evaluator"]["type"] == "python"
    p = parse_agent_output("I need more info.\nCLARIFY: What metric?")
    assert p["mode"] == "clarify" and "metric" in p["questions"]
    p = parse_agent_output("```json\n{bad json}\n```")
    assert p["call"] is None and "JSON parse error" in p["error"]
    assert parse_agent_output("just prose")["mode"] == "none"


def test_parse_judge():
    good = '```json\n{"scores": {"1": 2, "2": 1, "3": 2, "4": 2, "5": 0, "6": 2, "7": 1}, "critique": "c"}\n```'
    assert parse_judge(good)["scores"]["5"] == 0
    assert parse_judge('{"scores": {"1": 3}}') is None
    assert parse_judge("no json") is None


# ---------------- stub server ----------------
def test_stub_answers_both_apis(stub):
    def post(path, body):
        req = urllib.request.Request(stub.url + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    a = post("/v1/messages", {"model": "claude-haiku-5-5", "messages": []})
    assert a["content"][0]["text"] == CANNED_TEXT and a["type"] == "message"
    o = post("/v1/chat/completions", {"model": "gpt-6-luna", "messages": []})
    assert o["choices"][0]["message"]["content"] == CANNED_TEXT
    env = stub.env({})
    assert env["ANTHROPIC_API_KEY"] == "stub" and env["OA_BACKEND"] == "mock"


def test_stub_env_not_leaked_into_process(stub):
    import os
    stub.env()
    assert os.environ.get("ANTHROPIC_API_KEY") != "stub"


# ---------------- formulation runner (real subprocess, stub APIs) ----------------
def _item(data, iid):
    return next(r for r in data.formulations if r["id"] == iid)


def test_runner_valid_api_gold(data, stub):
    item = _item(data, "f01")  # python gold that calls api.anthropic.com via urllib
    v = check_validity({"mode": "call", "call": item["gold"], "error": None}, item, stub.env())
    assert v["valid"], v


def test_runner_api_gold_without_stub_is_invalid(data, stub):
    item = _item(data, "f01")
    env = {k: v for k, v in stub.env().items() if k not in ("OA_STUB_URL", "ANTHROPIC_API_KEY")}
    v = check_validity({"mode": "call", "call": item["gold"], "error": None}, item, env)
    assert not v["valid"] and "errored on every seed example" in v["reason"]


def test_runner_valid_shell_gold_uses_context_files(data, stub):
    item = _item(data, "f04")
    v = check_validity({"mode": "call", "call": item["gold"], "error": None}, item, stub.env())
    assert v["valid"], v


def test_runner_invalid_cases(data, stub):
    item = _item(data, "f04")
    crash = _py_call(evaluator={"type": "python", "code": "def evaluate(c, e):\n    raise ValueError('boom')\n"})
    v = check_validity({"mode": "call", "call": crash, "error": None}, item, stub.env())
    assert not v["valid"] and "boom" in v["reason"]
    missing = _py_call(evaluator={"type": "shell", "command": "python nope.py {{candidate_file}}"})
    v = check_validity({"mode": "call", "call": missing, "error": None}, item, stub.env())
    assert not v["valid"] and "shell exited" in v["reason"]
    out_of_range = _py_call(evaluator={"type": "python", "code": "def evaluate(c, e):\n    return 7, {}\n"})
    v = check_validity({"mode": "call", "call": out_of_range, "error": None}, item, stub.env())
    assert not v["valid"] and "[0,1]" in v["reason"]


def test_validity_clarify_rules(data, stub):
    clarify_item = _item(data, "f22")
    assert check_validity({"mode": "clarify"}, clarify_item, stub.env())["valid"]
    assert not check_validity({"mode": "call", "call": _py_call(), "error": None},
                              clarify_item, stub.env())["valid"]
    assert not check_validity({"mode": "clarify"}, _item(data, "f04"), stub.env())["valid"]


# ---------------- cost + cache ----------------
def test_cost_aggregation_and_cache(tmp_path):
    tracker = LockedCostTracker()
    llm = MockLLM(fn=lambda p: "answer", cost_per_call=0.01)
    a = CachedLLM(llm, tracker, "agent", tmp_path)
    j = CachedLLM(llm, tracker, "judge", tmp_path)
    a.complete("p1", system="s1", model="m", max_tokens=5)
    a.complete("p1", system="s1", model="m", max_tokens=5)  # hit
    a.complete("p1", system="s2", model="m", max_tokens=5)  # different system -> miss
    j.complete("p2", model="m", max_tokens=5, effort="low")
    d = tracker.to_dict()
    assert d["by_role"]["agent"]["calls"] == 2 and d["by_role"]["judge"]["calls"] == 1
    assert d["total_usd"] == pytest.approx(0.03)
    assert d["cache_hits"] == {"agent": 1}
    assert len(llm.calls) == 3
    assert (tmp_path / ".gitignore").exists()


def test_cache_skips_empty(tmp_path):
    tracker = LockedCostTracker()
    llm = MockLLM(fn=lambda p: "  ")
    c = CachedLLM(llm, tracker, "router", tmp_path)
    c.complete("p", model="m", max_tokens=5)
    c.complete("p", model="m", max_tokens=5)
    assert len(llm.calls) == 2


# ---------------- leakage ----------------
def test_leakage_metric(data):
    clean = leakage_report("---\nname: optimize-anything\ndescription: d\n---\nshort", data)
    assert clean["all_train_items"]["pct"] == 0.0 and not clean["flag"]
    copied = "\n".join(r["prompt"] for r in data.items("trigger", "train"))
    leaky = leakage_report(copied, data)
    assert leaky["trigger_prompts"]["pct"] > 0.9 and leaky["flag"]


# ---------------- scorer + engine wiring with a fake LLM ----------------
class FakeLLM:
    """Router picks the skill for gold-trigger prompts; agent emits the gold call; judge 10/14."""

    def __init__(self, data):
        self.by_request = {r["request"]: r for r in data.formulations}
        self.trigger_prompts = {r["prompt"]: r["label"] for r in data.triggers}
        self.calls: list[dict] = []

    def complete(self, prompt, *, system=None, model, max_tokens, effort=None):
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if system and "skill router" in system:
            req = prompt.split("<request>\n")[1].split("\n</request>")[0]
            text = "optimize-anything" if self.trigger_prompts.get(req) == "trigger" else "none"
        elif system and system.startswith("You are an AI coding agent"):
            req = prompt.split("User request:\n")[1].split("\n\nFiles in the working directory")[0]
            gold = self.by_request[req]["gold"]
            text = "CLARIFY: what metric?" if gold.get("clarify") else f"```json\n{json.dumps(gold)}\n```"
        elif system and system.startswith("You grade"):
            text = '```json\n{"scores": {"1": 2, "2": 2, "3": 1, "4": 2, "5": 1, "6": 1, "7": 1}, "critique": "ok"}\n```'
        else:
            text = "```\nproposal\n```"
        return LLMResponse(text=text, model=model, cost_usd=0.001, backend="fake")


def test_score_skill_with_fake_llm(data, stub, tmp_path):
    tracker = LockedCostTracker()
    scorer = SkillScorer(data, FakeLLM(data), tracker, env_fn=stub.env, cache_dir=tmp_path, limit=4)
    try:
        res = scorer.score_skill(SKILL, "val")["val"]
    finally:
        scorer.close()
    c = res["components"]
    assert c["trigger_f1"] == 1.0 and c["formulation_validity"] == 1.0
    assert c["formulation_quality"] == pytest.approx(10 / 14)
    assert res["weighted"] == pytest.approx(0.3 + 0.3 + 0.3 * 10 / 14 + 0.1 * c["brevity"])
    assert set(tracker.to_dict()["by_role"]) == {"router", "agent", "judge"}


def test_malformed_candidate_scores_zero(data, stub, tmp_path):
    scorer = SkillScorer(data, FakeLLM(data), LockedCostTracker(), env_fn=stub.env, cache_dir=tmp_path)
    try:
        res = scorer.score_skill("no frontmatter", "val")["val"]
        assert res["weighted"] == 0.0 and res["malformed"]
        score, asi = scorer.evaluate("no frontmatter", {"kind": "formulation", "split": "train", "id": "f01"})
        assert score == 0.0 and "malformed" in asi["error"]
    finally:
        scorer.close()


def test_judge_failure_raises_evaluation_error(data, stub, tmp_path):
    class BadJudge(FakeLLM):
        def complete(self, prompt, *, system=None, model, max_tokens, effort=None):
            if system and system.startswith("You grade"):
                return LLMResponse(text="no scores here", model=model)
            return super().complete(prompt, system=system, model=model, max_tokens=max_tokens)
    scorer = SkillScorer(data, BadJudge(data), LockedCostTracker(), env_fn=stub.env, cache_dir=None)
    try:
        with pytest.raises(EvaluationError):
            scorer.evaluate(SKILL, {"kind": "formulation", "split": "train", "id": "f13"})
    finally:
        scorer.close()


def test_engine_extract_candidate_keeps_skill_fences():
    """The engine proposer must return a fenced SKILL.md intact (it has inner ``` fences)."""
    from optimize_anything.proposer.prompts import extract_candidate
    assert extract_candidate(f"Here you go:\n````markdown\n{SKILL}\n````\nDone.") == SKILL.strip()
    assert extract_candidate(f"Here:\n```markdown\n{SKILL}\n```\n") == SKILL.strip()


def test_run_controller_plateau_and_cost():
    cli = _load_cli()
    t = LockedCostTracker()
    rc = cli.RunController(t, cap_usd=100.0, patience=3, min_gain=0.01)
    assert rc.check(1, 0.50, 0.0) is None
    assert rc.check(2, 0.505, 1.0) is None   # gain <= 0.01 does not reset
    assert rc.check(3, 0.505, 2.0) is None
    assert "plateau" in rc.check(4, 0.509, 3.0)
    off = cli.RunController(t, cap_usd=100.0, patience=None)
    assert all(off.check(i, 0.5, 0.0) is None for i in range(1, 10))
    rc = cli.RunController(t, cap_usd=10.0, patience=3)
    rc.check(1, 0.5, 0.0)
    rc.check(2, 0.6, 4.0)
    assert "cost" in rc.check(3, 0.7, 8.0)   # 8 + ~4 per iteration >= 10


def test_engine_wiring_val_isolation(data, stub, tmp_path):
    """One real engine iteration: the proposer only ever sees train items, never val."""
    cli = _load_cli()
    fake = FakeLLM(data)
    tracker = LockedCostTracker()
    scorer = SkillScorer(data, fake, tracker, env_fn=stub.env, cache_dir=tmp_path / "c", limit=3)
    proposer_llm = MockLLM(fn=lambda p: f"````markdown\n{SKILL}\n<!-- v2 -->\n````")
    from optimize_anything.core.cost import TrackedLLM
    proposer = ReflectiveProposer(TrackedLLM(proposer_llm, tracker, "propose"), model="m", effort=None)
    controller = cli.RunController(tracker, cap_usd=100.0)
    engine = Engine(seed_candidate={"skill": SKILL}, evaluator=EngineEvaluator(scorer),
                    proposer=proposer, objective=cli.OBJECTIVE, background=cli.BACKGROUND,
                    dataset=scorer.build_examples("train"), valset=scorer.build_examples("val"),
                    config=Config(max_iterations=2, run_dir=str(tmp_path / "run"), seed=1),
                    on_event=controller, cost_tracker=tracker)
    controller.engine = engine
    try:
        engine.run()
    finally:
        scorer.close()
    assert proposer_llm.calls, "proposer never ran"
    prompts = "\n".join(c["prompt"] for c in proposer_llm.calls)
    for r in data.items("trigger", "val"):
        assert r["prompt"][:80] not in prompts
    for r in data.items("formulation", "val"):
        assert r["request"][:80] not in prompts
    assert "propose" in tracker.to_dict()["by_role"]
    assert json.loads((tmp_path / "run" / "checkpoint.json").read_text())["candidates"]


def test_stoprun_recovery_with_eval_workers(data, stub, tmp_path):
    """A controller stop raises out of engine.run(); the checkpoint still saves and loads."""
    cli = _load_cli()
    tracker = LockedCostTracker()
    scorer = SkillScorer(data, FakeLLM(data), tracker, env_fn=stub.env, cache_dir=None, limit=3)
    from optimize_anything.core.cost import TrackedLLM
    proposer_llm = MockLLM(fn=lambda p: f"````markdown\n{SKILL}\n<!-- v2 -->\n````", cost_per_call=5.0)
    proposer = ReflectiveProposer(TrackedLLM(proposer_llm, tracker, "propose"), model="m", effort=None)
    controller = cli.RunController(tracker, cap_usd=6.0, patience=50)
    extra = {"eval_workers": 4} if "eval_workers" in Config.__dataclass_fields__ else {}
    engine = Engine(seed_candidate={"skill": SKILL}, evaluator=EngineEvaluator(scorer),
                    proposer=proposer, objective="o", background="b",
                    dataset=scorer.build_examples("train"), valset=scorer.build_examples("val"),
                    config=Config(max_iterations=20, run_dir=str(tmp_path / "run"), seed=3,
                                  skip_perfect_score=False, **extra),
                    on_event=controller, cost_tracker=tracker)
    controller.engine = engine
    try:
        with pytest.raises(cli.StopRun, match="cost"):
            engine.run()
        engine._checkpoint()
    finally:
        scorer.close()
    ck = cli.load_checkpoint(tmp_path / "run")
    assert ck["candidates"][0]["components"]["skill"] == SKILL
    assert controller.stop_reason.startswith("cost")
    assert len(proposer_llm.calls) <= 2


def test_quota_error_is_flagged_and_stops_run(tmp_path):
    from optimize_anything.llm import LLMError
    cli = _load_cli()

    class Limited:
        def complete(self, prompt, **kw):
            raise LLMError("claude -p error (exit 1): You've hit your 5-hour limit · resets 11:20pm")
    tracker = LockedCostTracker()
    c = CachedLLM(Limited(), tracker, "agent", tmp_path)
    with pytest.raises(LLMError):
        c.complete("p", model="m", max_tokens=5)
    assert "5-hour limit" in tracker.quota_error
    rc = cli.RunController(tracker, cap_usd=100.0, patience=None)
    with pytest.raises(cli.StopRun, match="quota"):
        rc({"type": "iteration_start", "iteration": 1})
