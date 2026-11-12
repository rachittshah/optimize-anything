"""Flag-gated engine upgrades: test split, paired_se promotion, successive halving, prescreen,
counterexample mining, mutation operators / bandit, Config.seed, shell example placeholders."""
from __future__ import annotations
import json
import math

import pytest

from optimize_anything.config import Config
from optimize_anything.core.engine import Engine
from optimize_anything.core.halving import rung_sizes, should_stop_early, shuffled_order
from optimize_anything.core.prescreen import components_key, is_near_duplicate, output_signature
from optimize_anything.core.promotion import paired_gain, should_promote
from optimize_anything.core.state import State
from optimize_anything.evaluators.base import EvaluationError
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.llm.mock import MockLLM
from optimize_anything.proposer.prompts import build_reflection_prompt
from optimize_anything.proposer.reflective import ReflectiveProposer
from optimize_anything.strategies.bandit import SlidingWindowUCB
from optimize_anything.strategies.hard_pool import HardPoolSampler, hard_ids
from optimize_anything.strategies.sampling import EpochBatchSampler

TRAIN = [{"id": i} for i in range(4)]
VAL = [{"id": 100 + i} for i in range(8)]


class TableEvaluator:
    """score = table[text][example id] (train ids 0-3 use `train`, val ids use `val`)."""

    def __init__(self, table: dict[str, dict[str, list[float]]], default: float = 0.0):
        self.table = table
        self.default = default
        self.calls: list[tuple[str, int | None]] = []

    def evaluate(self, candidate, example=None):
        ex_id = None if example is None else example["id"]
        self.calls.append((candidate, ex_id))
        row = self.table.get(candidate)
        if row is None or ex_id is None:
            return self.default, {"note": "unknown"}
        if ex_id >= 100:
            return row["val"][ex_id - 100], {"split": "val"}
        return row["train"][ex_id % len(row["train"])], {"split": "train"}


class SeqProposer:
    """Returns the given texts in order, then echoes the current text (a no-op proposal)."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls: list[dict] = []

    def propose(self, current_text, component_name, objective, asi_entries, background=None):
        self.calls.append({"current": current_text})
        return self.texts.pop(0) if self.texts else current_text


class OperatorProposer(SeqProposer):
    def propose(self, current_text, component_name, objective, asi_entries, background=None,
                operator: str = "rewrite"):
        self.calls.append({"current": current_text, "operator": operator})
        return self.texts.pop(0) if self.texts else current_text


def _engine(evaluator, proposer, config: Config, **kw) -> Engine:
    return Engine(seed_candidate={"p": "A"}, evaluator=evaluator, proposer=proposer,
                  objective="o", dataset=TRAIN, valset=VAL, config=config, **kw)


SEED_ROW = {"train": [0.5] * 4, "val": [0.5] * 8}


# ---------------- Config ----------------

def test_config_defaults_match_ab_results():
    # Adopted from the bench A/B (RESULTS.md): paired_se promotion + exact-duplicate prescreen;
    # everything that didn't win stays off.
    c = Config()
    assert (c.seed, c.promotion, c.val_eval, c.prescreen, c.counterexample_mining,
            c.mutation_operator, c.operator_bandit) == (None, "paired_se", "full", True, False,
                                                        "rewrite", False)
    assert c.dedupe_threshold is None
    assert c.resolved_seed() == 42
    assert Config(seed=0).resolved_seed() == 0  # not `seed or 42`


@pytest.mark.parametrize("field,value", [("promotion", "best"), ("val_eval", "half"),
                                         ("mutation_operator", "edit")])
def test_config_rejects_unknown_flag_values(field, value):
    with pytest.raises(ValueError):
        Config(**{field: value})


def test_default_and_seed_42_give_identical_traces():
    def run(config):
        ev = TableEvaluator({"A": SEED_ROW, "B": {"train": [0.9] * 4, "val": [0.6] * 8}})
        pr = SeqProposer(["B", "C"])
        res = _engine(ev, pr, config).run()
        return ev.calls, [c["current"] for c in pr.calls], res.best_candidate, res.total_evals

    assert run(Config(max_iterations=6)) == run(Config(max_iterations=6, seed=42))


def test_seed_reaches_sampler_and_frontier():
    e = _engine(TableEvaluator({}), SeqProposer([]), Config(seed=7))
    assert e.sampler._rng.random() == __import__("random").Random(7).random()
    assert e.state.frontier._rng.random() == __import__("random").Random(7).random()


# ---------------- 1. test split ----------------

def test_testset_reported_and_not_counted_in_total_evals(tmp_path):
    table = {"A": SEED_ROW, "B": {"train": [1.0] * 4, "val": [0.8] * 8}}
    test = [{"id": 100 + i} for i in range(4)]  # reuse val rows as "test" scores
    base = _engine(TableEvaluator(table), SeqProposer(["B"]), Config(max_iterations=2)).run()
    res = _engine(TableEvaluator(table), SeqProposer(["B"]),
                  Config(max_iterations=2, run_dir=str(tmp_path)), testset=test).run()
    assert res.total_evals == base.total_evals
    assert res.test_evals == 8  # seed sweep + best sweep
    assert res.seed_test == pytest.approx(0.5)
    assert res.best_test == pytest.approx(0.8)
    assert res.val_test_gap == pytest.approx(res.best_score - 0.8)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["best_test"] == pytest.approx(0.8) and saved["test_evals"] == 8
    assert base.best_test is None and base.test_evals == 0


def test_testset_single_sweep_when_best_is_seed_and_survives_failures():
    class Flaky(TableEvaluator):
        def evaluate(self, candidate, example=None):
            if example and example["id"] == 101 and len(self.calls) > 20:
                raise EvaluationError("judge down")
            return super().evaluate(candidate, example)

    ev = Flaky({"A": SEED_ROW})
    res = _engine(ev, SeqProposer([]), Config(max_iterations=1),
                  testset=[{"id": 100}, {"id": 101}]).run()
    assert res.best_candidate == {"p": "A"}
    assert res.test_evals == 2
    assert res.best_test == res.seed_test


# ---------------- 2. paired_se promotion ----------------

def test_paired_gain_and_should_promote():
    inc = {"a": 0.5, "b": 0.5}
    assert should_promote({"a": 0.7, "b": 0.7}, inc)  # constant diff -> SE 0, gain > 0
    assert not should_promote({"a": 0.5, "b": 0.5}, inc)
    noisy = {"a": 1.0, "b": 0.1}  # mean +0.05, big SE
    assert not should_promote(noisy, inc, max_drop=None)
    assert not should_promote({"a": 1.0, "b": 0.2}, inc, z=0.0, max_drop=0.25)  # drop 0.3
    assert should_promote({"a": 1.0, "b": 0.2}, inc, z=0.0, max_drop=None)
    assert should_promote({"a": 0.6, "zz": 0.0}, inc)  # pairs only common keys
    assert not should_promote({"zz": 1.0}, inc)  # no common keys
    mean, se, diffs = paired_gain({"a": 1.0, "b": 0.0}, inc)
    assert mean == 0 and se == pytest.approx(0.5) and len(diffs) == 2


def _noisy_table():
    return {
        "A": SEED_ROW,
        # higher mean (0.5125) but a coin flip per example vs the seed
        "B": {"train": [1.0] * 4, "val": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1]},
    }


def test_max_promotion_returns_noisy_argmax():
    res = _engine(TableEvaluator(_noisy_table()), SeqProposer(["B"]),
                  Config(max_iterations=2, promotion="max")).run()
    assert res.best_candidate == {"p": "B"}


def test_paired_se_keeps_incumbent_against_noise(tmp_path):
    events: list[dict] = []
    res = _engine(TableEvaluator(_noisy_table()), SeqProposer(["B"]),
                  Config(max_iterations=2, promotion="paired_se", promotion_max_drop=None,
                         run_dir=str(tmp_path)), on_event=events.append).run()
    assert res.best_candidate == {"p": "A"}
    assert res.best_score == pytest.approx(0.5)
    assert len(res.all_candidates) == 2  # frontier/state unchanged: B is still in the pool
    assert res.promotions_blocked == 1
    assert any(e["type"] == "promotion_blocked" for e in events)
    ckpt = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ckpt["best_idx"] == 0 and ckpt["promotions_blocked"] == 1


def test_paired_se_promotes_clear_winner():
    table = {"A": SEED_ROW, "C": {"train": [1.0] * 4, "val": [0.7] * 8}}
    events: list[dict] = []
    res = _engine(TableEvaluator(table), SeqProposer(["C"]),
                  Config(max_iterations=2, promotion="paired_se"), on_event=events.append).run()
    assert res.best_candidate == {"p": "C"}
    assert [e["candidate_idx"] for e in events if e["type"] == "new_best_found"] == [1]


# ---------------- 3. successive halving ----------------

def test_rung_sizes_and_order():
    assert rung_sizes(8) == [2, 4, 8]
    assert rung_sizes(10) == [3, 5, 10]
    assert rung_sizes(2) == [1, 2]
    assert rung_sizes(1) == [1]
    assert sorted(shuffled_order(8, 3)) == list(range(8))
    assert shuffled_order(8, 3) == shuffled_order(8, 3)


def test_should_stop_early_rules():
    parent = {"ex_0": 0.5, "ex_1": 0.5}
    front = {"ex_0": 0.5, "ex_1": 0.5}.get
    assert should_stop_early({"ex_0": 0.0}, parent, front)
    assert not should_stop_early({"ex_0": 0.6}, parent, front)  # better than parent
    assert not should_stop_early({"ex_0": 0.0, "ex_1": 0.9}, parent,
                                 lambda k: front(k))  # mean 0.45 < 0.5 but beats frontier on ex_1
    assert not should_stop_early({"ex_0": 0.45}, parent, front, margin=0.1)
    assert not should_stop_early({"ex_9": 0.0}, parent, lambda k: 1.0)  # no common keys


def test_halving_rejects_bad_child_early():
    table = {"A": SEED_ROW, "BAD": {"train": [1.0] * 4, "val": [0.0] * 8}}
    ev = TableEvaluator(table)
    events: list[dict] = []
    res = _engine(ev, SeqProposer(["BAD"]), Config(max_iterations=2, val_eval="halving"),
                  on_event=events.append).run()
    assert len(res.all_candidates) == 1
    assert res.halving_rejects == 1 and res.halving_evals_saved == 6
    assert sum(1 for c, i in ev.calls if c == "BAD" and i is not None and i >= 100) == 2
    assert any(e["type"] == "candidate_rejected" and e["reason"] == "halving" for e in events)


def test_halving_fully_evaluates_good_child_with_original_keys():
    table = {"A": SEED_ROW, "GOOD": {"train": [1.0] * 4, "val": [0.6] * 8}}
    full = _engine(TableEvaluator(table), SeqProposer(["GOOD"]), Config(max_iterations=2))
    full.run()
    halv = _engine(TableEvaluator(table), SeqProposer(["GOOD"]),
                   Config(max_iterations=2, val_eval="halving"))
    res = halv.run()
    assert res.best_candidate == {"p": "GOOD"} and res.halving_rejects == 0
    assert halv.state.scores[1] == full.state.scores[1]
    assert list(halv.state.scores[1]) == [f"ex_{i}" for i in range(8)]


# ---------------- 4. prescreen ----------------

def test_prescreen_helpers():
    assert components_key({"p": "a  b\n"}) == components_key({"p": "a b"})
    long = "x" * 300 + " rule"
    assert is_near_duplicate(long, long + ".", 0.98)
    assert not is_near_duplicate("short text", "totally different words", 0.98)
    assert output_signature([(1.0, {"a": 1}), None]) == output_signature([(1.0, {"a": 1}), None])
    assert output_signature([(1.0, {"a": 1})]) != output_signature([(1.0, {"a": 2})])


@pytest.mark.parametrize("child,reason", [("  A \n", "prescreen_duplicate"),
                                          ("L" * 300 + "!", "prescreen_near_duplicate")])
def test_prescreen_skips_before_child_evals(child, reason):
    seed = "L" * 300 if reason == "prescreen_near_duplicate" else "A"
    table = {seed: SEED_ROW}
    ev = TableEvaluator(table)
    events: list[dict] = []
    e = Engine(seed_candidate={"p": seed}, evaluator=ev, proposer=SeqProposer([child]),
               objective="o", dataset=TRAIN, valset=VAL,
               config=Config(max_iterations=2, prescreen=True, cache_evals=False,
                             dedupe_threshold=0.98),
               on_event=events.append)
    res = e.run()
    assert res.prescreen_skips == 2  # iteration 2 echoes the parent: an exact duplicate
    assert not any(c == child for c, _ in ev.calls)
    reasons = [x["reason"] for x in events if x["type"] == "candidate_rejected"]
    assert reasons == [reason, "prescreen_duplicate"]


def test_prescreen_off_evaluates_duplicates():
    ev = TableEvaluator({"A": SEED_ROW})
    _engine(ev, SeqProposer(["  A \n"]), Config(max_iterations=2, cache_evals=False, prescreen=False)).run()
    assert any(c == "  A \n" for c, _ in ev.calls)


def test_prescreen_signature_event():
    table = {"A": SEED_ROW, "A2": SEED_ROW}  # different text, identical outputs
    events: list[dict] = []
    res = _engine(TableEvaluator(table), SeqProposer(["A2"]),
                  Config(max_iterations=2, prescreen=True), on_event=events.append).run()
    reasons = [e["reason"] for e in events if e["type"] == "candidate_rejected"]
    assert reasons == ["prescreen_signature", "prescreen_duplicate"]
    assert res.prescreen_skips == 2


# ---------------- 5. counterexample mining ----------------

def test_hard_ids():
    assert hard_ids({0: 1.0, 1: 0.0, 2: 1.0}) == {1}
    assert hard_ids({0: 0.5, 1: 0.5}) == {0, 1}
    assert hard_ids({0: 1.0}) == set()
    assert hard_ids({}) == set()


def test_hard_pool_sampler_mixes_and_advances_epoch():
    plain = EpochBatchSampler(3, seed=1)
    mixed = HardPoolSampler(EpochBatchSampler(3, seed=1), seed=1)
    ids = list(range(9))
    for it in range(5):
        base = plain.next_batch(ids, it)
        batch = mixed.next_batch(ids, it, {8})
        assert 8 in batch and len(batch) == 3 and len(set(batch)) == 3
        assert [i for i in batch if i != 8][:2] == [i for i in base if i != 8][:2]
    assert mixed.next_batch(ids, 5, set()) == plain.next_batch(ids, 5)


def test_mining_draws_failing_train_examples_more_often():
    train = [{"id": i} for i in range(8)]
    row = {"train": [1.0] * 6 + [0.0, 0.0], "val": [0.5] * 8}

    def hard_calls(config):
        ev = TableEvaluator({"A": row})
        Engine(seed_candidate={"p": "A"}, evaluator=ev, proposer=SeqProposer([]), objective="o",
               dataset=train, valset=VAL, config=config).run()
        return sum(1 for _, i in ev.calls if i in (6, 7))

    base = Config(max_iterations=12, reflection_minibatch_size=2, cache_evals=False,
                  skip_perfect_score=False)
    mined = Config(max_iterations=12, reflection_minibatch_size=2, cache_evals=False,
                   skip_perfect_score=False, counterexample_mining=True)
    assert hard_calls(mined) > hard_calls(base)


# ---------------- 6. mutation operators + bandit ----------------

def test_prompt_operators():
    base = dict(current_text="t", component_name="p", objective="o", asi_entries=[])
    assert build_reflection_prompt(**base).endswith("Your improved version:")
    assert build_reflection_prompt(**base) == build_reflection_prompt(**base, operator="rewrite")
    assert "verbatim" in build_reflection_prompt(**base, operator="section_edit")
    assert "shorter" in build_reflection_prompt(**base, operator="compress")


def test_reflective_proposer_uses_operator_prompt():
    llm = MockLLM(responses=["```\nnew\n```"])
    prop = ReflectiveProposer(llm, model="m", effort=None)
    assert prop.propose("t", "p", "o", [], operator="section_edit") == "new"
    assert "verbatim" in llm.calls[-1]["prompt"]


def test_fixed_section_edit_operator_reaches_proposer():
    pr = OperatorProposer(["B"])
    res = _engine(TableEvaluator({"A": SEED_ROW}), pr,
                  Config(max_iterations=2, mutation_operator="section_edit")).run()
    assert {c["operator"] for c in pr.calls} == {"section_edit"}
    assert res.operator_counts == {"section_edit": len(pr.calls)}


def test_default_operator_call_is_unchanged_and_old_proposers_work():
    pr = OperatorProposer(["B"])
    _engine(TableEvaluator({"A": SEED_ROW}), pr, Config(max_iterations=2)).run()
    assert all(c["operator"] == "rewrite" for c in pr.calls)  # kwarg omitted -> default
    old = SeqProposer(["B"])
    res = _engine(TableEvaluator({"A": SEED_ROW}), old,
                  Config(max_iterations=2, mutation_operator="section_edit")).run()
    assert res.operator_unsupported == len(old.calls) >= 1


def test_sliding_window_ucb():
    b = SlidingWindowUCB(window=4)
    assert b.select() == "rewrite"
    b.update("rewrite", 0.0)
    assert b.select() == "section_edit"
    b.update("section_edit", 1.0)
    assert b.select(["rewrite", "section_edit"]) == "section_edit"  # compress ineligible
    for _ in range(4):
        b.update("section_edit", 1.0)
    assert len(b.history) == 4
    assert b.select(["rewrite", "section_edit"]) == "rewrite"  # rewrite fell out of window


def test_operator_bandit_plays_arms_and_skips_compress_for_short_text():
    pr = OperatorProposer([])
    res = _engine(TableEvaluator({"A": SEED_ROW}), pr,
                  Config(max_iterations=5, operator_bandit=True, skip_perfect_score=False)).run()
    ops = [c["operator"] for c in pr.calls]
    assert set(ops) == {"rewrite", "section_edit"}
    assert sum(res.operator_counts.values()) == len(ops)


# ---------------- 8. shell placeholders ----------------

def test_shell_example_file_and_json_placeholders():
    read = "import json,sys; print(json.load(open(sys.argv[1]))['x'])"
    ev = ShellEvaluator(command=f'python3 -c "{read}" {{{{example_file}}}}')
    assert ev.evaluate("cand", {"x": 0.75})[0] == 0.75
    parse = "import json,sys; print(json.loads(sys.argv[1])['x'])"
    ev = ShellEvaluator(command=f'python3 -c "{parse}" {{{{example_json}}}}')
    assert ev.evaluate("cand", {"x": 0.25, "q": "it's \"quoted\" $HOME"})[0] == 0.25


def test_shell_placeholders_not_expanded_inside_candidate():
    ev = ShellEvaluator(command="cat {{candidate_file}}; echo 0.5", score_pattern=r"(0\.5)")
    score, asi = ev.evaluate("literal {{example_file}}", {"x": 1})
    assert score == 0.5 and "literal {{example_file}}" in asi["stdout"]


# ---------------- checkpoint ----------------

def test_state_counters_roundtrip(tmp_path):
    s = State(component_names=["p"])
    s.prescreen_skips, s.halving_rejects, s.operator_counts = 2, 3, {"rewrite": 4}
    s.save(tmp_path / "c.json")
    t = State.load(tmp_path / "c.json")
    assert (t.prescreen_skips, t.halving_rejects, t.operator_counts) == (2, 3, {"rewrite": 4})
    legacy = json.loads((tmp_path / "c.json").read_text())
    for k in ("prescreen_skips", "halving_rejects", "halving_evals_saved", "promotions_blocked",
              "operator_counts"):
        legacy.pop(k)
    (tmp_path / "old.json").write_text(json.dumps(legacy))
    assert State.load(tmp_path / "old.json").halving_rejects == 0


def test_testset_cost_counted_once_and_consistent(tmp_path):
    from optimize_anything.core.cost import CostTracker

    class PaidEval(TableEvaluator):
        def __init__(self, table, tracker):
            super().__init__(table)
            self.tracker = tracker

        def evaluate(self, candidate, example=None):
            from optimize_anything.llm.base import LLMResponse
            self.tracker.record("judge", LLMResponse(text="", model="m", input_tokens=0,
                                                     output_tokens=0, cost_usd=0.01,
                                                     backend="mock"))
            return super().evaluate(candidate, example)

    tracker = CostTracker()
    ev = PaidEval({"A": SEED_ROW}, tracker)
    res = _engine(ev, SeqProposer([]), Config(max_iterations=1, run_dir=str(tmp_path)),
                  testset=[{"id": 100}], cost_tracker=tracker).run()
    assert res.total_cost_usd == pytest.approx(tracker.total_usd)
    assert res.total_cost_usd == pytest.approx(res.cost["total_usd"])
    assert tracker.total_usd == pytest.approx(0.01 * (len(ev.calls)))  # test eval included
    ckpt = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ckpt["test_evals"] == res.test_evals == 1


def test_val_failed_event_when_all_val_evals_fail():
    class FailVal(TableEvaluator):
        def evaluate(self, candidate, example=None):
            if candidate == "B" and example and example["id"] >= 100:
                raise EvaluationError("down")
            return super().evaluate(candidate, example)

    events: list[dict] = []
    res = _engine(FailVal({"A": SEED_ROW, "B": {"train": [1.0] * 4, "val": [1.0] * 8}}),
                  SeqProposer(["B"]), Config(max_iterations=2), on_event=events.append).run()
    assert len(res.all_candidates) == 1
    assert any(e.get("reason") == "val_failed" for e in events)


def test_cli_status_reports_paired_se_incumbent(tmp_path, capsys):
    import argparse
    from optimize_anything.cli import _status

    _engine(TableEvaluator(_noisy_table()), SeqProposer(["B"]),
            Config(max_iterations=2, promotion="paired_se", promotion_max_drop=None,
                   run_dir=str(tmp_path))).run()
    _status(argparse.Namespace(run_dir=str(tmp_path)))
    assert json.loads(capsys.readouterr().out)["best_score"] == pytest.approx(0.5)


# ---------------- perfect-skip bug / train saturation / stop_reason ----------------

def test_train_perfect_seed_stops_as_saturated_instead_of_burning_iterations(tmp_path):
    # Seed is perfect on every train example but not on val: nothing to reflect on.
    table = {"A": {"train": [1.0] * 4, "val": [0.5] * 8}}
    events: list[dict] = []
    pr = SeqProposer([])
    res = _engine(TableEvaluator(table), pr, Config(max_iterations=200, run_dir=str(tmp_path)),
                  on_event=events.append).run()
    assert res.stop_reason == "train_saturated"
    assert pr.calls == [] and res.skipped_perfect >= 2
    assert res.total_iterations == 0  # no proposal happened
    assert any(e["type"] == "train_saturated" for e in events)
    assert json.loads((tmp_path / "result.json").read_text())["stop_reason"] == "train_saturated"
    assert [e for e in events if e["type"] == "optimization_complete"][0]["stop_reason"] \
        == "train_saturated"


def test_perfect_parent_is_reselected_without_consuming_iterations():
    # P is train-perfect, S is not; both stay on the val frontier (each wins half the keys).
    table = {
        "S": {"train": [0.5] * 4, "val": [0.5] * 8},
        "P": {"train": [1.0] * 4, "val": [1.0] * 4 + [0.0] * 4},
    }
    pr = SeqProposer(["P"])
    res = Engine(seed_candidate={"p": "S"}, evaluator=TableEvaluator(table), proposer=pr,
                 objective="o", dataset=TRAIN, valset=VAL,
                 config=Config(max_iterations=8, max_perfect_resamples=30)).run()
    assert len(res.all_candidates) == 2
    assert res.stop_reason == "iterations"
    assert len(pr.calls) == 8  # every counted iteration reached the proposer
    assert res.skipped_perfect >= 1


def test_skip_perfect_off_still_proposes():
    table = {"A": {"train": [1.0] * 4, "val": [0.5] * 8}}
    pr = SeqProposer([])
    res = _engine(TableEvaluator(table), pr,
                  Config(max_iterations=3, skip_perfect_score=False)).run()
    assert len(pr.calls) == 3 and res.skipped_perfect == 0 and res.stop_reason == "iterations"


def test_single_task_perfect_seed_saturates():
    class Perfect:
        def evaluate(self, candidate, example=None):
            return 1.0, {}

    res = Engine(seed_candidate={"p": "x"}, evaluator=Perfect(), proposer=SeqProposer([]),
                 config=Config(max_iterations=50)).run()
    assert res.stop_reason == "train_saturated"


@pytest.mark.parametrize("config,reason", [
    (Config(max_iterations=2), "iterations"),
    (Config(max_iterations=100, max_metric_calls=20, cache_evals=False), "budget"),
    (Config(max_iterations=100, no_improvement_patience=2), "no_improvement"),
])
def test_stop_reason_per_stopper(config, reason):
    res = _engine(TableEvaluator({"A": SEED_ROW}), SeqProposer([]), config).run()
    assert res.stop_reason == reason


def test_composite_stopper_reason_names():
    from optimize_anything.strategies.stopping import (
        BudgetStopper, CompositeStopper, IterationStopper, TimeoutStopper,
    )
    c = CompositeStopper([IterationStopper(5), BudgetStopper(10), TimeoutStopper(0)])
    assert c.should_stop(iteration=1, total_evals=10)
    assert c.triggered == ["budget", "timeout"] and c.reason == "budget"


def test_exhausted_perfect_resamples_count_the_iteration():
    table = {
        "S": {"train": [0.5] * 4, "val": [0.5] * 8},
        "P": {"train": [1.0] * 4, "val": [1.0] * 4 + [0.0] * 4},
    }
    pr = SeqProposer(["P"])
    res = Engine(seed_candidate={"p": "S"}, evaluator=TableEvaluator(table), proposer=pr,
                 objective="o", dataset=TRAIN, valset=VAL,
                 config=Config(max_iterations=30, max_perfect_resamples=0)).run()
    assert res.stop_reason == "iterations"  # bounded: no infinite loop when not saturated
    assert len(pr.calls) + res.skipped_perfect == 30


# ---------------- 9. proposal memory ----------------

class MemoryProposer(SeqProposer):
    def propose(self, current_text, component_name, objective, asi_entries, background=None,
                rejected_edits=None):
        self.calls.append({"current": current_text, "memory": rejected_edits})
        return self.texts.pop(0) if self.texts else current_text


def _worse_table():
    return {"A": SEED_ROW, "W1": {"train": [0.2] * 4, "val": [0.2] * 8},
            "W2": {"train": [0.1] * 4, "val": [0.1] * 8}}


def test_proposal_memory_feeds_rejected_edits_back():
    pr = MemoryProposer(["W1", "W2"])
    _engine(TableEvaluator(_worse_table()), pr,
            Config(max_iterations=3, proposal_memory=True)).run()
    assert pr.calls[0]["memory"] is None
    mem = pr.calls[1]["memory"]
    assert len(mem) == 1 and "+W1" in mem[0]["diff"] and "-A" in mem[0]["diff"]
    assert mem[0]["parent_score"] == pytest.approx(0.5)
    assert mem[0]["child_score"] == pytest.approx(0.2)
    assert len(pr.calls[2]["memory"]) == 2


def test_proposal_memory_off_by_default_and_capped():
    pr = MemoryProposer(["W1", "W2"])
    _engine(TableEvaluator(_worse_table()), pr, Config(max_iterations=3)).run()
    assert all(c["memory"] is None for c in pr.calls)
    pr = MemoryProposer(["W1", "W2", "W1", "W2"])
    _engine(TableEvaluator(_worse_table()), pr,
            Config(max_iterations=5, proposal_memory=True, proposal_memory_size=2)).run()
    assert len(pr.calls[4]["memory"]) == 2


def test_proposal_memory_with_old_proposer_and_long_diff_truncated():
    _engine(TableEvaluator(_worse_table()), SeqProposer(["W1"]),
            Config(max_iterations=2, proposal_memory=True)).run()  # no kwarg: still works
    e = _engine(TableEvaluator({}), SeqProposer([]), Config(proposal_memory=True))
    e._remember(0, "p", "a\n" * 10, "b\n" * 2000, 0.5, 0.1)
    assert len(e._memory[(0, "p")][0]["diff"]) <= 620


def test_prompt_rejected_edits_section_and_mock_echo_unaffected():
    edits = [{"diff": "-old\n+new", "parent_score": 0.5, "child_score": 0.25, "note": None},
             {"diff": "+huge", "note": "edit too large"}]
    prompt = build_reflection_prompt("cur", "p", "o", [], rejected_edits=edits)
    assert "don't repeat them" in prompt and "~~~diff\n-old\n+new\n~~~" in prompt
    assert "0.250 vs parent 0.500" in prompt and "edit too large" in prompt
    assert build_reflection_prompt("cur", "p", "o", [], rejected_edits=[]) == \
        build_reflection_prompt("cur", "p", "o", [])
    # the diffs (in ~~~ fences) never change what the mock proposer extracts
    a = ReflectiveProposer(MockLLM(), model="m", effort=None)
    b = ReflectiveProposer(MockLLM(), model="m", effort=None)
    assert a.propose("cur", "p", "o", [], rejected_edits=edits) == b.propose("cur", "p", "o", [])


# ---------------- 11. max_edit_ratio ----------------

def test_max_edit_ratio_rejects_big_rewrites_before_eval():
    seed = "keep this line exactly as it is\n" * 5
    big = "completely different content here!\n" * 5
    small = seed + "one more rule\n"
    table = {seed: SEED_ROW, small: {"train": [1.0] * 4, "val": [0.9] * 8}}
    ev = TableEvaluator(table)
    events: list[dict] = []
    pr = MemoryProposer([big, small])
    res = Engine(seed_candidate={"p": seed}, evaluator=ev, proposer=pr, objective="o",
                 dataset=TRAIN, valset=VAL, on_event=events.append,
                 config=Config(max_iterations=2, max_edit_ratio=0.3,
                               proposal_memory=True)).run()
    assert res.edit_size_rejects == 1
    assert not any(c == big for c, _ in ev.calls)
    assert res.best_candidate == {"p": small}
    assert any(e.get("reason") == "edit_size" for e in events)
    assert "edit too large" in pr.calls[1]["memory"][0]["note"]


def test_max_edit_ratio_off_by_default():
    res = _engine(TableEvaluator({"A": SEED_ROW, "ZZZZ": {"train": [1.0] * 4, "val": [0.9] * 8}}),
                  SeqProposer(["ZZZZ"]), Config(max_iterations=2)).run()
    assert res.edit_size_rejects == 0 and res.best_candidate == {"p": "ZZZZ"}


# ---------------- 10. eval_workers ----------------

class SlowCounting(TableEvaluator):
    def evaluate(self, candidate, example=None):
        import time
        time.sleep(0.002)
        return super().evaluate(candidate, example)


def _run_workers(workers: int, **cfg):
    table = {"A": SEED_ROW, "B": {"train": [0.9] * 4, "val": [0.6] * 8},
             "C": {"train": [1.0] * 4, "val": [0.7] * 8}}
    ev = SlowCounting(table)
    e = _engine(ev, SeqProposer(["B", "C"]),
                Config(max_iterations=6, eval_workers=workers, **cfg))
    return e, e.run(), ev


def test_eval_workers_match_sequential_results():
    e1, r1, _ = _run_workers(1)
    e4, r4, ev4 = _run_workers(4)
    assert r4.best_candidate == r1.best_candidate
    assert e4.state.scores == e1.state.scores
    assert r4.total_evals == r1.total_evals == len(ev4.calls)
    assert r4.cache_hits == r1.cache_hits


def test_eval_workers_never_overshoot_budget():
    for budget in (5, 9, 13, 17):
        e, res, ev = _run_workers(4, max_metric_calls=budget, cache_evals=False)
        assert res.total_evals <= budget and len(ev.calls) == res.total_evals


def test_eval_workers_dedupe_within_batch():
    ev = TableEvaluator({"A": SEED_ROW})
    e = _engine(ev, SeqProposer([]), Config(eval_workers=4))
    out = e._eval_many("A", [{"id": 1}, {"id": 1}, {"id": 2}])
    assert [r[0] for r in out] == [0.5, 0.5, 0.5]
    assert len(ev.calls) == 2 and e.state.cache_hits == 1


def test_cost_tracker_thread_safe():
    from concurrent.futures import ThreadPoolExecutor
    from optimize_anything.core.cost import CostTracker
    from optimize_anything.llm.base import LLMResponse

    t = CostTracker()
    r = LLMResponse(text="", model="m", input_tokens=1, output_tokens=1, cost_usd=0.001,
                    backend="mock")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: t.record("judge", r), range(2000)))
    assert t.calls == 2000 and t.total_usd == pytest.approx(2.0)


def test_eval_workers_budget_truncated_child_never_enters_state():
    full_keys = {f"ex_{i}" for i in range(8)}
    for val_eval in ("full", "halving"):
        for budget in range(9, 30):
            events: list[dict] = []
            table = {"A": SEED_ROW, "B": {"train": [0.9] * 4, "val": [0.9] * 8},
                     "C": {"train": [1.0] * 4, "val": [1.0] * 8}}
            e = _engine(TableEvaluator(table), SeqProposer(["B", "C"]),
                        Config(max_iterations=6, eval_workers=4, max_metric_calls=budget,
                               cache_evals=False, val_eval=val_eval, promotion="paired_se"),
                        on_event=events.append)
            res = e.run()
            assert res.total_evals <= budget
            assert all(set(sc) == full_keys for sc in e.state.scores[1:]), (val_eval, budget)
            assert e.state.candidates[e._best_index()].components == res.best_candidate


def test_eval_workers_single_example_respects_budget():
    ev = TableEvaluator({"A": SEED_ROW})
    e = _engine(ev, SeqProposer([]), Config(eval_workers=4, max_metric_calls=0 or 1,
                                            cache_evals=False))
    e.state.total_evals = 1
    assert e._eval_many("A", [{"id": 1}]) == [None] and e._truncated == 1 and ev.calls == []



def test_near_duplicate_filter_off_by_default():
    seed = "L" * 300
    child = seed + "!"
    ev = TableEvaluator({seed: SEED_ROW})
    res = Engine(seed_candidate={"p": seed}, evaluator=ev, proposer=SeqProposer([child]),
                 objective="o", dataset=TRAIN, valset=VAL,
                 config=Config(max_iterations=2, prescreen=True)).run()
    assert Config().dedupe_threshold is None
    assert any(c == child for c, _ in ev.calls)  # evaluated: only exact dups are screened
    assert res.prescreen_skips == 1  # iteration 2's echo is an exact duplicate


# ---------------- min_improvement ----------------

def test_no_improvement_stopper_min_improvement():
    from optimize_anything.strategies.stopping import NoImprovementStopper

    s = NoImprovementStopper(patience=2, min_improvement=0.05)
    assert not s.should_stop(iteration=0, best_score=0.50)  # first observation counts
    assert not s.should_stop(iteration=1, best_score=0.52)  # +0.02: stale 1
    assert s.should_stop(iteration=2, best_score=0.54)  # +0.04 total from 0.50: stale 2
    s = NoImprovementStopper(patience=2, min_improvement=0.05)
    s.should_stop(iteration=0, best_score=0.50)
    s.should_stop(iteration=1, best_score=0.53)
    assert not s.should_stop(iteration=2, best_score=0.56)  # +0.06 cumulative: resets
    assert NoImprovementStopper(2).min_improvement == 0.0


def test_min_improvement_config_reaches_stopper():
    e = _engine(TableEvaluator({}), SeqProposer([]),
                Config(no_improvement_patience=3, min_improvement=0.01))
    nis = [s for s in e.stopper.stoppers if s.name == "no_improvement"][0]
    assert nis.min_improvement == 0.01


# ---------------- harness requests ----------------

def test_extract_candidate_keeps_inner_fences():
    from optimize_anything.proposer.prompts import extract_candidate

    skill = "# Skill\n\n```bash\nls -la\n```\n\nMore text\n\n```python\nprint(1)\n```\nEnd"
    assert extract_candidate(f"Here:\n````markdown\n{skill}\n````\nDone.") == skill
    assert extract_candidate(f"Here:\n```markdown\n{skill}\n```\n") == skill  # 3-tick outer
    assert extract_candidate("```\nplain\n```") == "plain"
    assert extract_candidate("inline ```x``` here") == "x"
    assert extract_candidate("no fence") == "no fence"
    assert extract_candidate("```\nunclosed") == "```\nunclosed"


def test_fence_hint_only_for_fenced_artifacts():
    base = dict(component_name="p", objective="o", asi_entries=[])
    assert "LONGER fence" in build_reflection_prompt(current_text="a ```x``` b", **base)
    assert "LONGER fence" not in build_reflection_prompt(current_text="plain", **base)


def test_optimize_anything_accepts_cost_tracker():
    from optimize_anything import optimize_anything
    from optimize_anything.core.cost import CostTracker

    from optimize_anything.llm.base import LLMResponse

    tracker = CostTracker()
    tracker.record("judge", LLMResponse(text="", model="m", input_tokens=0, output_tokens=0,
                                        cost_usd=0.25, backend="mock"))  # evaluator-side spend
    res = optimize_anything(seed_candidate={"p": "x"}, evaluator=lambda c, e: 0.5,
                            objective="o", config=Config(max_iterations=2),
                            cost_tracker=tracker)
    assert res.total_cost_usd == pytest.approx(0.25)  # same tracker: counts toward max_cost_usd



# ---------------- mock backend proposals ----------------

def test_mock_proposer_returns_tweaked_current_text():
    prop = ReflectiveProposer(MockLLM(), model="m", effort=None)
    first = prop.propose("keep me", "p", "o", [{"score": 0.1, "feedback": {"e": "x"}}])
    second = prop.propose("keep me", "p", "o", [])
    assert first.startswith("keep me") and first != "keep me" and first != second
    fenced = "# Skill\n```bash\nls\n```\nend"
    assert prop.propose(fenced, "p", "o", []).startswith(fenced)  # inner fences survive


def test_mock_backend_run_exercises_acceptance():
    from optimize_anything import optimize_anything

    res = optimize_anything(seed_candidate={"t": "x"}, evaluator=lambda c, e: min(1.0, len(c) / 100),
                            objective="longer", config=Config(max_iterations=3, backend="mock"))
    assert len(res.all_candidates) > 1 and res.best_score > res.seed_score
