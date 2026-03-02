# engine/tests/test_pareto.py
from optimize_anything.core.pareto import ParetoFrontier


def test_pareto_single_objective():
    frontier = ParetoFrontier()
    frontier.update("ex1", 0.5, candidate_idx=0)
    frontier.update("ex1", 0.8, candidate_idx=1)
    assert frontier.best_score("ex1") == 0.8
    assert 1 in frontier.programs_at("ex1")
    assert 0 not in frontier.programs_at("ex1")


def test_pareto_equal_scores_kept():
    frontier = ParetoFrontier()
    frontier.update("ex1", 0.8, candidate_idx=0)
    frontier.update("ex1", 0.8, candidate_idx=1)
    assert 0 in frontier.programs_at("ex1")
    assert 1 in frontier.programs_at("ex1")


def test_pareto_select_weighted():
    frontier = ParetoFrontier()
    frontier.update("ex1", 0.9, candidate_idx=0)
    frontier.update("ex2", 0.9, candidate_idx=0)
    frontier.update("ex3", 0.7, candidate_idx=1)
    # candidate 0 appears in 2 frontier keys, candidate 1 in 1
    # so candidate 0 should be selected more often
    selections = [frontier.select_candidate([0.85, 0.7]) for _ in range(100)]
    assert selections.count(0) > selections.count(1)


def test_pareto_aggregated_score():
    frontier = ParetoFrontier()
    frontier.update("ex1", 0.8, candidate_idx=0)
    frontier.update("ex2", 0.6, candidate_idx=0)
    assert frontier.aggregated_score(0) == 0.7  # mean
