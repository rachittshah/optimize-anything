"""Dominance pruning + selection must match GEPA exactly (gepa-ai/gepa@ce5b0d6 gepa_utils.py)."""
from __future__ import annotations
import random

from optimize_anything.core.pareto import ParetoFrontier


def _gepa_remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    """Verbatim port of gepa.gepa_utils.remove_dominated_programs (asserts kept)."""
    program_fronts = {}
    remaining_counts = {}
    for key, front in program_at_pareto_front_valset.items():
        remaining_counts[key] = len(front)
        for p in front:
            program_fronts.setdefault(p, []).append(key)
    dominated = set()
    programs = list(program_fronts)
    if scores is None:
        scores = dict.fromkeys(programs, 1)
    programs = sorted(programs, key=lambda x: scores[x], reverse=False)
    for y in programs:
        fronts = program_fronts[y]
        if all(remaining_counts[key] > 1 for key in fronts):
            dominated.add(y)
            for key in fronts:
                remaining_counts[key] -= 1
    dominators = {p for p in programs if p not in dominated}
    for front in program_at_pareto_front_valset.values():
        if not front:
            continue
        assert any(p in front for p in dominators)
    return {val_id: {i for i in front if i in dominators}
            for val_id, front in program_at_pareto_front_valset.items()}


def _gepa_select(fronts, scores, rng):
    pruned = _gepa_remove_dominated_programs(fronts, scores=scores)
    freq = {}
    for front in pruned.values():
        for i in front:
            freq[i] = freq.get(i, 0) + 1
    return rng.choice([i for i, f in freq.items() for _ in range(f)])


def _frontier(fronts: dict[str, set[int]]) -> ParetoFrontier:
    f = ParetoFrontier(seed=0)
    for key, progs in fronts.items():
        for p in sorted(progs):
            f.update(key, 1.0, p)
    return f


def test_superset_program_with_lowest_score_is_removed():
    f = _frontier({"k1": {0, 1}, "k2": {0, 2}})  # A=0 on both keys, B=1 on k1, C=2 on k2
    assert f.remove_dominated([0.1, 0.5, 0.6]) == {1, 2}  # A lowest -> removed


def test_identical_key_sets_lower_scored_removed():
    f = _frontier({"k1": {0, 1}, "k2": {0, 1}})
    assert f.remove_dominated([0.7, 0.3]) == {0}


def test_property_matches_gepa_port_on_random_fronts():
    rng = random.Random(1234)
    for _ in range(500):
        n_prog = rng.randint(1, 8)
        n_keys = rng.randint(1, 10)
        fronts = {f"ex_{k}": set(rng.sample(range(n_prog), rng.randint(1, n_prog)))
                  for k in range(n_keys)}
        scores = [round(rng.random(), 1) for _ in range(n_prog)]  # coarse -> many ties
        f = _frontier(fronts)
        assert f.pruned_fronts(scores) == _gepa_remove_dominated_programs(f._programs, scores)
        assert f.pruned_fronts() == _gepa_remove_dominated_programs(f._programs)
        seed = rng.randint(0, 10**6)
        f._rng = random.Random(seed)
        picks = [f.select_candidate(scores) for _ in range(5)]
        ref_rng = random.Random(seed)
        assert picks == [_gepa_select(f._programs, scores, ref_rng) for _ in range(5)]
