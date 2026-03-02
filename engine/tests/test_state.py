# engine/tests/test_state.py
from optimize_anything.core.state import State
from optimize_anything.core.candidate import Candidate


def test_state_add_candidate():
    state = State(component_names=["prompt"])
    candidate = Candidate(components={"prompt": "hello"})
    idx = state.add_candidate(candidate, scores={"ex1": 0.8}, parent_ids=[])
    assert idx == 0
    assert state.candidates[0].components["prompt"] == "hello"
    assert state.iteration == -1
    assert state.total_evals == 0


def test_state_best_candidate():
    state = State(component_names=["prompt"])
    state.add_candidate(Candidate({"prompt": "v1"}), {"ex1": 0.5}, [])
    state.add_candidate(Candidate({"prompt": "v2"}), {"ex1": 0.9}, [])
    best = state.best_candidate()
    assert best.components["prompt"] == "v2"


def test_state_tracks_budget():
    state = State(component_names=["prompt"])
    state.record_evals(5)
    state.record_evals(3)
    assert state.total_evals == 8
