# engine/tests/test_checkpoint.py
import json
from pathlib import Path
from optimize_anything.core.state import State
from optimize_anything.core.candidate import Candidate


def test_save_and_load_state(tmp_path):
    state = State(component_names=["prompt", "format"])
    state.add_candidate(
        Candidate({"prompt": "v1", "format": "json"}),
        scores={"ex0": 0.5, "ex1": 0.7},
        parent_ids=[],
    )
    state.add_candidate(
        Candidate({"prompt": "v2", "format": "json"}),
        scores={"ex0": 0.8, "ex1": 0.6},
        parent_ids=[0],
    )
    state.iteration = 5
    state.record_evals(42)

    save_path = tmp_path / "checkpoint.json"
    state.save(save_path)

    loaded = State.load(save_path)
    assert loaded.iteration == 5
    assert loaded.total_evals == 42
    assert len(loaded.candidates) == 2
    assert loaded.candidates[0].components["prompt"] == "v1"
    assert loaded.candidates[1].parent_ids == [0]
    assert loaded.best_score() > 0
