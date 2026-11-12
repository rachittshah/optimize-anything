from __future__ import annotations

import json
import subprocess
import sys

from optimize_anything.cli import _dry_run
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType

GOOD = "def evaluate(candidate, example):\n    return 0.5, {'len': len(candidate)}"
BROKEN = "def evaluate(candidate, example):\n    raise ValueError('boom')"


def _py(code: str) -> EvaluatorConfig:
    return EvaluatorConfig(type=EvaluatorType.PYTHON, code=code)


def test_dry_run_scores_seed_on_at_most_three_examples():
    out = _dry_run({"t": "hello"}, _py(GOOD), [{"i": i} for i in range(5)], None, Config(backend="mock"))
    assert out["ok"] is True
    assert out["examples_evaluated"] == 3
    assert out["scores"] == [0.5, 0.5, 0.5]
    assert out["example_source"] == "dataset"
    assert out["errors"] == []


def test_dry_run_prefers_valset_and_reports_evaluator_errors():
    out = _dry_run({"t": "x"}, _py(BROKEN), [{"i": 0}], [{"v": 1}], Config(backend="mock"))
    assert out["ok"] is False
    assert out["example_source"] == "valset"
    assert len(out["errors"]) == 1 and "boom" in out["errors"][0]["error"]


def test_dry_run_warns_on_perfect_seed():
    perfect = "def evaluate(candidate, example):\n    return 1.0, {'ok': True}"
    out = _dry_run({"t": "x"}, _py(perfect), None, None, Config(backend="mock"))
    assert out["ok"] is True
    assert any("no learning signal" in w for w in out["warnings"])


def test_cli_dry_run_prints_one_json_object(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"seed_candidate": {"t": "x"}, "evaluator": {"type": "python", "code": GOOD},
                               "config": {"backend": "mock", "run_dir": str(tmp_path / "run")}}))
    r = subprocess.run([sys.executable, "-m", "optimize_anything.cli", "run", "--config", str(cfg), "--dry-run"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["dry_run"] is True and out["ok"] is True
    assert not (tmp_path / "run" / "checkpoint.json").exists()  # nothing was optimized
