# engine/src/optimize_anything/evaluators/python_eval.py
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class PythonEvaluator:
    def __init__(self, code: str, timeout: int = 30):
        self.code = code
        self.timeout = timeout

    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        runner = self._build_runner(candidate, example)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(runner)
            f.flush()
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                return 0.0, {"error": result.stderr, "stdout": result.stdout}

            output = json.loads(result.stdout.strip().split("\n")[-1])
            return float(output["score"]), output.get("asi", {})

        except subprocess.TimeoutExpired:
            return 0.0, {"error": f"Timeout after {self.timeout}s"}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return 0.0, {"error": f"Output parse error: {e}"}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _build_runner(self, candidate: str, example: dict | None) -> str:
        return f'''
import json

{self.code}

_candidate = json.loads({json.dumps(json.dumps(candidate))})
_example = json.loads({json.dumps(json.dumps(example))})

_result = evaluate(_candidate, _example)
if isinstance(_result, tuple):
    _score, _asi = _result
else:
    _score, _asi = float(_result), {{}}

print(json.dumps({{"score": _score, "asi": _asi}}))
'''
