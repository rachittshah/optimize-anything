# engine/src/optimize_anything/evaluators/shell_eval.py
from __future__ import annotations
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class ShellEvaluator:
    def __init__(
        self,
        command: str,
        score_pattern: str = r"([\d.]+)",
        timeout: int = 30,
    ):
        self.command = command
        self.score_pattern = score_pattern
        self.timeout = timeout

    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        # Write candidate to temp file for safe access
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(candidate)
            candidate_file = f.name

        cmd = self.command.replace("{{candidate}}", candidate)
        cmd = cmd.replace("{{candidate_file}}", candidate_file)

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=self.timeout,
            )
            stdout = result.stdout
            stderr = result.stderr

            match = re.search(self.score_pattern, stdout)
            if match:
                score = float(match.group(1))
            else:
                score = 0.0

            return score, {"stdout": stdout, "stderr": stderr, "returncode": result.returncode}

        except subprocess.TimeoutExpired:
            return 0.0, {"error": f"Timeout after {self.timeout}s"}
        finally:
            Path(candidate_file).unlink(missing_ok=True)
