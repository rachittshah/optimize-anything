# engine/src/optimize_anything/evaluators/shell_eval.py
from __future__ import annotations
import json
import re
import shlex
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

    def cache_identity(self) -> str:
        return json.dumps(["shell", self.command, self.score_pattern])

    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        # Placeholders:
        #   {{candidate}}       raw candidate text
        #   {{candidate_file}}  path to a temp file holding the candidate
        #   {{example_file}}    path to a temp file holding the example as JSON
        #   {{example_json}}    the example as JSON, shell-quoted (one argument; don't add quotes)
        # All are substituted in one pass, so placeholder-like text inside a value is left alone.
        temp_files: list[str] = []

        def _temp(text: str, suffix: str) -> str:
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write(text)
            temp_files.append(f.name)
            return f.name

        example_json = json.dumps(example, default=str)
        values: dict[str, str] = {}
        if "{{candidate}}" in self.command:
            values["candidate"] = candidate
        if "{{candidate_file}}" in self.command:
            values["candidate_file"] = _temp(candidate, ".txt")
        if "{{example_file}}" in self.command:
            values["example_file"] = _temp(example_json, ".json")
        if "{{example_json}}" in self.command:
            values["example_json"] = shlex.quote(example_json)
        cmd = re.sub(
            r"\{\{(candidate|candidate_file|example_file|example_json)\}\}",
            lambda m: values[m.group(1)], self.command,
        )

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
            for path in temp_files:
                Path(path).unlink(missing_ok=True)
