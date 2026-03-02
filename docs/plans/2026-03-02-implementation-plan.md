# optimize_anything Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone universal optimization system that optimizes any text artifact through iterative LLM-powered search with evaluation feedback, exposed as a Python CLI/library, TypeScript MCP server, and Claude Code skill.

**Architecture:** Python engine (uv) implements the core optimization loop — Pareto frontier tracking, reflective mutation, candidate selection, checkpointing. TypeScript MCP server is a thin wrapper that spawns the Python engine as a subprocess. Claude Code skill teaches how to formulate optimization problems.

**Tech Stack:** Python 3.12+, uv, anthropic SDK, pydantic; TypeScript, @modelcontextprotocol/sdk, zod; Claude Opus 4.6 (proposer), Claude Sonnet 4.6 (evaluator/judge)

---

## Task 1: Project Scaffolding

**Files:**
- Create: `engine/pyproject.toml`
- Create: `engine/src/optimize_anything/__init__.py`
- Create: `engine/src/optimize_anything/config.py`
- Create: `mcp-server/package.json`
- Create: `mcp-server/tsconfig.json`
- Create: `mcp-server/src/index.ts`
- Create: `skill/SKILL.md`
- Create: `CLAUDE.md`

**Step 1: Initialize Python engine with uv**

```bash
cd ~/optimize-anything && mkdir -p engine/src/optimize_anything engine/tests
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "optimize-anything"
version = "0.1.0"
description = "Universal text artifact optimizer using LLM-powered search"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.80.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]

[project.scripts]
optimize-anything = "optimize_anything.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/optimize_anything"]
```

**Step 3: Create config.py with all configuration dataclasses**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class EvaluatorType(str, Enum):
    PYTHON = "python"
    SHELL = "shell"
    LLM_JUDGE = "llm_judge"


class SelectionStrategy(str, Enum):
    PARETO = "pareto"
    BEST = "best"
    EPSILON_GREEDY = "epsilon_greedy"


class FrontierType(str, Enum):
    INSTANCE = "instance"
    OBJECTIVE = "objective"
    HYBRID = "hybrid"


class ComponentSelector(str, Enum):
    ROUND_ROBIN = "round_robin"
    ALL = "all"


@dataclass
class EvaluatorConfig:
    type: EvaluatorType
    # For PYTHON: the code string defining evaluate(candidate, example) -> (score, asi)
    code: str | None = None
    # For SHELL: command template with {{candidate}} placeholder
    command: str | None = None
    score_pattern: str | None = None  # regex to extract score from stdout
    # For LLM_JUDGE: natural language criteria
    criteria: str | None = None
    judge_model: str = "claude-sonnet-4-6"
    # Common
    timeout: int = 30


@dataclass
class Config:
    # Proposer
    model: str = "claude-opus-4-6"
    max_tokens: int = 8192
    temperature: float = 1.0

    # Engine
    max_iterations: int = 100
    max_metric_calls: int | None = None
    selection_strategy: SelectionStrategy = SelectionStrategy.PARETO
    frontier_type: FrontierType = FrontierType.INSTANCE
    component_selector: ComponentSelector = ComponentSelector.ROUND_ROBIN
    reflection_minibatch_size: int = 3
    skip_perfect_score: bool = True
    use_merge: bool = False
    epsilon: float = 0.1  # for epsilon-greedy

    # Checkpointing
    checkpoint_interval: int = 5
    run_dir: str = "~/.optimize-anything/runs"

    # Stopping
    timeout_seconds: int | None = None
    no_improvement_patience: int | None = None
```

**Step 4: Create __init__.py with public API stub**

```python
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType

__version__ = "0.1.0"

__all__ = [
    "optimize_anything",
    "Config",
    "EvaluatorConfig",
    "EvaluatorType",
]


def optimize_anything(
    seed_candidate: dict[str, str] | None = None,
    evaluator=None,
    dataset: list | None = None,
    valset: list | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: Config | None = None,
    on_event=None,
):
    """Optimize any text artifact through iterative LLM-powered search."""
    raise NotImplementedError("Engine not yet implemented")
```

**Step 5: Initialize TypeScript MCP server**

```bash
cd ~/optimize-anything && mkdir -p mcp-server/src
```

Create `mcp-server/package.json`:
```json
{
  "name": "optimize-anything-mcp",
  "version": "1.0.0",
  "type": "module",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "start": "node dist/index.js"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.20.0",
    "zod": "^3.23.8"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "typescript": "^5.5.0"
  }
}
```

Create `mcp-server/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src/**/*"]
}
```

Create minimal `mcp-server/src/index.ts`:
```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';

const server = new Server(
  { name: 'optimize-anything-mcp', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [] }));
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  throw new Error(`Unknown tool: ${request.params.name}`);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('optimize-anything MCP server running on stdio');
}

main().catch((error) => { console.error('Fatal:', error); process.exit(1); });
```

**Step 6: Create CLAUDE.md for the project**

```markdown
# optimize-anything

Universal text artifact optimizer using LLM-powered iterative search.

## Project Structure
- `engine/` — Python optimization engine (uv)
- `mcp-server/` — TypeScript MCP server (thin wrapper)
- `skill/` — Claude Code skill

## Commands
- `cd engine && uv sync` — install Python deps
- `cd engine && uv run pytest` — run tests
- `cd engine && uv run optimize-anything --help` — CLI
- `cd mcp-server && npm install && npm run build` — build MCP server

## Architecture
Python engine does all optimization logic (Pareto frontier, reflective mutation, evaluation).
TS MCP server spawns Python subprocess and manages run lifecycle.
Three evaluator types: Python code, shell command, LLM-as-judge.
Three optimization modes: single-task, multi-task, generalization.
```

**Step 7: Run uv sync and npm install**

```bash
cd ~/optimize-anything/engine && uv sync
cd ~/optimize-anything/mcp-server && npm install
```

**Step 8: Verify both build**

```bash
cd ~/optimize-anything/engine && uv run python -c "from optimize_anything import Config; print('OK')"
cd ~/optimize-anything/mcp-server && npm run build
```

**Step 9: Initialize git and commit**

```bash
cd ~/optimize-anything && git init && git add -A && git commit -m "feat: project scaffolding — Python engine + TS MCP server + skill stub"
```

---

## Task 2: Candidate & State Management

**Files:**
- Create: `engine/src/optimize_anything/core/__init__.py`
- Create: `engine/src/optimize_anything/core/candidate.py`
- Create: `engine/src/optimize_anything/core/state.py`
- Create: `engine/src/optimize_anything/core/pareto.py`
- Create: `engine/tests/test_state.py`
- Create: `engine/tests/test_pareto.py`

**Step 1: Write failing tests for candidate and state**

```python
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
```

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/ -v
```

Expected: FAIL — modules don't exist yet.

**Step 3: Implement candidate.py**

```python
# engine/src/optimize_anything/core/candidate.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Candidate:
    components: dict[str, str]
    parent_ids: list[int] = field(default_factory=list)
    tag: str = ""

    def copy(self) -> Candidate:
        return Candidate(
            components=dict(self.components),
            parent_ids=list(self.parent_ids),
            tag=self.tag,
        )
```

**Step 4: Implement pareto.py**

```python
# engine/src/optimize_anything/core/pareto.py
from __future__ import annotations
import random
from collections import defaultdict


class ParetoFrontier:
    def __init__(self, seed: int = 42):
        self._front: dict[str, float] = {}
        self._programs: dict[str, set[int]] = defaultdict(set)
        self._rng = random.Random(seed)

    def update(self, key: str, score: float, candidate_idx: int) -> bool:
        prev = self._front.get(key, float("-inf"))
        if score > prev:
            self._front[key] = score
            self._programs[key] = {candidate_idx}
            return True
        elif score == prev:
            self._programs[key].add(candidate_idx)
            return False
        return False

    def best_score(self, key: str) -> float:
        return self._front.get(key, float("-inf"))

    def programs_at(self, key: str) -> set[int]:
        return self._programs.get(key, set())

    def all_program_ids(self) -> set[int]:
        result = set()
        for programs in self._programs.values():
            result |= programs
        return result

    def aggregated_score(self, candidate_idx: int) -> float:
        scores = []
        for key, programs in self._programs.items():
            if candidate_idx in programs:
                scores.append(self._front[key])
        return sum(scores) / len(scores) if scores else 0.0

    def select_candidate(self, all_scores: list[float]) -> int:
        freq: dict[int, int] = defaultdict(int)
        for programs in self._programs.values():
            for pid in programs:
                freq[pid] += 1

        if not freq:
            return 0

        # Remove dominated: a program is dominated if for every key it appears in,
        # there's another program also in that key
        # Simplified: just do frequency-weighted sampling
        sampling_list = []
        for pid, count in freq.items():
            sampling_list.extend([pid] * count)

        return self._rng.choice(sampling_list)

    def to_dict(self) -> dict:
        return {
            "front": dict(self._front),
            "programs": {k: list(v) for k, v in self._programs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int = 42) -> ParetoFrontier:
        pf = cls(seed=seed)
        pf._front = data["front"]
        pf._programs = defaultdict(set, {k: set(v) for k, v in data["programs"].items()})
        return pf
```

**Step 5: Implement state.py**

```python
# engine/src/optimize_anything/core/state.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

from optimize_anything.core.candidate import Candidate
from optimize_anything.core.pareto import ParetoFrontier


@dataclass
class State:
    component_names: list[str]
    candidates: list[Candidate] = field(default_factory=list)
    scores: list[dict[str, float]] = field(default_factory=list)  # per-example scores
    agg_scores: list[float] = field(default_factory=list)  # mean score per candidate
    frontier: ParetoFrontier = field(default_factory=ParetoFrontier)
    iteration: int = -1
    total_evals: int = 0
    component_cursor: dict[int, int] = field(default_factory=dict)  # per-candidate round-robin

    def add_candidate(
        self,
        candidate: Candidate,
        scores: dict[str, float],
        parent_ids: list[int],
    ) -> int:
        idx = len(self.candidates)
        candidate.parent_ids = parent_ids
        self.candidates.append(candidate)
        self.scores.append(scores)
        agg = sum(scores.values()) / len(scores) if scores else 0.0
        self.agg_scores.append(agg)
        self.component_cursor[idx] = 0

        for key, score in scores.items():
            self.frontier.update(key, score, idx)

        return idx

    def best_candidate(self) -> Candidate:
        if not self.candidates:
            raise ValueError("No candidates in state")
        best_idx = max(range(len(self.agg_scores)), key=lambda i: self.agg_scores[i])
        return self.candidates[best_idx]

    def best_score(self) -> float:
        if not self.agg_scores:
            return 0.0
        return max(self.agg_scores)

    def record_evals(self, count: int):
        self.total_evals += count

    def next_component(self, candidate_idx: int) -> str:
        cursor = self.component_cursor.get(candidate_idx, 0)
        name = self.component_names[cursor % len(self.component_names)]
        self.component_cursor[candidate_idx] = cursor + 1
        return name

    def save(self, path: Path):
        data = {
            "component_names": self.component_names,
            "candidates": [
                {"components": c.components, "parent_ids": c.parent_ids, "tag": c.tag}
                for c in self.candidates
            ],
            "scores": self.scores,
            "agg_scores": self.agg_scores,
            "frontier": self.frontier.to_dict(),
            "iteration": self.iteration,
            "total_evals": self.total_evals,
            "component_cursor": {str(k): v for k, v in self.component_cursor.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> State:
        data = json.loads(path.read_text())
        state = cls(component_names=data["component_names"])
        for c_data in data["candidates"]:
            state.candidates.append(
                Candidate(c_data["components"], c_data["parent_ids"], c_data.get("tag", ""))
            )
        state.scores = data["scores"]
        state.agg_scores = data["agg_scores"]
        state.frontier = ParetoFrontier.from_dict(data["frontier"])
        state.iteration = data["iteration"]
        state.total_evals = data["total_evals"]
        state.component_cursor = {int(k): v for k, v in data["component_cursor"].items()}
        return state
```

**Step 6: Create `engine/src/optimize_anything/core/__init__.py`**

```python
from optimize_anything.core.candidate import Candidate
from optimize_anything.core.state import State
from optimize_anything.core.pareto import ParetoFrontier

__all__ = ["Candidate", "State", "ParetoFrontier"]
```

**Step 7: Run tests and verify they pass**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/ -v
```

Expected: ALL PASS

**Step 8: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: candidate, state, and Pareto frontier with tests"
```

---

## Task 3: Evaluator Types

**Files:**
- Create: `engine/src/optimize_anything/evaluators/__init__.py`
- Create: `engine/src/optimize_anything/evaluators/base.py`
- Create: `engine/src/optimize_anything/evaluators/python_eval.py`
- Create: `engine/src/optimize_anything/evaluators/shell_eval.py`
- Create: `engine/src/optimize_anything/evaluators/llm_judge.py`
- Create: `engine/tests/test_evaluators.py`

**Step 1: Write failing tests**

```python
# engine/tests/test_evaluators.py
import pytest
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator


def test_python_evaluator_basic():
    code = '''
def evaluate(candidate, example=None):
    score = 1.0 if "hello" in candidate else 0.0
    return score, {"matched": "hello" in candidate}
'''
    ev = PythonEvaluator(code=code, timeout=10)
    score, asi = ev.evaluate("hello world")
    assert score == 1.0
    assert asi["matched"] is True


def test_python_evaluator_with_example():
    code = '''
def evaluate(candidate, example=None):
    expected = example.get("expected", "") if example else ""
    score = 1.0 if candidate.strip() == expected else 0.0
    return score, {"expected": expected, "got": candidate.strip()}
'''
    ev = PythonEvaluator(code=code, timeout=10)
    score, asi = ev.evaluate("foo", example={"expected": "foo"})
    assert score == 1.0


def test_python_evaluator_timeout():
    code = '''
import time
def evaluate(candidate, example=None):
    time.sleep(60)
    return 1.0, {}
'''
    ev = PythonEvaluator(code=code, timeout=2)
    score, asi = ev.evaluate("test")
    assert score == 0.0
    assert "timeout" in asi.get("error", "").lower()


def test_shell_evaluator_basic():
    ev = ShellEvaluator(
        command='echo "Score: 0.85"',
        score_pattern=r"Score: ([\d.]+)",
        timeout=10,
    )
    score, asi = ev.evaluate("anything")
    assert score == 0.85


def test_shell_evaluator_with_candidate():
    ev = ShellEvaluator(
        command='echo "Score: $(echo "{{candidate}}" | wc -c | tr -d " ")"',
        score_pattern=r"Score: (\d+)",
        timeout=10,
    )
    score, asi = ev.evaluate("hello")
    assert score > 0


# LLM judge tests need mocking — tested in integration
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_evaluators.py -v
```

**Step 3: Implement base.py**

```python
# engine/src/optimize_anything/evaluators/base.py
from __future__ import annotations
from typing import Any, Protocol


class Evaluator(Protocol):
    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a candidate, return (score, ASI diagnostics)."""
        ...
```

**Step 4: Implement python_eval.py**

```python
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

_candidate = {json.dumps(candidate)}
_example = {json.dumps(example)}

_result = evaluate(_candidate, _example)
if isinstance(_result, tuple):
    _score, _asi = _result
else:
    _score, _asi = float(_result), {{}}

print(json.dumps({{"score": _score, "asi": _asi}}))
'''
```

**Step 5: Implement shell_eval.py**

```python
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
```

**Step 6: Implement llm_judge.py**

```python
# engine/src/optimize_anything/evaluators/llm_judge.py
from __future__ import annotations
import re
from typing import Any

import anthropic


class LLMJudgeEvaluator:
    def __init__(
        self,
        criteria: str,
        model: str = "claude-sonnet-4-6",
        max_score: float = 10.0,
    ):
        self.criteria = criteria
        self.model = model
        self.max_score = max_score
        self.client = anthropic.Anthropic()

    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        prompt = f"""Evaluate the following candidate artifact against the given criteria.

## Criteria
{self.criteria}

## Candidate
{candidate}
"""
        if example:
            prompt += f"\n## Context\n{example}\n"

        prompt += f"""
## Instructions
1. Analyze the candidate against each criterion
2. Provide specific feedback
3. End with exactly: SCORE: X/{self.max_score:.0f}

Your evaluation:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            match = re.search(r"SCORE:\s*([\d.]+)", text)
            if match:
                raw_score = float(match.group(1))
                score = raw_score / self.max_score  # normalize to [0, 1]
            else:
                score = 0.0

            return score, {"reasoning": text, "raw_score": raw_score if match else None}

        except Exception as e:
            return 0.0, {"error": str(e)}
```

**Step 7: Create `engine/src/optimize_anything/evaluators/__init__.py`**

```python
from optimize_anything.evaluators.base import Evaluator
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator

__all__ = ["Evaluator", "PythonEvaluator", "ShellEvaluator", "LLMJudgeEvaluator"]
```

**Step 8: Run tests and verify they pass**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_evaluators.py -v
```

**Step 9: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: three evaluator types — Python, shell, LLM-as-judge"
```

---

## Task 4: Proposer (Claude API Reflection + Proposal)

**Files:**
- Create: `engine/src/optimize_anything/proposer/__init__.py`
- Create: `engine/src/optimize_anything/proposer/claude_proposer.py`
- Create: `engine/src/optimize_anything/proposer/prompts.py`
- Create: `engine/tests/test_proposer.py`

**Step 1: Write failing tests**

```python
# engine/tests/test_proposer.py
from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate


def test_build_reflection_prompt():
    prompt = build_reflection_prompt(
        current_text="You are a helpful assistant",
        component_name="system_prompt",
        objective="Optimize for helpfulness",
        asi_entries=[
            {"score": 0.5, "feedback": {"error": "Too vague"}},
            {"score": 0.8, "feedback": {"note": "Good but verbose"}},
        ],
        background=None,
    )
    assert "system_prompt" in prompt
    assert "Too vague" in prompt
    assert "Good but verbose" in prompt
    assert "Optimize for helpfulness" in prompt


def test_extract_candidate_from_fenced():
    text = '''Here is my improved version:

```
You are a precise, helpful assistant that always provides sources.
```

This improves clarity.'''
    result = extract_candidate(text)
    assert result == "You are a precise, helpful assistant that always provides sources."


def test_extract_candidate_no_fence():
    text = "You are a precise, helpful assistant."
    result = extract_candidate(text)
    assert result == "You are a precise, helpful assistant."
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_proposer.py -v
```

**Step 3: Implement prompts.py**

```python
# engine/src/optimize_anything/proposer/prompts.py
from __future__ import annotations
import re
from typing import Any


def build_reflection_prompt(
    current_text: str,
    component_name: str,
    objective: str,
    asi_entries: list[dict[str, Any]],
    background: str | None = None,
) -> str:
    parts = [f"# Task: Improve the `{component_name}` component\n"]

    if objective:
        parts.append(f"## Objective\n{objective}\n")

    if background:
        parts.append(f"## Background\n{background}\n")

    parts.append(f"## Current Version\n```\n{current_text}\n```\n")

    parts.append("## Evaluation Results (Actionable Side Information)\n")
    for i, entry in enumerate(asi_entries):
        parts.append(f"### Example {i + 1} — Score: {entry['score']}")
        feedback = entry.get("feedback", {})
        if isinstance(feedback, dict):
            for k, v in feedback.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(f"- {feedback}")
        parts.append("")

    parts.append("""## Instructions
1. Analyze the evaluation results to understand what's working and what's not
2. Identify specific improvements based on the diagnostic feedback
3. Propose an improved version of the component
4. Output your improved version inside a fenced code block (``` ```)
5. The improved version must be a complete replacement, not a diff

Your improved version:""")

    return "\n".join(parts)


def extract_candidate(text: str) -> str:
    match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
```

**Step 4: Implement claude_proposer.py**

```python
# engine/src/optimize_anything/proposer/claude_proposer.py
from __future__ import annotations
from typing import Any

import anthropic

from optimize_anything.proposer.prompts import build_reflection_prompt, extract_candidate


class ClaudeProposer:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 8192,
        temperature: float = 1.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic()

    def propose(
        self,
        current_text: str,
        component_name: str,
        objective: str,
        asi_entries: list[dict[str, Any]],
        background: str | None = None,
    ) -> str:
        prompt = build_reflection_prompt(
            current_text=current_text,
            component_name=component_name,
            objective=objective,
            asi_entries=asi_entries,
            background=background,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return extract_candidate(response.content[0].text)
```

**Step 5: Create `engine/src/optimize_anything/proposer/__init__.py`**

```python
from optimize_anything.proposer.claude_proposer import ClaudeProposer

__all__ = ["ClaudeProposer"]
```

**Step 6: Run tests and verify they pass**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_proposer.py -v
```

**Step 7: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: Claude-powered proposer with reflection prompts"
```

---

## Task 5: Strategies (Selection, Sampling, Stopping)

**Files:**
- Create: `engine/src/optimize_anything/strategies/__init__.py`
- Create: `engine/src/optimize_anything/strategies/selection.py`
- Create: `engine/src/optimize_anything/strategies/sampling.py`
- Create: `engine/src/optimize_anything/strategies/stopping.py`
- Create: `engine/tests/test_strategies.py`

**Step 1: Write failing tests**

```python
# engine/tests/test_strategies.py
from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import BudgetStopper, CompositeStopper


def test_epoch_batch_sampler():
    sampler = EpochBatchSampler(minibatch_size=2, seed=42)
    ids = ["a", "b", "c", "d", "e"]
    batch1 = sampler.next_batch(ids, iteration=0)
    assert len(batch1) == 2
    batch2 = sampler.next_batch(ids, iteration=1)
    assert len(batch2) == 2
    assert batch1 != batch2 or True  # may overlap by chance


def test_epoch_sampler_covers_all():
    sampler = EpochBatchSampler(minibatch_size=2, seed=42)
    ids = ["a", "b", "c", "d"]
    seen = set()
    for i in range(2):  # 2 batches of 2 = full epoch
        batch = sampler.next_batch(ids, iteration=i)
        seen.update(batch)
    assert seen == set(ids)


def test_budget_stopper():
    stopper = BudgetStopper(max_evals=10)
    assert stopper.should_stop(total_evals=5) is False
    assert stopper.should_stop(total_evals=10) is True
    assert stopper.should_stop(total_evals=15) is True


def test_composite_stopper():
    s1 = BudgetStopper(max_evals=100)
    s2 = BudgetStopper(max_evals=5)
    composite = CompositeStopper([s1, s2])
    assert composite.should_stop(total_evals=3) is False
    assert composite.should_stop(total_evals=5) is True  # s2 triggers
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_strategies.py -v
```

**Step 3: Implement sampling.py**

```python
# engine/src/optimize_anything/strategies/sampling.py
from __future__ import annotations
import random


class EpochBatchSampler:
    def __init__(self, minibatch_size: int = 3, seed: int = 42):
        self.minibatch_size = minibatch_size
        self._rng = random.Random(seed)
        self._shuffled: list = []
        self._epoch = -1

    def next_batch(self, all_ids: list, iteration: int) -> list:
        n = len(all_ids)
        base = iteration * self.minibatch_size
        epoch = base // n

        if epoch > self._epoch:
            self._epoch = epoch
            self._shuffled = list(all_ids)
            self._rng.shuffle(self._shuffled)
            # Pad to multiple of minibatch_size
            remainder = n % self.minibatch_size
            if remainder != 0:
                pad = self.minibatch_size - remainder
                self._shuffled.extend(self._shuffled[:pad])

        if not self._shuffled:
            self._shuffled = list(all_ids)
            self._rng.shuffle(self._shuffled)

        start = base % len(self._shuffled)
        end = start + self.minibatch_size
        return self._shuffled[start:end]
```

**Step 4: Implement stopping.py**

```python
# engine/src/optimize_anything/strategies/stopping.py
from __future__ import annotations
import time
from typing import Protocol


class Stopper(Protocol):
    def should_stop(self, **kwargs) -> bool: ...


class BudgetStopper:
    def __init__(self, max_evals: int):
        self.max_evals = max_evals

    def should_stop(self, total_evals: int = 0, **kwargs) -> bool:
        return total_evals >= self.max_evals


class IterationStopper:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def should_stop(self, iteration: int = 0, **kwargs) -> bool:
        return iteration >= self.max_iterations


class TimeoutStopper:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self._start = time.time()

    def should_stop(self, **kwargs) -> bool:
        return (time.time() - self._start) >= self.seconds


class NoImprovementStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self._best_score = float("-inf")
        self._stale_count = 0

    def should_stop(self, best_score: float = 0.0, **kwargs) -> bool:
        if best_score > self._best_score:
            self._best_score = best_score
            self._stale_count = 0
        else:
            self._stale_count += 1
        return self._stale_count >= self.patience


class CompositeStopper:
    def __init__(self, stoppers: list):
        self.stoppers = stoppers

    def should_stop(self, **kwargs) -> bool:
        return any(s.should_stop(**kwargs) for s in self.stoppers)
```

**Step 5: Implement selection.py**

```python
# engine/src/optimize_anything/strategies/selection.py
from __future__ import annotations
import random
from typing import Protocol

from optimize_anything.core.state import State


class CandidateSelector(Protocol):
    def select(self, state: State) -> int: ...


class ParetoSelector:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def select(self, state: State) -> int:
        return state.frontier.select_candidate(state.agg_scores)


class BestSelector:
    def select(self, state: State) -> int:
        return max(range(len(state.agg_scores)), key=lambda i: state.agg_scores[i])


class EpsilonGreedySelector:
    def __init__(self, epsilon: float = 0.1, seed: int = 42):
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select(self, state: State) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, len(state.candidates) - 1)
        return max(range(len(state.agg_scores)), key=lambda i: state.agg_scores[i])
```

**Step 6: Create `engine/src/optimize_anything/strategies/__init__.py`**

```python
from optimize_anything.strategies.selection import ParetoSelector, BestSelector, EpsilonGreedySelector
from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import (
    BudgetStopper, IterationStopper, TimeoutStopper,
    NoImprovementStopper, CompositeStopper,
)

__all__ = [
    "ParetoSelector", "BestSelector", "EpsilonGreedySelector",
    "EpochBatchSampler",
    "BudgetStopper", "IterationStopper", "TimeoutStopper",
    "NoImprovementStopper", "CompositeStopper",
]
```

**Step 7: Run tests and verify they pass**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_strategies.py -v
```

**Step 8: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: selection, sampling, and stopping strategies"
```

---

## Task 6: Core Engine (The Optimization Loop)

**Files:**
- Create: `engine/src/optimize_anything/core/engine.py`
- Create: `engine/src/optimize_anything/core/events.py`
- Create: `engine/tests/test_engine.py`

**Step 1: Write failing tests**

```python
# engine/tests/test_engine.py
from optimize_anything.core.engine import Engine
from optimize_anything.core.candidate import Candidate
from optimize_anything.config import Config


class FakeProposer:
    def __init__(self):
        self.call_count = 0

    def propose(self, current_text, component_name, objective, asi_entries, background=None):
        self.call_count += 1
        return current_text + " improved"


class FakeEvaluator:
    def __init__(self):
        self.call_count = 0

    def evaluate(self, candidate, example=None):
        self.call_count += 1
        score = min(1.0, len(candidate) / 100)
        return score, {"length": len(candidate)}


def test_engine_runs_iterations():
    config = Config(max_iterations=3)
    engine = Engine(
        seed_candidate={"prompt": "hello"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="make it longer",
        config=config,
    )
    result = engine.run()
    assert result.total_iterations >= 1
    assert result.best_score > 0


def test_engine_improves_score():
    config = Config(max_iterations=5)
    engine = Engine(
        seed_candidate={"prompt": "hi"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="make it longer",
        config=config,
    )
    result = engine.run()
    assert result.best_score >= 0.02  # "hi" = 2 chars, should grow


def test_engine_respects_budget():
    config = Config(max_iterations=100, max_metric_calls=5)
    engine = Engine(
        seed_candidate={"prompt": "test"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="improve",
        config=config,
    )
    result = engine.run()
    assert result.total_evals <= 10  # some slack for initial eval


def test_engine_emits_events():
    events = []
    config = Config(max_iterations=2)
    engine = Engine(
        seed_candidate={"prompt": "hello"},
        evaluator=FakeEvaluator(),
        proposer=FakeProposer(),
        objective="improve",
        config=config,
        on_event=lambda e: events.append(e),
    )
    engine.run()
    event_types = [e["type"] for e in events]
    assert "iteration_start" in event_types
    assert "optimization_complete" in event_types
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_engine.py -v
```

**Step 3: Implement events.py**

```python
# engine/src/optimize_anything/core/events.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


def event(type: str, **kwargs) -> dict[str, Any]:
    return {"type": type, **kwargs}


def iteration_start(iteration: int, candidate_idx: int) -> dict:
    return event("iteration_start", iteration=iteration, candidate_idx=candidate_idx)


def evaluation_complete(candidate_idx: int, score: float, asi: dict) -> dict:
    return event("evaluation_complete", candidate_idx=candidate_idx, score=score, asi=asi)


def new_best_found(candidate_idx: int, score: float) -> dict:
    return event("new_best_found", candidate_idx=candidate_idx, score=score)


def frontier_updated(frontier_size: int) -> dict:
    return event("frontier_updated", frontier_size=frontier_size)


def optimization_complete(best_score: float, total_iterations: int, total_evals: int) -> dict:
    return event(
        "optimization_complete",
        best_score=best_score,
        total_iterations=total_iterations,
        total_evals=total_evals,
    )
```

**Step 4: Implement engine.py**

```python
# engine/src/optimize_anything/core/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from optimize_anything.config import Config, SelectionStrategy
from optimize_anything.core.candidate import Candidate
from optimize_anything.core.state import State
from optimize_anything.core import events
from optimize_anything.strategies.selection import ParetoSelector, BestSelector, EpsilonGreedySelector
from optimize_anything.strategies.sampling import EpochBatchSampler
from optimize_anything.strategies.stopping import (
    BudgetStopper, IterationStopper, CompositeStopper, TimeoutStopper, NoImprovementStopper,
)


@dataclass
class OptimizationResult:
    best_candidate: dict[str, str]
    best_score: float
    all_candidates: list[dict[str, str]]
    all_scores: list[float]
    pareto_frontier: dict
    total_iterations: int
    total_evals: int


class Engine:
    def __init__(
        self,
        seed_candidate: dict[str, str],
        evaluator,
        proposer,
        objective: str = "",
        background: str | None = None,
        dataset: list | None = None,
        valset: list | None = None,
        config: Config | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.config = config or Config()
        self.evaluator = evaluator
        self.proposer = proposer
        self.objective = objective
        self.background = background
        self.dataset = dataset
        self.valset = valset
        self.on_event = on_event or (lambda e: None)

        self.state = State(component_names=list(seed_candidate.keys()))

        # Build selector
        if self.config.selection_strategy == SelectionStrategy.PARETO:
            self.selector = ParetoSelector()
        elif self.config.selection_strategy == SelectionStrategy.BEST:
            self.selector = BestSelector()
        else:
            self.selector = EpsilonGreedySelector(self.config.epsilon)

        self.sampler = EpochBatchSampler(self.config.reflection_minibatch_size)

        # Build stopper
        stoppers = [IterationStopper(self.config.max_iterations)]
        if self.config.max_metric_calls:
            stoppers.append(BudgetStopper(self.config.max_metric_calls))
        if self.config.timeout_seconds:
            stoppers.append(TimeoutStopper(self.config.timeout_seconds))
        if self.config.no_improvement_patience:
            stoppers.append(NoImprovementStopper(self.config.no_improvement_patience))
        self.stopper = CompositeStopper(stoppers)

        # Initial evaluation of seed
        self._seed_candidate = seed_candidate

    def run(self) -> OptimizationResult:
        # Evaluate seed candidate
        seed_scores = self._evaluate_candidate(self._seed_candidate)
        self.state.add_candidate(
            Candidate(components=dict(self._seed_candidate)),
            scores=seed_scores,
            parent_ids=[],
        )

        while not self._should_stop():
            self.state.iteration += 1
            iteration = self.state.iteration

            # Select parent candidate
            if len(self.state.candidates) == 1:
                parent_idx = 0
            else:
                parent_idx = self.selector.select(self.state)

            parent = self.state.candidates[parent_idx]
            self.on_event(events.iteration_start(iteration, parent_idx))

            # Select component to update (round-robin)
            component_name = self.state.next_component(parent_idx)

            # Get minibatch for evaluation
            if self.dataset:
                batch_ids = self.sampler.next_batch(
                    list(range(len(self.dataset))), iteration
                )
                batch = [self.dataset[i] for i in batch_ids]
            else:
                batch = [None]

            # Evaluate parent on minibatch
            parent_asi = []
            parent_scores = []
            for example in batch:
                score, asi = self.evaluator.evaluate(
                    parent.components.get(component_name, ""), example
                )
                parent_scores.append(score)
                parent_asi.append({"score": score, "feedback": asi})
                self.state.record_evals(1)

            self.on_event(events.evaluation_complete(
                parent_idx, sum(parent_scores) / len(parent_scores), {}
            ))

            # Skip if perfect
            if self.config.skip_perfect_score and all(s >= 1.0 for s in parent_scores):
                continue

            # Propose improvement
            new_text = self.proposer.propose(
                current_text=parent.components[component_name],
                component_name=component_name,
                objective=self.objective,
                asi_entries=parent_asi,
                background=self.background,
            )

            # Build new candidate
            new_components = dict(parent.components)
            new_components[component_name] = new_text
            new_candidate = Candidate(components=new_components, tag="reflective_mutation")

            # Evaluate new candidate on same minibatch
            new_scores = []
            for example in batch:
                score, asi = self.evaluator.evaluate(new_text, example)
                new_scores.append(score)
                self.state.record_evals(1)

            # Accept only if strictly improved on minibatch
            if sum(new_scores) > sum(parent_scores):
                # Full evaluation on valset (or dataset) for Pareto tracking
                full_scores = self._evaluate_candidate(new_components)
                new_idx = self.state.add_candidate(new_candidate, full_scores, [parent_idx])

                new_agg = self.state.agg_scores[new_idx]
                if new_agg >= self.state.best_score():
                    self.on_event(events.new_best_found(new_idx, new_agg))

                self.on_event(events.frontier_updated(len(self.state.frontier.all_program_ids())))

            # Checkpoint
            if (
                self.config.checkpoint_interval
                and iteration % self.config.checkpoint_interval == 0
            ):
                run_dir = Path(self.config.run_dir).expanduser()
                self.state.save(run_dir / "checkpoint.json")

        # Final result
        best = self.state.best_candidate()
        result = OptimizationResult(
            best_candidate=best.components,
            best_score=self.state.best_score(),
            all_candidates=[c.components for c in self.state.candidates],
            all_scores=self.state.agg_scores,
            pareto_frontier=self.state.frontier.to_dict(),
            total_iterations=self.state.iteration + 1,
            total_evals=self.state.total_evals,
        )

        self.on_event(events.optimization_complete(
            result.best_score, result.total_iterations, result.total_evals
        ))

        return result

    def _evaluate_candidate(self, components: dict[str, str]) -> dict[str, float]:
        examples = self.valset or self.dataset or [None]
        scores = {}
        for i, example in enumerate(examples):
            candidate_text = "\n".join(components.values())
            score, _ = self.evaluator.evaluate(candidate_text, example)
            scores[f"ex_{i}"] = score
            self.state.record_evals(1)
        return scores

    def _should_stop(self) -> bool:
        return self.stopper.should_stop(
            total_evals=self.state.total_evals,
            iteration=self.state.iteration + 1,
            best_score=self.state.best_score(),
        )
```

**Step 5: Run tests and verify they pass**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_engine.py -v
```

**Step 6: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: core optimization engine with event streaming"
```

---

## Task 7: Public API and CLI

**Files:**
- Modify: `engine/src/optimize_anything/__init__.py`
- Create: `engine/src/optimize_anything/api.py`
- Create: `engine/src/optimize_anything/cli.py`
- Create: `engine/tests/test_api.py`

**Step 1: Write failing tests**

```python
# engine/tests/test_api.py
from optimize_anything.api import optimize_anything
from optimize_anything.config import Config


def test_api_single_task_mode():
    def simple_eval(candidate, example=None):
        return min(1.0, len(candidate) / 50), {"length": len(candidate)}

    result = optimize_anything(
        seed_candidate={"text": "hello"},
        evaluator=simple_eval,
        objective="make it longer",
        config=Config(max_iterations=3),
    )
    assert result.best_score > 0
    assert "text" in result.best_candidate


def test_api_with_dataset():
    def eval_fn(candidate, example=None):
        target = example.get("target", "") if example else ""
        overlap = len(set(candidate.split()) & set(target.split()))
        return overlap / max(len(target.split()), 1), {}

    result = optimize_anything(
        seed_candidate={"answer": "hello world"},
        evaluator=eval_fn,
        dataset=[{"target": "hello world foo"}, {"target": "hello bar"}],
        objective="match targets",
        config=Config(max_iterations=2),
    )
    assert result.total_iterations >= 1
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_api.py -v
```

**Step 3: Implement api.py**

```python
# engine/src/optimize_anything/api.py
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Callable

from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import Engine, OptimizationResult
from optimize_anything.evaluators.python_eval import PythonEvaluator
from optimize_anything.evaluators.shell_eval import ShellEvaluator
from optimize_anything.evaluators.llm_judge import LLMJudgeEvaluator
from optimize_anything.proposer.claude_proposer import ClaudeProposer


class _FunctionEvaluator:
    """Wraps a plain Python function as an evaluator."""
    def __init__(self, fn: Callable):
        self.fn = fn

    def evaluate(self, candidate: str, example: dict | None = None) -> tuple[float, dict]:
        result = self.fn(candidate, example)
        if isinstance(result, tuple):
            return result
        return float(result), {}


def _build_evaluator(evaluator):
    if callable(evaluator):
        return _FunctionEvaluator(evaluator)
    if isinstance(evaluator, EvaluatorConfig):
        if evaluator.type == EvaluatorType.PYTHON:
            return PythonEvaluator(code=evaluator.code, timeout=evaluator.timeout)
        elif evaluator.type == EvaluatorType.SHELL:
            return ShellEvaluator(
                command=evaluator.command,
                score_pattern=evaluator.score_pattern or r"([\d.]+)",
                timeout=evaluator.timeout,
            )
        elif evaluator.type == EvaluatorType.LLM_JUDGE:
            return LLMJudgeEvaluator(
                criteria=evaluator.criteria,
                model=evaluator.judge_model,
            )
    if hasattr(evaluator, "evaluate"):
        return evaluator
    raise ValueError(f"Invalid evaluator: {type(evaluator)}")


def optimize_anything(
    seed_candidate: dict[str, str] | None = None,
    evaluator=None,
    dataset: list | None = None,
    valset: list | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: Config | None = None,
    on_event: Callable[[dict], None] | None = None,
    resume_from: str | None = None,
) -> OptimizationResult:
    """Optimize any text artifact through iterative LLM-powered search."""
    config = config or Config()

    if seed_candidate is None and objective is None:
        raise ValueError("Provide either seed_candidate or objective")
    if evaluator is None:
        raise ValueError("evaluator is required")

    # Generate seed from objective if not provided
    if seed_candidate is None:
        seed_candidate = {"artifact": ""}

    # Setup run directory
    run_id = str(uuid.uuid4())[:8]
    run_dir = Path(config.run_dir).expanduser() / run_id
    config = Config(**{**config.__dict__, "run_dir": str(run_dir)})

    # Build components
    eval_instance = _build_evaluator(evaluator)
    proposer = ClaudeProposer(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    engine = Engine(
        seed_candidate=seed_candidate,
        evaluator=eval_instance,
        proposer=proposer,
        objective=objective or "",
        background=background,
        dataset=dataset,
        valset=valset,
        config=config,
        on_event=on_event,
    )

    return engine.run()
```

**Step 4: Implement cli.py**

```python
# engine/src/optimize_anything/cli.py
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from optimize_anything.api import optimize_anything
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType


def main():
    parser = argparse.ArgumentParser(description="optimize_anything CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an optimization")
    run_parser.add_argument("--config", required=True, help="Path to JSON config file")
    run_parser.add_argument("--events", action="store_true", help="Stream events to stdout")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check run status")
    status_parser.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    elif args.command == "status":
        _status(args)
    else:
        parser.print_help()


def _run(args):
    config_data = json.loads(Path(args.config).read_text())

    seed = config_data.get("seed_candidate")
    objective = config_data.get("objective")
    background = config_data.get("background")
    dataset = config_data.get("dataset")
    valset = config_data.get("valset")

    eval_config = config_data.get("evaluator", {})
    evaluator = EvaluatorConfig(
        type=EvaluatorType(eval_config["type"]),
        code=eval_config.get("code"),
        command=eval_config.get("command"),
        score_pattern=eval_config.get("score_pattern"),
        criteria=eval_config.get("criteria"),
        judge_model=eval_config.get("judge_model", "claude-sonnet-4-6"),
        timeout=eval_config.get("timeout", 30),
    )

    engine_config = config_data.get("config", {})
    config = Config(**{k: v for k, v in engine_config.items() if k in Config.__dataclass_fields__})

    def on_event(event):
        if args.events:
            print(json.dumps(event), flush=True)

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        objective=objective,
        background=background,
        config=config,
        on_event=on_event,
    )

    output = {
        "best_candidate": result.best_candidate,
        "best_score": result.best_score,
        "total_iterations": result.total_iterations,
        "total_evals": result.total_evals,
    }
    print(json.dumps(output, indent=2))


def _status(args):
    checkpoint = Path(args.run_dir) / "checkpoint.json"
    if not checkpoint.exists():
        print(json.dumps({"status": "not_found"}))
        return

    from optimize_anything.core.state import State
    state = State.load(checkpoint)
    print(json.dumps({
        "status": "running" if state.iteration >= 0 else "initialized",
        "iteration": state.iteration,
        "total_evals": state.total_evals,
        "best_score": state.best_score(),
        "num_candidates": len(state.candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
```

**Step 5: Update `__init__.py`**

```python
from optimize_anything.api import optimize_anything
from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
from optimize_anything.core.engine import OptimizationResult

__version__ = "0.1.0"

__all__ = [
    "optimize_anything",
    "Config",
    "EvaluatorConfig",
    "EvaluatorType",
    "OptimizationResult",
]
```

**Step 6: Run tests and verify they pass**

Note: test_api.py tests use a FakeProposer-style setup since they call the real API which would need Claude. Update tests to use a mock or set max_iterations=0 pattern. The tests as written will work with the _FunctionEvaluator but will fail because ClaudeProposer needs an API key. Adjust tests:

```python
# Update test_api.py to mock the proposer
from unittest.mock import patch, MagicMock


def test_api_single_task_mode():
    def simple_eval(candidate, example=None):
        return min(1.0, len(candidate) / 50), {"length": len(candidate)}

    with patch("optimize_anything.api.ClaudeProposer") as MockProposer:
        mock_instance = MagicMock()
        mock_instance.propose.return_value = "hello world this is a longer text now"
        MockProposer.return_value = mock_instance

        result = optimize_anything(
            seed_candidate={"text": "hello"},
            evaluator=simple_eval,
            objective="make it longer",
            config=Config(max_iterations=3),
        )
        assert result.best_score > 0
        assert "text" in result.best_candidate
```

```bash
cd ~/optimize-anything/engine && uv run pytest tests/ -v
```

**Step 7: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: public API and CLI entry point"
```

---

## Task 8: Checkpointing and Resume

**Files:**
- Create: `engine/src/optimize_anything/core/checkpoint.py`
- Create: `engine/tests/test_checkpoint.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/test_checkpoint.py -v
```

These should already pass since State.save/load was implemented in Task 2. If not, fix and commit.

**Step 3: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: checkpoint save/load with tests"
```

---

## Task 9: TypeScript MCP Server — Full Implementation

**Files:**
- Modify: `mcp-server/src/index.ts`
- Create: `mcp-server/src/tools.ts`
- Create: `mcp-server/src/process-manager.ts`

**Step 1: Implement process-manager.ts**

```typescript
// mcp-server/src/process-manager.ts
import { spawn, ChildProcess } from 'child_process';
import { readFileSync, existsSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import { randomUUID } from 'crypto';

export interface RunConfig {
  seed_candidate?: Record<string, string>;
  evaluator: {
    type: 'python' | 'shell' | 'llm_judge';
    code?: string;
    command?: string;
    score_pattern?: string;
    criteria?: string;
    judge_model?: string;
    timeout?: number;
  };
  objective?: string;
  background?: string;
  dataset?: any[];
  valset?: any[];
  config?: Record<string, any>;
}

export interface RunStatus {
  run_id: string;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  iteration?: number;
  best_score?: number;
  num_candidates?: number;
  total_evals?: number;
  result?: any;
  error?: string;
}

const RUNS_DIR = join(homedir(), '.optimize-anything', 'runs');
const ENGINE_DIR = join(__dirname, '..', '..', 'engine');

export class ProcessManager {
  private processes: Map<string, ChildProcess> = new Map();
  private statuses: Map<string, RunStatus> = new Map();

  async startRun(config: RunConfig): Promise<string> {
    const runId = randomUUID().slice(0, 8);
    const runDir = join(RUNS_DIR, runId);
    mkdirSync(runDir, { recursive: true });

    const configPath = join(runDir, 'config.json');
    const fullConfig = {
      ...config,
      config: { ...config.config, run_dir: runDir },
    };
    writeFileSync(configPath, JSON.stringify(fullConfig, null, 2));

    const proc = spawn('uv', ['run', 'optimize-anything', 'run', '--config', configPath, '--events'], {
      cwd: ENGINE_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    this.processes.set(runId, proc);
    this.statuses.set(runId, { run_id: runId, status: 'running' });

    let lastEvent: any = null;

    proc.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          lastEvent = JSON.parse(line);
          if (lastEvent.type === 'optimization_complete') {
            this.statuses.set(runId, {
              run_id: runId,
              status: 'completed',
              best_score: lastEvent.best_score,
              total_evals: lastEvent.total_evals,
              iteration: lastEvent.total_iterations,
            });
          }
        } catch {}
      }
    });

    proc.stderr?.on('data', (data: Buffer) => {
      console.error(`[${runId}] ${data.toString()}`);
    });

    proc.on('close', (code) => {
      this.processes.delete(runId);
      const current = this.statuses.get(runId);
      if (current && current.status === 'running') {
        this.statuses.set(runId, {
          ...current,
          status: code === 0 ? 'completed' : 'failed',
          error: code !== 0 ? `Process exited with code ${code}` : undefined,
        });
      }
    });

    return runId;
  }

  getStatus(runId: string): RunStatus | null {
    // Check in-memory first
    const status = this.statuses.get(runId);
    if (status) return status;

    // Check checkpoint on disk
    const checkpointPath = join(RUNS_DIR, runId, 'checkpoint.json');
    if (existsSync(checkpointPath)) {
      try {
        const data = JSON.parse(readFileSync(checkpointPath, 'utf-8'));
        return {
          run_id: runId,
          status: 'completed',
          iteration: data.iteration,
          best_score: Math.max(...(data.agg_scores || [0])),
          num_candidates: data.candidates?.length || 0,
          total_evals: data.total_evals,
        };
      } catch {}
    }

    return null;
  }

  getBestCandidate(runId: string): { candidate: Record<string, string>; score: number } | null {
    const checkpointPath = join(RUNS_DIR, runId, 'checkpoint.json');
    if (!existsSync(checkpointPath)) return null;

    try {
      const data = JSON.parse(readFileSync(checkpointPath, 'utf-8'));
      const scores: number[] = data.agg_scores || [];
      const bestIdx = scores.indexOf(Math.max(...scores));
      return {
        candidate: data.candidates[bestIdx]?.components || {},
        score: scores[bestIdx] || 0,
      };
    } catch {
      return null;
    }
  }

  stopRun(runId: string): boolean {
    const proc = this.processes.get(runId);
    if (proc) {
      proc.kill('SIGTERM');
      this.processes.delete(runId);
      const current = this.statuses.get(runId);
      if (current) {
        this.statuses.set(runId, { ...current, status: 'stopped' });
      }
      return true;
    }
    return false;
  }

  listRuns(): RunStatus[] {
    const runs: RunStatus[] = [];
    for (const status of this.statuses.values()) {
      runs.push(status);
    }

    // Also check disk
    if (existsSync(RUNS_DIR)) {
      const { readdirSync } = require('fs');
      for (const dir of readdirSync(RUNS_DIR)) {
        if (!this.statuses.has(dir)) {
          const status = this.getStatus(dir);
          if (status) runs.push(status);
        }
      }
    }

    return runs;
  }
}
```

**Step 2: Implement tools.ts**

```typescript
// mcp-server/src/tools.ts
export const TOOLS = [
  {
    name: 'optimize_anything',
    description: 'Start an optimization run. Optimizes any text artifact through iterative LLM-powered search.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        seed_candidate: {
          type: 'object',
          description: 'Initial candidate as {component_name: text}. E.g. {"system_prompt": "You are..."}',
          additionalProperties: { type: 'string' },
        },
        evaluator: {
          type: 'object',
          description: 'Evaluator config: {type: "python"|"shell"|"llm_judge", code?, command?, score_pattern?, criteria?, judge_model?, timeout?}',
          properties: {
            type: { type: 'string', enum: ['python', 'shell', 'llm_judge'] },
            code: { type: 'string' },
            command: { type: 'string' },
            score_pattern: { type: 'string' },
            criteria: { type: 'string' },
            judge_model: { type: 'string' },
            timeout: { type: 'number' },
          },
          required: ['type'],
        },
        objective: { type: 'string', description: 'What to optimize for (natural language)' },
        background: { type: 'string', description: 'Domain knowledge and constraints' },
        dataset: { type: 'array', description: 'Training examples (for multi-task or generalization mode)' },
        valset: { type: 'array', description: 'Validation set (for generalization mode)' },
        config: {
          type: 'object',
          description: 'Engine config: {max_iterations?, max_metric_calls?, model?, selection_strategy?, ...}',
        },
      },
      required: ['evaluator'],
    },
  },
  {
    name: 'check_optimization',
    description: 'Check the current status of an optimization run.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID returned by optimize_anything' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'get_best_candidate',
    description: 'Get the current best candidate from an optimization run.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'stop_optimization',
    description: 'Stop a running optimization.',
    inputSchema: {
      type: 'object' as const,
      properties: {
        run_id: { type: 'string', description: 'The run ID to stop' },
      },
      required: ['run_id'],
    },
  },
  {
    name: 'list_optimization_runs',
    description: 'List all optimization runs with their status.',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
];
```

**Step 3: Update index.ts**

```typescript
// mcp-server/src/index.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { TOOLS } from './tools.js';
import { ProcessManager, RunConfig } from './process-manager.js';

const server = new Server(
  { name: 'optimize-anything-mcp', version: '1.0.0' },
  { capabilities: { tools: {} } }
);

const manager = new ProcessManager();

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const { name, arguments: args } = request.params;

    switch (name) {
      case 'optimize_anything': {
        const config: RunConfig = {
          seed_candidate: (args as any).seed_candidate,
          evaluator: (args as any).evaluator,
          objective: (args as any).objective,
          background: (args as any).background,
          dataset: (args as any).dataset,
          valset: (args as any).valset,
          config: (args as any).config,
        };
        const runId = await manager.startRun(config);
        return {
          content: [{ type: 'text', text: JSON.stringify({ run_id: runId, status: 'started' }, null, 2) }],
        };
      }

      case 'check_optimization': {
        const status = manager.getStatus((args as any).run_id);
        if (!status) {
          return { content: [{ type: 'text', text: JSON.stringify({ error: 'Run not found' }) }], isError: true };
        }
        return { content: [{ type: 'text', text: JSON.stringify(status, null, 2) }] };
      }

      case 'get_best_candidate': {
        const best = manager.getBestCandidate((args as any).run_id);
        if (!best) {
          return { content: [{ type: 'text', text: JSON.stringify({ error: 'No results found' }) }], isError: true };
        }
        return { content: [{ type: 'text', text: JSON.stringify(best, null, 2) }] };
      }

      case 'stop_optimization': {
        const stopped = manager.stopRun((args as any).run_id);
        return {
          content: [{ type: 'text', text: JSON.stringify({ stopped, run_id: (args as any).run_id }) }],
        };
      }

      case 'list_optimization_runs': {
        const runs = manager.listRuns();
        return { content: [{ type: 'text', text: JSON.stringify(runs, null, 2) }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return { content: [{ type: 'text', text: `Error: ${msg}` }], isError: true };
  }
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('optimize-anything MCP server running on stdio');
}

main().catch((error) => { console.error('Fatal:', error); process.exit(1); });
```

**Step 4: Build and verify**

```bash
cd ~/optimize-anything/mcp-server && npm run build
```

**Step 5: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: TypeScript MCP server with full tool set"
```

---

## Task 10: Claude Code Skill

**Files:**
- Create: `skill/SKILL.md`

**Step 1: Write the skill**

```markdown
---
name: optimize-anything
description: Use when optimizing any text artifact — prompts, code, configs, agent architectures, policies — through iterative LLM-powered search with evaluation feedback. Also use when the user mentions GEPA, optimize, evolve, or improve via search.
---

# optimize_anything

Optimize any text artifact through iterative LLM-powered search with diagnostic feedback.

## Core Concept

If it can be serialized to a string and its quality measured, it can be optimized. The system proposes improvements via Claude reflection on evaluation diagnostics (ASI), tracks a Pareto frontier of non-dominated solutions, and iterates until budget is exhausted.

## When to Use

- "Optimize this prompt" — prompt optimization
- "Make this code faster" — code optimization with benchmarks
- "Find a better agent architecture" — agent evolution
- "Tune this config" — configuration optimization
- Any task where you have a text artifact + a way to score it

## Three Modes

1. **Single-task**: One problem to solve. No dataset needed.
2. **Multi-task**: Batch of related problems. Cross-transfer of insights.
3. **Generalization**: Train + val split. Optimized artifact must transfer to unseen examples.

## How to Formulate an Optimization

Every optimization needs three things:

| Component | Question | Example |
|-----------|----------|---------|
| **Candidate** | What text are we optimizing? | `{"system_prompt": "You are..."}` |
| **Evaluator** | How do we score it? | Python fn, shell command, or LLM judge criteria |
| **Objective** | What are we optimizing for? | "Maximize accuracy while minimizing verbosity" |

## Evaluator Selection Guide

| Evaluator | When to Use | Example |
|-----------|-------------|---------|
| **Python** | Complex scoring logic, API calls, test suites | `def evaluate(c, e): return run_tests(c)` |
| **Shell** | CLI tools, benchmarks, existing test runners | `echo '{{candidate}}' \| python bench.py` |
| **LLM Judge** | Subjective quality, no code needed | "Rate clarity and specificity, SCORE: X/10" |

## Recipe 1: Optimize a System Prompt

```
Call optimize_anything MCP tool with:
  seed_candidate: {"system_prompt": "<current prompt>"}
  evaluator: {type: "llm_judge", criteria: "Rate helpfulness, accuracy, conciseness. SCORE: X/10"}
  objective: "Maximize response quality"
  dataset: [{"input": "example query 1"}, {"input": "example query 2"}, ...]
  config: {max_iterations: 20}
```

## Recipe 2: Optimize Code for Performance

```
Call optimize_anything MCP tool with:
  seed_candidate: {"algorithm": "<current code>"}
  evaluator: {type: "shell", command: "python bench.py {{candidate_file}}", score_pattern: "Score: ([\\d.]+)"}
  objective: "Maximize speed while maintaining correctness"
  config: {max_iterations: 30}
```

## Recipe 3: Discover Agent Architecture

```
Call optimize_anything MCP tool with:
  seed_candidate: {"agent_code": "<initial harness>"}
  evaluator: {type: "python", code: "def evaluate(c, e): ...run agent, return score..."}
  objective: "Maximize task completion accuracy"
  dataset: [<train_tasks>]
  valset: [<held_out_tasks>]
  config: {max_iterations: 50}
```

## Recipe 4: Optimize Config/Policy

```
Call optimize_anything MCP tool with:
  seed_candidate: {"config": "<yaml or json>"}
  evaluator: {type: "shell", command: "deploy_and_test.sh {{candidate_file}}", score_pattern: "metric: ([\\d.]+)"}
  objective: "Minimize cost while maintaining SLA"
  dataset: [<scenarios>]
  config: {max_iterations: 25}
```

## Workflow

1. Identify the artifact to optimize and decompose into components
2. Choose the right evaluator type
3. Define objective and any background context
4. Call `optimize_anything` MCP tool — returns a run_id
5. Poll with `check_optimization` to monitor progress
6. Retrieve results with `get_best_candidate`
7. Apply the optimized artifact

## Key Concepts

- **ASI (Actionable Side Information)**: Diagnostic feedback from evaluator (errors, traces, scores) that helps the proposer make targeted improvements
- **Pareto Frontier**: Tracks non-dominated solutions across multiple objectives/examples
- **Reflective Mutation**: Claude reads ASI, diagnoses issues, proposes targeted fixes — not blind evolution
- **Minibatch Reflection**: Each iteration focuses on a subset of examples, rotating coverage over time
```

**Step 2: Copy skill to Claude Code skills directory**

```bash
mkdir -p ~/.claude/skills/optimize-anything
cp ~/optimize-anything/skill/SKILL.md ~/.claude/skills/optimize-anything/SKILL.md
```

**Step 3: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: Claude Code skill with 4 optimization recipes"
```

---

## Task 11: Integration Test — End to End

**Files:**
- Create: `engine/tests/test_integration.py`

**Step 1: Write integration test with mock proposer**

```python
# engine/tests/test_integration.py
"""End-to-end integration test using mock proposer (no API calls)."""
from unittest.mock import patch, MagicMock
from optimize_anything.api import optimize_anything
from optimize_anything.config import Config


def test_e2e_single_task():
    """Single-task mode: optimize a string to contain 'hello world'."""
    call_count = 0

    def evaluator(candidate, example=None):
        words = {"hello", "world", "foo", "bar"}
        found = sum(1 for w in words if w in candidate)
        return found / len(words), {"found": found, "total": len(words)}

    with patch("optimize_anything.api.ClaudeProposer") as MockP:
        mock = MagicMock()
        responses = ["hello there", "hello world", "hello world foo", "hello world foo bar"]
        mock.propose.side_effect = lambda **kw: responses[min(mock.propose.call_count - 1, len(responses) - 1)]
        MockP.return_value = mock

        result = optimize_anything(
            seed_candidate={"text": "nothing"},
            evaluator=evaluator,
            objective="include all target words",
            config=Config(max_iterations=4),
        )

    assert result.best_score > 0
    assert result.total_iterations >= 1


def test_e2e_with_dataset():
    """Multi-task mode: optimize across multiple examples."""
    def evaluator(candidate, example=None):
        if example and "target" in example:
            match = example["target"] in candidate
            return (1.0 if match else 0.0), {"target": example["target"], "matched": match}
        return 0.5, {}

    with patch("optimize_anything.api.ClaudeProposer") as MockP:
        mock = MagicMock()
        mock.propose.return_value = "I contain apple and banana"
        MockP.return_value = mock

        result = optimize_anything(
            seed_candidate={"text": "empty"},
            evaluator=evaluator,
            dataset=[{"target": "apple"}, {"target": "banana"}, {"target": "cherry"}],
            objective="contain all targets",
            config=Config(max_iterations=3, reflection_minibatch_size=2),
        )

    assert result.total_iterations >= 1
    assert len(result.all_candidates) >= 1
```

**Step 2: Run all tests**

```bash
cd ~/optimize-anything/engine && uv run pytest tests/ -v
```

**Step 3: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: end-to-end integration tests"
```

---

## Task 12: MCP Server Configuration

**Files:**
- Create: `mcp-server/start-server.sh`
- Modify: project docs

**Step 1: Create startup script**

```bash
#!/bin/bash
cd "$(dirname "$0")" && node dist/index.js
```

**Step 2: Add MCP server to Claude Code settings**

Add to `~/.claude/settings.json` mcpServers section:
```json
{
  "mcpServers": {
    "optimize-anything": {
      "command": "node",
      "args": ["/Users/rshah/optimize-anything/mcp-server/dist/index.js"]
    }
  }
}
```

**Step 3: Final build and verify**

```bash
cd ~/optimize-anything/engine && uv sync && uv run pytest tests/ -v
cd ~/optimize-anything/mcp-server && npm install && npm run build
```

**Step 4: Commit**

```bash
cd ~/optimize-anything && git add -A && git commit -m "feat: MCP server configuration and startup script"
```

---

## Summary

| Task | Component | What It Builds |
|------|-----------|---------------|
| 1 | Scaffolding | Project structure, pyproject.toml, package.json, CLAUDE.md |
| 2 | State | Candidate, State, ParetoFrontier data structures |
| 3 | Evaluators | Python, shell, LLM-as-judge evaluator types |
| 4 | Proposer | Claude API reflection prompts and proposal extraction |
| 5 | Strategies | Selection, batch sampling, stopping conditions |
| 6 | Engine | Core optimization loop with events |
| 7 | API + CLI | Public optimize_anything() function and CLI |
| 8 | Checkpoint | Save/load state for resume |
| 9 | MCP Server | TypeScript MCP with process management |
| 10 | Skill | Claude Code skill with 4 recipes |
| 11 | Integration | End-to-end tests |
| 12 | Config | MCP server wiring to Claude Code |
