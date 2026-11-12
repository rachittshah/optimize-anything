#!/usr/bin/env python3
"""Child process: run one agent-emitted optimize_anything call through the engine.

Usage: python run_formulation.py SPEC.json   (cwd = temp dir holding the item's context_files)

SPEC = {"call": <tool arguments>, "max_metric_calls": int}. Builds the evaluator exactly as the
CLI/MCP path does (cli._run), wraps it to record every (score, asi), runs the engine for one
iteration with the mock proposer, and prints one JSON line:
  {"ok": true, "n_seed": k, "records": [{"score", "error", "asi"}...], "seed_score": x}
or {"ok": false, "stage": "...", "error": "..."}.
Must run with the stub env (OA_BACKEND=mock, stub API keys/base URLs, sitecustomize).
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path


def _short(v, n: int = 400) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s if len(s) <= n else s[:n] + "…"


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    call = spec["call"]
    out: dict = {"ok": False}
    try:
        from optimize_anything.api import _build_evaluator
        from optimize_anything.config import Config, EvaluatorConfig, EvaluatorType
        from optimize_anything.core.engine import Engine
        from optimize_anything.llm import MockLLM
        from optimize_anything.proposer.reflective import ReflectiveProposer
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "stage": "import", "error": repr(e)}))
        return 0

    stage = "evaluator"
    try:
        ev = call["evaluator"]
        cfg = EvaluatorConfig(  # same construction as cli._run (the MCP path)
            type=EvaluatorType(ev["type"]),
            code=ev.get("code"),
            command=ev.get("command"),
            score_pattern=ev.get("score_pattern"),
            criteria=ev.get("criteria"),
            rubric=ev.get("rubric"),
            judge_model=ev.get("judge_model"),
            timeout=ev.get("timeout", 30),
        )
        inner = _build_evaluator(cfg, MockLLM(), "mock-judge", "low")
        pattern = cfg.score_pattern or r"([\d.]+)"
        records: list[dict] = []

        class Recorder:
            accepts_dict = getattr(inner, "accepts_dict", False)

            def cache_identity(self) -> str:
                ident = getattr(inner, "cache_identity", None)
                return ident() if callable(ident) else "recorder"

            def evaluate(self, candidate, example=None):
                score, asi = inner.evaluate(candidate, example)
                err = asi.get("error") if isinstance(asi, dict) else None
                if cfg.type == EvaluatorType.SHELL and not err:
                    stdout = asi.get("stdout", "")
                    if asi.get("returncode", 0) != 0:
                        err = f"shell exited {asi.get('returncode')}: {_short(asi.get('stderr', ''), 300)}"
                    elif not re.search(pattern, stdout):
                        err = f"score_pattern {pattern!r} did not match stdout: {_short(stdout, 200)}"
                ok_score = isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0
                if not ok_score and not err:
                    err = f"score {score!r} is not a float in [0,1]"
                records.append({"score": score if isinstance(score, (int, float)) else None,
                                "error": _short(err, 600) if err else None,
                                "asi": _short(asi, 300)})
                return score, asi

        stage = "config"
        given = {k: v for k, v in (call.get("config") or {}).items() if k in Config.__dataclass_fields__}
        cap = int(spec.get("max_metric_calls", 20))
        mmc = given.get("max_metric_calls")
        given.update({
            "backend": "mock", "max_iterations": 1, "run_dir": str(Path.cwd() / "_oa_run"),
            "run_id": None, "max_metric_calls": min(int(mmc), cap) if mmc else cap,
            "timeout_seconds": None, "max_cost_usd": None,
        })
        config = Config(**given)
        seed = call.get("seed_candidate") or {"artifact": ""}
        dataset, valset = call.get("dataset"), call.get("valset")
        n_seed = len(valset or dataset or [None])

        stage = "engine"
        engine = Engine(
            seed_candidate=seed, evaluator=Recorder(),
            proposer=ReflectiveProposer(MockLLM(), model="mock", effort=None),
            objective=call.get("objective") or "", background=call.get("background"),
            dataset=dataset, valset=valset, config=config,
        )
        result = engine.run()
        out = {"ok": True, "n_seed": n_seed, "records": records,
               "seed_score": result.seed_score}
    except Exception as e:  # noqa: BLE001 - report any failure as invalid
        out = {"ok": False, "stage": stage, "error": f"{type(e).__name__}: {e}",
               "trace": traceback.format_exc()[-800:]}
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
