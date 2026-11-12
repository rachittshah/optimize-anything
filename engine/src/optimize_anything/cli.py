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
    run_parser.add_argument("--preset", choices=["default", "cheap", "quality"],
                            help="Model preset (overrides config.preset)")
    run_parser.add_argument("--backend", choices=["auto", "anthropic", "openai", "claude-cli", "codex-cli", "mock"],
                            help="LLM backend (overrides config.backend)")
    run_parser.add_argument("--proposer-model", help="Proposer model (overrides config)")
    run_parser.add_argument("--judge-model", help="LLM-judge model (overrides config)")
    run_parser.add_argument("--proposer-effort", choices=["low", "medium", "high", "xhigh", "max"])
    run_parser.add_argument("--judge-effort", choices=["low", "medium", "high", "xhigh", "max"])
    run_parser.add_argument("--max-cost-usd", type=float, help="Stop once LLM spend reaches this")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Only score the seed on <=3 examples (<=45s) and report scores/errors")

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
    testset = config_data.get("testset")  # held out: only scored at the end

    eval_config = config_data.get("evaluator", {})
    evaluator = EvaluatorConfig(
        type=EvaluatorType(eval_config["type"]),
        code=eval_config.get("code"),
        command=eval_config.get("command"),
        score_pattern=eval_config.get("score_pattern"),
        criteria=eval_config.get("criteria"),
        rubric=eval_config.get("rubric"),
        judge_model=eval_config.get("judge_model"),
        timeout=eval_config.get("timeout", 30),
    )

    engine_config = config_data.get("config", {})
    overrides = {
        "preset": args.preset,
        "backend": args.backend,
        "proposer_model": args.proposer_model,
        "judge_model": args.judge_model,
        "proposer_effort": args.proposer_effort,
        "judge_effort": args.judge_effort,
        "max_cost_usd": args.max_cost_usd,
    }
    engine_config = {**engine_config, **{k: v for k, v in overrides.items() if v is not None}}
    config = Config(**{k: v for k, v in engine_config.items() if k in Config.__dataclass_fields__})

    if getattr(args, "dry_run", False):
        print(json.dumps(_dry_run(seed, evaluator, dataset, valset, config), indent=2))
        return

    def on_event(event):
        if args.events:
            print(json.dumps(event), flush=True)

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        dataset=dataset,
        valset=valset,
        testset=testset,
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
        "seed_score": result.seed_score,
        "seed_test": result.seed_test,
        "best_test": result.best_test,
        "val_test_gap": result.val_test_gap,
        "total_cost_usd": result.total_cost_usd,
        "cost_per_point": result.cost_per_point,
        "cost": result.cost,
        "run_dir": result.run_dir,
        "failed_evals": result.failed_evals,
        "cache_hits": result.cache_hits,
    }
    print(json.dumps(output, indent=2))


DRY_RUN_MAX_EXAMPLES = 3
DRY_RUN_DEADLINE_S = 45.0


def _dry_run(seed, evaluator_cfg: EvaluatorConfig, dataset, valset, config: Config) -> dict:
    """Score the seed on <=3 examples within 45s and report scores + evaluator errors.
    Nothing is optimized; the proposer is never called (an llm_judge evaluator does call its judge)."""
    import dataclasses
    import time

    from optimize_anything.api import _build_evaluator
    from optimize_anything.core.cost import CostTracker, TrackedLLM
    from optimize_anything.evaluators.base import EvaluationError
    from optimize_anything.llm import make_llm, resolve_backend, resolve_effort, resolve_model

    start = time.monotonic()
    out: dict = {"dry_run": True, "ok": False, "examples_evaluated": 0, "scores": [],
                 "errors": [], "asi_samples": [], "warnings": []}
    if not seed:
        seed = {"artifact": ""}
        out["warnings"].append("no seed_candidate: scored an empty artifact")
    examples = list((valset or dataset or [None])[:DRY_RUN_MAX_EXAMPLES])
    out["example_source"] = "valset" if valset else ("dataset" if dataset else "none (single-task)")

    tracker = CostTracker()
    judge_llm, judge_model, judge_effort = None, "", None
    try:
        if evaluator_cfg.type == EvaluatorType.LLM_JUDGE:
            backend = resolve_backend(config.backend)
            out["backend"] = backend
            make_llm(backend, base_url=config.openai_base_url, api_key_env=config.openai_api_key_env)  # fail fast
            judge_model = resolve_model("judge", backend, evaluator_cfg.judge_model or config.judge_model,
                                        config.preset)
            judge_effort = resolve_effort("judge", config.judge_effort, config.preset)
    except Exception as e:  # no backend / bad model: report instead of crashing
        out["errors"].append({"stage": "setup", "error": f"{type(e).__name__}: {e}"})
        out["elapsed_s"] = round(time.monotonic() - start, 2)
        return out

    for i, example in enumerate(examples):
        remaining = DRY_RUN_DEADLINE_S - (time.monotonic() - start)
        if remaining < 1:
            out["warnings"].append(f"stopped at example {i}: the 45s dry-run cap was reached "
                                   "(evaluator too slow for an interactive check)")
            break
        cfg = dataclasses.replace(evaluator_cfg, timeout=max(1, min(evaluator_cfg.timeout, int(remaining))))
        if evaluator_cfg.type == EvaluatorType.LLM_JUDGE:  # judge call bounded by the time left
            judge_llm = TrackedLLM(make_llm(out["backend"], base_url=config.openai_base_url,
                                            api_key_env=config.openai_api_key_env,
                                            timeout=max(1.0, remaining - 1)), tracker, "judge")
        ev = _build_evaluator(cfg, judge_llm, judge_model, judge_effort)
        if len(seed) == 1:
            payload = next(iter(seed.values()))
        elif getattr(ev, "accepts_dict", False):
            payload = dict(seed)
        else:
            payload = "\n".join(seed.values())
        try:
            score, asi = ev.evaluate(payload, example)
        except EvaluationError as e:
            out["errors"].append({"example": i, "error": f"evaluator infrastructure failure: {e}"})
            continue
        except Exception as e:
            out["errors"].append({"example": i, "error": f"{type(e).__name__}: {e}"})
            continue
        out["examples_evaluated"] += 1
        out["scores"].append(score)
        asi = asi if isinstance(asi, dict) else {"asi": asi}
        if asi.get("error"):  # Python/shell evaluators report failures as score 0 + asi.error
            out["errors"].append({"example": i, "error": str(asi["error"])[-500:]})
            continue
        sample = json.dumps(asi, default=str)
        out["asi_samples"].append(sample if len(sample) <= 400 else sample[:400] + "…")

    scores = out["scores"]
    if scores:
        out["seed_score_mean"] = sum(scores) / len(scores)
        if any(s < 0 or s > 1 for s in scores):
            out["warnings"].append("scores outside [0, 1]: normalize so 1.0 = perfect")
        if all(s >= 1.0 for s in scores):
            out["warnings"].append("seed already scores 1.0 on every checked example: no learning signal; "
                                   "add harder examples or a stricter evaluator")
        elif len(scores) > 1 and len(set(scores)) == 1 and not out["errors"]:
            out["warnings"].append("all checked examples got the same score: check the evaluator discriminates")
        if out["asi_samples"] and all(s == "{}" for s in out["asi_samples"]):
            out["warnings"].append("evaluator returns no diagnostics (asi is empty): return errors, "
                                   "expected vs got, etc. so the proposer can target failures")
    out["ok"] = bool(scores) and not out["errors"]
    out["cost_usd"] = round(tracker.total_usd, 6)
    out["elapsed_s"] = round(time.monotonic() - start, 2)
    return out


def _status(args):
    checkpoint = Path(args.run_dir) / "checkpoint.json"
    if not checkpoint.exists():
        print(json.dumps({"status": "not_found"}))
        return

    from optimize_anything.core.state import State
    state = State.load(checkpoint)
    raw = json.loads(checkpoint.read_text())
    best_idx = raw.get("best_idx")  # reported best (differs from argmax under promotion=paired_se)
    best_score = (state.agg_scores[best_idx] if isinstance(best_idx, int)
                  and 0 <= best_idx < len(state.agg_scores) else state.best_score())
    print(json.dumps({
        "status": "running" if state.iteration >= 0 else "initialized",
        "iteration": state.iteration,
        "total_evals": state.total_evals,
        "best_score": best_score,
        "num_candidates": len(state.candidates),
        "cost": raw.get("cost"),
    }, indent=2))


if __name__ == "__main__":
    main()
