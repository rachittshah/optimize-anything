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
