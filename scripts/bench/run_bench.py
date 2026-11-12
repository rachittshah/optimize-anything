"""Run the optimize-anything engine benchmark.

    cd engine && uv run python ../scripts/bench/run_bench.py --tasks all --seeds 0 1 2 \
        --max-iterations 10 --proposer claude --variant-name sota --out ../scripts/bench/results/sota.jsonl

--engine-src <path> puts another checkout's engine/src first on sys.path (e.g. a `git worktree` of main).
Works with the Engine API on `main`; extra Config fields are passed only if the Config has them.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import random
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--max-metric-calls", type=int, default=None)
    p.add_argument("--proposer", choices=["claude", "mock"], default="claude")
    p.add_argument("--proposer-model", default="claude-haiku-5-5")
    p.add_argument("--variant-name", default="default")
    p.add_argument("--config-json", default="{}", help="extra Config overrides (JSON object or @file)")
    p.add_argument("--background", choices=["minimal", "full"], default="minimal",
                   help="minimal: proposer infers hidden rules from ASI; full: give it the whole spec (easy)")
    p.add_argument("--engine-src", default=None, help="engine src dir to import optimize_anything from")
    p.add_argument("--out", default=str(BENCH_DIR / "results" / "results.jsonl"))
    return p.parse_args(argv)


def build_config(Config, overrides: dict, run_dir: str) -> tuple[object, dict, list[str]]:
    fields = Config.__dataclass_fields__
    wanted = {"checkpoint_interval": 0, "run_dir": run_dir, **overrides}
    kwargs, dropped = {}, []
    defaults = Config()
    for k, v in wanted.items():
        if k not in fields:
            dropped.append(k)
            continue
        default = getattr(defaults, k)
        if isinstance(default, enum.Enum) and not isinstance(v, enum.Enum):
            v = type(default)(v)
        kwargs[k] = v
    return Config(**kwargs), kwargs, dropped


class CountingEvaluator:
    """Counts real evaluator calls and records full-valset sweeps as (evals_so_far, val_score)."""

    def __init__(self, inner, val_ids: list[str]):
        self.inner = inner
        self.val_ids = set(val_ids)
        self.calls = 0
        self._val_scores: dict[str, dict[str, float]] = {}
        self._done: set[str] = set()
        self.val_sweeps: list[tuple[int, float]] = []
        self.accepts_dict = getattr(inner, "accepts_dict", False)

    def evaluate(self, candidate, example=None):
        score, asi = self.inner.evaluate(candidate, example)
        self.calls += 1
        if example is not None and example.get("id") in self.val_ids:
            key = json.dumps(candidate, sort_keys=True) if isinstance(candidate, dict) else str(candidate)
            per = self._val_scores.setdefault(key, {})
            per[example["id"]] = score
            if key not in self._done and len(per) == len(self.val_ids):
                self._done.add(key)
                self.val_sweeps.append((self.calls, sum(per.values()) / len(per)))
        return score, asi


def best_so_far(points: list[tuple[int, float]]) -> list[list[float]]:
    out, best = [], float("-inf")
    for x, y in points:
        if y > best:
            best = y
            out.append([x, round(y, 6)])
    return out


ENGINE_COUNTERS = (
    "cache_hits", "failed_evals", "skipped_perfect", "stop_reason", "prescreen_skips",
    "halving_rejects", "halving_evals_saved", "promotions_blocked", "operator_counts",
    "operator_unsupported", "edit_size_rejects",
)


def run_one(task_name: str, seed: int, args, overrides: dict, Engine, Config, bench_tasks, make_proposer, oa_file):
    task = bench_tasks.get_task(task_name)
    inner = task.evaluator()
    train, val, test = task.splits["train"], task.splits["val"], task.splits["test"]
    counting = CountingEvaluator(inner, [ex["id"] for ex in val])
    proposer = make_proposer(args.proposer, args.proposer_model, task.reference_candidate)

    events: list[dict] = []
    iter_starts = 0

    def on_event(e):
        nonlocal iter_starts
        if e.get("type") == "iteration_start":
            iter_starts += 1
        events.append({"evals": counting.calls, **{k: v for k, v in e.items() if k != "asi"}})

    random.seed(seed)
    ov = {"max_iterations": args.max_iterations, **overrides}
    if args.max_metric_calls is not None:
        ov["max_metric_calls"] = args.max_metric_calls
    ov.setdefault("seed", seed)
    with tempfile.TemporaryDirectory(prefix="oa-bench-run-") as run_dir:
        config, used, dropped = build_config(Config, ov, run_dir)
        engine = Engine(
            seed_candidate=dict(task.seed_candidate), evaluator=counting, proposer=proposer,
            objective=task.objective,
            background=task.spec if args.background == "full" else task.background,
            dataset=train, valset=val, config=config, on_event=on_event,
        )
        t0 = time.monotonic()
        result = engine.run()
        wall = time.monotonic() - t0

    def split_scores(cand):  # harness-side scoring, never counted in total_evals
        return {k: bench_tasks.score_split(inner, cand, v) for k, v in (("train", train), ("val", val), ("test", test))}

    seed_s = split_scores(task.seed_candidate)
    best_s = split_scores(result.best_candidate)
    traj = best_so_far(counting.val_sweeps)
    if not traj:  # engine didn't do full val sweeps we could see; fall back to all_scores
        traj = best_so_far([(i, s) for i, s in enumerate(result.all_scores)])
    used_json = {k: (v.value if isinstance(v, enum.Enum) else v) for k, v in used.items() if k != "run_dir"}
    return {
        "variant": args.variant_name, "task": task_name, "seed": seed,
        "proposer": args.proposer, "background": args.background, "proposer_model": args.proposer_model if args.proposer == "claude" else None,
        "seed_train": seed_s["train"], "seed_val": seed_s["val"], "seed_test": seed_s["test"],
        "best_train": best_s["train"], "best_val": best_s["val"], "best_test": best_s["test"],
        "train_val_gap": best_s["train"] - best_s["val"],
        "engine_best_score": result.best_score,
        "total_evals": result.total_evals, "evaluator_calls": counting.calls,
        "iterations": result.total_iterations, "iterations_observed": iter_starts,
        "num_candidates": len(result.all_candidates),
        "proposer_calls": proposer.calls, "proposer_failures": proposer.failures,
        "proposer_cost_usd": round(proposer.total_cost_usd, 6),
        "proposer_latency_s": round(proposer.total_latency_s, 2),
        "proposer_last_error": getattr(proposer, "last_error", None),
        "wall_s": round(wall, 2),
        "max_iterations": args.max_iterations, "max_metric_calls": args.max_metric_calls,
        "val_trajectory": traj,  # best-so-far [evaluator_calls, val_score]
        "new_best_events": [[e["evals"], e.get("score")] for e in events if e.get("type") == "new_best_found"],
        "config_used": used_json, "config_dropped": dropped,
        "engine_module": oa_file,
        "best_candidate": result.best_candidate,
        # Engine counters (None when the engine under test predates them, e.g. --engine-src main)
        **{k: getattr(result, k, None) for k in ENGINE_COUNTERS},
    }


def main(argv=None):
    args = parse_args(argv)
    if args.engine_src:
        sys.path.insert(0, str(Path(args.engine_src).expanduser().resolve()))
        for m in [m for m in sys.modules if m == "optimize_anything" or m.startswith("optimize_anything.")]:
            del sys.modules[m]
    sys.path.insert(0, str(BENCH_DIR))

    import optimize_anything
    from optimize_anything.config import Config
    from optimize_anything.core.engine import Engine
    import bench_tasks
    from bench_proposer import make_proposer

    raw = args.config_json
    overrides = json.loads(Path(raw[1:]).read_text() if raw.startswith("@") else raw)
    tasks = list(bench_tasks.TASK_NAMES) if args.tasks == ["all"] else args.tasks
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"engine: {optimize_anything.__file__}", file=sys.stderr)
    records = []
    for task_name in tasks:
        for seed in args.seeds:
            rec = run_one(task_name, seed, args, overrides, Engine, Config, bench_tasks, make_proposer,
                          optimize_anything.__file__)
            records.append(rec)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['variant']}] {task_name} seed={seed}: test {rec['seed_test']:.3f}->{rec['best_test']:.3f} "
                  f"val {rec['seed_val']:.3f}->{rec['best_val']:.3f} evals={rec['total_evals']} "
                  f"calls={rec['proposer_calls']} fail={rec['proposer_failures']} "
                  f"${rec['proposer_cost_usd']:.4f} {rec['wall_s']}s", file=sys.stderr)
    return records


if __name__ == "__main__":
    main()
