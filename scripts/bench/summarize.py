"""Aggregate run_bench JSONL files into a markdown table per (variant, task).

    python scripts/bench/summarize.py scripts/bench/results/*.jsonl

AUC = mean best-so-far val score over evaluator calls 0..X, where X is the task's shared budget
(max_metric_calls if all runs set the same one, else the largest evaluator_calls seen for that task).
Before the first val sweep the curve uses seed_val. Higher AUC = better sample efficiency.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def auc(rec: dict, x_max: float) -> float:
    if x_max <= 0:
        return rec["best_val"]
    pts = [(0.0, rec["seed_val"])] + [(float(x), float(y)) for x, y in rec["val_trajectory"]]
    area, best = 0.0, pts[0][1]
    for (x0, _), (x1, y1) in zip(pts, pts[1:] + [(x_max, None)]):
        x0c, x1c = min(x0, x_max), min(x1, x_max)
        area += best * max(0.0, x1c - x0c)
        if y1 is not None:
            best = max(best, y1)
    return area / x_max


def ms(vals: list[float], fmt: str = "{:.3f}") -> str:
    if not vals:
        return "-"
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{fmt.format(statistics.mean(vals))}±{fmt.format(sd)}"


def summarize(records: list[dict]) -> str:
    budgets = defaultdict(set)
    calls = defaultdict(list)
    for r in records:
        budgets[r["task"]].add(r.get("max_metric_calls"))
        calls[r["task"]].append(r.get("evaluator_calls", r["total_evals"]))
    x_max = {}
    for t in budgets:
        b = budgets[t]
        x_max[t] = next(iter(b)) if len(b) == 1 and None not in b else max(calls[t])

    groups = defaultdict(list)
    for r in records:
        groups[(r["variant"], r["task"])].append(r)

    lines = [
        "| variant | task | n | seed_test | best_test | best_val | train-val gap | evals | proposer calls | $ | AUC(val vs evals) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (variant, task), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append("| " + " | ".join([
            variant, task, str(len(rs)),
            ms([r["seed_test"] for r in rs]),
            ms([r["best_test"] for r in rs]),
            ms([r["best_val"] for r in rs]),
            ms([r["train_val_gap"] for r in rs]),
            ms([r["total_evals"] for r in rs], "{:.0f}"),
            ms([r["proposer_calls"] for r in rs], "{:.1f}"),
            ms([r["proposer_cost_usd"] for r in rs], "{:.4f}"),
            ms([auc(r, x_max[task]) for r in rs]),
        ]) + " |")
    lines.append("")
    lines.append("AUC budget per task (evaluator calls): " + ", ".join(f"{t}={x_max[t]}" for t in sorted(x_max)))
    return "\n".join(lines)


def compare(records: list[dict], baseline: str, metric: str = "best_test",
            n_boot: int = 4000, seed: int = 0) -> str:
    """Each variant vs `baseline`: mean over tasks of the per-task mean difference in `metric`,
    with a 95% stratified bootstrap CI (runs resampled within each task), plus the cost ratio."""
    import random
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    cost = defaultdict(list)
    for r in records:
        by[r["variant"]][r["task"]].append(float(r[metric]))
        cost[r["variant"]].append(float(r.get("proposer_cost_usd") or 0.0))
    if baseline not in by:
        return f"(baseline {baseline!r} not found)"
    base = by[baseline]
    lines = [f"Compare vs `{baseline}` on {metric} (mean over tasks; 95% bootstrap CI)", "",
             "| variant | Δ mean | 95% CI | P(Δ>0) | cost ratio |", "|---|---|---|---|---|"]
    for v in sorted(by):
        if v == baseline:
            continue
        tasks = [t for t in by[v] if t in base]
        if not tasks:
            continue
        point = statistics.mean(statistics.mean(by[v][t]) - statistics.mean(base[t]) for t in tasks)
        boots = []
        for _ in range(n_boot):
            d = []
            for t in tasks:
                a = [rng.choice(by[v][t]) for _ in by[v][t]]
                b = [rng.choice(base[t]) for _ in base[t]]
                d.append(statistics.mean(a) - statistics.mean(b))
            boots.append(statistics.mean(d))
        boots.sort()
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
        p_pos = sum(x > 0 for x in boots) / n_boot
        cr = statistics.mean(cost[v]) / max(1e-12, statistics.mean(cost[baseline]))
        lines.append(f"| {v} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p_pos:.2f} | {cr:.2f}x |")
    return "\n".join(lines)


def main(argv=None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    baseline = None
    if "--baseline" in args:
        i = args.index("--baseline")
        baseline = args[i + 1]
        del args[i:i + 2]
    paths = args
    if not paths:
        sys.exit("usage: summarize.py [--baseline VARIANT] results.jsonl [more.jsonl ...]")
    records = []
    for p in paths:
        with open(p) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    print(summarize(records))
    if baseline:
        print()
        print(compare(records, baseline))


if __name__ == "__main__":
    main()
