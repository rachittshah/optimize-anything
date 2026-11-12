#!/usr/bin/env python3
"""Score and hill-climb skill/SKILL.md with the optimize-anything engine.

  cd engine && uv run python ../scripts/optimize_skill.py score --skill ../skill/SKILL.md --split both
  cd engine && uv run python ../scripts/optimize_skill.py optimize --max-cost 50 --out ../scripts/skill_eval/results/<name>
  cd engine && uv run python ../scripts/optimize_skill.py check-golds      # gold calls pass validity (no LLM)

Official score = 0.3 trigger_F1 + 0.3 formulation_validity + 0.3 formulation_quality + 0.1 brevity,
reported for train and val (held-out 30%) separately. See scripts/skill_eval/ for the pieces.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_anything.config import Config  # noqa: E402
from optimize_anything.core.engine import Engine  # noqa: E402
from optimize_anything.llm import make_llm, resolve_backend, resolve_effort, resolve_model  # noqa: E402
from optimize_anything.proposer.reflective import ReflectiveProposer  # noqa: E402

from skill_eval.common import CACHE_DIR, REPO, WEIGHTS, leakage_report, load_data  # noqa: E402
from skill_eval.formulation import check_validity  # noqa: E402
from skill_eval.llm_cache import LockedCostTracker  # noqa: E402
from skill_eval.scorer import EngineEvaluator, SkillScorer  # noqa: E402
from skill_eval.stub_server import StubServer  # noqa: E402

OBJECTIVE = (
    "Improve SKILL.md, the agent skill that teaches coding agents when and how to call the "
    "optimize_anything MCP tool. Maximize: 0.3 trigger F1 (does a skill router, seeing only the "
    "frontmatter name + description among competing skills, load it exactly when appropriate) + "
    "0.3 formulation validity + 0.3 formulation quality (does an agent following the skill emit a "
    "correct, runnable optimize_anything call, or ask to clarify when it must) + 0.1 brevity."
)

BACKGROUND = """How candidates are scored (each example's score = 0.9 * component + 0.1 * brevity):
- Trigger examples: a router model sees ONLY the frontmatter `name` + `description`, mixed with ~14 \
other skills (prompt-writing, llm-eval-builder, performance-profiling, web-performance, \
hyperparameter-tuning, code-review, ...) and answers which skill to load, or none. Trigger exactly \
when a request implies an artifact made of text + an evaluator (metric, tests, labels, benchmark, \
rubric) + repeated propose-evaluate cycles. Near-misses must NOT trigger: one-shot edits or \
critiques with no evaluator, building evals, writing tests, fine-tuning weights, numeric HPO with \
Optuna, speeding up one query, questions about the method. The description is the only routing \
signal, so it must name what makes a request a fit and what does not.
- Formulation examples: an agent gets the full SKILL.md as instructions plus a user request and \
files, and must output the optimize_anything tool arguments as one JSON object, or `CLARIFY:` \
questions when a sound call is impossible (no artifact, no success signal, no data). Validity = \
the JSON matches the tool schema and actually runs through the engine on the seed without \
evaluator errors (context files sit in the working directory). Quality = a judge scores 7 \
criteria 0-2: evaluator fits the signal; ASI richness; mode and split; objective; budget; \
clarify when needed; safety and runnability.

Formulation rules the skill should teach:
- Verifiable signal (labels, expected outputs, tests, benchmark) -> python or shell evaluator, \
not llm_judge. llm_judge only for subjective targets, with concrete anchored criteria ending in \
"SCORE: X/10".
- Python evaluators: stdlib only; define evaluate(candidate, example); accept candidate as str OR \
dict (multi-component); handle example=None; return (score in [0,1], JSON-serializable dict). \
Return a rich ASI dict: input, expected vs got, failing checks, parse errors. They run in a \
subprocess, so put test data in dataset/valset rather than reading files.
- Shell evaluators: read the candidate via {{candidate_file}}, never inline {{candidate}} \
(unquoted injection); print a score already in [0,1] matched by score_pattern; examples reach \
the command only via {{example_file}} / {{example_json}}, otherwise the run is single-task.
- Modes: single-task (no dataset), multi-task (dataset), generalization (dataset + disjoint \
valset) when the user cares about unseen inputs. Never leak val examples into dataset or the seed.
- Budget: config.max_metric_calls / max_iterations sized to eval cost and any budget the user \
states; evaluator timeout above one eval's runtime.
- Objective: specific and measurable, naming the metric and any hard constraint.
- Clarify only for blocking gaps, and say what each answer unlocks.

Constraints on the skill: agent-neutral (it must work in Claude Code, Codex, OpenCode and Gemini \
CLI); frontmatter has only `name: optimize-anything` and `description` (<= 1024 chars); no \
agent-specific tool names; the MCP tool is called optimize_anything. Teach general rules and \
patterns, not answers to specific examples. Shorter is better: full brevity credit at <= 150 \
lines, zero at 300."""


# ---------------- run control: plateau + $ cap via on_event ----------------
class StopRun(Exception):
    pass


class RunController:
    """Stop after `patience` completed iterations without > min_gain in val best, or before the
    next iteration would cross the engine's $ cap. Raised from on_event at iteration_start; the
    harness then recovers the candidates from checkpoint.json."""

    def __init__(self, tracker: LockedCostTracker, cap_usd: float, patience: int | None = 3,
                 min_gain: float = 0.01, log_path: Path | None = None):
        self.tracker, self.cap, self.patience, self.min_gain = tracker, cap_usd, patience, min_gain
        self.log_path = log_path
        self.engine: Engine | None = None
        self.ref_best: float | None = None
        self.ref_iter = 0
        self.iter_costs: list[float] = []
        self._last_spend = 0.0
        self.stop_reason: str | None = None
        self.best_history: list[float] = []
        self.n_starts = 0  # own count: the engine's iteration numbering is not relied on

    def log(self, event: dict) -> None:
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": time.time(), **event}, default=str) + "\n")

    def check(self, iteration: int, best: float, spend: float) -> str | None:
        """Called at the start of `iteration`; iterations 1..iteration-1 are complete."""
        if self.ref_best is None:
            self.ref_best, self._last_spend = best, spend
        else:
            self.iter_costs.append(spend - self._last_spend)
            self._last_spend = spend
        self.best_history.append(best)
        done = iteration - 1
        if best > self.ref_best + self.min_gain:
            self.ref_best, self.ref_iter = best, done
        if self.patience is not None and done - self.ref_iter >= self.patience:
            return (f"plateau: {done - self.ref_iter} iterations without > {self.min_gain:.2f} "
                    f"gain in val best ({self.ref_best:.4f})")
        recent = self.iter_costs[-5:]
        est = sum(recent) / len(recent) if recent else 0.0
        if spend + est >= self.cap:
            return f"cost: spent ${spend:.2f}, next iteration (~${est:.2f}) would cross ${self.cap:.2f}"
        return None

    def __call__(self, event: dict) -> None:
        self.log(event)
        t = event.get("type")
        if t == "new_best_found":
            print(f"  new best val={event['score']:.4f}  spend=${self.tracker.total_usd:.2f}", flush=True)
        if self.tracker.quota_error and t in ("iteration_start", "proposal_complete",
                                              "evaluation_complete", "new_best_found"):
            self.stop_reason = f"quota: {self.tracker.quota_error}"
            self.log({"type": "harness_stop", "reason": self.stop_reason})
            raise StopRun(self.stop_reason)
        if t == "iteration_start" and self.engine is not None:
            self.n_starts += 1
            best = self.engine.state.best_score()
            reason = self.check(self.n_starts, best, self.tracker.total_usd)
            print(f"[iter {self.n_starts}] val best={best:.4f} spend=${self.tracker.total_usd:.2f}",
                  flush=True)
            if reason:
                self.stop_reason = reason
                self.log({"type": "harness_stop", "reason": reason})
                raise StopRun(reason)


# ---------------- reporting ----------------
COLS = ["trigger_f1", "formulation_validity", "formulation_quality", "brevity"]


def fmt(v) -> str:
    return "  n/a " if v is None else f"{v:6.3f}"


def table(rows: dict[str, dict]) -> str:
    head = f"{'':16} {'trigF1':>6} {'valid':>6} {'qual':>6} {'brev':>6} {'WEIGHT':>6}  n(trig/form) failed"
    out = [head]
    for label, r in rows.items():
        c = r["components"]
        out.append(f"{label:16} " + " ".join(fmt(c.get(k)) for k in COLS) + f" {fmt(r['weighted'])}"
                   f"  {r.get('n_trigger', 0)}/{r.get('n_formulation', 0)}"
                   f"       {len(r.get('failed', []))}")
    return "\n".join(out)


def slim(score: dict) -> dict:
    """Score dict without the bulky agent outputs (those go to a separate file)."""
    return {s: {k: v for k, v in r.items() if k != "agent_outputs"} for s, r in score.items()}


def build_scorer(args, tracker: LockedCostTracker, stub: StubServer) -> SkillScorer:
    backend = resolve_backend(args.backend)
    if backend == "mock":
        print("WARNING: mock backend; scores are meaningless", file=sys.stderr)
    llm = make_llm(backend, timeout=args.llm_timeout)
    cache = None if args.no_cache else CACHE_DIR / "llm"
    return SkillScorer(load_data(), llm, tracker, env_fn=stub.env, cache_dir=cache,
                       workers=args.workers, limit=args.limit)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ---------------- commands ----------------
def cmd_score(args) -> int:
    text = Path(args.skill).read_text()
    tracker = LockedCostTracker()
    t0 = time.time()
    with StubServer(CACHE_DIR / "stub_site") as stub:
        scorer = build_scorer(args, tracker, stub)
        try:
            score = scorer.score_skill(text, args.split)
        finally:
            scorer.close()
    wall = time.time() - t0
    print(table(score))
    for s, r in score.items():
        if r.get("malformed"):
            print(f"[{s}] MALFORMED: {r['flags']}")
            continue
        print(f"[{s}] confusion={r['trigger_confusion']} misrouted={[m['id'] + '->' + m['pick'] for m in r['misrouted']]}")
        print(f"[{s}] invalid={[(f['id'], f['validity_reason'][:90]) for f in r['formulations'] if not f['validity']]}")
        print(f"[{s}] valid_low_quality={r['valid_low_quality']} invalid_high_quality={r['invalid_high_quality']}")
        if r["failed"]:
            print(f"[{s}] FAILED (infra, excluded): {r['failed']}")
        if r["flags"]:
            print(f"[{s}] flags: {r['flags']}")
    spend = tracker.to_dict()
    if tracker.quota_error:
        print(f"WARNING: LLM quota hit, failed items excluded: {tracker.quota_error}")
    print(f"cost ${spend['total_usd']:.3f} | calls {spend['calls']} | cache hits {spend['cache_hits']} | "
          f"wall {wall:.0f}s | by role " +
          ", ".join(f"{k} ${v['usd']:.3f}/{v['calls']}" for k, v in spend["by_role"].items()))
    if args.out:
        out = Path(args.out)
        write_json(out / "score.json", slim(score))
        write_json(out / "agent_outputs.json", {s: r.get("agent_outputs", {}) for s, r in score.items()})
        write_json(out / "spend.json", {**spend, "wall_s": wall})
        write_json(out / "leakage.json", leakage_report(text, scorer.data))
        print(f"wrote {out}")
    return 0


def cmd_check_golds(args) -> int:
    data = load_data()
    with StubServer(CACHE_DIR / "stub_site") as stub:
        env = stub.env()

        def one(it):
            parsed = ({"mode": "clarify"} if it["gold"].get("clarify")
                      else {"mode": "call", "call": it["gold"], "error": None})
            return it["id"], check_validity(parsed, it, env)

        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(one, data.formulations))
    bad = 0
    for iid, v in res:
        bad += not v["valid"]
        print(f"{iid} {'OK ' if v['valid'] else 'BAD'} {v['reason'][:120]} {v.get('warnings') or ''}")
    print(f"{len(res) - bad}/{len(res)} gold formulations valid")
    return 1 if bad else 0


def load_checkpoint(run_dir: Path) -> dict:
    return json.loads((run_dir / "checkpoint.json").read_text())


def cmd_optimize(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_text = Path(args.skill).read_text()
    tracker = LockedCostTracker()
    t0 = time.time()
    phases: dict[str, float] = {}
    with StubServer(CACHE_DIR / "stub_site") as stub:
        scorer = build_scorer(args, tracker, stub)
        try:
            # 1. Official "before" score; also warms every seed item the engine will ask for.
            print("scoring seed (train + val)...", flush=True)
            seed_score = scorer.score_skill(seed_text, "both")
            phases["seed_scoring"] = tracker.total_usd
            print(table({f"seed {s}": r for s, r in seed_score.items()}), flush=True)
            write_json(out / "seed_score.json", slim(seed_score))

            # 2. Engine run. Reserve ~one full scoring pass for the final official score.
            reserve = min(max(phases["seed_scoring"], 1.0), 0.25 * args.max_cost)
            engine_cap = args.max_cost - reserve
            # Plateau stop is the engine's (no_improvement_patience + min_improvement on val best);
            # the controller only stops before an iteration that would cross the $ cap, and logs.
            controller = RunController(tracker, engine_cap, patience=None,
                                       log_path=out / "history.jsonl")
            backend = resolve_backend(args.backend)
            llm = make_llm(backend, timeout=args.llm_timeout)
            # Proposer calls are disk-cached too, so a re-run after an interruption replays the
            # same trajectory for free up to where it stopped.
            from skill_eval.llm_cache import CachedLLM
            proposer = ReflectiveProposer(  # engine proposer: keeps inner fences (extract_candidate)
                CachedLLM(llm, tracker, "propose", None if args.no_cache else CACHE_DIR / "llm"),
                model=resolve_model("proposer", backend, args.proposer_model, args.preset),
                effort=resolve_effort("proposer", args.proposer_effort, args.preset), max_tokens=16000)
            extra = dict(promotion=args.promotion, eval_workers=args.workers,
                         no_improvement_patience=args.patience, min_improvement=args.min_gain,
                         prescreen=not args.no_prescreen, proposal_memory=not args.no_proposal_memory)
            config = Config(preset=args.preset, backend=backend, max_iterations=args.max_iterations,
                            max_cost_usd=engine_cap, cache_evals=True, seed=args.seed,
                            reflection_minibatch_size=args.minibatch,
                            run_dir=str(out / "engine_run"), checkpoint_interval=1, **extra)
            engine = Engine(
                seed_candidate={"skill": seed_text}, evaluator=EngineEvaluator(scorer),
                proposer=proposer, objective=OBJECTIVE, background=BACKGROUND,
                dataset=scorer.build_examples("train"), valset=scorer.build_examples("val"),
                config=config, on_event=controller, cost_tracker=tracker,
            )
            controller.engine = engine
            print(f"engine: {len(engine.dataset)} train / {len(engine.valset)} val examples, "
                  f"cap ${engine_cap:.2f} (reserve ${reserve:.2f}), extra config {extra}", flush=True)
            stop_reason = None
            try:
                result = engine.run()
                stop_reason = getattr(result, "stop_reason", None) or "engine"
            except StopRun as e:
                stop_reason = str(e)
            except Exception as e:  # noqa: BLE001 - e.g. proposer LLMError: finalize what we have
                stop_reason = f"error: {type(e).__name__}: {str(e)[:300]}"
                print(f"engine run failed, finalizing from checkpoint: {stop_reason}", flush=True)
            if not stop_reason.startswith(("engine", "budget", "iterations", "cost", "timeout",
                                           "no_improvement", "train_saturated")):
                try:
                    engine._checkpoint()  # persist the latest state before reading it back
                except Exception as e:  # noqa: BLE001
                    print(f"checkpoint after stop failed ({e}); using the last saved one", flush=True)
            phases["engine"] = tracker.total_usd - phases["seed_scoring"]

            ck = load_checkpoint(out / "engine_run")
            texts = [c["components"]["skill"] for c in ck["candidates"]]
            order = sorted(range(len(texts)), key=lambda i: ck["agg_scores"][i], reverse=True)
            seen, top = set(), []
            for i in order:
                if texts[i] not in seen:
                    seen.add(texts[i])
                    top.append(i)
            # paired_se: the engine's promoted incumbent, not simply the top val mean.
            best_i = ck.get("best_idx", top[0])
            if best_i in top:
                top.remove(best_i)
            top.insert(0, best_i)
            best_text = texts[best_i]

            if tracker.quota_error and not str(stop_reason).startswith("quota"):
                stop_reason = f"quota: {tracker.quota_error} (engine stop was {stop_reason})"
            # 3. Official "after" score (val items are mostly cached from the engine's val eval).
            print("scoring best (train + val)...", flush=True)
            before_final = tracker.total_usd
            best_score = scorer.score_skill(best_text, "both")
            phases["final_scoring"] = tracker.total_usd - before_final
        finally:
            scorer.close()

    wall = time.time() - t0
    (out / "best_SKILL.md").write_text(best_text)
    for rank, i in enumerate(top[:3], 1):
        (out / f"top{rank}_cand{i}_val{ck['agg_scores'][i]:.3f}.md").write_text(texts[i])
    write_json(out / "best_score.json", slim(best_score))
    write_json(out / "agent_outputs_best.json", {s: r.get("agent_outputs", {}) for s, r in best_score.items()})
    leak = {"seed": leakage_report(seed_text, scorer.data), "best": leakage_report(best_text, scorer.data)}
    write_json(out / "leakage.json", leak)
    spend = {**tracker.to_dict(), "phases_usd": phases, "cap_usd": args.max_cost, "wall_s": wall}
    write_json(out / "spend.json", spend)
    summary = {
        "stop_reason": stop_reason, "iterations": ck.get("iteration"), "candidates": len(texts),
        "incomplete": bool(tracker.quota_error) or any(r.get("failed") for r in best_score.values()),
        "quota_error": tracker.quota_error,
        "failed_evals": ck.get("failed_evals"), "total_evals": ck.get("total_evals"),
        "cache_hits_engine": ck.get("cache_hits"),
        "best_candidate_idx": best_i, "best_val_proxy": ck["agg_scores"][best_i],
        "seed_val_proxy": ck["agg_scores"][0], "val_best_history": controller.best_history,
        "official": {f"{who}_{s}": {"weighted": sc[s]["weighted"], **sc[s]["components"]}
                     for who, sc in (("seed", seed_score), ("best", best_score)) for s in sc},
        "cross_counts": {f"{who}_{s}": {"valid_low_quality": sc[s].get("valid_low_quality"),
                                        "invalid_high_quality": sc[s].get("invalid_high_quality")}
                         for who, sc in (("seed", seed_score), ("best", best_score)) for s in sc},
        "leakage_pct": {k: v["all_train_items"]["pct"] for k, v in leak.items()},
        "leakage_flag": {k: v["flag"] for k, v in leak.items()},
        "total_usd": tracker.total_usd, "wall_s": wall, "weights": WEIGHTS,
    }
    write_json(out / "summary.json", summary)
    print(table({**{f"seed {s}": r for s, r in seed_score.items()},
                 **{f"best {s}": r for s, r in best_score.items()}}))
    if summary["incomplete"]:
        print("WARNING: INCOMPLETE run (LLM quota or failed evals); re-run after the limit resets. "
              "Cached calls replay for free.")
    print(f"stop: {stop_reason} | spend ${tracker.total_usd:.2f} | wall {wall / 60:.1f} min | "
          f"leakage seed {summary['leakage_pct']['seed']:.0%} best {summary['leakage_pct']['best']:.0%}")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_skill = str(REPO / "skill" / "SKILL.md")

    def common(p):
        p.add_argument("--skill", default=default_skill)
        p.add_argument("--limit", type=int, default=None, help="subsample N items per type per split")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--backend", default="auto", help="engine LLM backend (auto -> claude-cli here)")
        p.add_argument("--llm-timeout", type=float, default=300.0)
        p.add_argument("--no-cache", action="store_true", help="bypass the disk LLM cache")

    p = sub.add_parser("score", help="official score of one SKILL.md")
    common(p)
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("optimize", help="hill-climb SKILL.md")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--max-cost", type=float, default=50.0, help="total $ cap (API-equivalent)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--minibatch", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--promotion", choices=["max", "paired_se"], default="paired_se")
    p.add_argument("--proposer-effort", default=None, help="None -> preset (default: medium)")
    p.add_argument("--no-prescreen", action="store_true")
    p.add_argument("--no-proposal-memory", action="store_true")
    p.add_argument("--preset", default="default")
    p.add_argument("--proposer-model", default=None)
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("check-golds", help="gold formulations pass the validity runner (no LLM)")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_check_golds)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
