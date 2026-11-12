#!/usr/bin/env python3
"""Validate the skill-eval datasets in scripts/skill_eval/data/.

Usage:
  python3 scripts/skill_eval/check_data.py                  # validate (offline)
  python3 scripts/skill_eval/check_data.py --live           # also run API-backed golds (needs ANTHROPIC_API_KEY)
  python3 scripts/skill_eval/check_data.py --assign-splits  # (re)write the seeded 70/30 stratified split field

Checks: JSON schema, unique ids, split balance, no dataset/valset overlap, and that every gold
evaluator runs on its own seed and returns a float in [0,1] below 1.0 (a perfect seed is never
optimized because skip_perfect_score is on). Python golds run in a subprocess the same way the
engine's PythonEvaluator does; shell golds run from a temp dir that holds the item's context_files.
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
SEED = 20261111
VAL_FRAC = 0.3
EVAL_TYPES = {"python", "shell", "llm_judge"}

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            err(f"{path.name}:{n}: invalid JSON ({e})")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def assign_splits(rows: list[dict], key: str) -> None:
    """Seeded stratified split. Each stratum with >= 2 items gets at least one train and one val item."""
    rng = random.Random(SEED)
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        strata[r[key]].append(r)
    for name in sorted(strata):
        group = sorted(strata[name], key=lambda r: r["id"])
        rng.shuffle(group)
        n_val = round(len(group) * VAL_FRAC)
        if len(group) >= 2:
            n_val = min(max(n_val, 1), len(group) - 1)
        for i, r in enumerate(group):
            r["split"] = "val" if i < n_val else "train"


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def paraphrase_check(rows: list[dict], field: str) -> None:
    train = [r for r in rows if r.get("split") == "train"]
    for v in (r for r in rows if r.get("split") == "val"):
        tv = tokens(v[field])
        for t in train:
            tt = tokens(t[field])
            j = len(tv & tt) / max(1, len(tv | tt))
            if j > 0.6:
                warnings.append(f"possible paraphrase: val {v['id']} ~ train {t['id']} (jaccard {j:.2f})")


# ---------------- triggers ----------------
def check_triggers(rows: list[dict]) -> None:
    ids = Counter(r.get("id") for r in rows)
    for i, c in ids.items():
        if c > 1:
            err(f"triggers: duplicate id {i}")
    for r in rows:
        for f in ("id", "prompt", "label", "kind", "note", "split"):
            if not r.get(f):
                err(f"triggers {r.get('id')}: missing {f}")
        want = "trigger" if r.get("kind") == "positive" else "no_trigger"
        if r.get("kind") not in ("positive", "near_miss", "unrelated"):
            err(f"triggers {r['id']}: bad kind {r.get('kind')}")
        elif r.get("label") != want:
            err(f"triggers {r['id']}: kind {r['kind']} must have label {want}")
    if len(rows) < 48:
        err(f"triggers: {len(rows)} items, need >= 48")
    nm = [r for r in rows if r.get("kind") == "near_miss"]
    if len(nm) < 16:
        err(f"triggers: {len(nm)} near-misses, need >= 16")
    for kind in ("positive", "near_miss", "unrelated"):
        splits = {r.get("split") for r in rows if r.get("kind") == kind}
        if splits != {"train", "val"}:
            err(f"triggers: kind {kind} not present in both splits ({splits})")
    paraphrase_check(rows, "prompt")


def check_distractors(items: list) -> None:
    if not isinstance(items, list) or len(items) < 12:
        err("distractors: need a list of >= 12 skills")
        return
    names = [d.get("name") for d in items]
    if len(set(names)) != len(names):
        err("distractors: duplicate names")
    for d in items:
        if not d.get("description", "").startswith("Use when"):
            err(f"distractors {d.get('name')}: description should start with 'Use when'")
        if "optimiz" in d.get("name", ""):
            err(f"distractors {d.get('name')}: name collides with the skill under test")


# ---------------- formulations ----------------
def run_python(code: str, candidate, example, timeout: int) -> tuple[float | None, dict]:
    """Mirror engine PythonEvaluator: subprocess, last stdout line is {"score", "asi"}."""
    runner = (
        f"import json\n{code}\n"
        f"_c = json.loads({json.dumps(json.dumps(candidate))})\n"
        f"_e = json.loads({json.dumps(json.dumps(example))})\n"
        "_r = evaluate(_c, _e)\n"
        "_s, _a = _r if isinstance(_r, tuple) else (float(_r), {})\n"
        "print(json.dumps({'score': _s, 'asi': _a}))\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runner.py"
        p.write_text(runner)
        try:
            res = subprocess.run([sys.executable, "-I", str(p)], capture_output=True, text=True,
                                 timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return None, {"error": f"timeout after {timeout}s"}
    if res.returncode != 0:
        return None, {"error": res.stderr[-500:]}
    try:
        out = json.loads(res.stdout.strip().splitlines()[-1])
        return float(out["score"]), out.get("asi", {})
    except Exception as e:  # noqa: BLE001 - report any parse failure
        return None, {"error": f"output parse error: {e}", "stdout": res.stdout[-300:]}


def run_shell(ev: dict, seed_text: str, files: dict) -> tuple[float | None, dict]:
    cmd = ev["command"]
    if "{{candidate}}" in cmd:
        return None, {"error": "uses {{candidate}} inline (unquoted substitution); use {{candidate_file}}"}
    with tempfile.TemporaryDirectory() as d:
        for name, content in files.items():
            (Path(d) / name).write_text(content)
        cand = Path(d) / "_candidate.txt"
        cand.write_text(seed_text)
        try:
            res = subprocess.run(cmd.replace("{{candidate_file}}", str(cand)), shell=True, cwd=d,
                                 capture_output=True, text=True, timeout=ev.get("timeout", 30))
        except subprocess.TimeoutExpired:
            return None, {"error": "timeout"}
    m = re.search(ev.get("score_pattern", r"([\d.]+)"), res.stdout)
    if not m:
        return None, {"error": "score_pattern did not match", "stdout": res.stdout[-300:], "stderr": res.stderr[-300:]}
    return float(m.group(1)), {"stdout": res.stdout.strip()[-200:]}


def check_gold(item: dict, live: bool, report: list[str]) -> None:
    iid, gold = item["id"], item["gold"]
    if gold.get("clarify"):
        if not gold.get("ask"):
            err(f"{iid}: clarify item needs a non-empty 'ask' list")
        report.append(f"{iid:4} clarify      asks={len(gold.get('ask', []))}")
        return
    for f in ("seed_candidate", "evaluator", "objective", "config"):
        if not gold.get(f):
            err(f"{iid}: gold missing {f}")
            return
    seed = gold["seed_candidate"]
    if not isinstance(seed, dict) or not all(isinstance(v, str) for v in seed.values()):
        err(f"{iid}: seed_candidate must be {{name: str}}")
        return
    ev = gold["evaluator"]
    if ev.get("type") not in EVAL_TYPES:
        err(f"{iid}: bad evaluator type {ev.get('type')}")
        return
    ds, vs = gold.get("dataset") or [], gold.get("valset") or []
    if vs and not ds:
        err(f"{iid}: valset without dataset")
    overlap = {json.dumps(x, sort_keys=True) for x in ds} & {json.dumps(x, sort_keys=True) for x in vs}
    if overlap:
        err(f"{iid}: {len(overlap)} example(s) in both dataset and valset")

    if ev["type"] == "llm_judge":
        if "SCORE" not in ev.get("criteria", ""):
            err(f"{iid}: llm_judge criteria must define the SCORE format")
        report.append(f"{iid:4} llm_judge    structure ok (not executed)")
        return

    joined = "\n".join(seed.values())
    if ev["type"] == "shell":
        if ds:
            warnings.append(f"{iid}: shell evaluator with a dataset; the shell evaluator ignores examples")
        score, asi = run_shell(ev, joined, item.get("context_files", {}))
        scores = [score]
    else:
        needs_api = "ANTHROPIC_API_KEY" in item.get("requires", [])
        if needs_api and not live:
            score, asi = run_python(ev["code"], seed if len(seed) > 1 else joined, (ds or [None])[0],
                                    ev.get("timeout", 30) + 30)
            env_has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if score is None:
                err(f"{iid}: API gold crashed instead of degrading: {asi.get('error', '')[:200]}")
            elif not env_has_key and not (score == 0.0 and "error" in asi):
                err(f"{iid}: without a key, expected score 0.0 with an 'error' in ASI, got {score} {asi}")
            report.append(f"{iid:4} python(api)  offline smoke ok, seed score not measured (use --live)")
            return
        examples = ds + vs or [None]
        variants = [seed] if len(seed) == 1 else [seed, joined]  # multi-component: dict and joined str
        scores = []
        for cand in variants:
            c = next(iter(seed.values())) if len(seed) == 1 else cand
            for ex in examples:
                s, a = run_python(ev["code"], c, ex, ev.get("timeout", 30) + 30)
                if s is None:
                    err(f"{iid}: python gold failed on example {ex!r:.80}: {a.get('error', '')[:300]}")
                    return
                try:
                    json.dumps(a)
                except TypeError:
                    err(f"{iid}: ASI not JSON-serializable")
                scores.append(s)
    if any(s is None for s in scores):
        err(f"{iid}: evaluator did not produce a score: {asi}")
        return
    bad = [s for s in scores if not 0.0 <= s <= 1.0]
    if bad:
        err(f"{iid}: scores outside [0,1]: {bad[:3]}")
    mean = sum(scores) / len(scores)
    if mean >= 1.0:
        err(f"{iid}: seed already scores 1.0, so it would never be optimized")
    report.append(f"{iid:4} {ev['type']:12} seed mean={mean:.3f} over {len(scores)} eval(s)")


def check_formulations(rows: list[dict], live: bool) -> list[str]:
    report: list[str] = []
    ids = Counter(r.get("id") for r in rows)
    for i, c in ids.items():
        if c > 1:
            err(f"formulations: duplicate id {i}")
    for r in rows:
        for f in ("id", "category", "request", "gold", "rubric_notes", "split"):
            if not r.get(f):
                err(f"formulations {r.get('id')}: missing {f}")
        for name, content in (r.get("context_files") or {}).items():
            if "/" in name or not isinstance(content, str):
                err(f"{r['id']}: context file {name!r} must be a flat name with str content")
    if len(rows) < 18:
        err(f"formulations: {len(rows)} items, need >= 18")
    for cat in sorted({r["category"] for r in rows}):
        splits = {r["split"] for r in rows if r["category"] == cat}
        if splits != {"train", "val"}:
            err(f"formulations: category {cat} not in both splits ({splits})")
    paraphrase_check(rows, "request")
    for r in rows:
        if "gold" in r:
            check_gold(r, live, report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="run API-backed python golds against the real API")
    ap.add_argument("--assign-splits", action="store_true", help="rewrite the split field (seeded, stratified)")
    args = ap.parse_args()

    trig = load_jsonl(DATA / "triggers.jsonl")
    form = load_jsonl(DATA / "formulations.jsonl")
    if args.assign_splits:
        assign_splits(trig, "kind")
        assign_splits(form, "category")
        write_jsonl(DATA / "triggers.jsonl", trig)
        write_jsonl(DATA / "formulations.jsonl", form)
        print(f"splits assigned (seed {SEED})")

    check_triggers(trig)
    check_distractors(json.loads((DATA / "distractors.json").read_text()))
    report = check_formulations(form, args.live)

    print("triggers   ", dict(sorted(Counter((r.get("split"), r.get("kind")) for r in trig).items())))
    print("formulations", dict(sorted(Counter(r.get("split") for r in form).items())),
          "categories:", len({r["category"] for r in form}))
    print("\n".join(report))
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("FAIL ", e)
    print("OK" if not errors else f"{len(errors)} failure(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
