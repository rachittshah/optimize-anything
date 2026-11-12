"""Pure helpers for the skill eval: data loading, frontmatter, F1, brevity, shards, leakage.

No LLM calls and no subprocesses here, so everything is unit-testable.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE_DIR = HERE / ".cache"
RESULTS_DIR = HERE / "results"
REPO = HERE.parent.parent
SKILL_NAME = "optimize-anything"
DESCRIPTION_LIMIT = 1024  # Agent Skills spec
BREVITY_FULL, BREVITY_ZERO = 150, 300  # lines


# ---------------- data ----------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@dataclass
class EvalData:
    triggers: list[dict]
    distractors: list[dict]
    formulations: list[dict]
    rubric: str

    def items(self, kind: str, split: str) -> list[dict]:
        rows = self.triggers if kind == "trigger" else self.formulations
        return [r for r in rows if split == "both" or r["split"] == split]


def load_data(data_dir: Path = DATA) -> EvalData:
    return EvalData(
        triggers=load_jsonl(data_dir / "triggers.jsonl"),
        distractors=json.loads((data_dir / "distractors.json").read_text()),
        formulations=load_jsonl(data_dir / "formulations.jsonl"),
        rubric=(data_dir / "rubric.md").read_text(),
    )


def subsample(rows: list[dict], n: int | None, key: str) -> list[dict]:
    """First n rows, interleaving strata (`key`) so small samples cover every stratum."""
    if not n or n >= len(rows):
        return list(rows)
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        strata[r[key]].append(r)
    out: list[dict] = []
    while len(out) < n:
        for name in sorted(strata):
            if strata[name] and len(out) < n:
                out.append(strata[name].pop(0))
    return out


# ---------------- frontmatter ----------------
class FrontmatterError(ValueError):
    pass


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        return inner.replace('\\"', '"') if v[0] == '"' else inner.replace("''", "'")
    return v


def parse_frontmatter(text: str) -> dict[str, str]:
    """Minimal YAML frontmatter parser for flat `key: value` maps.

    Supports quoted values, block scalars (`>`, `|`, with `-`/`+`), and indented plain
    continuation lines. Raises FrontmatterError if there is no `---` block.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError("no frontmatter: file must start with a '---' line")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise FrontmatterError("frontmatter not closed with '---'") from None
    out: dict[str, str] = {}
    key: str | None = None
    block: str | None = None  # ">" or "|" while reading a block scalar
    buf: list[str] = []

    def flush() -> None:
        if key is None:
            return
        if block == "|":
            out[key] = "\n".join(buf).strip("\n")
        else:
            out[key] = " ".join(s.strip() for s in buf if s.strip()).strip()

    for raw in lines[1:end]:
        m = re.match(r"^([A-Za-z0-9_-]+):(.*)$", raw)
        if m and not raw.startswith((" ", "\t")):
            flush()
            key, rest = m.group(1), m.group(2).strip()
            if rest[:1] in (">", "|") and re.fullmatch(r"[>|][-+]?", rest):
                block, buf = rest[0], []
            else:
                block, buf = None, [_unquote(rest)] if rest else []
        elif key is not None and (raw.startswith((" ", "\t")) or not raw.strip()):
            buf.append(raw.strip() if block != "|" else raw.lstrip())
        elif raw.strip().startswith("#"):
            continue
        else:
            raise FrontmatterError(f"cannot parse frontmatter line: {raw!r}")
    flush()
    return out


def check_skill_text(text: str) -> tuple[dict[str, str] | None, list[str], list[str]]:
    """(frontmatter or None, fatal errors, warnings). Fatal = the candidate can't be routed."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        fm = parse_frontmatter(text)
    except FrontmatterError as e:
        return None, [str(e)], warnings
    if fm.get("name") != SKILL_NAME:
        errors.append(f"frontmatter name must be {SKILL_NAME!r}, got {fm.get('name')!r}")
    if not fm.get("description"):
        errors.append("frontmatter description is missing or empty")
    extra = sorted(set(fm) - {"name", "description"})
    if extra:
        warnings.append(f"frontmatter has keys beyond name+description (not agent-neutral): {extra}")
    if len(fm.get("description", "")) > DESCRIPTION_LIMIT:
        warnings.append(f"description is {len(fm['description'])} chars, over the "
                        f"{DESCRIPTION_LIMIT}-char Agent Skills limit (brevity forced to 0)")
    return (fm if not errors else None), errors, warnings


# ---------------- metrics ----------------
def f1_score(golds: list[bool], preds: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(golds, preds))
    fp = sum(p and not g for g, p in zip(golds, preds))
    fn = sum(g and not p for g, p in zip(golds, preds))
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def shard_score(golds: list[bool], preds: list[bool]) -> float:
    """Per-shard F1; accuracy when the shard has no gold positives (F1 is undefined there)."""
    if not any(golds):
        return sum(g == p for g, p in zip(golds, preds)) / max(1, len(golds))
    return f1_score(golds, preds)


def brevity_score(text: str, description: str | None = None) -> float:
    if description is not None and len(description) > DESCRIPTION_LIMIT:
        return 0.0
    n = len(text.splitlines())
    if n <= BREVITY_FULL:
        return 1.0
    if n >= BREVITY_ZERO:
        return 0.0
    return (BREVITY_ZERO - n) / (BREVITY_ZERO - BREVITY_FULL)


WEIGHTS = {"trigger_f1": 0.3, "formulation_validity": 0.3, "formulation_quality": 0.3, "brevity": 0.1}


def weighted_score(components: dict[str, float | None]) -> float | None:
    if any(components.get(k) is None for k in WEIGHTS):
        return None
    return sum(w * components[k] for k, w in WEIGHTS.items())


def make_shards(items: list[dict], size: int = 5) -> list[list[dict]]:
    """Stratified shards of ~size items: round-robin by kind over id-sorted items."""
    if not items:
        return []
    n = max(1, round(len(items) / size))
    shards: list[list[dict]] = [[] for _ in range(n)]
    cursor = 0
    for kind in ("positive", "near_miss", "unrelated"):
        for r in sorted((r for r in items if r["kind"] == kind), key=lambda r: r["id"]):
            shards[cursor % n].append(r)
            cursor += 1
    for r in sorted((r for r in items if r["kind"] not in ("positive", "near_miss", "unrelated")),
                    key=lambda r: r["id"]):
        shards[cursor % n].append(r)
        cursor += 1
    return [s for s in shards if s]


# ---------------- reward-hacking: n-gram leakage ----------------
STOPWORDS = frozenset("""
a an the and or but if then else of to in on for with by from at as is are was were be been being
it its this that these those i you we they he she my our your their me us them do does did done
not no so than too very can could should would will shall may might must have has had having
what which who whom when where why how all any each some such into over under about up down out
""".split())


def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOPWORDS]


def ngrams(tokens: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def json_strings(obj) -> list[str]:
    """All string leaves of a JSON value (dict keys excluded)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in json_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in json_strings(v)]
    return []


def leakage_report(candidate: str, data: EvalData, n: int = 5, flag_at: float = 0.20) -> dict:
    """Share of train items that share a content-word n-gram (stopwords removed) with the
    candidate. Trigger prompts and formulation (request + gold JSON) are reported separately."""
    cand = ngrams(content_tokens(candidate), n)

    def grams(t) -> set[tuple[str, ...]]:
        # Lists of strings = separate segments (n-grams never span two JSON values).
        segs = t if isinstance(t, list) else [t]
        return set().union(*(ngrams(content_tokens(x), n) for x in segs)) if segs else set()

    def frac(texts: dict) -> tuple[float, list[dict]]:
        hits = []
        for iid, t in texts.items():
            shared = cand & grams(t)
            if shared:
                hits.append({"id": iid, "shared": [" ".join(g) for g in sorted(shared)[:3]],
                             "n_shared": len(shared)})
        return (len(hits) / len(texts) if texts else 0.0), hits

    trig = {r["id"]: r["prompt"] for r in data.items("trigger", "train")}
    req = {r["id"]: r["request"] for r in data.items("formulation", "train")}
    # Gold: string values only, so the tool's own field names don't count as leakage.
    gold = {r["id"]: json_strings(r["gold"]) for r in data.items("formulation", "train")}
    allt = {**trig, **{k: [req[k], *gold[k]] for k in req}}
    out = {}
    for name, texts in (("trigger_prompts", trig), ("formulation_requests", req),
                        ("gold_json", gold), ("all_train_items", allt)):
        pct, hits = frac(texts)
        out[name] = {"pct": round(pct, 3), "n_items": len(texts), "hits": hits[:10]}
    out["flag"] = out["all_train_items"]["pct"] > flag_at
    out["flag_threshold"] = flag_at
    return out
