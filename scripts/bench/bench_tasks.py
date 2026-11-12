"""Deterministic benchmark tasks for the optimize-anything engine.

Each task has: a seed candidate (dict[str, str]), an objective/background shown to
the proposer, fixed train/val/test splits (scripts/bench/data/<task>.json) and an
evaluator with `evaluate(candidate, example) -> (score, asi)`.

Evaluators accept either a str or a dict[str, str] candidate, because the engine on
`main` passes a single component's text during minibatch evals and
"\\n".join(components.values()) during full evals; newer engines may pass a dict.
Untrusted candidate code (regex, Python) runs in a `python -I` subprocess with a timeout.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parent / "data"
SUBPROCESS_TIMEOUT_S = 5.0


def load_splits(task_name: str) -> dict[str, list[dict]]:
    data = json.loads((DATA_DIR / f"{task_name}.json").read_text())
    return {k: data[k] for k in ("train", "val", "test")}


def _as_text(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return "\n".join(str(v) for v in candidate.values())
    return str(candidate)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _run_sandboxed(script: str, payload: dict, timeout: float = SUBPROCESS_TIMEOUT_S) -> tuple[Any, str | None]:
    """Run `script` in an isolated interpreter (stdin = JSON payload). Returns (result, error)."""
    with tempfile.TemporaryDirectory(prefix="oa-bench-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                input=json.dumps(payload), capture_output=True, text=True,
                timeout=timeout, cwd=cwd, env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {timeout:.0f}s (infinite loop or catastrophic backtracking?)"
    if proc.returncode != 0:
        return None, f"runner crashed: {proc.stderr.strip()[-500:]}"
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1]), None
    except (json.JSONDecodeError, IndexError) as e:
        return None, f"runner output unparseable: {e}; stdout tail={proc.stdout[-200:]!r}"


class _BatchCachedEvaluator:
    """Runs a candidate on ALL task inputs in one subprocess, caches per candidate."""

    accepts_dict = True  # newer engines pass the components dict when this is set

    def __init__(self, all_examples: list[dict]):
        self._inputs = {ex["id"]: ex["input"] for ex in all_examples}
        self._cache: dict[str, tuple[dict | None, str | None]] = {}

    def _results_for(self, text: str) -> tuple[dict | None, str | None]:
        key = _hash(text)
        if key not in self._cache:
            self._cache[key] = self._compute(text)
        return self._cache[key]

    def _compute(self, text: str) -> tuple[dict | None, str | None]:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1: regex_emails
# ---------------------------------------------------------------------------

_REGEX_RUNNER = r"""
import json, re, sys
p = json.loads(sys.stdin.read())
try:
    rx = re.compile(p["pattern"])
except re.error as e:
    print(json.dumps({"compile_error": str(e)})); sys.exit(0)
out = {}
for k, s in p["inputs"].items():
    out[k] = rx.fullmatch(s) is not None
print(json.dumps({"matches": out}))
"""


class RegexEmailEvaluator(_BatchCachedEvaluator):
    def _compute(self, text):
        pattern = text.strip()
        res, err = _run_sandboxed(_REGEX_RUNNER, {"pattern": pattern, "inputs": self._inputs})
        if err:
            return None, err
        if "compile_error" in res:
            return None, f"regex does not compile: {res['compile_error']}"
        return res["matches"], None

    def evaluate(self, candidate, example=None):
        if example is None:
            return 0.0, {"error": "regex_emails needs an example"}
        matches, err = self._results_for(_as_text(candidate))
        expected = "match" if example["label"] else "no-match"
        if err:
            return 0.0, {"input": example["input"], "expected": expected, "error": err}
        got = "match" if matches[example["id"]] else "no-match"
        score = 1.0 if got == expected else 0.0
        asi = {"input": repr(example["input"]), "expected": expected, "got": got}
        if score < 1.0:
            asi["hint"] = example["hint"]
        return score, asi


REGEX_SEED = r"[a-z]+@[a-z]+\.com"
REGEX_REFERENCE = (
    r"(?=[^@]{1,64}@)[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
)
REGEX_OBJECTIVE = (
    "Write ONE Python regular expression (used with re.fullmatch on the whole string) that "
    "accepts valid email addresses and rejects invalid ones. Output only the regex."
)
REGEX_SPEC = """Validity spec (simplified, NOT full RFC 5322):
- Exactly one '@'. No whitespace, quotes, brackets or other characters outside the sets below.
- Local part: 1-64 chars from letters, digits and . _ % + - ; no leading/trailing dot, no consecutive dots.
- Domain: two or more dot-separated labels. Each label is 1-63 letters/digits/hyphens and does not
  start or end with a hyphen. No underscores. The last label (TLD) is 2+ letters only.
- No trailing dot after the domain, no IP-literal domains."""


REGEX_BACKGROUND = ("A simplified (not RFC 5322) notion of email validity is used; infer the exact rules "
                    "from the per-example feedback.")

# ---------------------------------------------------------------------------
# Task 2: parse_dates
# ---------------------------------------------------------------------------

_PARSE_RUNNER = r"""
import json, signal, sys
p = json.loads(sys.stdin.read())
class _TO(Exception): pass
def _alarm(*a): raise _TO()
signal.signal(signal.SIGALRM, _alarm)
ns = {"__name__": "candidate"}
try:
    exec(compile(p["code"], "<candidate>", "exec"), ns)
    parse = ns["parse"]
except Exception as e:
    print(json.dumps({"load_error": f"{type(e).__name__}: {e}"})); sys.exit(0)
out = {}
for k, s in p["inputs"].items():
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        r = parse(s)
        out[k] = {"got": r if isinstance(r, str) else repr(r)}
    except _TO:
        out[k] = {"got": None, "exception": "timeout (>0.5s on this input)"}
    except BaseException as e:
        out[k] = {"got": None, "exception": f"{type(e).__name__}: {e}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps({"results": out}))
"""


class ParseDatesEvaluator(_BatchCachedEvaluator):
    def _compute(self, text):
        code = text
        res, err = _run_sandboxed(_PARSE_RUNNER, {"code": code, "inputs": self._inputs})
        if err:
            return None, err
        if "load_error" in res:
            return None, f"candidate code failed to load / define parse(): {res['load_error']}"
        return res["results"], None

    def evaluate(self, candidate, example=None):
        if example is None:
            return 0.0, {"error": "parse_dates needs an example"}
        results, err = self._results_for(_as_text(candidate))
        if err:
            return 0.0, {"input": repr(example["input"]), "expected": example["expected"], "error": err}
        r = results[example["id"]]
        score = 1.0 if r["got"] == example["expected"] else 0.0
        asi = {"input": repr(example["input"]), "expected": example["expected"], "got": repr(r["got"])}
        if "exception" in r:
            asi["exception"] = r["exception"]
        return score, asi


DATES_SEED = '''import re

def parse(s: str) -> str:
    if re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", s):
        return s
    return "INVALID"
'''

DATES_REFERENCE = '''import re
from datetime import date

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
FULL = ["january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december"]
for i, m in enumerate(FULL, 1):
    MONTHS[m] = i
WEEKDAYS = r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|nesday|sday|urday)?"

def _ok(y, m, d):
    try:
        if not 1000 <= y <= 9999:
            return "INVALID"
        return date(y, m, d).isoformat()
    except ValueError:
        return "INVALID"

def parse(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"^" + WEEKDAYS + r",?\\s+", "", t)
    m = re.fullmatch(r"(\\d{4})[-/.](\\d{1,2})[-/.](\\d{1,2})(?:t\\d{2}:\\d{2}(?::\\d{2})?)?", t)
    if m:
        return _ok(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\\d{1,2})[-/.](\\d{1,2})[-/.](\\d{4})", t)
    if m:
        return _ok(int(m[3]), int(m[2]), int(m[1]))
    m = re.fullmatch(r"(\\d{4})(\\d{2})(\\d{2})", t)
    if m:
        return _ok(int(m[1]), int(m[2]), int(m[3]))
    day = r"(\\d{1,2})(?:st|nd|rd|th)?"
    mon = r"([a-z]+)\\.?"
    m = re.fullmatch(mon + r"\\s+" + day + r",?\\s+(\\d{4})", t)
    if m and m[1] in MONTHS:
        return _ok(int(m[3]), MONTHS[m[1]], int(m[2]))
    m = re.fullmatch(r"(?:the\\s+)?" + day + r"\\s+(?:of\\s+)?" + mon + r",?\\s+(\\d{4})", t)
    if m and m[2] in MONTHS:
        return _ok(int(m[3]), MONTHS[m[2]], int(m[1]))
    return "INVALID"
'''

DATES_OBJECTIVE = (
    "Write Python source defining `parse(s: str) -> str` that normalizes a messy date string to "
    "ISO 'YYYY-MM-DD', or returns exactly 'INVALID'. Standard library only. Output the full module."
)
DATES_SPEC = """Conventions:
- Ignore surrounding whitespace and letter case. An optional leading weekday name (e.g. 'Tuesday, ') is ignored.
- Numeric with year FIRST (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD; month/day may be 1 or 2 digits) is year-month-day.
  A trailing ISO time part like 'T10:30' or 'T10:30:00' is ignored.
- Numeric with year LAST (separators / - .) is DAY-first: DD/MM/YYYY (European). Never month-first.
- 8 digits with no separator = YYYYMMDD.
- Month names: full names, 3-letter abbreviations (optionally followed by '.'), and 'Sept'.
  Accepted orders: 'March 5, 2024', 'Mar 5 2024', '5 March 2024', '5th March 2024', '23rd of October 2015'.
- Years must be 4 digits in 1000..9999. Two-digit years are INVALID.
- The date must exist on the Gregorian calendar (leap years: divisible by 4, except centuries not divisible by 400).
- Anything else (missing day, garbage, extra parts) is INVALID."""


DATES_BACKGROUND = ("Inputs come in many human formats. Some conventions are hidden (e.g. how ambiguous numeric "
                    "dates are read); infer them from the per-example feedback.")

# ---------------------------------------------------------------------------
# Task 3: ticket_router (multi-component)
# ---------------------------------------------------------------------------

QUEUES = ("billing", "technical", "account", "shipping", "general")
PRIORITIES = ("P1", "P2", "P3")
_SECTION_RE = re.compile(r"^###\s*(keyword_rules|priority_rules)\s*$", re.MULTILINE)


def split_router_sections(candidate: Any) -> dict[str, str | None]:
    """Return {"keyword_rules": text|None, "priority_rules": text|None} (None = section absent)."""
    if isinstance(candidate, dict):
        return {k: (_strip_header(candidate[k]) if k in candidate else None)
                for k in ("keyword_rules", "priority_rules")}
    text = str(candidate)
    out: dict[str, str | None] = {"keyword_rules": None, "priority_rules": None}
    marks = list(_SECTION_RE.finditer(text))
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[mk.group(1)] = text[mk.end():end].strip()
    return out


def _strip_header(text: str) -> str:
    return _SECTION_RE.sub("", text, count=1).strip()


def parse_keyword_rules(text: str | None) -> tuple[dict[str, str], str | None]:
    if text is None:
        return {}, None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```\w*\n?|```$", "", body).strip()
    if not body:
        return {}, None
    try:
        rules = json.loads(body)
    except json.JSONDecodeError as e:
        return {}, f"keyword_rules is not valid JSON: {e}"
    if not isinstance(rules, dict):
        return {}, "keyword_rules must be a JSON object {keyword: queue}"
    bad = {k: v for k, v in rules.items() if v not in QUEUES}
    if bad:
        return {}, f"keyword_rules has unknown queues {bad}; allowed: {list(QUEUES)}"
    return {str(k).lower(): v for k, v in rules.items()}, None


_ATOM_RE = re.compile(r'^(not\s+)?contains\s+"([^"]+)"$|^queue\s*(==|!=)\s*(\w+)$')
_RULE_RE = re.compile(r"^if\s+(.+?)\s*->\s*(P[123])$")
_DEFAULT_RE = re.compile(r"^default\s*->\s*(P[123])$")


def parse_priority_rules(text: str | None) -> tuple[list, list[str]]:
    """DSL, one rule per line, first match wins:
       if contains "outage" -> P1
       if contains "refund" and queue == billing -> P2
       if not contains "question" and queue != general -> P2
       default -> P3
    """
    rules, errors = [], []
    if text is None:
        return rules, errors
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("###") else ""
        if not line or line.startswith("```"):
            continue
        m = _DEFAULT_RE.match(line)
        if m:
            rules.append(([], m.group(1), line))
            continue
        m = _RULE_RE.match(line)
        if not m:
            errors.append(f"line {ln} unparseable: {line!r}")
            continue
        atoms, ok = [], True
        for part in re.split(r"\s+and\s+", m.group(1)):
            a = _ATOM_RE.match(part.strip())
            if not a:
                errors.append(f"line {ln} bad condition: {part!r}")
                ok = False
                break
            if a.group(2) is not None:
                atoms.append(("contains", a.group(2).lower(), bool(a.group(1))))
            else:
                atoms.append(("queue", a.group(4), a.group(3) == "!="))
        if ok:
            rules.append((atoms, m.group(2), line))
    return rules, errors


def route_ticket(text: str, kw_rules: dict[str, str], pr_rules: list) -> dict[str, Any]:
    low = text.lower()
    hits = [k for k in kw_rules if k in low]
    # Longest matching keyword wins; ties go to the earlier rule.
    kw = max(hits, key=len) if hits else None
    queue = kw_rules[kw] if kw else "general"
    priority, fired = "P3", "(no priority rule fired; implicit P3)"
    for atoms, prio, src in pr_rules:
        ok = True
        for kind, val, neg in atoms:
            hit = (val in low) if kind == "contains" else (queue == val)
            if hit == neg:
                ok = False
                break
        if ok:
            priority, fired = prio, src
            break
    return {"queue": queue, "priority": priority,
            "queue_rule": f'"{kw}" -> {queue}' if kw else "(no keyword matched; default general)",
            "priority_rule": fired}


class TicketRouterEvaluator:
    accepts_dict = True

    def evaluate(self, candidate, example=None):
        if example is None:
            return 0.0, {"error": "ticket_router needs an example"}
        sec = split_router_sections(candidate)
        kw, kw_err = parse_keyword_rules(sec["keyword_rules"])
        pr, pr_errs = parse_priority_rules(sec["priority_rules"])
        r = route_ticket(example["input"], kw, pr)
        q_ok = 0.0 if kw_err else float(r["queue"] == example["queue"])
        p_ok = float(r["priority"] == example["priority"])
        asi = {
            "ticket": example["input"],
            "expected": f"queue={example['queue']} priority={example['priority']}",
            "got": f"queue={r['queue']} priority={r['priority']}",
            "queue_rule_fired": r["queue_rule"],
            "priority_rule_fired": r["priority_rule"],
        }
        missing = [k for k, v in sec.items() if v is None]
        if missing:
            asi["note"] = f"sections not present in this evaluation (treated as empty): {missing}"
        if kw_err:
            asi["keyword_rules_error"] = kw_err
        if pr_errs:
            asi["priority_rules_errors"] = "; ".join(pr_errs[:5])
        return 0.5 * q_ok + 0.5 * p_ok, asi


ROUTER_SEED = {
    "keyword_rules": '### keyword_rules\n{"refund": "billing"}',
    "priority_rules": "### priority_rules\ndefault -> P3",
}
ROUTER_REFERENCE = {
    "keyword_rules": "### keyword_rules\n" + json.dumps({
        "charged": "billing", "charged twice": "billing", "double charged": "billing", "refund": "billing",
        "invoice": "billing", "receipt": "billing", "subscription fee": "billing", "credit card": "billing",
        "overcharged": "billing", "payment failed": "billing", "billed": "billing",
        "password": "account", "log in": "account", "login": "account", "2fa": "account",
        "two-factor": "account", "locked out": "account", "username": "account", "delete my account": "account",
        "email address on my account": "account", "sign in": "account",
        "package": "shipping", "delivery": "shipping", "tracking": "shipping", "courier": "shipping",
        "shipped": "shipping", "arrived damaged": "shipping", "shipping address": "shipping",
        "error": "technical", "crash": "technical", "api": "technical", "bug": "technical",
        "500": "technical", "timeout": "technical", "sync": "technical", "not loading": "technical",
        "webhook": "technical", "down": "technical",
        "vulnerability": "technical", "fraud": "billing",
    }, indent=1),
    "priority_rules": "\n".join([
        "### priority_rules",
        'if contains "outage" -> P1',
        'if contains "data loss" -> P1',
        'if contains "all users" -> P1',
        'if contains "security" -> P1',
        'if contains "hacked" -> P1',
        'if contains "production is down" -> P1',
        'if contains "charged twice" -> P2',
        'if contains "double charged" -> P2',
        'if contains "overcharged" -> P2',
        'if contains "locked out" -> P2',
        'if contains "fraud" -> P1',
        'if contains "never arrived" -> P2',
        'if contains "urgent" and queue != general -> P2',
        "default -> P3",
    ]),
}
ROUTER_OBJECTIVE = (
    "Improve a support-ticket router made of two components: `keyword_rules` (a JSON object mapping a "
    "lowercase keyword/phrase to a queue) and `priority_rules` (a tiny rule DSL). Maximize the fraction of "
    "tickets routed to the right queue and priority. Keep the '### <component>' header line as the first line."
)
ROUTER_SPEC = """Router semantics:
- Queue: every keyword that appears (case-insensitive substring) in the ticket matches; the LONGEST matching
  keyword wins (ties: earlier in the JSON). No match -> "general". Queues: billing, technical, account, shipping, general.
- Priority DSL, one rule per line, evaluated top to bottom, first match wins, '#' starts a comment:
    if contains "outage" -> P1
    if contains "refund" and queue == billing -> P2
    if not contains "question" and queue != general -> P2
    default -> P3
  Conditions: contains "<text>", not contains "<text>", queue == <q>, queue != <q>, joined with 'and'.
  No rule fires -> P3.
Gold policy: billing = charges/refunds/invoices (money beats account words); account = login/password/2FA/profile;
shipping = deliveries/packages; technical = errors/bugs/API/performance; general = everything else.
P1 = outages, data loss, security incidents, fraud, many users affected; P2 = money disputes, locked-out users,
missing deliveries, urgent requests outside general; P3 = everything else."""


# Minimal background: the router mechanics (needed to write valid rules), not the gold labelling policy.
ROUTER_BACKGROUND = ROUTER_SPEC.split("Gold policy:")[0].rstrip()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class BenchTask:
    name: str
    seed_candidate: dict[str, str]
    reference_candidate: dict[str, str]
    objective: str
    background: str
    make_evaluator: Callable[[list[dict]], Any]
    spec: str = ""  # full hidden spec; pass as background only with --background full
    splits: dict[str, list[dict]] = field(default_factory=dict)

    def evaluator(self):
        return self.make_evaluator(self.splits["train"] + self.splits["val"] + self.splits["test"])


def get_task(name: str) -> BenchTask:
    specs = {
        "regex_emails": (
            {"regex": REGEX_SEED}, {"regex": REGEX_REFERENCE}, REGEX_OBJECTIVE, REGEX_BACKGROUND,
            RegexEmailEvaluator, REGEX_SPEC),
        "parse_dates": (
            {"parser": DATES_SEED}, {"parser": DATES_REFERENCE}, DATES_OBJECTIVE, DATES_BACKGROUND,
            ParseDatesEvaluator, DATES_SPEC),
        "ticket_router": (
            dict(ROUTER_SEED), dict(ROUTER_REFERENCE), ROUTER_OBJECTIVE, ROUTER_BACKGROUND,
            lambda _examples: TicketRouterEvaluator(), ROUTER_SPEC),
    }
    if name not in specs:
        raise KeyError(f"unknown task {name!r}; choose from {list(specs)}")
    seed, ref, obj, bg, make, spec = specs[name]
    return BenchTask(name, seed, ref, obj, bg, make, spec, load_splits(name))


TASK_NAMES = ("regex_emails", "parse_dates", "ticket_router")


def score_split(evaluator, candidate, examples: list[dict]) -> float:
    if not examples:
        return 0.0
    return sum(evaluator.evaluate(candidate, ex)[0] for ex in examples) / len(examples)
