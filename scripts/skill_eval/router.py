"""Trigger routing: would an agent's skill router load the candidate for this request?

The router sees only skill names + descriptions (as real agents do), shuffled per item.
"""
from __future__ import annotations

import random
import re
import zlib

from optimize_anything.evaluators.base import EvaluationError
from optimize_anything.llm import PRESETS, LLMClient, LLMError

ROUTER_MODEL = PRESETS["default"]["cheap"]  # claude-haiku-5-5 (llm/models.py)
ROUTER_SYSTEM = ("You are the skill router inside an AI coding agent. You decide which skill's "
                 "instructions get loaded before the agent handles a user request.")
QUESTION = ("Which skill (if any) should be loaded for this user request? "
            "Reply with exactly the skill name or none.")


def build_menu(candidate: dict[str, str], distractors: list[dict], item_id: str) -> list[dict]:
    """Candidate + distractors, shuffled deterministically by item id."""
    menu = [{"name": candidate["name"], "description": candidate["description"]}] + \
           [{"name": d["name"], "description": d["description"]} for d in distractors]
    random.Random(zlib.crc32(item_id.encode())).shuffle(menu)
    return menu


def router_prompt(menu: list[dict], request: str) -> str:
    lines = [f"- {m['name']}: {m['description']}" for m in menu]
    lines.append("- none: no skill applies; handle the request without loading a skill.")
    return ("Available skills:\n" + "\n".join(lines) +
            f"\n\nUser request:\n<request>\n{request}\n</request>\n\n{QUESTION}")


def parse_choice(text: str, names: list[str]) -> str:
    """Map the reply to a menu name, 'none', or '?unparsed:<reply>'."""
    t = text.strip().strip("`*\"'. \n").lower()
    lower = {n.lower(): n for n in names}
    if t in lower:
        return lower[t]
    if t == "none":
        return "none"
    # Longest name mentioned anywhere (avoids prefix collisions).
    found = [n for n in sorted(names, key=len, reverse=True)
             if re.search(rf"(?<![\w-]){re.escape(n.lower())}(?![\w-])", t)]
    if found:
        return found[0]
    if re.search(r"\bnone\b", t):
        return "none"
    return f"?unparsed:{text.strip()[:60]}"


def route(llm: LLMClient, candidate: dict[str, str], distractors: list[dict], item: dict,
          effort: str = "low") -> str:
    menu = build_menu(candidate, distractors, item["id"])
    try:
        resp = llm.complete(router_prompt(menu, item["prompt"]), system=ROUTER_SYSTEM,
                            model=ROUTER_MODEL, max_tokens=50, effort=effort)
    except LLMError as e:
        raise EvaluationError(f"router call failed: {e}") from e
    return parse_choice(resp.text, [m["name"] for m in menu])
