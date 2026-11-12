from __future__ import annotations
import re
from typing import Any


REWRITE_INSTRUCTIONS = """## Instructions
1. Analyze the evaluation results to understand what's working and what's not
2. Identify specific improvements based on the diagnostic feedback
3. Propose an improved version of the component
4. Output your improved version inside a fenced code block (``` ```)
5. The improved version must be a complete replacement, not a diff

Your improved version:"""

# SEPO-style minimal edit: fix only what the failures implicate, keep the rest verbatim.
SECTION_EDIT_INSTRUCTIONS = """## Instructions
1. Find the specific parts of the current version that the failing examples implicate
2. Make the smallest targeted edit that fixes those failures
3. Keep every other line exactly as it is (verbatim: same wording, order and formatting)
4. Do not restructure, reformat or rephrase parts unrelated to the failures
5. Output the FULL updated component (not a diff) inside one fenced code block (``` ```)

Your edited version:"""

COMPRESS_INSTRUCTIONS = """## Instructions
1. Make the component shorter while preserving its behaviour on the examples above
2. Remove redundancy, repetition and instructions that do not affect the results
3. Keep every rule that a passing example depends on, and fix failures if a fix is short
4. Output the FULL shortened component (not a diff) inside one fenced code block (``` ```)

Your compressed version:"""

OPERATOR_INSTRUCTIONS = {
    "rewrite": REWRITE_INSTRUCTIONS,
    "section_edit": SECTION_EDIT_INSTRUCTIONS,
    "compress": COMPRESS_INSTRUCTIONS,
}


def build_reflection_prompt(
    current_text: str,
    component_name: str,
    objective: str,
    asi_entries: list[dict[str, Any]],
    background: str | None = None,
    operator: str = "rewrite",
    rejected_edits: list[dict[str, Any]] | None = None,
) -> str:
    parts = [f"# Task: Improve the `{component_name}` component\n"]

    if objective:
        parts.append(f"## Objective\n{objective}\n")

    if background:
        parts.append(f"## Background\n{background}\n")

    parts.append(f"## Current Version\n```\n{current_text}\n```\n")

    parts.append("## Evaluation Results (Actionable Side Information)\n")
    for i, entry in enumerate(asi_entries):
        parts.append(f"### Example {i + 1} — Score: {entry['score']}")
        feedback = entry.get("feedback", {})
        if isinstance(feedback, dict):
            for k, v in feedback.items():
                parts.append(f"- **{k}**: {v}")
        else:
            parts.append(f"- {feedback}")
        parts.append("")

    if rejected_edits:
        # ~~~ fences so fenced-code extractors never mistake a diff for the artifact
        parts.append("## Previously tried edits that did not help — don't repeat them\n")
        for i, e in enumerate(rejected_edits):
            label = e.get("note") or (
                f"minibatch score {e.get('child_score'):.3f} vs parent {e.get('parent_score'):.3f}"
                if e.get("child_score") is not None and e.get("parent_score") is not None
                else "rejected")
            parts.append(f"### Rejected edit {i + 1} ({label})\n~~~diff\n{e.get('diff', '')}\n~~~\n")
    if "```" in current_text:  # only then: keeps prompts for fence-free artifacts unchanged
        parts.append("Note: the component itself contains ``` code fences. Wrap your whole "
                     "answer in a LONGER fence (```` four backticks) so inner fences survive.\n")
    parts.append(OPERATOR_INSTRUCTIONS.get(operator, REWRITE_INSTRUCTIONS))

    return "\n".join(parts)


_OPEN_FENCE = re.compile(r"^[ \t]*(`{3,})[\w+.-]*[ \t]*$")


def extract_candidate(text: str) -> str:
    """Body of the answer's code block, keeping fences nested inside it.

    Opening fence = first line that is only backticks (3+) plus an optional language tag.
    - 4+ backticks: closes at the next bare fence of at least that length (inner ``` survive).
    - 3 backticks: closes at the LAST bare ``` line, so an artifact with inner fences isn't cut
      at its first inner closing fence.
    Falls back to the first inline ```...``` match, then to the whole text.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _OPEN_FENCE.match(line)
        if not m:
            continue
        n = len(m.group(1))
        close = re.compile(rf"^[ \t]*`{{{n},}}[ \t]*$")
        closers = [j for j in range(i + 1, len(lines)) if close.match(lines[j])]
        if closers:
            end = closers[0] if n > 3 else closers[-1]
            return "\n".join(lines[i + 1:end]).strip()
        break
    match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
