from __future__ import annotations
import re
from typing import Any


def build_reflection_prompt(
    current_text: str,
    component_name: str,
    objective: str,
    asi_entries: list[dict[str, Any]],
    background: str | None = None,
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

    parts.append("""## Instructions
1. Analyze the evaluation results to understand what's working and what's not
2. Identify specific improvements based on the diagnostic feedback
3. Propose an improved version of the component
4. Output your improved version inside a fenced code block (``` ```)
5. The improved version must be a complete replacement, not a diff

Your improved version:""")

    return "\n".join(parts)


def extract_candidate(text: str) -> str:
    match = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
