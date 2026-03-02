# engine/src/optimize_anything/evaluators/llm_judge.py
from __future__ import annotations
import re
from typing import Any

import anthropic


class LLMJudgeEvaluator:
    def __init__(
        self,
        criteria: str,
        model: str = "claude-sonnet-4-6",
        max_score: float = 10.0,
    ):
        self.criteria = criteria
        self.model = model
        self.max_score = max_score
        self.client = anthropic.Anthropic()

    def evaluate(
        self, candidate: str, example: dict[str, Any] | None = None
    ) -> tuple[float, dict[str, Any]]:
        prompt = f"""Evaluate the following candidate artifact against the given criteria.

## Criteria
{self.criteria}

## Candidate
{candidate}
"""
        if example:
            prompt += f"\n## Context\n{example}\n"

        prompt += f"""
## Instructions
1. Analyze the candidate against each criterion
2. Provide specific feedback
3. End with exactly: SCORE: X/{self.max_score:.0f}

Your evaluation:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            match = re.search(r"SCORE:\s*([\d.]+)", text)
            if match:
                raw_score = float(match.group(1))
                score = raw_score / self.max_score  # normalize to [0, 1]
            else:
                score = 0.0

            return score, {"reasoning": text, "raw_score": raw_score if match else None}

        except Exception as e:
            return 0.0, {"error": str(e)}
