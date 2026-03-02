# engine/src/optimize_anything/core/candidate.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Candidate:
    components: dict[str, str]
    parent_ids: list[int] = field(default_factory=list)
    tag: str = ""

    def copy(self) -> Candidate:
        return Candidate(
            components=dict(self.components),
            parent_ids=list(self.parent_ids),
            tag=self.tag,
        )
