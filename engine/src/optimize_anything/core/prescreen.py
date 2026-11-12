"""Cheap checks that reject a proposal before (or instead of) spending evals on it.

Used when Config.prescreen is on:
- exact duplicate (whitespace-normalized) of the parent or any existing candidate
- near duplicate of the parent (difflib ratio >= threshold, ShinkaEvolve-style)
- identical per-example output signature on the minibatch (score + ASI hash)
"""
from __future__ import annotations
import difflib
import hashlib
import json
from typing import Any


def normalize(text: str) -> str:
    return " ".join(text.split())


def components_key(components: dict[str, str]) -> str:
    """Whitespace-insensitive identity of a full candidate."""
    return json.dumps({k: normalize(v) for k, v in sorted(components.items())})


def is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    """difflib ratio >= threshold. quick_ratio is an upper bound, so it filters cheaply."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
        return False
    return sm.ratio() >= threshold


def output_signature(results: list[tuple[float, dict] | None]) -> list[Any]:
    """Per-example (score, sha256(asi)) signature; failed evals are None."""
    sig: list[Any] = []
    for r in results:
        if r is None:
            sig.append(None)
            continue
        asi = json.dumps(r[1], sort_keys=True, default=str)
        sig.append((r[0], hashlib.sha256(asi.encode()).hexdigest()))
    return sig
