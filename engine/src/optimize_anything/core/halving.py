"""Paired successive halving for full val evaluation (Config.val_eval = "halving").

A child is evaluated on growing prefixes of a fixed shuffled val order (|V|/4, |V|/2, |V|).
At each non-final rung it is compared with its parent on the same examples; a clearly worse
child that isn't strictly better than the frontier anywhere is dropped early.
"""
from __future__ import annotations
import math
import random
from typing import Callable


def rung_sizes(n: int) -> list[int]:
    """Strictly increasing prefix sizes ending at n, e.g. 10 -> [3, 5, 10], 2 -> [1, 2]."""
    if n <= 0:
        return []
    sizes: list[int] = []
    for s in (math.ceil(n / 4), math.ceil(n / 2), n):
        if not sizes or s > sizes[-1]:
            sizes.append(s)
    return sizes


def shuffled_order(n: int, seed: int) -> list[int]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def should_stop_early(
    child: dict[str, float],
    parent: dict[str, float],
    frontier_best: Callable[[str], float],
    margin: float = 0.0,
) -> bool:
    """Reject if child_mean < parent_mean - margin on common keys AND the child is not strictly
    better than the frontier best on any example it was scored on. No common keys -> keep going."""
    common = [k for k in child if k in parent]
    if not common:
        return False
    child_mean = sum(child[k] for k in common) / len(common)
    parent_mean = sum(parent[k] for k in common) / len(common)
    if child_mean >= parent_mean - margin:
        return False
    return not any(child[k] > frontier_best(k) for k in child)
