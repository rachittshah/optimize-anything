"""Which candidate the run reports as best (Config.promotion).

"max": argmax of mean val score (original behaviour).
"paired_se": a challenger replaces the incumbent only if its mean paired per-example gain beats
z * SE of those differences, and (optionally) no single example drops by more than max_drop.
Frontier / parent selection never use this; it only changes the returned best.
"""
from __future__ import annotations
import math


def paired_gain(new: dict[str, float], incumbent: dict[str, float]) -> tuple[float, float, list[float]]:
    """(mean diff, SE of diffs, diffs) over the keys both candidates were scored on."""
    common = sorted(set(new) & set(incumbent))
    diffs = [new[k] - incumbent[k] for k in common]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, []
    mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0, diffs
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(var / n), diffs


def should_promote(
    new: dict[str, float],
    incumbent: dict[str, float],
    z: float = 1.0,
    max_drop: float | None = 0.25,
) -> bool:
    """True if `new` beats `incumbent` by more than z*SE of the paired differences.

    Zero SE (one common example, or constant differences) reduces to "mean gain > 0".
    No common examples -> never promote.
    """
    mean, se, diffs = paired_gain(new, incumbent)
    if not diffs:
        return False
    if max_drop is not None and min(diffs) < -max_drop:
        return False
    return mean > 0 and mean > z * se
