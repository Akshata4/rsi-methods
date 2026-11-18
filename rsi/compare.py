"""Paired comparison of two improvers on the same held-out datasets.

Course lab 09.05: compare improvers under the same conditions. Both get
the same datasets, the same experiment budget, the same model, and the same
fixed harness contract. Datasets use different metrics and scales, so we
compare per dataset (relative gain of challenger over champion) and never
average raw scores.
"""

from dataclasses import dataclass

from rsi.run_task import RunResult

FAILED_RUN_PENALTY = -1.0  # a run that never produced a metric counts as a 100% loss


@dataclass
class Comparison:
    per_dataset: dict[str, float]  # dataset -> relative gain of challenger (+ = better)
    mean_gain: float
    wins: int
    n: int
    accepted: bool
    reason: str


def relative_gain(champion: RunResult, challenger: RunResult) -> float:
    c, p = challenger.best_metric, champion.best_metric
    if c is None and p is None:
        return 0.0
    if c is None:
        return FAILED_RUN_PENALTY
    if p is None:
        return -FAILED_RUN_PENALTY
    gain = (c - p) / max(abs(p), 1e-9)
    return -gain if champion.minimize else gain


def compare(champion: list[RunResult], challenger: list[RunResult], min_gain: float = 0.01) -> Comparison:
    """Accept the challenger only if it is better on average by at least
    `min_gain` (relative) AND wins on a majority of datasets. The margin
    stops us from promoting a child that only got lucky by noise."""
    champ_by = {r.dataset: r for r in champion}
    per = {r.dataset: relative_gain(champ_by[r.dataset], r) for r in challenger}
    n = len(per)
    wins = sum(g > 0 for g in per.values())
    mean = sum(per.values()) / n if n else 0.0
    accepted = mean >= min_gain and wins > n / 2
    reason = (
        f"mean relative gain {mean:+.2%} (need >= {min_gain:+.2%}), "
        f"wins {wins}/{n} (need majority) -> {'ACCEPT' if accepted else 'REJECT'}"
    )
    return Comparison(per, mean, wins, n, accepted, reason)
