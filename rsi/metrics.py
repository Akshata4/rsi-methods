"""Run-level metrics: how well did ONE Tumble session improve train.py?

All computed from the workspace's experiment history, so they can be
recomputed later from any saved result.json.

    final_best           best metric the session reached
    baseline             the first metric that ran successfully (the agent's own starting point)
    gain_over_baseline   relative improvement from baseline to final_best (+ = better)
    experiments_to_best  1-based index of the experiment that first reached final_best
    wasted_rate          share of experiments that errored or were reverted
                         (the very first successful run is the baseline, not waste)
    failure_rate         share of experiments that errored or timed out
    best_so_far          best metric after each experiment, length = budget
                         (None before the first success; padded if the agent stopped early)
"""


def _better(a: float, b: float, minimize: bool) -> bool:
    return a < b if minimize else a > b


def relative_gain(new: float | None, ref: float | None, minimize: bool) -> float | None:
    """(new - ref) / |ref|, sign-flipped for lower-is-better metrics. + always means better."""
    if new is None or ref is None:
        return None
    gain = (new - ref) / max(abs(ref), 1e-9)
    return -gain if minimize else gain


def run_metrics(history: list[dict], minimize: bool, budget: int) -> dict:
    metrics_seen = [e["metric"] for e in history]
    successes = [m for m in metrics_seen if m is not None]
    n = len(history)

    best = None
    curve: list[float | None] = []
    experiments_to_best = None
    for i, m in enumerate(metrics_seen, start=1):
        if m is not None and (best is None or _better(m, best, minimize)):
            best = m
            experiments_to_best = i
        curve.append(best)
    curve += [best] * (budget - len(curve))  # stopping early = flat curve from there

    failures = sum(m is None for m in metrics_seen)
    kept = sum(bool(e["kept"]) for e in history)
    baseline = successes[0] if successes else None

    return {
        "final_best": best,
        "baseline": baseline,
        "gain_over_baseline": relative_gain(best, baseline, minimize),
        "experiments_to_best": experiments_to_best,
        "experiments_used": n,
        # failed + reverted experiments; kept ones (including the baseline) are not waste
        "wasted_rate": (n - kept) / n if n else None,
        "failure_rate": failures / n if n else None,
        "best_so_far": curve[:budget],
    }
