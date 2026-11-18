"""The orchestrator: choose the next experiment's stage from the results so far.

Deterministic and harness-side, so routing is reproducible and the agent
can't skip it. Adaptive routing targets the current bottleneck:

    no working script yet    -> fix_errors (or a fresh baseline after too many fixes)
    the last run crashed     -> fix_errors, up to max_fix_attempts in a row,
                                then abandon the idea and move to the next stage
    a fix worked             -> resume the stage the crash interrupted
    the stage just improved  -> stay (stay_on_gain)
    the stage has stalled    -> next non-stalled stage in `order`
    every stage has stalled  -> back to the stage with the best mean gain so far

With rules.adaptive = False it ignores results and walks `order` in a loop
(still routing crashes to fix_errors) -- the ablation that shows what
adaptive routing itself adds.
"""

from dataclasses import dataclass

BASELINE, FIX = "baseline", "fix_errors"


@dataclass
class Outcome:
    stage: str
    crashed: bool
    gain: float | None     # relative gain over the best before this experiment (None if crashed or no best yet)
    kept: bool | None = None  # the harness's (noise-aware) keep decision, when available


class Router:
    def __init__(self, workflow: dict):
        self.order = list(workflow["order"])
        self.r = workflow["rules"]

    def first(self) -> tuple[str, str]:
        return BASELINE, "every session starts with a baseline"

    def _improved(self, o: Outcome) -> bool:
        """Progress = the harness kept it AND it cleared min_gain. Under version-2 scoring
        'kept' already means consistent across folds, so noise no longer counts as progress
        (version 1's router stayed in a stage on noise-level gains)."""
        if o.gain is None or o.gain <= self.r["min_gain"]:
            return False
        return o.kept is not False

    def _work_stages(self, outcomes: list[Outcome]) -> list[Outcome]:
        return [o for o in outcomes if o.stage not in (BASELINE, FIX)]

    def _stalled(self, outcomes: list[Outcome], stage: str) -> bool:
        tries = [o for o in outcomes if o.stage == stage]
        k = self.r["plateau_patience"]
        return len(tries) >= k and not any(self._improved(o) for o in tries[-k:])

    def _after(self, stage: str | None, outcomes: list[Outcome]) -> tuple[str, str]:
        """Next non-stalled stage in order after `stage` (cyclic)."""
        start = self.order.index(stage) + 1 if stage in self.order else 0
        for i in range(len(self.order)):
            cand = self.order[(start + i) % len(self.order)]
            if cand != stage and not self._stalled(outcomes, cand):
                return cand, f"moving on to {cand}"
        gains: dict[str, list[float]] = {}
        for o in self._work_stages(outcomes):
            gains.setdefault(o.stage, []).append(o.gain if o.gain is not None else -1.0)
        if gains:
            best = max(gains, key=lambda s: sum(gains[s]) / len(gains[s]))
            return best, f"every stage has stalled; returning to the most productive one ({best})"
        return self.order[0], "starting the workflow"

    def next(self, outcomes: list[Outcome]) -> tuple[str, str]:
        last = outcomes[-1]
        crash_streak = 0
        for o in reversed(outcomes):
            if not o.crashed:
                break
            crash_streak += 1
        have_working = any(not o.crashed for o in outcomes)

        if not have_working:
            if crash_streak <= self.r["max_fix_attempts"]:
                return FIX, "there is no working script yet"
            return BASELINE, "too many failed fixes; write a simpler baseline from scratch"

        work = self._work_stages(outcomes)
        current = work[-1].stage if work else None

        if not self.r["adaptive"]:  # ablation: crashes still go to fix_errors, otherwise walk the order
            if last.crashed and crash_streak <= self.r["max_fix_attempts"]:
                return FIX, "the last experiment crashed"
            stage = self.order[len(work) % len(self.order)]
            return stage, "fixed order"

        if last.crashed:
            if crash_streak <= self.r["max_fix_attempts"]:
                return FIX, "the last experiment crashed"
            return self._after(current, outcomes)[0], "repeated crashes; abandoning that idea"
        if last.stage == FIX:
            if current:
                return current, f"the fix worked; resuming {current}"
            return self._after(None, outcomes)
        if last.stage == BASELINE:
            return self._after(None, outcomes)
        if self.r["stay_on_gain"] and self._improved(last):
            return last.stage, f"{last.stage} just improved the score; keep pushing it"
        if self._stalled(outcomes, last.stage):
            nxt, why = self._after(last.stage, outcomes)
            return nxt, f"{last.stage} has stalled; {why}"
        return last.stage, f"{last.stage} has not stalled yet"
