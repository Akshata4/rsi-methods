"""Detect the three structural anomalies of SO-RSI in a session's experiment trace.

Everything here is deterministic and runs in the harness: the agent cannot
switch it off, and an anomaly is a fact about the trace, not the agent's opinion.

    recurrence   the same error comes back, or the same kind of change is
                 reverted twice in a row ("patching the symptom again")
    opposing     an edit brings back lines that a recent edit removed
                 (flip-flopping: A -> B, then B -> A). Merely modifying a line a
                 recent edit added (e.g. tuning 100 -> 200) is NOT opposing.
    mismatch     the agent predicted an improvement (or no change) and the
                 script crashed or the score clearly moved the other way
"""

import difflib
import re
from dataclasses import dataclass, field

from rsi.budget.features import edit_features

MISMATCH_DROP = 0.01      # expected "improve" but the score fell by more than 1% (relative)
NO_CHANGE_BAND = 0.02     # expected "no_change" but the score moved by more than 2%
OPPOSING_WINDOW = 3       # look this many experiments back for undone edits
EDIT_KINDS = ("model_changed", "ensemble_added", "preprocessing_touched", "log_features_added",
              "log_hyperparams_changed", "search_added")


@dataclass
class Step:
    """One experiment, as the detector sees it."""
    iteration: int
    description: str
    expected: str            # "improve" | "no_change" | "worse"
    metric: float | None
    best_before: float | None
    kept: bool
    error_signature: str | None
    added: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)
    kinds: frozenset = frozenset()


@dataclass
class Anomaly:
    kind: str                # "recurrence" | "opposing" | "mismatch"
    evidence: str
    iterations: list[int]


def error_signature(error: str | None) -> str | None:
    """The last line of a traceback with literal values blanked out, so
    'unseen labels: 'Missing'' and 'unseen labels: 'Other'' count as the same error."""
    if not error:
        return None
    lines = [l.strip() for l in error.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else error
    last = re.sub(r"'[^']*'|\"[^\"]*\"", "<v>", last)
    last = re.sub(r"\b\d+(\.\d+)?\b", "<n>", last)
    return last[:200]


def _significant(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith(("#", "import ", "from ", "print(")) and len(s) > 3


def edit_lines(parent: str, candidate: str) -> tuple[set[str], set[str]]:
    added, removed = set(), set()
    for line in difflib.unified_diff(parent.splitlines(), candidate.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++") and _significant(line[1:]):
            added.add(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---") and _significant(line[1:]):
            removed.add(line[1:].strip())
    return added, removed


def edit_kinds(parent: str, candidate: str) -> frozenset:
    f = edit_features(parent, candidate, 0.0)
    return frozenset(k for k in EDIT_KINDS if f[k] > 0)


def _relative(metric: float, best: float, minimize: bool) -> float:
    gain = (metric - best) / max(abs(best), 1e-9)
    return -gain if minimize else gain


def detect(steps: list[Step], minimize: bool) -> list[Anomaly]:
    """Anomalies raised by the most recent step, given everything before it."""
    if not steps:
        return []
    cur, past = steps[-1], steps[:-1]
    found: list[Anomaly] = []

    # Recurrence (a): the same error signature as an earlier failed experiment.
    if cur.error_signature:
        earlier = [s.iteration for s in past if s.error_signature == cur.error_signature]
        if earlier:
            found.append(Anomaly("recurrence", f"the same error happened again: {cur.error_signature}",
                                 earlier + [cur.iteration]))
    # Recurrence (b): the same kind of change reverted twice in a row (no crash involved).
    elif past and not cur.kept and not past[-1].kept and past[-1].error_signature is None \
            and cur.kinds and cur.kinds == past[-1].kinds:
        found.append(Anomaly("recurrence", "the same kind of change (" + ", ".join(sorted(cur.kinds))
                             + ") was tried twice in a row and reverted both times",
                             [past[-1].iteration, cur.iteration]))

    # Opposing: this edit restores lines that a recent edit removed -- a real reversal.
    # (cur.removed & prev.added alone is just editing a recently added line; the
    # first real run flagged plain hyperparameter tuning that way.)
    for prev in past[-OPPOSING_WINDOW:]:
        restored = cur.added & prev.removed
        if restored:
            example = sorted(restored, key=len)[0][:100]
            found.append(Anomaly("opposing", f"this edit brings back what experiment {prev.iteration} removed "
                                 f"(e.g. `{example}`)", [prev.iteration, cur.iteration]))
            break

    # Expectation mismatch: the prediction was clearly wrong.
    if cur.expected in ("improve", "no_change"):
        if cur.metric is None:
            found.append(Anomaly("mismatch", f"expected '{cur.expected}' but the script failed "
                                 f"({cur.error_signature})", [cur.iteration]))
        elif cur.best_before is not None:
            rel = _relative(cur.metric, cur.best_before, minimize)
            if cur.expected == "improve" and rel < -MISMATCH_DROP:
                found.append(Anomaly("mismatch", f"expected an improvement but the score fell {rel:+.1%} "
                                     f"vs the best so far", [cur.iteration]))
            elif cur.expected == "no_change" and abs(rel) > NO_CHANGE_BAND:
                found.append(Anomaly("mismatch", f"expected no change but the score moved {rel:+.1%}",
                                     [cur.iteration]))
    return found
