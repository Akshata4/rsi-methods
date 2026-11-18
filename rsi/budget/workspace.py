"""The Budget RSI candidate protocol on top of Tumble's workspace.

After a normal baseline experiment, each round the agent writes k candidate
scripts (candidate_1.py ... candidate_k.py) and calls submit_candidates. The
harness -- never the agent -- decides which candidate to execute:

    policy "all"     run every candidate (label collection: each idea gets a true outcome)
    policy "scorer"  run only the candidate the scorer predicts is best
    policy "random"  run one candidate chosen at random
    policy "first"   run candidate_1 (the agent's first idea; closest to plain Tumble)

Every executed candidate is logged with its features and outcome to
`label_log` (outside the workspace), which is the scorer's training data.
Each submit_candidates call is one experiment of the budget, whatever the policy.
"""

import json
import os
import random

from rsi.budget.features import edit_features
from rsi.budget.scorer import EditScorer, label_for
from tumble_core.workspace import TOOL_SCHEMAS, Workspace

POLICIES = ("all", "scorer", "random", "first")

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_candidates",
        "description": (
            "Submit the candidate scripts candidate_1.py ... candidate_k.py you wrote this round. The harness "
            "selects one, runs it, and keeps it as train.py only if it beats your best score. Uses one "
            "experiment from your budget. Candidate files are deleted afterwards."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "descriptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One short description per candidate, in order (candidate_1 first).",
                },
            },
            "required": ["descriptions"],
        },
    },
}


class CandidateWorkspace(Workspace):
    def __init__(self, root: str, executor, max_experiments: int, k: int, policy: str,
                 scorer: EditScorer | None = None, label_log: str | None = None, seed: int = 0, context: dict | None = None):
        super().__init__(root, executor, max_experiments)
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if policy == "scorer" and (scorer is None or not scorer.fitted):
            raise ValueError("policy 'scorer' needs a fitted EditScorer")
        self.k, self.policy, self.scorer = k, policy, scorer
        self.label_log = label_log
        self.rng = random.Random(seed)
        self.context = context or {}  # dataset / repeat etc., copied into every label row

    def tool_schemas(self) -> list[dict]:
        return TOOL_SCHEMAS + [SUBMIT_SCHEMA]

    def tool_functions(self) -> dict:
        return super().tool_functions() | {"submit_candidates": self.submit_candidates}

    def _candidate_path(self, i: int) -> str:
        return os.path.join(self.root, f"candidate_{i}.py")

    def _committed_train(self) -> str:
        return self._git("show", "HEAD:train.py").stdout

    def _log(self, rows: list[dict]) -> None:
        if self.label_log:
            with open(self.label_log, "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def _choose(self, features: list[dict]) -> tuple[int, list[float] | None]:
        if self.policy == "scorer":
            predicted = self.scorer.predict(features)
            return max(range(self.k), key=lambda i: predicted[i]), predicted
        if self.policy == "random":
            return self.rng.randrange(self.k), None
        return 0, None  # "first" (and the nominal pick for "all", replaced by the best actual result)

    def submit_candidates(self, descriptions: list[str] | None = None) -> str:
        state = self.load_state()
        iteration = state["iteration"] + 1
        refusal = self._budget_refusal(iteration)
        if refusal:
            return refusal
        if state["best_metric"] is None:
            return ("Error: no baseline yet. Your first experiment must be train.py via run_experiment; "
                    "submit_candidates is for every experiment after that.")
        missing = [f"candidate_{i}.py" for i in range(1, self.k + 1) if not os.path.exists(self._candidate_path(i))]
        if missing:
            return f"Error: missing {', '.join(missing)}. Write all {self.k} candidates, then call submit_candidates again."

        descriptions = list(descriptions or [])
        descriptions += [f"candidate {i + 1}" for i in range(len(descriptions), self.k)]
        parent, parent_score = self._committed_train(), state["best_metric"]
        codes = []
        for i in range(1, self.k + 1):
            with open(self._candidate_path(i)) as f:
                codes.append(f.read())
        features = [edit_features(parent, code, parent_score) for code in codes]
        chosen, predicted = self._choose(features)

        to_run = range(self.k) if self.policy == "all" else [chosen]
        outcomes, folds = {}, {}
        for i in to_run:
            outcomes[i] = self.execute(codes[i])
            folds[i] = self.last_fold_scores
        if self.policy == "all":  # collection mode keeps the best of all candidates
            scored = [i for i in to_run if outcomes[i][0] is not None]
            if scored:
                chosen = (min if state["minimize"] else max)(scored, key=lambda i: outcomes[i][0])

        self._log([{
            **self.context, "round": iteration, "candidate": i + 1, "policy": self.policy,
            "picked": i == chosen, "description": descriptions[i], "features": features[i],
            "predicted": predicted[i] if predicted else None, "parent_score": parent_score,
            "metric": outcomes[i][0], "error": outcomes[i][1],
            "label": label_for(outcomes[i][0], parent_score, state["minimize"]),
        } for i in to_run])

        for i in range(1, self.k + 1):
            os.remove(self._candidate_path(i))

        metric, error = outcomes[chosen]
        extra = {"candidates": self.k, "picked_candidate": chosen + 1, "executed": len(to_run)}
        result = self._keep_or_revert(state, iteration, descriptions[chosen], codes[chosen], metric, error,
                                      executions=len(to_run), extra=extra, fold_scores=folds[chosen])
        lines = [f"Candidate {chosen + 1} of {self.k} was selected and run: {descriptions[chosen]}", result]
        if self.policy == "all":  # collection mode: every candidate ran, so report them all
            for i in to_run:
                if i != chosen:
                    m, e = outcomes[i]
                    lines.append(f"  candidate {i + 1} also ran: " + (f"{m}" if m is not None else f"FAILED ({(e or '')[:200]})"))
        return "\n".join(lines)
