"""The Improver: the object that recursive self-improvement rewrites.

The three RSI objects (course lab 09.01), mapped onto Tumble:

    task model  = train.py               -- rewritten by the agent loop, many times per run
    improver    = research_prompt        -- how the agent goes about improving train.py
    meta-level  = meta_prompt            -- how the improver rewrites *itself*

An Improver carries both prompts. A child improver is written by its parent's
meta_prompt (see meta.py), and the child may rewrite the meta_prompt too. So
whatever improves the next generation is itself a product of the previous one.
That inheritance chain is what makes this recursive rather than just "someone
tuned the prompt".

What an improver may NOT change is HARNESS_CONTRACT below: the tool rules,
the predictions protocol, the hidden-label scoring and the budget. If an improver could edit
those, it could "improve" by changing how it's graded.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field

# Fixed by the harness. Every improver's research_prompt is wrapped in this.
HARNESS_CONTRACT = """\
Dataset preview:
{preview}

Harness rules (fixed -- these are not yours to change):
- Your workspace has 'dataset.csv' (training data, including the target
  column '{target_column}') and 'test_features.csv' (the same feature columns,
  no target). The preview above is of dataset.csv.
- No train.py exists yet. Write the first version yourself before calling
  run_experiment. It must train on dataset.csv and write 'predictions.csv'
  with one column named 'prediction' and exactly one row per row of
  test_features.csv, in the same order.{metric_note}
- The harness scores predictions.csv against hidden labels for
  test_features.csv: that {metric_name} ({direction}) is your score. You never
  see those labels. Any validation split you make inside dataset.csv is for
  your own judgment only; you may print it, but it is not your score.
- Available Python libraries: numpy, pandas, scikit-learn (plus the standard
  library). Nothing else is installed and nothing can be installed -- for
  gradient boosting use sklearn's HistGradientBoostingClassifier/Regressor.
{protocol}
- Use read_file to see the current train.py and get_history to see every
  attempt so far.
- You have a budget of {round_budget} experiments. When it is used up, or
  further work won't help, write REPORT.md (via write_file) summarizing what
  you tried, what worked, and your final best {metric_name}, then stop by
  replying with plain text and no tool calls.
"""

# How experiments are proposed. "single" is plain Tumble; "candidates" is the
# Budget RSI protocol: several candidate edits per round, the harness picks one.
PROTOCOLS = {
    "single": """\
- Use write_file to propose train.py (or your next change), then call
  run_experiment with a one-sentence description of what you changed.
- run_experiment keeps your change if it beats the best {metric_name} so far
  and reverts it otherwise (or on error). The harness decides, not you.""",
    "candidates": """\
- Experiment 1: write train.py with write_file and call run_experiment with a
  one-sentence description. That is your baseline.
- Every later experiment: write {k} DIFFERENT candidate versions of the whole
  script as candidate_1.py ... candidate_{k}.py (each a complete train.py
  following the rules above, each trying a different idea), then call
  submit_candidates with one short description per candidate, in order.
- The harness selects ONE candidate and runs it -- you do not choose which.
  It keeps it as train.py if it beats the best {metric_name} so far and reverts
  otherwise. Candidate files are deleted after each submit_candidates call.
- Each submit_candidates call uses one experiment from your budget.""",
    "sorsi": """\
- Use write_file to propose train.py (or your next change), then call
  run_experiment with a one-sentence description, an expected_direction
  ("improve", "no_change" or "worse") and a one-sentence expectation saying
  what you expect and why.
- run_experiment keeps your change if it beats the best {metric_name} so far
  and reverts it otherwise (or on error). The harness decides, not you.
- The harness watches your experiments for three anomalies: recurrence (the
  same error again, or the same kind of change reverted twice in a row),
  opposing edits (undoing a recent edit), and expectation mismatch (your
  expectation was clearly wrong). When one is detected, run_experiment is
  blocked until you call record_diagnosis with the root cause (WHY it
  happens, not what to try next), your evidence, and the change that
  addresses that cause. Your next experiment then tests the diagnosis.
- Before diagnosing you may call run_probe with a short Python script to
  investigate (it runs in your workspace, prints its output; at most {probes}
  probes per session). Probes do not use your experiment budget.{lessons}""",
}

# Generation 0 = Tumble's original instructions, unchanged in spirit.
SEED_RESEARCH_PROMPT = """\
You are an autonomous ML research scientist. You've been given a dataset and a
metric to optimize; your job is to write and iteratively improve a training
script, train.py, that models the data as well as possible."""

SEED_META_PROMPT = """\
You improve the instructions of an autonomous ML research agent.

You will receive (1) the agent's current research instructions, (2) these
meta-instructions you are reading now, and (3) evidence: the experiment
histories and final scores from runs that used the current instructions.

Study the evidence. Look for wasted experiments (errors, repeated ideas,
changes that were reverted), missed opportunities, and patterns in what
helped. Then write improved research instructions that would get a better
final score within the same experiment budget on *new, unseen* datasets --
general strategy, not facts about these specific datasets.

You may also rewrite these meta-instructions if you see a better way to do
this improvement step; future generations will use your version.

Reply with ONLY a JSON object:
{"research_prompt": "...", "meta_prompt": "...", "rationale": "what you changed and why, citing the evidence"}"""

MAX_PROMPT_CHARS = 6000

# How each metric must be computed, so every improver is scored the same way.
METRIC_NOTES = {
    "roc_auc": " Each prediction must be the predicted probability (or a\n"
               "  score) that target == 1; the score is ROC-AUC.",
    "accuracy": " Each prediction must be the predicted class label; the score is accuracy.",
    "rmse": " Each prediction must be the predicted value; the score is RMSE.",
}


@dataclass
class Improver:
    research_prompt: str
    meta_prompt: str
    generation: int = 0
    parent_id: str | None = None
    rationale: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @classmethod
    def seed(cls) -> "Improver":
        return cls(SEED_RESEARCH_PROMPT, SEED_META_PROMPT, generation=0, rationale="seed: Tumble's original prompt", id="gen000-seed")

    def system_prompt(self, preview: str, target_column: str, metric_name: str, minimize: bool, round_budget: int,
                      protocol: str = "single", k: int = 3, protocol_vars: dict | None = None) -> str:
        contract = HARNESS_CONTRACT.format(
            protocol=PROTOCOLS[protocol].format(metric_name=metric_name, k=k, **(protocol_vars or {})),
            preview=preview,
            target_column=target_column,
            metric_name=metric_name,
            direction="lower is better" if minimize else "higher is better",
            round_budget=round_budget,
            metric_note=METRIC_NOTES.get(metric_name, ""),
        )
        return f"{self.research_prompt.strip()}\n\n{contract}"

    def validate(self) -> list[str]:
        problems = []
        for name in ("research_prompt", "meta_prompt"):
            text = getattr(self, name)
            if not isinstance(text, str) or len(text.strip()) < 50:
                problems.append(f"{name} is missing or too short")
            elif len(text) > MAX_PROMPT_CHARS:
                problems.append(f"{name} is longer than {MAX_PROMPT_CHARS} chars")
        return problems

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.id}.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Improver":
        with open(path) as f:
            return cls(**json.load(f))
