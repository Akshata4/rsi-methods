"""RSIFlow on top of Tumble's workspace: every experiment belongs to a stage.

The tools are plain Tumble's. What changes:
- the orchestrator (router.py) assigns each experiment's stage; the result of
  every run_experiment ends with "NEXT STAGE -> <stage>: <instruction>";
- each experiment is tagged with its stage, its relative gain, and whether
  the edit actually matched the stage (from the same static edit features
  Budget RSI uses), and logged to flow.jsonl outside the workspace.

Those per-stage statistics are the execution feedback the meta-agent uses to
evolve the workflow between generations.
"""

import json
import os
import time

from rsi.budget.features import edit_features
from rsi.flow.router import FIX, Outcome, Router
from tumble_core.workspace import Workspace

# Which edit features show that an experiment did the kind of work its stage asked for.
STAGE_EVIDENCE = {
    "preprocess": ("preprocessing_touched",),
    "features": ("log_features_added",),
    "model": ("model_changed",),
    "tune": ("log_hyperparams_changed",),
    "ensemble": ("ensemble_added",),
}


class FlowWorkspace(Workspace):
    def __init__(self, root: str, executor, max_experiments: int, workflow: dict, session_dir: str,
                 context: dict | None = None):
        super().__init__(root, executor, max_experiments)
        self.workflow = workflow
        self.router = Router(workflow)
        self.flow_log = os.path.join(session_dir, "flow.jsonl")
        self.context = context or {}
        self.outcomes: list[Outcome] = []
        self.stage, self.reason = self.router.first()

    def _log(self, event: str, **data) -> None:
        with open(self.flow_log, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **self.context, **data}) + "\n")

    def first_directive(self) -> str:
        return f"{self.stage}: {self.workflow['stages'][self.stage]}"

    def run_experiment(self, description: str) -> str:
        state = self.load_state()
        iteration = state["iteration"] + 1
        refusal = self._budget_refusal(iteration)
        if refusal:
            return refusal
        if not os.path.exists(self.train_script):
            return "Error: train.py does not exist yet. Write it with write_file first."

        parent = self._git("show", "HEAD:train.py").stdout
        with open(self.train_script) as f:
            code = f.read()
        best_before = state["best_metric"]
        stage = self.stage
        feats = edit_features(parent, code, best_before or 0.0) if parent else {}
        expected = STAGE_EVIDENCE.get(stage)
        on_stage = None if expected is None or not parent else any(feats.get(k, 0) > 0 for k in expected)

        metric, error = self.execute(code)
        gain = None
        if metric is not None and best_before is not None:
            gain = (metric - best_before) / max(abs(best_before), 1e-9)
            if state["minimize"]:
                gain = -gain
        message = self._keep_or_revert(state, iteration, description, code, metric, error,
                                       extra={"stage": stage, "on_stage": on_stage, "gain": gain})

        self.outcomes.append(Outcome(stage, metric is None, gain, kept=state["history"][-1]["kept"]))
        self.stage, self.reason = self.router.next(self.outcomes)
        self._log("experiment", iteration=iteration, stage=stage, metric=metric, gain=gain,
                  kept=state["history"][-1]["kept"], crashed=metric is None, on_stage=on_stage,
                  description=description, next_stage=self.stage, reason=self.reason)
        instruction = self.workflow["stages"][self.stage]
        return (f"{message}\nNEXT STAGE -> {self.stage}: {instruction}\n(orchestrator: {self.reason})")


def flow_protocol_vars(workflow: dict, first_directive: str) -> dict:
    from rsi.flow.workflow import render_stages
    return {"stages": render_stages(workflow), "first_stage": first_directive, "fix_stage": FIX}
