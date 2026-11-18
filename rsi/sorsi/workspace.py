"""SO-RSI on top of Tumble's workspace: expect, detect, diagnose, then patch.

Changes to the experiment loop:
- run_experiment also takes the agent's expectation (expected_direction +
  one sentence), so expectation mismatch can be detected.
- After every experiment the harness runs the anomaly detector. If anything
  fires, run_experiment is BLOCKED until the agent calls record_diagnosis
  with a root cause, its evidence, and the change that addresses the cause.
- run_probe lets the agent investigate with a short script first (dev data
  only -- the hidden labels never touch disk). Probes don't use the
  experiment budget but are capped and counted.
- The next experiment after a diagnosis tests it; the outcome is written to
  this session's memory.jsonl (see memory.py).

Every step, anomaly, probe and diagnosis is also logged to trace.jsonl
(outside the workspace) for the report.
"""

import json
import os
import time

from rsi.sorsi.detector import Step, detect, edit_kinds, edit_lines, error_signature
from tumble_core.workspace import TOOL_SCHEMAS, Workspace

EXPECTED = ("improve", "no_change", "worse")
MAX_PROBES = 6
PROBE_TIMEOUT = 30
PROBE_OUTPUT_CHARS = 4000
MIN_DIAGNOSIS_CHARS = 15

RUN_EXPERIMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_experiment",
        "description": (
            "Run the current train.py as one experiment. The harness scores predictions.csv against hidden labels "
            "and keeps the change if it beats the best score so far, otherwise reverts it. Blocked while an "
            "anomaly is waiting for record_diagnosis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "One sentence: what you changed."},
                "expected_direction": {"type": "string", "enum": list(EXPECTED),
                                       "description": "What you expect this change to do to the score."},
                "expectation": {"type": "string", "description": "One sentence: what you expect and why."},
            },
            "required": ["description", "expected_direction", "expectation"],
        },
    },
}
PROBE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_probe",
        "description": (
            f"Run a short diagnostic Python script in your workspace (it can read dataset.csv, test_features.csv and "
            f"train.py) and see its printed output. Use it to test a hypothesis about WHY something fails. Does not "
            f"use your experiment budget; at most {MAX_PROBES} probes per session, {PROBE_TIMEOUT}s each."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python script to run."},
                "purpose": {"type": "string", "description": "One sentence: what hypothesis this probe tests."},
            },
            "required": ["code", "purpose"],
        },
    },
}
DIAGNOSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_diagnosis",
        "description": "Record your diagnosis of the pending anomaly. Required before the next run_experiment.",
        "parameters": {
            "type": "object",
            "properties": {
                "anomaly": {"type": "string", "enum": ["recurrence", "opposing", "mismatch"]},
                "root_cause": {"type": "string", "description": "WHY it happens (the mechanism), not what to try next."},
                "evidence": {"type": "string", "description": "What you observed (e.g. probe output) that supports it."},
                "next_change": {"type": "string", "description": "The change that addresses the root cause."},
            },
            "required": ["anomaly", "root_cause", "evidence", "next_change"],
        },
    },
}


class SORSIWorkspace(Workspace):
    def __init__(self, root: str, executor, max_experiments: int, session_dir: str, context: dict | None = None):
        super().__init__(root, executor, max_experiments)
        self.trace_log = os.path.join(session_dir, "trace.jsonl")
        self.memory_log = os.path.join(session_dir, "memory.jsonl")
        self.context = context or {}
        self._steps: list[Step] = []
        self._pending: list[dict] = []       # anomalies waiting for a diagnosis
        self._open_diagnosis: dict | None = None  # recorded, not yet tested by an experiment
        self._probes = 0

    def tool_schemas(self) -> list[dict]:
        base = [s for s in TOOL_SCHEMAS if s["function"]["name"] != "run_experiment"]
        return base + [RUN_EXPERIMENT_SCHEMA, PROBE_SCHEMA, DIAGNOSIS_SCHEMA]

    def tool_functions(self) -> dict:
        return super().tool_functions() | {"run_experiment": self.run_experiment, "run_probe": self.run_probe,
                                           "record_diagnosis": self.record_diagnosis}

    def _log(self, event: str, **data) -> None:
        with open(self.trace_log, "a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **self.context, **data}) + "\n")

    def _write_memory(self, entry: dict) -> None:
        with open(self.memory_log, "a") as f:
            f.write(json.dumps({**self.context, **entry}) + "\n")

    def run_experiment(self, description: str, expected_direction: str = "improve", expectation: str = "") -> str:
        if self._pending:
            kinds = ", ".join(a["kind"] for a in self._pending)
            return (f"Blocked: anomaly pending ({kinds}). Investigate with run_probe if useful, then call "
                    f"record_diagnosis before running another experiment.")
        if expected_direction not in EXPECTED:
            return f"Error: expected_direction must be one of {EXPECTED}."
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
        metric, error = self.execute(code)
        message = self._keep_or_revert(state, iteration, description, code, metric, error,
                                       extra={"expected": expected_direction, "expectation": expectation})
        kept = state["history"][-1]["kept"]
        added, removed = edit_lines(parent, code)
        step = Step(iteration, description, expected_direction, metric, best_before, kept, error_signature(error),
                    added, removed, edit_kinds(parent, code) if parent else frozenset())
        self._steps.append(step)
        self._log("step", iteration=iteration, description=description, expected=expected_direction,
                  expectation=expectation, metric=metric, kept=kept, error_signature=step.error_signature)

        if self._open_diagnosis:  # this experiment tests the previous diagnosis
            d = self._open_diagnosis
            confirmed = (metric is not None) if d["trigger_error"] else kept
            self._write_memory({**d, "tested_by": iteration, "confirmed": confirmed})
            self._log("diagnosis_tested", iteration=iteration, anomaly=d["anomaly"], confirmed=confirmed)
            self._open_diagnosis = None

        anomalies = detect(self._steps, state["minimize"])
        if not anomalies:
            return message
        self._pending = [{"kind": a.kind, "evidence": a.evidence, "iterations": a.iterations,
                          "error": step.error_signature} for a in anomalies]
        for a in self._pending:
            self._log("anomaly", iteration=iteration, **a)
        lines = [message, "", "ANOMALY DETECTED -- diagnose before patching:"]
        lines += [f"- {a['kind']}: {a['evidence']}" for a in self._pending]
        lines += ["run_experiment is blocked until you call record_diagnosis. Ask WHY this happens; you may use "
                  "run_probe to test a hypothesis first."]
        return "\n".join(lines)

    def run_probe(self, code: str, purpose: str = "") -> str:
        if self._probes >= MAX_PROBES:
            return f"Refused: probe limit reached ({MAX_PROBES} per session)."
        self._probes += 1
        result = self.executor.run(code, timeout=PROBE_TIMEOUT, script="probe.py")
        if result.timed_out:
            output = f"(probe timed out after {PROBE_TIMEOUT}s)"
        else:
            output = (result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr.strip() else "")).strip()
        output = output[:PROBE_OUTPUT_CHARS] + ("\n[truncated]" if len(output) > PROBE_OUTPUT_CHARS else "")
        self._log("probe", purpose=purpose, n=self._probes, exit_code=result.exit_code, output_chars=len(output))
        return f"Probe {self._probes}/{MAX_PROBES} output:\n{output or '(no output)'}"

    def record_diagnosis(self, anomaly: str, root_cause: str, evidence: str, next_change: str) -> str:
        if not self._pending:
            return "No anomaly is pending; nothing to diagnose. Continue with your next experiment."
        too_short = [n for n, v in (("root_cause", root_cause), ("evidence", evidence), ("next_change", next_change))
                     if len((v or "").strip()) < MIN_DIAGNOSIS_CHARS]
        if too_short:
            return f"Diagnosis rejected: {', '.join(too_short)} too short. Explain the mechanism and your evidence."
        trigger = next((a for a in self._pending if a["kind"] == anomaly), self._pending[0])
        self._open_diagnosis = {
            "anomaly": trigger["kind"], "trigger": trigger["evidence"], "trigger_error": trigger["error"],
            "root_cause": root_cause, "evidence": evidence, "next_change": next_change,
            "diagnosed_after": self._steps[-1].iteration,
        }
        self._log("diagnosis", **self._open_diagnosis, pending=[a["kind"] for a in self._pending])
        self._pending = []
        return ("Diagnosis recorded. Your next experiment tests it: make the change that addresses the root cause, "
                "then call run_experiment.")
