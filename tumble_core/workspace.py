"""The agent's git-backed workspace and its tools (from tumble/tools.py).

Changes from Tumble:
- the experiment budget is a constructor argument, not a module constant, so
  two improvers can be compared under exactly the same budget;
- W&B run logging is removed (the RSI layer keeps its own records);
- any executor with run(code, timeout) works (LocalExecutor or Tumble's SandboxSession).

The keep/revert rule is still deterministic and owned by the harness, not the
LLM. That matters even more for RSI: an improver may rewrite its own strategy,
but it must never be able to rewrite how it is scored.

Scoring is harness-side (change from Tumble, where train.py printed its own
metric). At init the dataset is split into a DEV part (dataset.csv, with
labels) and a HIDDEN part whose features are given as test_features.csv. Its
labels are kept only in this object's memory, never on disk, so nothing the
agent's code does -- e.g. target-encoding over the whole file before its own
split -- can leak them. train.py writes predictions.csv for the hidden rows and
the harness scores it. (Found the hard way: an agent target-encoded over all
rows and "improved" readmission AUC from 0.60 to a fake 0.80.)
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

RUN_TIMEOUT_SECONDS = 60
HIDDEN_FRACTION = 0.2
HIDDEN_SPLIT_SEED = 0
DEV_FILE, FEATURES_FILE, PREDICTIONS_FILE = "dataset.csv", "test_features.csv", "predictions.csv"

# metric name -> score(y_true, predictions)
METRICS = {
    "roc_auc": roc_auc_score,
    "accuracy": lambda y, p: accuracy_score(y, np.asarray(p).round().astype(int)),
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
}


def split_hidden(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The one dev/hidden split everybody uses (harness, reference baselines).
    Stratified for classification targets so the hidden set keeps the class balance."""
    y = df[target]
    stratify = y if y.nunique() <= 20 else None
    dev, hidden = train_test_split(df, test_size=HIDDEN_FRACTION, random_state=HIDDEN_SPLIT_SEED, stratify=stratify)
    return dev.reset_index(drop=True), hidden.reset_index(drop=True)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full text contents of a file in the experiment workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace root, e.g. 'train.py'."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file's full contents in the experiment workspace. Use this to propose your next change to train.py before calling run_experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root, e.g. 'train.py'."},
                    "content": {"type": "string", "description": "Full text content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in the experiment workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative path to list. Use '.' for the workspace root."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": (
                "Run the current train.py as one experiment. The harness scores the predictions.csv it writes "
                "against hidden labels and automatically keeps the change (git commit) if it beats the best score "
                "so far, or reverts it (git checkout) if it doesn't or the script errors. This decision is NOT "
                "yours. Hard-capped per session -- calling it after the budget is used returns a refusal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "One sentence describing what you changed in this attempt."},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "Return every experiment so far: iteration, metric, kept or reverted, description.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class Workspace:
    def __init__(self, root: str, executor, max_experiments: int):
        self.root = os.path.abspath(root)
        self.executor = executor
        self.max_experiments = max_experiments
        self.state_path = os.path.join(self.root, "state.json")
        self.train_script = os.path.join(self.root, "train.py")
        self.dev_path = os.path.join(self.root, DEV_FILE)
        self._hidden_labels: np.ndarray | None = None  # memory only -- never written to disk
        self._metric_fn = None

    def init(self, dataset_csv_path: str, target_column: str, metric_name: str, minimize: bool) -> None:
        if metric_name not in METRICS:
            raise ValueError(f"unsupported metric {metric_name!r}; supported: {sorted(METRICS)}")
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root)
        dev, hidden = split_hidden(pd.read_csv(dataset_csv_path), target_column)
        dev.to_csv(self.dev_path, index=False)
        hidden.drop(columns=[target_column]).to_csv(os.path.join(self.root, FEATURES_FILE), index=False)
        self._hidden_labels = hidden[target_column].to_numpy()
        self._metric_fn = METRICS[metric_name]
        self._git("init")
        self._save_state(
            {
                "best_metric": None,
                "iteration": -1,
                "metric_name": metric_name,
                "minimize": minimize,
                "target_column": target_column,
                "history": [],
                "executions": 0,  # train.py runs actually executed (Budget RSI can run fewer than it considers)
            }
        )

    def tool_schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def tool_functions(self) -> dict:
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_dir": self.list_dir,
            "run_experiment": self.run_experiment,
            "get_history": self.get_history,
        }

    # --- state ---------------------------------------------------------

    def load_state(self) -> dict:
        with open(self.state_path) as f:
            return json.load(f)

    def _save_state(self, state: dict) -> None:
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _resolve(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.root, path))
        if os.path.commonpath([full, self.root]) != self.root:
            raise ValueError(f"'{path}' resolves outside the workspace ({self.root})")
        return full

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)

    # --- file tools ----------------------------------------------------

    def read_file(self, path: str) -> str:
        try:
            with open(self._resolve(path)) as f:
                return f.read()
        except (ValueError, OSError) as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        if os.path.basename(path) in {"state.json", DEV_FILE, FEATURES_FILE}:
            return f"Refused: {path} is managed by the harness."
        try:
            with open(self._resolve(path), "w") as f:
                f.write(content)
            return f"Wrote {len(content)} characters to {path}"
        except (ValueError, OSError) as e:
            return f"Error writing {path}: {e}"

    def list_dir(self, path: str) -> str:
        try:
            return "\n".join(sorted(p for p in os.listdir(self._resolve(path)) if p != ".git"))
        except (ValueError, OSError) as e:
            return f"Error listing {path}: {e}"

    # --- experiments ---------------------------------------------------

    def _score_predictions(self) -> tuple[float | None, str | None]:
        """Score predictions.csv against the hidden labels. Returns (metric, error)."""
        path = os.path.join(self.root, PREDICTIONS_FILE)
        if not os.path.exists(path):
            return None, f"train.py did not write {PREDICTIONS_FILE}"
        try:
            preds = pd.read_csv(path)
        except Exception as e:
            return None, f"could not read {PREDICTIONS_FILE}: {e}"
        if "prediction" not in preds.columns:
            return None, f"{PREDICTIONS_FILE} needs a 'prediction' column (found {list(preds.columns)})"
        if len(preds) != len(self._hidden_labels):
            return None, f"{PREDICTIONS_FILE} has {len(preds)} rows; {FEATURES_FILE} has {len(self._hidden_labels)}"
        p = pd.to_numeric(preds["prediction"], errors="coerce")
        if p.isna().any():
            return None, f"{PREDICTIONS_FILE} has {int(p.isna().sum())} missing or non-numeric predictions"
        try:
            return float(self._metric_fn(self._hidden_labels, p.to_numpy())), None
        except Exception as e:
            return None, f"scoring failed: {e}"

    def _record(self, state: dict, iteration: int, description: str, metric: float | None, kept: bool,
                error: str | None = None, extra: dict | None = None) -> None:
        entry = {"iteration": iteration, "metric": metric, "kept": kept, "description": description}
        if error is not None:
            entry["error"] = error
        if extra:
            entry.update(extra)
        state["iteration"] = iteration
        state["history"].append(entry)
        self._save_state(state)

    def execute(self, code: str) -> tuple[float | None, str | None]:
        """Run `code` as train.py and score its predictions on the hidden labels.
        Returns (metric, error). Does NOT commit or revert anything; callers decide."""
        stale = os.path.join(self.root, PREDICTIONS_FILE)
        if os.path.exists(stale):
            os.remove(stale)  # never score a previous experiment's predictions
        result = self.executor.run(code, timeout=RUN_TIMEOUT_SECONDS)
        if result.timed_out:
            return None, "timed out"
        if result.exit_code != 0:
            return None, result.stderr.strip()[-500:] or f"exit code {result.exit_code}"
        return self._score_predictions()

    def _budget_refusal(self, iteration: int) -> str | None:
        if iteration >= self.max_experiments:
            return (
                f"Experiment budget exhausted ({self.max_experiments}/{self.max_experiments} used). "
                "Write your final report to REPORT.md now and stop."
            )
        return None

    def _keep_or_revert(self, state: dict, iteration: int, description: str, code: str,
                        metric: float | None, error: str | None, executions: int = 1, extra: dict | None = None) -> str:
        """Apply the deterministic keep/revert rule to one scored version of train.py."""
        metric_name = state["metric_name"]
        state["executions"] = state.get("executions", 0) + executions
        if metric is None:
            self._git("checkout", "--", "train.py")
            self._record(state, iteration, description, None, False, error, extra)
            if error == "timed out":
                return f"Iteration {iteration}: TIMED OUT after {RUN_TIMEOUT_SECONDS}s. Change reverted."
            return f"Iteration {iteration}: FAILED ({error}). Change reverted; train.py is back to the last kept version."

        best = state["best_metric"]
        kept = best is None or (metric < best if state["minimize"] else metric > best)
        if kept:
            with open(self.train_script, "w") as f:
                f.write(code)
            self._git("add", "train.py")
            self._git(
                "-c", "user.name=rsi-harness", "-c", "user.email=rsi@local",
                "commit", "-m", f"iteration {iteration}: {metric_name}={metric} -- {description}",
            )
            state["best_metric"] = metric
        else:
            self._git("checkout", "--", "train.py")
        self._record(state, iteration, description, metric, kept, None, extra)

        verdict = "NEW BEST -- change kept and committed" if kept else f"did not beat the best ({best}) -- change reverted"
        return f"Iteration {iteration}: {metric_name}={metric}. {verdict}. Current best: {state['best_metric']}."

    def run_experiment(self, description: str) -> str:
        state = self.load_state()
        iteration = state["iteration"] + 1
        refusal = self._budget_refusal(iteration)
        if refusal:
            return refusal
        if not os.path.exists(self.train_script):
            return "Error: train.py does not exist yet. Write it with write_file first."
        with open(self.train_script) as f:
            code = f.read()
        metric, error = self.execute(code)
        return self._keep_or_revert(state, iteration, description, code, metric, error)

    def get_history(self) -> str:
        state = self.load_state()
        best = state["best_metric"] if state["best_metric"] is not None else "none yet"
        lines = [f"Best {state['metric_name']} so far: {best} (after {state['iteration'] + 1}/{self.max_experiments} attempts)", ""]
        for e in state["history"]:
            m = e["metric"] if e["metric"] is not None else f"FAILED ({e.get('error', 'unknown')})"
            lines.append(f"  iter {e['iteration']}: {m} ({'kept' if e['kept'] else 'reverted'}) -- {e['description']}")
        return "\n".join(lines)
