"""The agent's git-backed workspace and its tools (from tumble/tools.py).

Changes from Tumble:
- the experiment budget is a constructor argument, not a module constant, so
  two improvers can be compared under exactly the same budget;
- W&B run logging is removed (the RSI layer keeps its own records);
- any executor with run(code, timeout) works (LocalExecutor or Tumble's SandboxSession).

The keep/revert rule is still deterministic and owned by the harness, not the
LLM. That matters even more for RSI: an improver may rewrite its own strategy,
but it must never be able to rewrite how it is scored.
"""

import json
import os
import shutil
import subprocess

RUN_TIMEOUT_SECONDS = 60

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
                "Run the current train.py as one experiment. Parses the METRIC line it prints and automatically "
                "keeps the change (git commit) if it beats the best metric so far, or reverts it (git checkout) if "
                "it doesn't or the script errors. This decision is NOT yours. Hard-capped per session -- calling it "
                "after the budget is used returns a refusal."
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

    def init(self, dataset_csv_path: str, target_column: str, metric_name: str, minimize: bool) -> None:
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root)
        shutil.copyfile(dataset_csv_path, os.path.join(self.root, "dataset.csv"))
        self._git("init")
        self._save_state(
            {
                "best_metric": None,
                "iteration": -1,
                "metric_name": metric_name,
                "minimize": minimize,
                "target_column": target_column,
                "history": [],
            }
        )

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
        if os.path.basename(path) in {"state.json", "dataset.csv"}:
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

    @staticmethod
    def _parse_metric(stdout: str) -> float | None:
        metric = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("METRIC:"):
                try:
                    metric = float(line.split("METRIC:", 1)[1].strip())
                except ValueError:
                    continue
        return metric

    def _record(self, state: dict, iteration: int, description: str, metric: float | None, kept: bool, error: str | None = None) -> None:
        entry = {"iteration": iteration, "metric": metric, "kept": kept, "description": description}
        if error is not None:
            entry["error"] = error
        state["iteration"] = iteration
        state["history"].append(entry)
        self._save_state(state)

    def run_experiment(self, description: str) -> str:
        state = self.load_state()
        iteration = state["iteration"] + 1
        metric_name = state["metric_name"]

        if iteration >= self.max_experiments:
            return (
                f"Experiment budget exhausted ({self.max_experiments}/{self.max_experiments} used). "
                "Write your final report to REPORT.md now and stop."
            )
        if not os.path.exists(self.train_script):
            return "Error: train.py does not exist yet. Write it with write_file first."

        with open(self.train_script) as f:
            code = f.read()
        result = self.executor.run(code, timeout=RUN_TIMEOUT_SECONDS)

        if result.timed_out:
            self._git("checkout", "--", "train.py")
            self._record(state, iteration, description, None, False, "timed out")
            return f"Iteration {iteration}: TIMED OUT after {RUN_TIMEOUT_SECONDS}s. Change reverted."

        metric = self._parse_metric(result.stdout) if result.exit_code == 0 else None
        if metric is None:
            error = result.stderr.strip()[-500:] or "no METRIC line in output"
            self._git("checkout", "--", "train.py")
            self._record(state, iteration, description, None, False, error)
            return f"Iteration {iteration}: FAILED ({error}). Change reverted; train.py is back to the last kept version."

        best = state["best_metric"]
        kept = best is None or (metric < best if state["minimize"] else metric > best)
        if kept:
            self._git("add", "train.py")
            self._git(
                "-c", "user.name=rsi-harness", "-c", "user.email=rsi@local",
                "commit", "-m", f"iteration {iteration}: {metric_name}={metric} -- {description}",
            )
            state["best_metric"] = metric
        else:
            self._git("checkout", "--", "train.py")
        self._record(state, iteration, description, metric, kept)

        verdict = "NEW BEST -- change kept and committed" if kept else f"did not beat the best ({best}) -- change reverted"
        return f"Iteration {iteration}: {metric_name}={metric}. {verdict}. Current best: {state['best_metric']}."

    def get_history(self) -> str:
        state = self.load_state()
        best = state["best_metric"] if state["best_metric"] is not None else "none yet"
        lines = [f"Best {state['metric_name']} so far: {best} (after {state['iteration'] + 1}/{self.max_experiments} attempts)", ""]
        for e in state["history"]:
            m = e["metric"] if e["metric"] is not None else f"FAILED ({e.get('error', 'unknown')})"
            lines.append(f"  iter {e['iteration']}: {m} ({'kept' if e['kept'] else 'reverted'}) -- {e['description']}")
        return "\n".join(lines)
