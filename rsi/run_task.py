"""Run one improver on one dataset: a single Tumble session.

This is the inner loop, i.e. first-order self-improvement of train.py.
The generation loop calls this many times: (improver x dataset) per generation.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field

from rsi.improver import Improver
from rsi.metrics import run_metrics
from tumble_core.agent_loop import run_agent
from tumble_core.dataset_preview import build_preview
from tumble_core.executor import LocalExecutor
from tumble_core.trajectory import Trajectory
from tumble_core.workspace import Workspace


@dataclass
class DatasetSpec:
    name: str
    path: str
    target: str
    metric: str = "accuracy"
    minimize: bool = False
    split: str = ""          # train / heldout / test, as assigned in the registry
    description: str = ""


@dataclass
class RunResult:
    improver_id: str
    dataset: str
    metric: str
    minimize: bool
    best_metric: float | None
    experiments_used: int
    status: str
    history: list
    report: str
    trajectory_path: str
    budget: int = 0
    metrics: dict = field(default_factory=dict)  # see rsi/metrics.py
    total_tokens: int = 0
    llm_calls: int = 0
    wall_seconds: float = 0.0


def load_datasets(registry_path: str) -> dict[str, DatasetSpec]:
    base = os.path.dirname(os.path.abspath(registry_path))
    with open(registry_path) as f:
        raw = json.load(f)
    known = set(DatasetSpec.__dataclass_fields__)
    return {d["name"]: DatasetSpec(**{k: v for k, v in d.items() if k in known} | {"path": os.path.join(base, d["path"])})
            for d in raw}


def registry_split(registry: dict[str, DatasetSpec], split: str) -> list[str]:
    return [name for name, spec in registry.items() if spec.split == split]


def _print_event(event: dict) -> None:
    if event["type"] == "tool" and event["name"] == "run_experiment":
        print(f"      {event['result'][:160]}")


def run_improver_on_dataset(
    improver: Improver, dataset: DatasetSpec, budget: int, out_dir: str, verbose: bool = True,
    make_workspace=None, protocol: str = "single", k: int = 3,
) -> RunResult:
    """Run a full Tumble session under `improver` and return its final result.
    Everything (workspace, trajectory, result.json) lands in out_dir.

    `make_workspace(workspace_dir, executor, budget)` swaps in a different
    workspace (e.g. Budget RSI's CandidateWorkspace); `protocol`/`k` select
    the matching instructions in the fixed harness contract."""
    workspace_dir = os.path.join(out_dir, "workspace")
    executor = LocalExecutor(workspace_dir)
    if make_workspace is None:
        workspace = Workspace(workspace_dir, executor, max_experiments=budget)
    else:
        workspace = make_workspace(workspace_dir, executor, budget)
    workspace.init(dataset.path, dataset.target, dataset.metric, dataset.minimize)

    preview = build_preview(workspace.dev_path, dataset.target)  # the agent's view: dev rows only
    system_prompt = improver.system_prompt(preview, dataset.target, dataset.metric, dataset.minimize, budget,
                                           protocol=protocol, k=k)
    prompt = f"Improve {dataset.metric} as much as you can within your budget of {budget} experiments, then write your final report."

    trajectory = Trajectory(os.path.join(out_dir, "trajectory.jsonl"))
    if verbose:
        print(f"    [{improver.id} on {dataset.name}]")
        trajectory.subscribe(_print_event)
    usage: dict = {}
    started = time.monotonic()
    try:
        turns_per_experiment = 4 if protocol == "single" else k + 3  # k candidate writes + submit + reads
        status = run_agent(prompt, system_prompt, workspace.tool_schemas(), workspace.tool_functions(), trajectory,
                           max_turns=turns_per_experiment * budget + 10, usage=usage)
    except Exception as e:  # an API failure loses this run, not the whole generation
        status = f"error: {type(e).__name__}: {e}"
        trajectory.log_notice(status)
    finally:
        trajectory.close()
        executor.close()

    state = workspace.load_state()
    report_path = os.path.join(workspace_dir, "REPORT.md")
    report = open(report_path).read() if os.path.exists(report_path) else ""

    result = RunResult(
        improver_id=improver.id,
        dataset=dataset.name,
        metric=dataset.metric,
        minimize=dataset.minimize,
        best_metric=state["best_metric"],
        experiments_used=state["iteration"] + 1,
        status=status,
        history=state["history"],
        report=report,
        trajectory_path=trajectory.path,
        budget=budget,
        metrics=run_metrics(state["history"], dataset.minimize, budget) | {"executions": state.get("executions")},
        total_tokens=usage.get("total_tokens", 0),
        llm_calls=usage.get("llm_calls", 0),
        wall_seconds=round(time.monotonic() - started, 1),
    )
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(asdict(result), f, indent=2)
    if verbose:
        m = result.metrics
        gain = f"{m['gain_over_baseline']:+.1%}" if m["gain_over_baseline"] is not None else "n/a"
        wasted = f"{m['wasted_rate']:.0%}" if m["wasted_rate"] is not None else "n/a"
        print(f"    => best {dataset.metric} = {result.best_metric} | gain over baseline {gain} | "
              f"best at exp {m['experiments_to_best']} | wasted {wasted} | {status}")
    return result
