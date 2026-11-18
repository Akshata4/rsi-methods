"""Entry point for the Budget RSI experiment (see rsi/budget/experiment.py).

    uv run python run_budget_rsi.py                       # collect -> analyze -> live, new run dir
    uv run python run_budget_rsi.py --phases collect analyze   # offline part only (cheaper)
    uv run python run_budget_rsi.py --out runs/budget-<ts>     # resume: finished sessions are reused

Defaults: collect on the registry's train + held-out datasets, live on its test datasets.
"""

import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv()

from rsi.budget.experiment import run  # noqa: E402
from rsi.run_task import load_datasets, registry_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Budget RSI: score candidate edits before running them.")
    p.add_argument("--registry", default="datasets/registry.json")
    p.add_argument("--out", help="run directory (default: runs/budget-<timestamp>); reuse one to resume")
    p.add_argument("--phases", nargs="+", default=["collect", "analyze", "live"], choices=["collect", "analyze", "live"])
    p.add_argument("--k", type=int, default=3, help="candidate edits per round")
    p.add_argument("--rounds", type=int, default=4, help="experiments per session (the budget)")
    p.add_argument("--collect-repeats", type=int, default=2)
    p.add_argument("--live-repeats", type=int, default=2)
    p.add_argument("--collect", nargs="+", help="datasets for label collection (default: train + held-out)")
    p.add_argument("--test", nargs="+", help="datasets for the live comparison (default: test split)")
    args = p.parse_args()

    registry = load_datasets(args.registry)
    collect_names = args.collect or registry_split(registry, "train") + registry_split(registry, "heldout")
    test_names = args.test or registry_split(registry, "test")
    if set(collect_names) & set(test_names):
        p.error(f"collect and test datasets must not overlap: {sorted(set(collect_names) & set(test_names))}")
    out = args.out or os.path.join("runs", time.strftime("budget-%Y%m%d-%H%M%S"))
    print(f"out={out}  collect={collect_names}  test={test_names}")
    run(out, [registry[n] for n in collect_names], [registry[n] for n in test_names], k=args.k, rounds=args.rounds,
        collect_repeats=args.collect_repeats, live_repeats=args.live_repeats, phases=tuple(args.phases))


if __name__ == "__main__":
    main()
