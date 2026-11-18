"""Entry point for the SO-RSI experiment (see rsi/sorsi/experiment.py).

    uv run python run_so_rsi.py                          # build memory -> test, new run dir
    uv run python run_so_rsi.py --phases build           # memory building only
    uv run python run_so_rsi.py --out runs/sorsi-<ts>    # resume: finished sessions are reused

Defaults: build on the registry's train + held-out datasets, test on its test datasets.
"""

import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv()

from rsi.run_task import load_datasets, registry_split  # noqa: E402
from rsi.sorsi.experiment import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="SO-RSI: diagnose why experiments fail before patching.")
    p.add_argument("--registry", default="datasets/registry.json")
    p.add_argument("--out", help="run directory (default: runs/sorsi-<timestamp>); reuse one to resume")
    p.add_argument("--phases", nargs="+", default=["build", "test"], choices=["build", "test"])
    p.add_argument("--budget", type=int, default=6, help="experiments per session")
    p.add_argument("--rounds", type=int, default=2, help="memory-building rounds over the build datasets")
    p.add_argument("--repeats", type=int, default=2, help="repeats per contender and test dataset")
    p.add_argument("--build", nargs="+", help="datasets for memory building (default: train + held-out)")
    p.add_argument("--test", nargs="+", help="datasets for the comparison (default: test split)")
    args = p.parse_args()

    registry = load_datasets(args.registry)
    build_names = args.build or registry_split(registry, "train") + registry_split(registry, "heldout")
    test_names = args.test or registry_split(registry, "test")
    if set(build_names) & set(test_names):
        p.error(f"build and test datasets must not overlap: {sorted(set(build_names) & set(test_names))}")
    out = args.out or os.path.join("runs", time.strftime("sorsi-%Y%m%d-%H%M%S"))
    print(f"out={out}  build={build_names}  test={test_names}")
    run(out, [registry[n] for n in build_names], [registry[n] for n in test_names], budget=args.budget,
        rounds=args.rounds, repeats=args.repeats, phases=tuple(args.phases))


if __name__ == "__main__":
    main()
