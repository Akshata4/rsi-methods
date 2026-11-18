"""Entry point for the RSI generation loop.

    uv run python run_rsi.py --budget 5 --generations 3 --repeats 3

By default the train / held-out / test datasets are the splits assigned in
datasets/registry.json (run scripts/make_datasets.py first). Override any of
them with --train / --heldout / --test.
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from rsi.evaluate import evaluate  # noqa: E402
from rsi.flow.workflow import SEED_WORKFLOW, static_variant  # noqa: E402
from rsi.generations import GenerationConfig, run_generations  # noqa: E402
from rsi.improver import Improver  # noqa: E402
from rsi.run_task import load_datasets, registry_split  # noqa: E402
from tumble_core.workspace import SCORING  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Recursive self-improvement on top of the Tumble loop.")
    p.add_argument("--registry", default="datasets/registry.json")
    p.add_argument("--train", nargs="+", help="datasets the improver learns from (evidence)")
    p.add_argument("--heldout", nargs="+", help="datasets the loop accepts/rejects children on")
    p.add_argument("--test", nargs="+", help="untouched datasets for the final evaluation (run after the loop)")
    p.add_argument("--no-eval", action="store_true", help="skip the final evaluation on test datasets")
    p.add_argument("--repeats", type=int, default=3, help="runs per (contender, test dataset) in the final evaluation")
    p.add_argument("--budget", type=int, default=6, help="experiments per Tumble session")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--seed-improver", help="path to an improver JSON to start from (default: Tumble's prompt)")
    p.add_argument("--folds", type=int, default=SCORING["folds"],
                   help="hidden folds per score (version 2 default 5; --folds 1 --keep-t 0 = version 1)")
    p.add_argument("--keep-t", type=float, default=SCORING["keep_t"],
                   help="paired-t threshold for keeping a change (0 = keep any improvement)")
    p.add_argument("--method", choices=["prompt", "flow"], default="prompt",
                   help="prompt: evolve the research prompt; flow: RSIFlow, evolve the workflow")
    args = p.parse_args()
    SCORING.update(folds=args.folds, keep_t=args.keep_t)

    registry = load_datasets(args.registry)
    args.train = args.train or registry_split(registry, "train")
    args.heldout = args.heldout or registry_split(registry, "heldout")
    args.test = [] if args.no_eval else (args.test or registry_split(registry, "test"))
    if not args.train or not args.heldout:
        p.error("no train/held-out datasets: pass --train/--heldout or assign splits in the registry")
    unknown = [n for n in args.train + args.heldout + args.test if n not in registry]
    if unknown:
        p.error(f"unknown datasets {unknown}; available: {sorted(registry)}")

    splits = {"train": set(args.train), "heldout": set(args.heldout), "test": set(args.test)}
    for a, b in [("train", "heldout"), ("train", "test"), ("heldout", "test")]:
        if splits[a] & splits[b]:
            p.error(f"{a} and {b} datasets must not overlap: {sorted(splits[a] & splits[b])}")

    print(f"train={args.train}  heldout={args.heldout}  test={args.test}")
    cfg = GenerationConfig(
        train=[registry[n] for n in args.train],
        heldout=[registry[n] for n in args.heldout],
        budget=args.budget,
        max_generations=args.generations,
        patience=args.patience,
        min_gain=args.min_gain,
        test=[registry[n] for n in args.test],
    )
    if args.seed_improver:
        seed = Improver.load(args.seed_improver)
    else:
        seed = Improver.seed_flow() if args.method == "flow" else None
    run_dir = run_generations(cfg, seed)
    if cfg.test:
        extra = None
        if args.method == "flow":  # RSIFlow: also compare against plain Tumble and fixed-order routing
            plain = Improver.seed()
            plain.id = "plain-tumble"
            extra = [("plain-tumble", plain),
                     ("flow-static", Improver.seed_flow(static_variant(SEED_WORKFLOW), id="flow-static"))]
        evaluate(run_dir, cfg.test, repeats=args.repeats, extra=extra)
        if args.method == "flow":
            from rsi.flow.report import write_flow_report
            print(f"[flow] {write_flow_report(run_dir)}")
    else:
        print(f"\nNo --test datasets given. Evaluate later with:\n"
              f"  uv run python -m rsi.evaluate {run_dir} --test <datasets> --repeats {args.repeats}")


if __name__ == "__main__":
    main()
