"""Final evaluation: did recursive self-improvement actually help?

Runs once, after the generation loop has made all its accept/reject decisions,
on TEST datasets the loop never saw. Every contender runs on every test
dataset `repeats` times, because LLM agents are noisy and one run proves nothing.

Contenders
    gen0                 the seed improver (Tumble's original prompt)
    gen1, gen2, ...      every champion the loop accepted
    gen0-2x-budget       BASELINE: the seed with double the experiments. Is the gain from
                         recursion, or would just running more experiments do as well?
    blind-rewrite        BASELINE: the seed rewritten the same number of times, but with no
                         run evidence. Is the gain from learning from evidence, or from
                         rewriting the prompt at all?

Outputs (in <run_dir>/eval/)
    summary.md           the results table and verdicts
    results.json         every number behind the table
    generations.png      held-out gain (what the loop selected on) vs test gain, per generation
    best_so_far.png      average best-so-far curve per contender

    uv run python -m rsi.evaluate runs/<timestamp> --test <dataset> <dataset> --repeats 3
"""

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from rsi.improver import Improver
from rsi.meta import propose_child
from rsi.metrics import relative_gain
from rsi.run_task import DatasetSpec, RunResult, load_datasets, run_improver_on_dataset

FAILED_RUN_GAIN = -1.0  # a run with no metric at all counts as a 100% loss (same as compare.py)
BOOTSTRAP_SAMPLES = 2000


# --- contenders -------------------------------------------------------------

def _lineage(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "lineage.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _champions(run_dir: str) -> list[Improver]:
    """gen0 plus every accepted child, in generation order."""
    improvers_dir = os.path.join(run_dir, "improvers")
    return [Improver.load(os.path.join(improvers_dir, f"{e['improver']}.json"))
            for e in _lineage(run_dir) if e["accepted"] and e["improver"]]


def _blind_rewrite(seed: Improver, n_rewrites: int, eval_dir: str) -> Improver:
    """Rewrite the seed n times with NO evidence. Cached so re-running evaluate costs nothing."""
    path = os.path.join(eval_dir, "improvers", "blind-rewrite.json")
    if os.path.exists(path):
        return Improver.load(path)
    current = seed
    for i in range(1, n_rewrites + 1):
        current = propose_child(current, evidence=[], child_id=f"blind-{i}")
    blind = Improver(current.research_prompt, current.meta_prompt, generation=n_rewrites,
                     parent_id=seed.id, rationale=current.rationale, id="blind-rewrite")
    blind.save(os.path.dirname(path))
    return blind


# --- running ------------------------------------------------------------------

def _run_cached(improver: Improver, dataset: DatasetSpec, budget: int, out_dir: str) -> RunResult:
    """Reuse a finished run if its result.json exists (lets you resume an interrupted eval)."""
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return RunResult(**json.load(f))
    return run_improver_on_dataset(improver, dataset, budget, out_dir)


# --- statistics -----------------------------------------------------------------

def _bootstrap_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = rng.choice(arr, size=(BOOTSTRAP_SAMPLES, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _verdict(lo: float, hi: float) -> str:
    if np.isnan(lo):
        return "too few runs"
    if lo > 0:
        return "clear gain"
    if hi < 0:
        return "clear loss"
    return "within noise"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(runs: dict[str, dict[str, list[RunResult]]], reference: str = "gen0") -> dict:
    """runs[contender][dataset] = list of repeat results. Gains are relative to the
    reference contender's MEAN final score on the same dataset."""
    ref_mean = {ds: _mean([r.best_metric for r in rs]) for ds, rs in runs[reference].items()}
    out = {}
    for name, by_ds in runs.items():
        gains, per_ds = [], {}
        for ds, rs in by_ds.items():
            scores = [r.best_metric for r in rs]
            ok = [s for s in scores if s is not None]
            per_ds[ds] = {
                "mean": float(np.mean(ok)) if ok else None,
                "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0,
                "failed_runs": len(scores) - len(ok),
                "n": len(scores),
            }
            for r in rs:
                g = relative_gain(r.best_metric, ref_mean[ds], r.minimize)
                gains.append(FAILED_RUN_GAIN if g is None else g)
        all_runs = [r for rs in by_ds.values() for r in rs]
        lo, hi = _bootstrap_ci(gains)
        out[name] = {
            "per_dataset": per_ds,
            "mean_gain_vs_gen0": float(np.mean(gains)) if gains else None,
            "ci95": [lo, hi],
            "verdict": "reference" if name == reference else _verdict(lo, hi),
            "wasted_rate": _mean([r.metrics.get("wasted_rate") for r in all_runs]),
            "failure_rate": _mean([r.metrics.get("failure_rate") for r in all_runs]),
            "experiments_to_best": _mean([r.metrics.get("experiments_to_best") for r in all_runs]),
            "gain_over_own_baseline": _mean([r.metrics.get("gain_over_baseline") for r in all_runs]),
            "tokens_per_run": _mean([r.total_tokens for r in all_runs]),
            "seconds_per_run": _mean([r.wall_seconds for r in all_runs]),
            "budget": all_runs[0].budget if all_runs else None,
        }
    # The reference's own spread is the noise floor: how much gen0 differs from itself.
    self_gains = [relative_gain(r.best_metric, ref_mean[ds], r.minimize)
                  for ds, rs in runs[reference].items() for r in rs]
    self_gains = [g for g in self_gains if g is not None]
    out[reference]["noise_floor"] = float(np.std(self_gains, ddof=1)) if len(self_gains) > 1 else None
    return out


def _mean_curves(runs: dict[str, dict[str, list[RunResult]]], reference: str = "gen0") -> dict[str, list[float | None]]:
    """Best-so-far after each experiment, as relative gain vs gen0's mean final score,
    averaged over datasets and repeats. Before a run's first success it is skipped."""
    ref_mean = {ds: _mean([r.best_metric for r in rs]) for ds, rs in runs[reference].items()}
    curves = {}
    for name, by_ds in runs.items():
        length = max(len(r.metrics.get("best_so_far", [])) for rs in by_ds.values() for r in rs)
        points: list[list[float]] = [[] for _ in range(length)]
        for ds, rs in by_ds.items():
            for r in rs:
                for i, v in enumerate(r.metrics.get("best_so_far", [])):
                    g = relative_gain(v, ref_mean[ds], r.minimize)
                    if g is not None:
                        points[i].append(g)
        curves[name] = [float(np.mean(p)) if p else None for p in points]
    return curves


# --- reporting ---------------------------------------------------------------------

def _fmt_pct(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.0%}"


def write_summary(path: str, summary: dict, datasets: list[DatasetSpec], repeats: int, lineage: list[dict]) -> None:
    head = ["Contender", "Budget"] + [f"{d.name} ({d.metric} {'↓' if d.minimize else '↑'})" for d in datasets] + \
           ["Gain vs gen0 [95% CI]", "Verdict", "Wasted", "Exp. to best", "Tokens/run"]
    lines = [
        "# RSI evaluation on untouched test datasets", "",
        f"Each contender ran on each test dataset {repeats} time(s). Scores are mean ± std over repeats.",
        "Gain is relative to gen0's mean score on the same dataset (+ = better), averaged over all runs.",
        "A gain only counts as real if its 95% bootstrap CI excludes 0.", "",
        "| " + " | ".join(head) + " |", "|" + "---|" * len(head),
    ]
    for name, s in summary.items():
        cells = [name, str(s["budget"])]
        for d in datasets:
            p = s["per_dataset"][d.name]
            cell = "failed" if p["mean"] is None else f"{p['mean']:.4g} ± {p['std']:.2g}"
            if p["failed_runs"]:
                cell += f" ({p['failed_runs']} failed)"
            cells.append(cell)
        lo, hi = s["ci95"]
        ci = "" if np.isnan(lo) else f" [{_fmt_pct(lo)}, {_fmt_pct(hi)}]"
        cells += [_fmt_pct(s["mean_gain_vs_gen0"]) + ci, s["verdict"], _fmt_pct(s["wasted_rate"], False),
                  "n/a" if s["experiments_to_best"] is None else f"{s['experiments_to_best']:.1f}",
                  "n/a" if s["tokens_per_run"] is None else f"{s['tokens_per_run']:,.0f}"]
        lines.append("| " + " | ".join(cells) + " |")

    noise = summary["gen0"].get("noise_floor")
    lines += ["", f"**Noise floor:** gen0 varies by ±{noise:.1%} against itself across repeats. "
              "Gains smaller than this are not evidence." if noise is not None else ""]

    accepted = sum(1 for e in lineage[1:] if e["accepted"])
    proposed = len(lineage) - 1
    lines += ["", "## Generation loop", "",
              f"- Children proposed: {proposed}, accepted: {accepted}"
              + (f" (acceptance rate {accepted / proposed:.0%})" if proposed else ""),
              f"- Meta-step tokens (writing children): {sum(e.get('meta_tokens', 0) for e in lineage):,}", "",
              "## How to read this", "",
              "- **genK vs gen0:** did recursion produce a better improver on data it never saw?",
              "- **genK vs gen0-2x-budget:** is recursion better than simply running more experiments?",
              "- **genK vs blind-rewrite:** does learning from run evidence matter, or does any rewrite help?",
              "- **Wasted / Exp. to best:** a better improver should waste fewer experiments and peak sooner.",
              "- **generations.png:** if test gain is far below held-out gain, the loop over-fit its held-out set."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --- main ---------------------------------------------------------------------------

def evaluate(run_dir: str, test: list[DatasetSpec], repeats: int = 3, budget: int | None = None,
             baselines: bool = True) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    budget = budget or cfg["budget"]
    loop_names = {d["name"] for d in cfg["train"] + cfg["heldout"]}
    leaked = loop_names & {d.name for d in test}
    if leaked:
        raise ValueError(f"test datasets were used by the generation loop: {sorted(leaked)}")

    eval_dir = os.path.join(run_dir, "eval")
    champions = _champions(run_dir)
    contenders: list[tuple[str, Improver, int]] = [(f"gen{c.generation}", c, budget) for c in champions]
    if baselines:
        seed = champions[0]
        contenders.append(("gen0-2x-budget", seed, 2 * budget))
        contenders.append(("blind-rewrite", _blind_rewrite(seed, max(1, len(champions) - 1), eval_dir), budget))

    runs: dict[str, dict[str, list[RunResult]]] = {}
    for name, improver, b in contenders:
        print(f"\n[eval] {name} ({improver.id}), budget {b}")
        runs[name] = {d.name: [_run_cached(improver, d, b, os.path.join(eval_dir, name, d.name, f"r{i}"))
                               for i in range(repeats)] for d in test}

    summary = summarize(runs)
    curves = _mean_curves(runs)
    lineage = _lineage(run_dir)
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "curves": curves, "repeats": repeats,
                   "test": [asdict(d) for d in test]}, f, indent=2)
    write_summary(os.path.join(eval_dir, "summary.md"), summary, test, repeats, lineage)

    from rsi.plots import plot_best_so_far, plot_generations  # matplotlib only needed here
    minimize = {d["name"]: d["minimize"] for d in cfg["heldout"]}
    plot_generations(os.path.join(eval_dir, "generations.png"), lineage, summary, minimize)
    plot_best_so_far(os.path.join(eval_dir, "best_so_far.png"), curves, budget)
    print(f"\n[eval] wrote {eval_dir}/summary.md, results.json, generations.png, best_so_far.png")
    return summary


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description="Evaluate an RSI run on untouched test datasets.")
    p.add_argument("run_dir")
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--budget", type=int, help="experiments per session (default: same as the loop)")
    p.add_argument("--registry", default="datasets/registry.json")
    p.add_argument("--no-baselines", action="store_true", help="skip gen0-2x-budget and blind-rewrite")
    args = p.parse_args()
    registry = load_datasets(args.registry)
    evaluate(args.run_dir, [registry[n] for n in args.test], args.repeats, args.budget, not args.no_baselines)


if __name__ == "__main__":
    main()
