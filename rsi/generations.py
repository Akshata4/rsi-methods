"""The generation loop: recursive self-improvement on top of Tumble.

    Gen 0   champion = seed improver (Tumble's original prompt)
            run it on TRAIN datasets    -> evidence for writing a child
            run it on HELD-OUT datasets -> the score to beat

    Gen g   child = champion.meta_prompt(champion, evidence)   # the improver writes its successor
            run child on HELD-OUT datasets (same budget as champion)
            paired comparison vs champion (compare.py)
            ACCEPT -> child becomes champion; run it on TRAIN to get its own evidence
            REJECT -> champion stays; its next child sees why the last one lost

    Stop    after max_generations, or after `patience` rejections in a row.

The child is judged only on held-out datasets it never saw in its evidence,
so it can't win by memorizing tricks for one dataset (course labs 09.05-09.06).
"""

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field

from rsi.compare import Comparison, compare
from rsi.improver import Improver
from rsi.meta import propose_child
from rsi.run_task import DatasetSpec, RunResult, run_improver_on_dataset
from tumble_core.workspace import SCORING


@dataclass
class GenerationConfig:
    train: list[DatasetSpec]
    heldout: list[DatasetSpec]
    budget: int = 6                  # experiments per Tumble session -- identical for every improver
    max_generations: int = 3
    patience: int = 2                # consecutive rejections before stopping
    min_gain: float = 0.01           # relative margin a child must beat the champion by
    out_root: str = "runs"
    # TEST datasets are recorded here but never touched by the loop: they are
    # used once, by rsi/evaluate.py, after all accept/reject decisions are made.
    test: list[DatasetSpec] = field(default_factory=list)


def _evaluate(improver: Improver, datasets: list[DatasetSpec], cfg: GenerationConfig, run_dir: str, split: str) -> list[RunResult]:
    print(f"  evaluating {improver.id} on {split}: {[d.name for d in datasets]}")
    return [
        run_improver_on_dataset(improver, d, cfg.budget, os.path.join(run_dir, improver.id, split, d.name))
        for d in datasets
    ]


def _scores(results: list[RunResult]) -> dict:
    return {r.dataset: r.best_metric for r in results}


def _run_stats(results: list[RunResult]) -> dict:
    """Run-level metrics per dataset, for the lineage log."""
    return {r.dataset: {k: r.metrics.get(k) for k in ("gain_over_baseline", "experiments_to_best", "wasted_rate")}
            | {"tokens": r.total_tokens} for r in results}


def _provenance() -> dict:
    """Which version of the harness produced this run: the git commit, and
    whether there were uncommitted changes (a 'dirty' run isn't reproducible
    from the commit alone)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()
    model = os.environ.get("LLM_MODEL") or ""
    return {
        "git_commit": git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "llm_provider": os.environ.get("LLM_PROVIDER", "wandb"),
        "llm_model": model or "(provider default)",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scoring": dict(SCORING),  # measurement version: folds=1, keep_t=0 is version 1
    }


def run_generations(cfg: GenerationConfig, seed: Improver | None = None) -> str:
    """Runs the loop and returns the run directory (champion id is in champion.txt)."""
    run_dir = os.path.join(cfg.out_root, time.strftime("%Y%m%d-%H%M%S"))
    improvers_dir = os.path.join(run_dir, "improvers")
    lineage_path = os.path.join(run_dir, "lineage.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(run_dir, "provenance.json"), "w") as f:
        json.dump(_provenance(), f, indent=2)

    def log(record: dict) -> None:
        with open(lineage_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    champion = seed or Improver.seed()
    champion.save(improvers_dir)
    print(f"[gen 0] seed improver {champion.id}")
    champ_train = _evaluate(champion, cfg.train, cfg, run_dir, "train")
    champ_heldout = _evaluate(champion, cfg.heldout, cfg, run_dir, "heldout")
    log({"generation": 0, "improver": champion.id, "parent": None, "accepted": True,
         "train": _scores(champ_train), "heldout": _scores(champ_heldout),
         "heldout_stats": _run_stats(champ_heldout), "meta_tokens": 0})

    rejections: list[str] = []
    for g in range(1, cfg.max_generations + 1):
        print(f"\n[gen {g}] champion {champion.id} writes its successor")
        child_id = f"gen{g:03d}-{champion.id[:6]}-child"
        try:
            meta_usage: dict = {}
            child = propose_child(champion, champ_train, child_id, feedback="\n\n".join(rejections), usage=meta_usage)
        except RuntimeError as e:
            print(f"  {e}")
            log({"generation": g, "improver": None, "parent": champion.id, "accepted": False, "reason": str(e)})
            rejections.append(f"(no valid child: {e})")
            if len(rejections) >= cfg.patience:
                break
            continue
        child.save(improvers_dir)
        print(f"  child {child.id} rationale: {child.rationale[:300]}")

        child_heldout = _evaluate(child, cfg.heldout, cfg, run_dir, "heldout")
        result: Comparison = compare(champ_heldout, child_heldout, cfg.min_gain)
        print(f"  {result.reason}")
        log({"generation": g, "improver": child.id, "parent": champion.id, "accepted": result.accepted,
             "heldout": _scores(child_heldout), "champion_heldout": _scores(champ_heldout),
             "per_dataset_gain": result.per_dataset, "reason": result.reason, "rationale": child.rationale,
             "heldout_stats": _run_stats(child_heldout), "meta_tokens": meta_usage.get("total_tokens", 0)})

        if result.accepted:
            champion, champ_heldout = child, child_heldout
            champ_train = _evaluate(champion, cfg.train, cfg, run_dir, "train")  # its own evidence for the next rewrite
            rejections = []
        else:
            rejections.append(f"Child {child.id}: {result.reason}\nIts rationale was: {child.rationale}")
            if len(rejections) >= cfg.patience:
                print(f"\n[stop] {cfg.patience} rejections in a row")
                break

    with open(os.path.join(run_dir, "champion.txt"), "w") as f:
        f.write(champion.id + "\n")
    print(f"\n[done] final champion: {champion.id} (generation {champion.generation}). Lineage: {lineage_path}")
    return run_dir
