"""The Budget RSI experiment, in four phases (course labs 10.03-10.05).

1. collect   On the train + held-out datasets, run the candidate protocol with
             policy "all": every candidate edit is executed, so each one gets a
             true outcome label. The first experiment of a session is the
             baseline, so the default budget of 4 gives 3 candidate rounds:
             4 datasets x 2 repeats x 3 rounds x 3 candidates = 72 labeled
             edits (the paper fits on 96; --rounds 5 gives 96).
2. analyze   Offline, at zero extra cost: because every candidate ran, we can
             replay each round and ask what each selection strategy WOULD have
             scored -- scorer pick vs random vs the agent's first idea vs the best
             possible pick. The scorer is always evaluated on a dataset it was not
             trained on (leave-one-dataset-out). Also: a learning curve (does the
             scorer pick better as it gets more of its own labeled outcomes?).
3. live      On the untouched test datasets, same budget for everyone:
             plain Tumble vs propose-3-and-run {first | random | scorer pick}.
4. refit     The scorer is refit on its live outcomes too -> scorer_v2 for the
             next generation. That self-update is the recursive step.
"""

import json
import os
import random
from dataclasses import asdict

import numpy as np

from rsi.budget.features import FEATURE_NAMES
from rsi.budget.scorer import CLIP, EditScorer
from rsi.budget.workspace import CandidateWorkspace
from rsi.evaluate import summarize
from rsi.generations import _provenance
from rsi.improver import Improver
from rsi.run_task import DatasetSpec, RunResult, run_improver_on_dataset

LIVE_CONTENDERS = ("tumble", "first", "random", "scorer")


# --- running sessions ----------------------------------------------------------

def _session(out_dir: str, dataset: DatasetSpec, rounds: int, k: int, policy: str | None,
             scorer: EditScorer | None, repeat: int) -> RunResult:
    """One Tumble session; policy None = plain Tumble. Cached: a finished session is not rerun."""
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return RunResult(**json.load(f))
    os.makedirs(out_dir, exist_ok=True)
    label_log = os.path.join(out_dir, "labels.jsonl")
    if os.path.exists(label_log):
        os.remove(label_log)  # an interrupted earlier attempt must not leave duplicate labels
    seed = random.Random(f"{dataset.name}-{repeat}-{policy}").randrange(2**31)

    if policy is None:
        return run_improver_on_dataset(Improver.seed(), dataset, rounds, out_dir)

    def make_workspace(root, executor, budget):
        return CandidateWorkspace(root, executor, budget, k=k, policy=policy, scorer=scorer, label_log=label_log,
                                  seed=seed, context={"dataset": dataset.name, "repeat": repeat})

    return run_improver_on_dataset(Improver.seed(), dataset, rounds, out_dir,
                                   make_workspace=make_workspace, protocol="candidates", k=k)


def load_labels(root: str) -> list[dict]:
    rows = []
    for dirpath, _, files in sorted(os.walk(root)):
        if "labels.jsonl" in files and os.path.exists(os.path.join(dirpath, "result.json")):
            with open(os.path.join(dirpath, "labels.jsonl")) as f:
                rows += [json.loads(line) for line in f if line.strip()]
    return rows


def collect(out: str, datasets: list[DatasetSpec], repeats: int, rounds: int, k: int) -> list[dict]:
    print(f"[collect] {len(datasets)} datasets x {repeats} repeats, {rounds} experiments, {k} candidates, all executed")
    for d in datasets:
        for r in range(repeats):
            _session(os.path.join(out, "collect", d.name, f"r{r}"), d, rounds, k, "all", None, r)
    rows = load_labels(os.path.join(out, "collect"))
    print(f"[collect] {len(rows)} labeled edits")
    return rows


# --- offline analysis -----------------------------------------------------------

def _rounds(rows: list[dict], k: int) -> dict[tuple, list[dict]]:
    """Group collected rows into complete rounds of k candidates."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["repeat"], row["round"]), []).append(row)
    return {key: sorted(g, key=lambda r: r["candidate"]) for key, g in groups.items() if len(g) == k}


def _fit(rows: list[dict], alpha: float = 1.0) -> EditScorer:
    return EditScorer(alpha=alpha).fit([r["features"] for r in rows], [r["label"] for r in rows])


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _bootstrap_ci(values: list[float], seed: int = 0, n: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    arr = np.array(values)
    means = np.random.default_rng(seed).choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(rows: list[dict], k: int, out_dir: str, alpha: float = 1.0) -> dict:
    """Leave-one-dataset-out replay of every collected round."""
    os.makedirs(out_dir, exist_ok=True)
    rounds = _rounds(rows, k)
    datasets = sorted({key[0] for key in rounds})
    picks = {"scorer": [], "random": [], "first": [], "oracle": []}
    scorer_minus_random, hits, preds_all, labels_all, crash_picks = [], 0, [], [], 0
    per_dataset = {}

    for held_out in datasets:
        train = [r for r in rows if r["dataset"] != held_out]
        scorer = _fit(train, alpha)
        ds_scores = {s: [] for s in picks}
        for key, cands in rounds.items():
            if key[0] != held_out:
                continue
            labels = [c["label"] for c in cands]
            pred = scorer.predict([c["features"] for c in cands])
            pick = int(np.argmax(pred))
            realized = {"scorer": labels[pick], "random": float(np.mean(labels)),  # expected value of a random pick
                        "first": labels[0], "oracle": max(labels)}
            for s, v in realized.items():
                picks[s].append(v)
                ds_scores[s].append(v)
            scorer_minus_random.append(realized["scorer"] - realized["random"])
            hits += labels[pick] == max(labels)
            crash_picks += cands[pick]["metric"] is None
            preds_all += pred
            labels_all += labels
        per_dataset[held_out] = {s: float(np.mean(v)) for s, v in ds_scores.items()}

    n_rounds = len(picks["scorer"])
    lo, hi = _bootstrap_ci(scorer_minus_random)
    result = {
        "n_labels": len(rows), "n_rounds": n_rounds, "k": k,
        "mean_realized_gain": {s: float(np.mean(v)) for s, v in picks.items()},
        "per_dataset": per_dataset,
        "scorer_minus_random": {"mean": float(np.mean(scorer_minus_random)), "ci95": [lo, hi]},
        "top1_hit_rate": hits / n_rounds if n_rounds else None,
        "scorer_pick_crash_rate": crash_picks / n_rounds if n_rounds else None,
        "candidate_crash_rate": float(np.mean([r["metric"] is None for r in rows])) if rows else None,
        "spearman_pred_vs_actual": _spearman(preds_all, labels_all),
        "learning_curve": learning_curve(rows, rounds, datasets, alpha),
    }
    final = _fit(rows, alpha)
    final.save(os.path.join(out_dir, "scorer_v1.json"))
    result["scorer_v1_weights"] = final.weights()
    with open(os.path.join(out_dir, "offline.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def learning_curve(rows, rounds, datasets, alpha, samples: int = 30, seed: int = 0) -> list[dict]:
    """Realized gain of scorer picks when it was trained on only n labeled edits."""
    rng = random.Random(seed)
    max_train = min(sum(r["dataset"] != d for r in rows) for d in datasets) if datasets else 0
    sizes = sorted({n for n in (12, 24, 36, 48, 60, 72, max_train) if 6 <= n <= max_train})
    curve = []
    for n in sizes:
        realized = []
        for held_out in datasets:
            pool = [r for r in rows if r["dataset"] != held_out]
            for _ in range(samples if n < len(pool) else 1):
                scorer = _fit(rng.sample(pool, n), alpha)
                for key, cands in rounds.items():
                    if key[0] == held_out:
                        pred = scorer.predict([c["features"] for c in cands])
                        realized.append(cands[int(np.argmax(pred))]["label"])
        curve.append({"n_labels": n, "mean_realized_gain": float(np.mean(realized))})
    return curve


# --- live comparison --------------------------------------------------------------

def live(out: str, test: list[DatasetSpec], repeats: int, rounds: int, k: int, scorer: EditScorer) -> dict:
    runs: dict[str, dict[str, list[RunResult]]] = {}
    for name in LIVE_CONTENDERS:
        policy = None if name == "tumble" else name
        print(f"\n[live] {name}")
        runs[name] = {d.name: [_session(os.path.join(out, "live", name, d.name, f"r{r}"), d, rounds, k, policy,
                                        scorer if name == "scorer" else None, r)
                               for r in range(repeats)] for d in test}
    summary = summarize(runs, reference="tumble")
    for name, by_ds in runs.items():
        execs = [r.metrics.get("executions") for rs in by_ds.values() for r in rs]
        execs = [e for e in execs if e is not None]
        summary[name]["executions_per_run"] = float(np.mean(execs)) if execs else None
    return summary


def refit(out: str, collected: list[dict], alpha: float = 1.0) -> tuple[EditScorer, int]:
    """Scorer v2: everything v1 learned from, plus the outcomes of its own live picks."""
    live_rows = load_labels(os.path.join(out, "live"))
    v2 = _fit(collected + live_rows, alpha)
    v2.save(os.path.join(out, "scorer_v2.json"))
    return v2, len(live_rows)


# --- report -----------------------------------------------------------------------

def _pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2%}"


def write_report(out: str, cfg: dict, offline: dict | None, live_summary: dict | None,
                 v1: EditScorer | None, v2: EditScorer | None, n_live_labels: int) -> str:
    L = ["# Budget RSI: scoring candidate edits before running them", "",
         f"k = {cfg['k']} candidates per round · {cfg['rounds']} experiments per session · "
         f"collection: {', '.join(cfg['collect'])} × {cfg['collect_repeats']} repeats · "
         f"test: {', '.join(cfg['test'])} × {cfg['live_repeats']} repeats", ""]
    if offline:
        g = offline["mean_realized_gain"]
        smr = offline["scorer_minus_random"]
        L += ["## 1. Offline replay (every candidate was run, so every pick can be scored)", "",
              f"{offline['n_labels']} labeled edits in {offline['n_rounds']} rounds. The scorer is always tested on a "
              f"dataset it was not trained on. Gain = relative improvement of the picked edit over its parent "
              f"(clipped to ±{CLIP:.0%}; a crash counts as −{CLIP:.0%}).", "",
              "| Pick strategy | Mean gain of the picked edit |", "|---|---|",
              f"| Best possible (oracle) | {_pct(g['oracle'])} |",
              f"| **Scorer (Budget RSI)** | **{_pct(g['scorer'])}** |",
              f"| Random candidate | {_pct(g['random'])} |",
              f"| Agent's first idea | {_pct(g['first'])} |", "",
              f"- Scorer − random: {_pct(smr['mean'])} [95% CI {_pct(smr['ci95'][0])}, {_pct(smr['ci95'][1])}]",
              f"- Scorer picked the best candidate in {offline['top1_hit_rate']:.0%} of rounds (random: {1 / cfg['k']:.0%})",
              f"- Scorer's picks crashed in {offline['scorer_pick_crash_rate']:.0%} of rounds; "
              f"{offline['candidate_crash_rate']:.0%} of all candidates crashed",
              f"- Rank correlation of predicted vs actual gain: {offline['spearman_pred_vs_actual']:.2f}", "",
              "The parent-score-only ablation from the paper cannot rank candidates here: all candidates in a round "
              "share one parent, so it reduces to a random pick.", "",
              "### Learning curve: does the scorer pick better with more of its own outcomes?", "",
              "| Labeled edits trained on | Mean gain of scorer's pick |", "|---|---|"]
        L += [f"| {p['n_labels']} | {_pct(p['mean_realized_gain'])} |" for p in offline["learning_curve"]]
        L += ["", "![Offline replay](offline_strategies.png)", "", "![Learning curve](learning_curve.png)", ""]
    if v1:
        L += ["## 2. What the scorer learned (standardized weights)", "",
              "Positive = edits with more of this tended to help; negative = tended to hurt.", "",
              "| Feature | Scorer v1 | Scorer v2 (after live outcomes) |", "|---|---|---|"]
        w1, w2 = v1.weights(), v2.weights() if v2 else {}
        L += [f"| {name} | {w1[name]:+.4f} | {w2[name]:+.4f} |" if w2 else f"| {name} | {w1[name]:+.4f} | — |"
              for name in FEATURE_NAMES]
        L += ["", f"v1 trained on {v1.n_train} edits" + (f"; v2 on {v2.n_train} (+{n_live_labels} from live picks)" if v2 else ""), ""]
    if live_summary:
        test = cfg["test"]
        L += ["## 3. Live comparison on untouched test datasets", "",
              "Same experiment budget for every contender. Gain is relative to plain Tumble's mean score.", "",
              "| Contender | " + " | ".join(f"{d} AUC" for d in test) + " | Gain vs Tumble [95% CI] | Verdict | Scripts executed / run | Tokens / run |",
              "|---|" + "---|" * (len(test) + 4)]
        names = {"tumble": "Plain Tumble (1 idea per experiment)", "first": "Propose 3, run the first",
                 "random": "Propose 3, run a random one", "scorer": "**Propose 3, run the scorer's pick**"}
        for name, s in live_summary.items():
            cells = []
            for d in test:
                p = s["per_dataset"][d]
                cells.append("failed" if p["mean"] is None else f"{p['mean']:.3f} ± {p['std']:.3f}"
                             + (f" ({p['failed_runs']} failed)" if p["failed_runs"] else ""))
            lo, hi = s["ci95"]
            ci = "" if np.isnan(lo) else f" [{_pct(lo)}, {_pct(hi)}]"
            ex = s.get("executions_per_run")
            L.append(f"| {names[name]} | " + " | ".join(cells) + f" | {_pct(s['mean_gain_vs_gen0'])}{ci} | {s['verdict']} | "
                     f"{'n/a' if ex is None else f'{ex:.1f}'} | {'n/a' if s['tokens_per_run'] is None else f'{s['tokens_per_run']:,.0f}'} |")
        noise = live_summary["tumble"].get("noise_floor")
        if noise is not None:
            L += ["", f"Noise floor: plain Tumble varies by ±{noise:.1%} against itself across repeats."]
        L += ["", "Reading it: *scorer vs random* isolates whether the learned ranking helps; *random vs first* "
              "shows whether the agent's own ordering carries information; *tumble vs first* shows the cost or "
              "benefit of asking for 3 ideas at all.", ""]
    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def run(out: str, collect_sets: list[DatasetSpec], test: list[DatasetSpec], k: int = 3, rounds: int = 4,
        collect_repeats: int = 2, live_repeats: int = 2, phases: tuple = ("collect", "analyze", "live"),
        alpha: float = 1.0) -> str:
    os.makedirs(out, exist_ok=True)
    cfg = {"k": k, "rounds": rounds, "collect": [d.name for d in collect_sets], "test": [d.name for d in test],
           "collect_repeats": collect_repeats, "live_repeats": live_repeats, "alpha": alpha,
           "collect_specs": [asdict(d) for d in collect_sets], "test_specs": [asdict(d) for d in test]}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if not os.path.exists(os.path.join(out, "provenance.json")):  # keep the original on resume
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(_provenance(), f, indent=2)

    rows = collect(out, collect_sets, collect_repeats, rounds, k) if "collect" in phases else \
        load_labels(os.path.join(out, "collect"))
    offline, v1, v2, live_summary, n_live = None, None, None, None, 0
    analysis_dir = os.path.join(out, "analysis")
    if "analyze" in phases or "live" in phases:
        if not rows:
            raise SystemExit("no labeled edits -- run the collect phase first")
        offline = analyze(rows, k, analysis_dir, alpha)
        v1 = EditScorer.load(os.path.join(analysis_dir, "scorer_v1.json"))
        from rsi.budget.plots import plot_learning_curve, plot_offline
        plot_offline(os.path.join(out, "offline_strategies.png"), offline)
        plot_learning_curve(os.path.join(out, "learning_curve.png"), offline)
        print(f"[analyze] mean gain of picks: " + ", ".join(f"{s} {_pct(v)}" for s, v in offline["mean_realized_gain"].items()))
    if "live" in phases:
        live_summary = live(out, test, live_repeats, rounds, k, v1)
        v2, n_live = refit(out, rows, alpha)
        with open(os.path.join(out, "live_summary.json"), "w") as f:
            json.dump(live_summary, f, indent=2)
    path = write_report(out, cfg, offline, live_summary, v1, v2, n_live)
    print(f"\n[done] {path}")
    return path
