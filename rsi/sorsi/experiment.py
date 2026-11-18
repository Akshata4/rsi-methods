"""The SO-RSI experiment (course labs 10.06-10.08), in two phases.

1. build   Rounds of SO-RSI sessions on the train + held-out datasets. Every
           session starts with the lessons confirmed so far, and adds its own
           diagnoses to the inquiry memory. Round by round we track thrashing
           (repeated errors, failures, anomalies) -- does growing, verified
           memory make later rounds thrash less? That is the recursive part.
2. test    On the untouched test datasets, same budget for everyone:
             naive        plain Tumble (first-order: patch, run, repeat)
             sorsi_nomem  SO-RSI's detect + diagnose + probe, no memory
             sorsi        the same, plus the lessons frozen from the build phase
"""

import json
import os
from dataclasses import asdict

import numpy as np

from rsi.evaluate import summarize
from rsi.generations import _provenance
from rsi.improver import Improver
from rsi.run_task import DatasetSpec, RunResult, run_improver_on_dataset
from rsi.sorsi.detector import error_signature
from rsi.sorsi.memory import InquiryMemory
from rsi.sorsi.workspace import MAX_PROBES, SORSIWorkspace

CONTENDERS = ("naive", "sorsi_nomem", "sorsi")
NAMES = {"naive": "Plain Tumble (first-order)", "sorsi_nomem": "SO-RSI, no memory",
         "sorsi": "**SO-RSI with inquiry memory**"}


def _session(out_dir: str, dataset: DatasetSpec, budget: int, mode: str, lessons: str, context: dict) -> RunResult:
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return RunResult(**json.load(f))
    os.makedirs(out_dir, exist_ok=True)
    for stale in ("trace.jsonl", "memory.jsonl"):  # from an interrupted earlier attempt
        if os.path.exists(os.path.join(out_dir, stale)):
            os.remove(os.path.join(out_dir, stale))
    if mode == "naive":
        return run_improver_on_dataset(Improver.seed(), dataset, budget, out_dir)
    with open(os.path.join(out_dir, "lessons.txt"), "w") as f:
        f.write(lessons)

    def make_workspace(root, executor, b):
        return SORSIWorkspace(root, executor, b, session_dir=out_dir, context=context)

    return run_improver_on_dataset(Improver.seed(), dataset, budget, out_dir, make_workspace=make_workspace,
                                   protocol="sorsi", protocol_vars={"probes": MAX_PROBES, "lessons": lessons})


def session_stats(session_dir: str, result: RunResult) -> dict:
    """Thrashing and inquiry statistics for one session."""
    signatures = [error_signature(e.get("error")) for e in result.history if e.get("metric") is None]
    seen, repeats = set(), 0
    for s in signatures:  # comparable across all contenders: computed from the plain history
        repeats += s in seen
        seen.add(s)
    events = []
    trace = os.path.join(session_dir, "trace.jsonl")
    if os.path.exists(trace):
        with open(trace) as f:
            events = [json.loads(line) for line in f if line.strip()]
    anomalies = [e for e in events if e["event"] == "anomaly"]
    tested = [e for e in events if e["event"] == "diagnosis_tested"]
    return {
        "best_metric": result.best_metric,
        "failure_rate": result.metrics.get("failure_rate"),
        "wasted_rate": result.metrics.get("wasted_rate"),
        "gain_over_baseline": result.metrics.get("gain_over_baseline"),
        "repeated_errors": repeats,
        "anomalies": {k: sum(a["kind"] == k for a in anomalies) for k in ("recurrence", "opposing", "mismatch")},
        "diagnoses": sum(e["event"] == "diagnosis" for e in events),
        "diagnoses_tested": len(tested),
        "diagnoses_confirmed": sum(bool(e["confirmed"]) for e in tested),
        "probes": sum(e["event"] == "probe" for e in events),
        "tokens": result.total_tokens,
    }


def build(out: str, datasets: list[DatasetSpec], rounds: int, budget: int) -> list[dict]:
    rows = []
    build_root = os.path.join(out, "build")
    for r in range(rounds):
        print(f"\n[build] round {r + 1}/{rounds}")
        for d in datasets:
            memory = InquiryMemory.from_runs(build_root)  # everything confirmed in finished sessions so far
            session_dir = os.path.join(build_root, f"round{r + 1}", d.name)
            result = _session(session_dir, d, budget, "sorsi", memory.render(),
                              {"dataset": d.name, "round": r + 1})
            rows.append({"round": r + 1, "dataset": d.name, "lessons_available": len(memory.lessons()),
                         **session_stats(session_dir, result)})
    return rows


def test(out: str, datasets: list[DatasetSpec], repeats: int, budget: int) -> tuple[dict, dict, str]:
    memory = InquiryMemory.from_runs(os.path.join(out, "build"))
    lessons = memory.render()  # frozen: test sessions never add to it
    with open(os.path.join(out, "frozen_lessons.txt"), "w") as f:
        f.write(lessons or "(no confirmed lessons)")
    runs: dict[str, dict[str, list[RunResult]]] = {}
    stats: dict[str, list[dict]] = {}
    for mode in CONTENDERS:
        print(f"\n[test] {mode}")
        runs[mode], stats[mode] = {}, []
        for d in datasets:
            runs[mode][d.name] = []
            for r in range(repeats):
                session_dir = os.path.join(out, "test", mode, d.name, f"r{r}")
                result = _session(session_dir, d, budget, mode, lessons if mode == "sorsi" else "",
                                  {"dataset": d.name, "repeat": r, "mode": mode})
                runs[mode][d.name].append(result)
                stats[mode].append(session_stats(session_dir, result))
    return summarize(runs, reference="naive"), stats, lessons


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _fmt(x, pct=False, digits=2):
    if x is None:
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.{digits}f}"


def write_report(out: str, cfg: dict, build_rows: list[dict] | None, summary: dict | None,
                 stats: dict | None, lessons: str | None) -> str:
    L = ["# SO-RSI: diagnose before patching", "",
         f"Budget {cfg['budget']} experiments per session · build: {', '.join(cfg['build'])} × {cfg['rounds']} rounds · "
         f"test: {', '.join(cfg['test'])} × {cfg['repeats']} repeats", ""]
    if build_rows:
        L += ["## 1. Building inquiry memory, round by round", "",
              "Each session starts with the lessons confirmed in earlier finished sessions.", "",
              "| Round | Sessions | Lessons available (mean) | Failure rate | Repeated errors / session | "
              "Anomalies / session | Diagnoses confirmed | Mean best AUC |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted({row["round"] for row in build_rows}):
            rows = [row for row in build_rows if row["round"] == r]
            tested = sum(row["diagnoses_tested"] for row in rows)
            confirmed = sum(row["diagnoses_confirmed"] for row in rows)
            L.append(f"| {r} | {len(rows)} | {_mean([row['lessons_available'] for row in rows]):.1f} | "
                     f"{_fmt(_mean([row['failure_rate'] for row in rows]), pct=True)} | "
                     f"{_mean([row['repeated_errors'] for row in rows]):.2f} | "
                     f"{_mean([sum(row['anomalies'].values()) for row in rows]):.2f} | "
                     f"{confirmed}/{tested} | {_fmt(_mean([row['best_metric'] for row in rows]), digits=3)} |")
        L += [""]
    if summary:
        test = cfg["test"]
        L += ["## 2. Comparison on untouched test datasets", "",
              "Same budget for every contender. Gain is relative to plain Tumble's mean score.", "",
              "| Contender | " + " | ".join(f"{d} AUC" for d in test) + " | Gain vs plain [95% CI] | Verdict |",
              "|---|" + "---|" * (len(test) + 2)]
        for mode in CONTENDERS:
            s = summary[mode]
            cells = []
            for d in test:
                p = s["per_dataset"][d]
                cells.append("failed" if p["mean"] is None else f"{p['mean']:.3f} ± {p['std']:.3f}"
                             + (f" ({p['failed_runs']} failed)" if p["failed_runs"] else ""))
            lo, hi = s["ci95"]
            ci = "" if np.isnan(lo) else f" [{lo:+.1%}, {hi:+.1%}]"
            gain = s["mean_gain_vs_gen0"]
            L.append(f"| {NAMES[mode]} | " + " | ".join(cells) + f" | {gain:+.1%}{ci} | {s['verdict']} |")
        noise = summary["naive"].get("noise_floor")
        if noise is not None:
            L += ["", f"Noise floor: plain Tumble varies by ±{noise:.1%} against itself across repeats."]
        L += ["", "### Thrashing and inquiry (per session, mean)", "",
              "| Contender | Failed experiments | Wasted experiments | Repeated errors | Anomalies (rec / opp / mis) | "
              "Diagnoses confirmed | Probes | Tokens |", "|---|---|---|---|---|---|---|---|"]
        for mode in CONTENDERS:
            rows = stats[mode]
            tested = sum(r["diagnoses_tested"] for r in rows)
            confirmed = sum(r["diagnoses_confirmed"] for r in rows)
            an = {k: _mean([r["anomalies"][k] for r in rows]) for k in ("recurrence", "opposing", "mismatch")}
            anomalies = "—" if mode == "naive" else " / ".join(f"{an[k]:.1f}" for k in an)
            L.append(f"| {NAMES[mode]} | {_fmt(_mean([r['failure_rate'] for r in rows]), pct=True)} | "
                     f"{_fmt(_mean([r['wasted_rate'] for r in rows]), pct=True)} | "
                     f"{_mean([r['repeated_errors'] for r in rows]):.2f} | {anomalies} | "
                     f"{'—' if mode == 'naive' else f'{confirmed}/{tested}'} | "
                     f"{'—' if mode == 'naive' else f'{_mean([r['probes'] for r in rows]):.1f}'} | "
                     f"{_mean([r['tokens'] for r in rows]):,.0f} |")
        L += ["", "Repeated errors are counted the same way for every contender (failed experiments whose error "
              "matches an earlier one in the session); anomalies are only detected for SO-RSI sessions.", ""]
    if lessons is not None:
        L += ["## 3. The inquiry memory used in testing", "", "```text", lessons.strip() or "(no confirmed lessons)",
              "```", ""]
    path = os.path.join(out, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def run(out: str, build_sets: list[DatasetSpec], test_sets: list[DatasetSpec], budget: int = 6, rounds: int = 2,
        repeats: int = 2, phases: tuple = ("build", "test")) -> str:
    os.makedirs(out, exist_ok=True)
    cfg = {"budget": budget, "rounds": rounds, "repeats": repeats, "build": [d.name for d in build_sets],
           "test": [d.name for d in test_sets], "build_specs": [asdict(d) for d in build_sets],
           "test_specs": [asdict(d) for d in test_sets], "max_probes": MAX_PROBES}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if not os.path.exists(os.path.join(out, "provenance.json")):  # keep the original on resume
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(_provenance(), f, indent=2)
    build_rows = build(out, build_sets, rounds, budget) if "build" in phases else None
    summary = stats = lessons = None
    if "test" in phases:
        summary, stats, lessons = test(out, test_sets, repeats, budget)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"build": build_rows, "summary": summary, "stats": stats}, f, indent=2)
    path = write_report(out, cfg, build_rows, summary, stats, lessons)
    print(f"\n[done] {path}")
    return path
