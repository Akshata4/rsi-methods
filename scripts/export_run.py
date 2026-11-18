"""Export a finished run from runs/ (gitignored) into results/ (committed).

    uv run python scripts/export_run.py runs/<run> --title "Short name" --notes "Context"

Recognizes the three run layouts in this repo:

    generation loop   prompt-only RSI and RSIFlow (run_rsi.py): lineage.jsonl, improvers/, eval/
    budget            Budget RSI (run_budget_rsi.py): collect/, analysis/, live/, report.md
    sorsi             SO-RSI (run_so_rsi.py): build/, test/, report.md

and copies the parts worth keeping for each -- config, provenance, the
reports and charts, the learned objects (prompts, workflows, scorer weights,
lessons) -- plus a per-session table (sessions.csv) and a readable
results/<run>/README.md, then regenerates the results log in the top-level
README.md. Bulky per-session workspaces and trajectories stay in runs/
unless --trajectories is given.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LOG_START, LOG_END = "<!-- results-log:start -->", "<!-- results-log:end -->"
SESSION_FIELDS = ["phase", "contender", "split", "dataset", "repeat", "budget", "best_metric",
                  "gain_over_baseline", "experiments_to_best", "experiments_used", "executions", "wasted_rate",
                  "failure_rate", "total_tokens", "wall_seconds", "status"]
METHOD_NAMES = {"prompt": "Prompt-only RSI loop", "flow": "RSIFlow", "budget": "Budget RSI", "sorsi": "SO-RSI"}


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _fmt(x, pct=False, digits=3):
    if x is None:
        return "—"
    return f"{x:+.1%}" if pct else f"{x:.{digits}f}"


def detect_method(run_dir: str) -> str:
    if os.path.exists(os.path.join(run_dir, "lineage.jsonl")):
        seeds = glob.glob(os.path.join(run_dir, "improvers", "*.json"))
        return "flow" if any(_load_json(p).get("workflow") for p in seeds) else "prompt"
    if os.path.isdir(os.path.join(run_dir, "collect")) or os.path.isdir(os.path.join(run_dir, "analysis")):
        return "budget"
    if os.path.isdir(os.path.join(run_dir, "build")):
        return "sorsi"
    raise SystemExit(f"{run_dir}: not a recognized run layout")


# --- sessions ------------------------------------------------------------------------

def _session_location(rel: list[str]) -> tuple[str, str, str, str]:
    """(phase, contender, split, repeat) from a result.json path relative to the run dir."""
    head = rel[0]
    if head in ("eval", "live", "test"):        # <phase>/<contender>/<dataset>/r<i>
        return head, rel[1], "test", rel[3].lstrip("r")
    if head == "collect":                       # collect/<dataset>/r<i>
        return "collect", "all-candidates", "train+heldout", rel[2].lstrip("r")
    if head == "build":                         # build/round<k>/<dataset>
        return "build", rel[1], "train+heldout", ""
    return "loop", rel[0], rel[1], ""           # <improver>/<split>/<dataset>


def collect_sessions(run_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "result.json"), recursive=True)):
        rel = os.path.relpath(path, run_dir).split(os.sep)
        r = _load_json(path)
        m = r.get("metrics", {})
        phase, contender, split, repeat = _session_location(rel)
        rows.append({
            "phase": phase, "contender": contender, "split": split, "dataset": r["dataset"], "repeat": repeat,
            "budget": r.get("budget"), "best_metric": r.get("best_metric"),
            "gain_over_baseline": m.get("gain_over_baseline"), "experiments_to_best": m.get("experiments_to_best"),
            "experiments_used": r.get("experiments_used"), "executions": m.get("executions"),
            "wasted_rate": m.get("wasted_rate"), "failure_rate": m.get("failure_rate"),
            "total_tokens": r.get("total_tokens"), "wall_seconds": r.get("wall_seconds"), "status": r.get("status"),
        })
    return rows


def _write_sessions(out: str, sessions: list[dict]) -> None:
    with open(os.path.join(out, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        w.writeheader()
        w.writerows(sessions)


def _copy(run_dir: str, out: str, names: list[str]) -> None:
    for name in names:
        src = os.path.join(run_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out, name), dirs_exist_ok=True)
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(out, name)), exist_ok=True)
            shutil.copy(src, os.path.join(out, name))


def _gather_jsonl(root: str, filename: str, event: set[str] | None = None) -> list[dict]:
    """Union of a per-session JSONL file over FINISHED sessions under root."""
    rows = []
    for dirpath, _, files in sorted(os.walk(root)):
        if filename in files and os.path.exists(os.path.join(dirpath, "result.json")):
            for row in _load_jsonl(os.path.join(dirpath, filename)):
                if event is None or row.get("event") in event:
                    rows.append({"session": os.path.relpath(dirpath, root), **row})
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --- README pieces --------------------------------------------------------------------

def _header(info: dict, run_id: str) -> list[str]:
    prov = info.get("provenance", {})
    L = [f"# {info['title']}", "", f"{METHOD_NAMES[info['method']]} · run `{run_id}`", ""]
    if info.get("notes"):
        L += [info["notes"], ""]
    commit = f"`{prov.get('git_commit')}`" + (" (with uncommitted changes)" if prov.get("git_dirty") else "")
    if prov.get("git_commit_source"):
        commit += f" ({prov['git_commit_source']})"
    L += [f"- Harness commit: {commit}", f"- Model: {prov.get('llm_provider', 'wandb')} / "
          f"{prov.get('llm_model', '(provider default)')}", f"- Settings: {info['settings']}", ""]
    return L


def _embed_report(path: str) -> list[str]:
    with open(path) as f:
        text = f.read().split("\n", 1)[1]  # drop its H1; ours is the title
    return [text.strip(), ""]


def _write_readme(out: str, lines: list[str]) -> None:
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# --- generation loop (prompt-only and RSIFlow) -----------------------------------------

def _generation_readme(out, run_id, info, cfg, lineage, improvers, sessions, has_eval, has_flow) -> None:
    L = _header(info, run_id)
    L += ["## Generation loop", "",
          f"Train: {', '.join(d['name'] for d in cfg['train'])} · held-out: {', '.join(d['name'] for d in cfg['heldout'])}"
          f" · test: {', '.join(d['name'] for d in cfg.get('test', [])) or '—'}", ""]
    heldout = [d["name"] for d in cfg["heldout"]]
    L += ["| Gen | Improver | Parent | " + " | ".join(f"{d} (held-out)" for d in heldout) + " | Decision |",
          "|---|---|---|" + "---|" * len(heldout) + "---|"]
    for e in lineage:
        scores = e.get("heldout") or {}
        decision = "seed" if e["generation"] == 0 else ("**accepted**" if e["accepted"] else "rejected")
        L.append(f"| {e['generation']} | `{e.get('improver') or '—'}` | `{e.get('parent') or '—'}` | "
                 + " | ".join(_fmt(scores.get(d)) for d in heldout) + f" | {decision} |")
    L += [""]
    for e in lineage[1:]:
        L += [f"**Gen {e['generation']} (`{e.get('improver')}`)**: {e.get('reason', '')}", ""]
        if e.get("rationale"):
            L += [f"> {e['rationale'].strip()}".replace("\n", "\n> "), ""]
    if has_eval:
        with open(os.path.join(out, "eval", "summary.md")) as f:
            summary = f.read().split("\n", 1)[1].replace("\n## ", "\n### ")
        L += ["## Final evaluation on test datasets", summary.strip(), "",
              "![Gain by generation](eval/generations.png)", "", "![Best score so far](eval/best_so_far.png)", ""]
    if has_flow:
        L += ["## RSIFlow", "", "Per-stage statistics, routing decisions and every workflow (with the meta-agent's "
              "reasons) are in [flow_report.md](flow_report.md).", ""]
    L += ["## Improvers", ""]
    for imp in improvers:
        body = json.dumps(imp["workflow"], indent=2) if imp.get("workflow") else imp["research_prompt"].strip()
        kind = "workflow" if imp.get("workflow") else "research prompt"
        L += [f"<details><summary><code>{imp['id']}</code> (generation {imp['generation']}, parent "
              f"<code>{imp.get('parent_id')}</code>): {kind}</summary>", "",
              "```" + ("json" if imp.get("workflow") else "text"), body, "```", "", "</details>", ""]
    L += ["## Files", "", "- `sessions.csv`: one row per Tumble session",
          "- `lineage.jsonl`: every generation's scores, decision and rationale",
          "- `improvers/`: every improver (prompts, and workflows for RSIFlow)",
          "- `config.json`, `provenance.json`: settings and harness version", f"- {len(sessions)} sessions in total"]
    _write_readme(out, L)


def _verdict(s: dict) -> str:
    """'+2.0% (clear gain)', noting crashed runs, which count as -100% and dominate small means."""
    failed = sum(p.get("failed_runs", 0) for p in s.get("per_dataset", {}).values())
    note = f", {failed} crashed run{'s' if failed > 1 else ''} counted as -100%" if failed else ""
    return f"{_fmt(s['mean_gain_vs_gen0'], pct=True)} ({s['verdict']}{note})"


def _generation_outcome(lineage, eval_results, flow: bool = False) -> str:
    proposed = len(lineage) - 1
    accepted = sum(1 for e in lineage[1:] if e["accepted"])
    champ = max((e["generation"] for e in lineage if e["accepted"]), default=0)
    text = f"{accepted}/{proposed} children accepted; champion gen{champ}"
    if eval_results:
        s = eval_results["summary"]
        ref = "the RSIFlow seed" if flow else "gen0"
        if champ and f"gen{champ}" in s:
            text += f"; test: gen{champ} vs {ref} {_verdict(s[f'gen{champ}'])}"
        labels = {"plain-tumble": "plain Tumble", "flow-static": "fixed-order routing",
                  "gen0-2x-budget": "2x budget", "blind-rewrite": "blind rewrite"}
        for name, label in labels.items():
            if name in s:
                text += f"; {label} vs {ref} {_verdict(s[name])}"
    return text


def export_generation(run_dir, out, info) -> list[dict]:
    _copy(run_dir, out, ["config.json", "lineage.jsonl", "champion.txt", "provenance.json", "improvers",
                         "flow_report.md"])
    eval_dir = os.path.join(run_dir, "eval")
    has_eval = os.path.exists(os.path.join(eval_dir, "summary.md"))
    eval_results = None
    if has_eval:
        _copy(eval_dir, os.path.join(out, "eval"), ["summary.md", "results.json", "generations.png", "best_so_far.png"])
        _copy(eval_dir, out, ["improvers"])
        eval_results = _load_json(os.path.join(eval_dir, "results.json"))
    cfg = _load_json(os.path.join(run_dir, "config.json"))
    lineage = _load_jsonl(os.path.join(run_dir, "lineage.jsonl"))
    improvers = sorted((_load_json(p) for p in glob.glob(os.path.join(out, "improvers", "*.json"))),
                       key=lambda i: (i["generation"], i["id"]))
    sessions = collect_sessions(run_dir)
    info["settings"] = (f"budget {cfg['budget']} · up to {cfg['max_generations']} generations · "
                        f"{(eval_results or {}).get('repeats', '—')} test repeats")
    info["outcome"] = _generation_outcome(lineage, eval_results, flow=info["method"] == "flow")
    _generation_readme(out, os.path.basename(run_dir), info, cfg, lineage, improvers, sessions, has_eval,
                       os.path.exists(os.path.join(run_dir, "flow_report.md")))
    return sessions


# --- Budget RSI ------------------------------------------------------------------------

def export_budget(run_dir, out, info) -> list[dict]:
    _copy(run_dir, out, ["config.json", "provenance.json", "report.md", "offline_strategies.png",
                         "learning_curve.png", "scorer_v2.json", "live_summary.json", "analysis/offline.json",
                         "analysis/scorer_v1.json"])
    labels = _gather_jsonl(os.path.join(run_dir, "collect"), "labels.jsonl") + \
        _gather_jsonl(os.path.join(run_dir, "live"), "labels.jsonl")
    _write_jsonl(os.path.join(out, "labels.jsonl"), labels)
    cfg = _load_json(os.path.join(run_dir, "config.json"))
    offline = _load_json(os.path.join(run_dir, "analysis", "offline.json"))
    live = _load_json(os.path.join(run_dir, "live_summary.json")) if os.path.exists(
        os.path.join(run_dir, "live_summary.json")) else None
    sessions = collect_sessions(run_dir)
    info["settings"] = (f"k {cfg['k']} candidates · budget {cfg['rounds']} · {cfg['collect_repeats']} collect / "
                        f"{cfg['live_repeats']} live repeats")
    g, smr = offline["mean_realized_gain"], offline["scorer_minus_random"]
    text = (f"offline: scorer {_fmt(g['scorer'], pct=True)} vs random {_fmt(g['random'], pct=True)} "
            f"(diff 95% CI [{smr['ci95'][0]:+.1%}, {smr['ci95'][1]:+.1%}])")
    if live:
        for name in ("scorer", "random", "first"):
            text += f"; live {name} pick vs plain Tumble {_verdict(live[name])}"
    info["outcome"] = text
    L = _header(info, os.path.basename(run_dir)) + _embed_report(os.path.join(run_dir, "report.md"))
    L += ["## Files", "", "- `labels.jsonl`: every executed candidate edit -- features, outcome, label, and for live "
          "runs which policy picked it", "- `analysis/offline.json`: the offline replay, learning curve and v1 weights",
          "- `analysis/scorer_v1.json`, `scorer_v2.json`: the learned scorer before and after its live outcomes",
          "- `live_summary.json`, `sessions.csv`, `config.json`, `provenance.json`",
          f"- {len(labels)} labeled edits, {len(sessions)} sessions"]
    _write_readme(out, L)
    return sessions


# --- SO-RSI ------------------------------------------------------------------------------

def export_sorsi(run_dir, out, info) -> list[dict]:
    _copy(run_dir, out, ["config.json", "provenance.json", "report.md", "results.json", "frozen_lessons.txt"])
    diagnoses = _gather_jsonl(run_dir, "trace.jsonl", {"anomaly", "diagnosis", "diagnosis_tested"})
    memory = _gather_jsonl(run_dir, "memory.jsonl")
    _write_jsonl(os.path.join(out, "inquiry_events.jsonl"), diagnoses)
    _write_jsonl(os.path.join(out, "memory.jsonl"), memory)
    cfg = _load_json(os.path.join(run_dir, "config.json"))
    results = _load_json(os.path.join(run_dir, "results.json"))
    sessions = collect_sessions(run_dir)
    info["settings"] = f"budget {cfg['budget']} · {cfg['rounds']} build rounds · {cfg['repeats']} test repeats"
    s = results.get("summary") or {}
    parts = []
    for name in ("sorsi_nomem", "sorsi"):
        if name in s:
            label = "with memory" if name == "sorsi" else "without memory"
            parts.append(f"SO-RSI {label} vs plain Tumble {_verdict(s[name])}")
    confirmed = sum(1 for m in memory if m.get("confirmed"))
    parts.append(f"{confirmed}/{len(memory)} tested diagnoses confirmed (build + test)")
    info["outcome"] = "; ".join(parts)
    L = _header(info, os.path.basename(run_dir)) + _embed_report(os.path.join(run_dir, "report.md"))
    L += ["## Files", "", "- `frozen_lessons.txt`: the inquiry memory given to the test sessions",
          "- `memory.jsonl`: every tested diagnosis (root cause, evidence, change, confirmed or not)",
          "- `inquiry_events.jsonl`: every anomaly, diagnosis and test of a diagnosis, per session",
          "- `results.json`, `sessions.csv`, `config.json`, `provenance.json`",
          f"- {len(sessions)} sessions"]
    _write_readme(out, L)
    return sessions


# --- results log -----------------------------------------------------------------------

def update_results_log() -> None:
    rows = []
    infos = []
    for info_path in glob.glob(os.path.join(RESULTS, "*", "run_info.json")):
        info = _load_json(info_path)
        info["run_id"] = os.path.basename(os.path.dirname(info_path))
        infos.append(info)
    for info in sorted(infos, key=lambda i: i.get("provenance", {}).get("started_at") or i["run_id"]):
        prov = info.get("provenance", {})
        rows.append(f"| [{info['run_id']}](results/{info['run_id']}/README.md) | {METHOD_NAMES[info['method']]} | "
                    f"{info['title']} | `{prov.get('git_commit')}`{' + uncommitted' if prov.get('git_dirty') else ''} | "
                    f"{info['settings']} | {info['outcome']} |")
    table = "\n".join([LOG_START, "| Run | Method | Title | Harness commit | Settings | Outcome |",
                       "|---|---|---|---|---|---|", *rows, LOG_END])
    readme = os.path.join(ROOT, "README.md")
    with open(readme) as f:
        text = f.read()
    if LOG_START not in text:
        raise SystemExit(f"README.md has no {LOG_START} marker -- add a Results log section first")
    start, end = text.index(LOG_START), text.index(LOG_END) + len(LOG_END)
    with open(readme, "w") as f:
        f.write(text[:start] + table + text[end:])


def export(run_dir: str, title: str, notes: str = "", trajectories: bool = False, commit: str | None = None,
           started_at: str | None = None) -> str:
    run_dir = os.path.abspath(run_dir.rstrip(os.sep))
    run_id = os.path.basename(run_dir)
    method = detect_method(run_dir)
    out = os.path.join(RESULTS, run_id)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    prov = _load_json(os.path.join(run_dir, "provenance.json")) if os.path.exists(
        os.path.join(run_dir, "provenance.json")) else {}
    if commit:  # runs started before provenance was recorded
        prov = {**prov, "git_commit": commit, "git_commit_source": "given at export"}
    if started_at and not prov.get("started_at"):
        prov["started_at"] = started_at
    if not prov.get("started_at"):
        prov["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(os.path.getmtime(os.path.join(run_dir, "config.json"))))
    info = {"title": title, "notes": notes, "method": method, "provenance": prov}

    exporter = {"prompt": export_generation, "flow": export_generation, "budget": export_budget,
                "sorsi": export_sorsi}[method]
    sessions = exporter(run_dir, out, info)
    _write_sessions(out, sessions)
    if prov:
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump(prov, f, indent=2)
    if trajectories:
        for path in glob.glob(os.path.join(run_dir, "**", "trajectory.jsonl"), recursive=True):
            dest = os.path.join(out, "trajectories",
                                os.path.relpath(os.path.dirname(path), run_dir).replace(os.sep, "__") + ".jsonl")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(path, dest)
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    update_results_log()
    print(f"exported {run_id} ({METHOD_NAMES[method]}) -> {os.path.relpath(out, ROOT)} ({len(sessions)} sessions)")
    print(f"outcome: {info['outcome']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export a finished run from runs/ into committed results/.")
    p.add_argument("run_dir")
    p.add_argument("--title", required=True, help="short name for the results log")
    p.add_argument("--notes", default="", help="a paragraph of context shown at the top of the run README")
    p.add_argument("--trajectories", action="store_true", help="also copy every session's trajectory.jsonl")
    p.add_argument("--commit", help="harness git commit, for runs started before provenance.json existed")
    p.add_argument("--started-at", help="'YYYY-MM-DD HH:MM:SS', for runs without provenance (orders the log)")
    a = p.parse_args()
    export(a.run_dir, a.title, a.notes, a.trajectories, a.commit, a.started_at)


if __name__ == "__main__":
    main()
