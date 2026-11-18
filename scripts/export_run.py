"""Export a finished run from runs/ (gitignored) into results/ (committed).

    uv run python scripts/export_run.py runs/<timestamp> --title "First real run" --notes "..."

Copies the parts worth keeping -- config, provenance, lineage, every improver's
prompts, the final evaluation (summary, numbers, charts) -- writes a per-session
table (sessions.csv) and a readable results/<run>/README.md, then regenerates
the results log table in the top-level README.md. Bulky per-session
workspaces and trajectories stay in runs/ unless --trajectories is given.
"""

import argparse
import csv
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LOG_START, LOG_END = "<!-- results-log:start -->", "<!-- results-log:end -->"
SESSION_FIELDS = ["phase", "improver", "split", "dataset", "repeat", "budget", "best_metric",
                  "gain_over_baseline", "experiments_to_best", "experiments_used", "wasted_rate",
                  "failure_rate", "total_tokens", "wall_seconds", "status"]


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(x, pct=False, digits=3):
    if x is None:
        return "—"
    return f"{x:+.1%}" if pct else f"{x:.{digits}f}"


def collect_sessions(run_dir: str) -> list[dict]:
    """One row per Tumble session, from every result.json in the run."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "result.json"), recursive=True)):
        rel = os.path.relpath(path, run_dir).split(os.sep)
        with open(path) as f:
            r = json.load(f)
        m = r.get("metrics", {})
        if rel[0] == "eval":   # eval/<contender>/<dataset>/r<i>/result.json
            phase, improver, split, repeat = "eval", rel[1], "test", rel[3].lstrip("r")
        else:                  # <improver>/<split>/<dataset>/result.json
            phase, improver, split, repeat = "loop", rel[0], rel[1], ""
        rows.append({
            "phase": phase, "improver": improver, "split": split, "dataset": r["dataset"], "repeat": repeat,
            "budget": r.get("budget"), "best_metric": r.get("best_metric"),
            "gain_over_baseline": m.get("gain_over_baseline"), "experiments_to_best": m.get("experiments_to_best"),
            "experiments_used": r.get("experiments_used"), "wasted_rate": m.get("wasted_rate"),
            "failure_rate": m.get("failure_rate"), "total_tokens": r.get("total_tokens"),
            "wall_seconds": r.get("wall_seconds"), "status": r.get("status"),
        })
    return rows


def write_run_readme(out: str, run_id: str, info: dict, cfg: dict, lineage: list[dict],
                     improvers: list[dict], sessions: list[dict], has_eval: bool) -> None:
    prov = info.get("provenance", {})
    L = [f"# {info['title']}", "", f"Run `{run_id}`", ""]
    if info.get("notes"):
        L += [info["notes"], ""]
    L += ["## Setup", "",
          f"- Harness commit: `{prov.get('git_commit')}`" + (" (with uncommitted changes)" if prov.get("git_dirty") else ""),
          f"- Model: {prov.get('llm_provider')} / {prov.get('llm_model')}",
          f"- Budget: {cfg['budget']} experiments per session · max generations: {cfg['max_generations']} · "
          f"patience: {cfg['patience']} · min gain: {cfg['min_gain']:.0%}",
          f"- Train: {', '.join(d['name'] for d in cfg['train'])}",
          f"- Held-out: {', '.join(d['name'] for d in cfg['heldout'])}",
          f"- Test: {', '.join(d['name'] for d in cfg.get('test', [])) or '—'}", "",
          "## Generation loop", ""]
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
            summary = f.read().split("\n", 1)[1]  # drop its H1
        summary = summary.replace("\n## ", "\n### ")  # nest its sections under ours
        L += ["## Final evaluation on test datasets", summary.strip(), "",
              "![Gain by generation](eval/generations.png)", "", "![Best score so far](eval/best_so_far.png)", ""]
    else:
        L += ["## Final evaluation", "", "No final evaluation in this run.", ""]

    L += ["## Improvers", "", "Each improver's full prompts are in `improvers/`. Research prompts:", ""]
    for imp in improvers:
        L += [f"<details><summary><code>{imp['id']}</code> (generation {imp['generation']}, parent "
              f"<code>{imp.get('parent_id')}</code>)</summary>", "", "```text", imp["research_prompt"].strip(), "```",
              "", "</details>", ""]
    L += ["## Files", "",
          "- `sessions.csv`: one row per Tumble session (scores, gain over baseline, wasted rate, tokens)",
          "- `lineage.jsonl`: every generation's scores, decision and rationale",
          "- `improvers/`: every improver's research and meta prompts",
          "- `config.json`, `provenance.json`: exact settings and harness version",
          f"- {len(sessions)} sessions in total"]
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def outcome(lineage: list[dict], eval_results: dict | None) -> str:
    proposed = len(lineage) - 1
    accepted = sum(1 for e in lineage[1:] if e["accepted"])
    champ = max((e["generation"] for e in lineage if e["accepted"]), default=0)
    text = f"{accepted}/{proposed} children accepted; champion gen{champ}"
    if eval_results:
        s = eval_results["summary"]
        name = f"gen{champ}"
        if champ and name in s:
            v = s[name]
            text += f"; test gain vs gen0 {_fmt(v['mean_gain_vs_gen0'], pct=True)} ({v['verdict']})"
        elif "gen0" in s:
            text += "; no champion beyond gen0 to test"
    return text


def update_results_log() -> None:
    rows = []
    for info_path in sorted(glob.glob(os.path.join(RESULTS, "*", "run_info.json"))):
        with open(info_path) as f:
            info = json.load(f)
        run_id = os.path.basename(os.path.dirname(info_path))
        prov, cfg = info.get("provenance", {}), info.get("config_summary", {})
        rows.append(f"| [{run_id}](results/{run_id}/README.md) | {info['title']} | `{prov.get('git_commit')}` | "
                    f"{cfg.get('budget')} / {cfg.get('max_generations')} / {cfg.get('repeats', '—')} | {info['outcome']} |")
    table = "\n".join([
        LOG_START,
        "| Run | Title | Harness commit | Budget / gens / repeats | Outcome |",
        "|---|---|---|---|---|",
        *rows,
        LOG_END,
    ])
    readme = os.path.join(ROOT, "README.md")
    with open(readme) as f:
        text = f.read()
    if LOG_START not in text:
        raise SystemExit(f"README.md has no {LOG_START} marker -- add a Results log section first")
    start, end = text.index(LOG_START), text.index(LOG_END) + len(LOG_END)
    with open(readme, "w") as f:
        f.write(text[:start] + table + text[end:])


def export(run_dir: str, title: str, notes: str = "", trajectories: bool = False, commit: str | None = None,
           repeats: int | None = None) -> str:
    run_dir = os.path.abspath(run_dir)
    run_id = os.path.basename(run_dir.rstrip(os.sep))
    out = os.path.join(RESULTS, run_id)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    for name in ("config.json", "lineage.jsonl", "champion.txt", "provenance.json"):
        if os.path.exists(os.path.join(run_dir, name)):
            shutil.copy(os.path.join(run_dir, name), out)
    shutil.copytree(os.path.join(run_dir, "improvers"), os.path.join(out, "improvers"))
    eval_dir = os.path.join(run_dir, "eval")
    has_eval = os.path.exists(os.path.join(eval_dir, "summary.md"))
    eval_results = None
    if has_eval:
        os.makedirs(os.path.join(out, "eval"))
        for name in ("summary.md", "results.json", "generations.png", "best_so_far.png"):
            if os.path.exists(os.path.join(eval_dir, name)):
                shutil.copy(os.path.join(eval_dir, name), os.path.join(out, "eval"))
        if os.path.exists(os.path.join(eval_dir, "improvers")):
            shutil.copytree(os.path.join(eval_dir, "improvers"), os.path.join(out, "improvers"), dirs_exist_ok=True)
        with open(os.path.join(eval_dir, "results.json")) as f:
            eval_results = json.load(f)
    if trajectories:
        for path in glob.glob(os.path.join(run_dir, "**", "trajectory.jsonl"), recursive=True):
            dest = os.path.join(out, "trajectories", os.path.relpath(os.path.dirname(path), run_dir).replace(os.sep, "__") + ".jsonl")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(path, dest)

    sessions = collect_sessions(run_dir)
    with open(os.path.join(out, "sessions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        w.writeheader()
        w.writerows(sessions)

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    prov = {}
    if os.path.exists(os.path.join(run_dir, "provenance.json")):
        with open(os.path.join(run_dir, "provenance.json")) as f:
            prov = json.load(f)
    if commit:  # runs started before provenance was recorded
        prov = {**prov, "git_commit": commit, "git_commit_source": "given at export"}
    lineage = _load_jsonl(os.path.join(run_dir, "lineage.jsonl"))
    improvers = []
    for path in sorted(glob.glob(os.path.join(out, "improvers", "*.json"))):
        with open(path) as f:
            improvers.append(json.load(f))
    improvers.sort(key=lambda i: (i["generation"], i["id"]))

    info = {
        "title": title, "notes": notes, "provenance": prov, "outcome": outcome(lineage, eval_results),
        "config_summary": {"budget": cfg["budget"], "max_generations": cfg["max_generations"],
                           "repeats": repeats or (eval_results or {}).get("repeats")},
    }
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    write_run_readme(out, run_id, info, cfg, lineage, improvers, sessions, has_eval)
    update_results_log()
    print(f"exported {run_id} -> {os.path.relpath(out, ROOT)} ({len(sessions)} sessions)")
    print(f"outcome: {info['outcome']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Export a finished run from runs/ into committed results/.")
    p.add_argument("run_dir")
    p.add_argument("--title", required=True, help="short name for the results log, e.g. 'First real run'")
    p.add_argument("--notes", default="", help="a paragraph of context shown at the top of the run README")
    p.add_argument("--trajectories", action="store_true", help="also copy every session's trajectory.jsonl")
    p.add_argument("--commit", help="harness git commit, for runs started before provenance.json existed")
    p.add_argument("--repeats", type=int, help="eval repeats, if the run had no final evaluation")
    a = p.parse_args()
    export(a.run_dir, a.title, a.notes, a.trajectories, a.commit, a.repeats)


if __name__ == "__main__":
    main()
