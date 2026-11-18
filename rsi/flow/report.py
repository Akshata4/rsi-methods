"""RSIFlow-specific report for a finished run: how the workflow was used and how it evolved.

    uv run python -m rsi.flow.report runs/<timestamp>

Writes <run_dir>/flow_report.md with
- per-stage statistics per improver (attempts, kept, crashes, mean gain, edits matching the stage),
- how often each routing decision fired,
- the champion workflow next to the seed (what the meta-agent changed).
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict


def _events(run_dir: str) -> list[dict]:
    out = []
    for path in glob.glob(os.path.join(run_dir, "**", "flow.jsonl"), recursive=True):
        rel = os.path.relpath(path, run_dir).split(os.sep)
        group = f"eval/{rel[1]}" if rel[0] == "eval" else f"{rel[0]} ({rel[1]})"
        with open(path) as f:
            out += [{**json.loads(line), "group": group} for line in f if line.strip()]
    return out


def _reason_kind(reason: str) -> str:
    for key, label in [("crashed", "crash -> fix"), ("fix worked", "fix worked -> resume"), ("improved", "gain -> stay"),
                       ("stalled", "stalled -> move on"), ("abandoning", "repeated crashes -> abandon"),
                       ("no working script", "no working script -> fix"), ("simpler baseline", "restart baseline"),
                       ("not stalled yet", "keep trying"), ("fixed order", "fixed order (ablation)"),
                       ("moving on", "baseline done -> first stage"), ("starting", "start")]:
        if key in reason:
            return label
    return reason[:40]


def write_flow_report(run_dir: str) -> str:
    events = [e for e in _events(run_dir) if e["event"] == "experiment"]
    L = ["# RSIFlow: workflow usage and evolution", ""]

    by_group = defaultdict(list)
    for e in events:
        by_group[e["group"]].append(e)
    L += ["## Per-stage statistics", "",
          "Gain = relative change vs the best score before that experiment (crashes excluded). "
          "Match = the edit actually did that stage's kind of change (static edit features).", ""]
    for group in sorted(by_group):
        L += [f"### {group}", "", "| Stage | Attempts | Kept | Crashes | Mean gain | Edits matching stage |",
              "|---|---|---|---|---|---|"]
        stages = defaultdict(list)
        for e in by_group[group]:
            stages[e["stage"]].append(e)
        for stage, es in stages.items():
            gains = [e["gain"] for e in es if e.get("gain") is not None]
            matched = [e["on_stage"] for e in es if e.get("on_stage") is not None]
            gain_txt = f"{sum(gains) / len(gains):+.2%}" if gains else "n/a"
            match_txt = f"{sum(matched)}/{len(matched)}" if matched else "n/a"
            L.append(f"| {stage} | {len(es)} | {sum(e['kept'] for e in es)} | {sum(e['crashed'] for e in es)} | "
                     f"{gain_txt} | {match_txt} |")
        L += [""]

    reasons = Counter(_reason_kind(e.get("reason", "")) for e in events)
    L += ["## Routing decisions (all sessions)", "", "| Decision | Count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    L += [""]

    improvers = {}
    for path in glob.glob(os.path.join(run_dir, "improvers", "*.json")):
        with open(path) as f:
            imp = json.load(f)
        improvers[imp["id"]] = imp
    champion_id = open(os.path.join(run_dir, "champion.txt")).read().strip() \
        if os.path.exists(os.path.join(run_dir, "champion.txt")) else None
    seed = next((i for i in improvers.values() if i["generation"] == 0), None)
    if seed and seed.get("workflow"):
        L += ["## Workflows", ""]
        for imp in sorted(improvers.values(), key=lambda i: (i["generation"], i["id"])):
            if not imp.get("workflow"):
                continue
            tag = " (champion)" if imp["id"] == champion_id else ""
            L += [f"<details><summary><code>{imp['id']}</code>, generation {imp['generation']}{tag}</summary>", "",
                  "```json", json.dumps(imp["workflow"], indent=2), "```", ""]
            if imp.get("rationale") and imp["generation"] > 0:
                L += [f"> {imp['rationale'].strip()}".replace("\n", "\n> "), ""]
            L += ["</details>", ""]
    path = os.path.join(run_dir, "flow_report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    print(write_flow_report(sys.argv[1]))
