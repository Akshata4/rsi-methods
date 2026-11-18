"""The self-rewrite step: a parent improver writes its own child.

The parent's meta_prompt is the system message, and the parent's own prompts
plus its run evidence are the input. The child that comes back carries a
new research_prompt and possibly a new meta_prompt, so the next rewrite is
done by instructions an earlier generation wrote.
"""

import json
import re

from rsi.improver import Improver
from rsi.run_task import RunResult
from tumble_core.llm_client import get_client_and_model

MAX_ATTEMPTS = 3


def stage_statistics(results: list[RunResult]) -> str:
    """RSIFlow execution feedback: per-stage outcomes across all runs."""
    stats: dict[str, dict] = {}
    for r in results:
        for e in r.history:
            if "stage" not in e:
                continue
            s = stats.setdefault(e["stage"], {"attempts": 0, "kept": 0, "crashes": 0, "gains": [], "on": 0, "on_n": 0})
            s["attempts"] += 1
            s["kept"] += bool(e["kept"])
            s["crashes"] += e["metric"] is None
            if e.get("gain") is not None:
                s["gains"].append(e["gain"])
            if e.get("on_stage") is not None:
                s["on_n"] += 1
                s["on"] += bool(e["on_stage"])
    if not stats:
        return ""
    lines = ["## Per-stage statistics (all runs above)",
             "| stage | attempts | kept | crashes | mean relative gain | edits matching the stage |", "|---|---|---|---|---|---|"]
    for name, s in stats.items():
        gain = f"{sum(s['gains']) / len(s['gains']):+.2%}" if s["gains"] else "n/a"
        match = f"{s['on']}/{s['on_n']}" if s["on_n"] else "n/a"
        lines.append(f"| {name} | {s['attempts']} | {s['kept']} | {s['crashes']} | {gain} | {match} |")
    return "\n".join(lines)


def summarize_evidence(results: list[RunResult]) -> str:
    if not results:
        return "(no evidence provided -- rely on general ML research knowledge)"
    blocks = []
    for r in results:
        direction = "lower is better" if r.minimize else "higher is better"
        lines = [f"## Dataset: {r.dataset} ({r.metric}, {direction})",
                 f"Final best: {r.best_metric}  |  experiments used: {r.experiments_used}  |  status: {r.status}"]
        for e in r.history:
            m = e["metric"] if e["metric"] is not None else f"FAILED: {e.get('error', '')[:150]}"
            stage = f"[{e['stage']}] " if "stage" in e else ""
            lines.append(f"- iter {e['iteration']}: {stage}{m} ({'kept' if e['kept'] else 'reverted'}) -- {e['description']}")
        if r.report:
            lines.append(f"Agent's own report (truncated):\n{r.report[:1200]}")
        blocks.append("\n".join(lines))
    stage_table = stage_statistics(results)
    return "\n\n".join(blocks + ([stage_table] if stage_table else []))


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def propose_child(
    parent: Improver, evidence: list[RunResult], child_id: str, feedback: str = "", usage: dict | None = None
) -> Improver:
    """Ask the parent (via its own meta_prompt) to write the next improver.
    Raises RuntimeError if no valid child is produced after a few attempts."""
    client, model = get_client_and_model()
    workflow_block = f"# Current workflow\n{json.dumps(parent.workflow, indent=2)}\n\n" if parent.workflow else ""
    user = (
        workflow_block +
        f"# Current research instructions\n{parent.research_prompt}\n\n"
        f"# Current meta-instructions\n{parent.meta_prompt}\n\n"
        f"# Evidence from runs with the current instructions\n{summarize_evidence(evidence)}"
    )
    if feedback:
        user += f"\n\n# Earlier children of these instructions that were REJECTED\n{feedback}"
    messages = [{"role": "system", "content": parent.meta_prompt}, {"role": "user", "content": user}]

    last_problem = ""
    for _ in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
        except Exception as e:  # API failure: report it as "no valid child", don't crash the generation loop
            raise RuntimeError(f"LLM call failed: {type(e).__name__}: {e}") from e
        if usage is not None and getattr(response, "usage", None) is not None:
            usage["total_tokens"] = usage.get("total_tokens", 0) + (response.usage.total_tokens or 0)
        text = response.choices[0].message.content
        data = _extract_json(text)
        if data is None:
            last_problem = "reply was not a JSON object"
        else:
            child = Improver(
                # A workflow improver evolves its workflow; its research prompt is inherited unless rewritten.
                research_prompt=str(data.get("research_prompt") or (parent.research_prompt if parent.workflow else "")),
                workflow=data.get("workflow") if parent.workflow is not None else None,
                # If the parent chose not to touch its meta_prompt, inherit it.
                meta_prompt=str(data.get("meta_prompt") or parent.meta_prompt),
                generation=parent.generation + 1,
                parent_id=parent.id,
                rationale=str(data.get("rationale", "")),
                id=child_id,
            )
            problems = child.validate()
            if not problems:
                return child
            last_problem = "; ".join(problems)
        messages += [
            {"role": "assistant", "content": text or ""},
            {"role": "user", "content": f"That reply was invalid ({last_problem}). Reply with ONLY the JSON object."},
        ]
    raise RuntimeError(f"parent {parent.id} failed to produce a valid child: {last_problem}")
