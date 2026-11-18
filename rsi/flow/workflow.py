"""The workflow: what RSIFlow's meta-agent evolves.

A workflow is plain data (JSON), so a child can be written by an LLM and then
checked here before it is ever used. It can change stage instructions, add or
remove optional stages, reorder them, and tune the routing rules -- but it can
never touch scoring, budget or tools; those stay in the harness.
"""

import copy

REQUIRED_STAGES = ("baseline", "fix_errors")  # the router relies on these two
MAX_STAGES = 10
MAX_INSTRUCTION_CHARS = 600
RULE_BOUNDS = {
    "max_fix_attempts": (1, 4),
    "plateau_patience": (1, 4),
    "min_gain": (0.0, 0.05),
}

SEED_WORKFLOW = {
    "stages": {
        "baseline": "Write a robust, simple first train.py: a pipeline with imputation, scaling, one-hot "
                    "encoding with handle_unknown='ignore', and logistic regression.",
        "fix_errors": "The last script crashed. Read the error, find its cause, and fix only that; keep "
                      "everything else unchanged.",
        "preprocess": "Improve data handling: missing values (including zeros that really mean missing), "
                      "encoding of categoricals, scaling, outliers.",
        "model": "Try a different model family suited to this data (e.g. tree ensembles vs linear models).",
        "features": "Engineer new features from domain knowledge: ratios, interactions, bins, missingness "
                    "indicators.",
        "tune": "Tune the current model's most important hyperparameters, one or two at a time.",
        "ensemble": "Combine the strongest, most different approaches so far (e.g. soft voting of two models).",
    },
    "order": ["preprocess", "model", "features", "tune", "ensemble"],
    "rules": {
        "adaptive": True,          # False = follow `order` blindly (the ablation)
        "max_fix_attempts": 2,     # crash fixes in a row before abandoning the idea
        "plateau_patience": 2,     # non-improving experiments in a stage before it counts as stalled
        "min_gain": 0.002,         # relative gain that counts as an improvement
        "stay_on_gain": True,      # keep working a stage while it improves
    },
}


def static_variant(workflow: dict) -> dict:
    """Same stages and instructions, but fixed-order routing: isolates the router's effect."""
    w = copy.deepcopy(workflow)
    w["rules"]["adaptive"] = False
    return w


def validate_workflow(w) -> list[str]:
    problems = []
    if not isinstance(w, dict):
        return ["workflow must be an object"]
    stages, order, rules = w.get("stages"), w.get("order"), w.get("rules")
    if not isinstance(stages, dict) or not stages:
        return ["workflow.stages must be a non-empty object of name -> instruction"]
    if len(stages) > MAX_STAGES:
        problems.append(f"at most {MAX_STAGES} stages")
    for name in REQUIRED_STAGES:
        if name not in stages:
            problems.append(f"stage '{name}' is required")
    for name, text in stages.items():
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            problems.append(f"bad stage name {name!r}")
        if not isinstance(text, str) or not 20 <= len(text) <= MAX_INSTRUCTION_CHARS:
            problems.append(f"stage {name!r}: instruction must be 20-{MAX_INSTRUCTION_CHARS} characters")
    if not isinstance(order, list) or not order:
        problems.append("workflow.order must be a non-empty list of stage names")
    else:
        for name in order:
            if name not in (stages or {}):
                problems.append(f"order mentions unknown stage {name!r}")
            if name in REQUIRED_STAGES:
                problems.append(f"{name!r} is routed automatically; leave it out of order")
    if not isinstance(rules, dict):
        problems.append("workflow.rules must be an object")
    else:
        for key, (lo, hi) in RULE_BOUNDS.items():
            v = rules.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
                problems.append(f"rules.{key} must be a number in [{lo}, {hi}]")
        for key in ("adaptive", "stay_on_gain"):
            if not isinstance(rules.get(key), bool):
                problems.append(f"rules.{key} must be true or false")
    return problems


def render_stages(w: dict) -> str:
    return "\n".join(f"  * {name}: {text}" for name, text in w["stages"].items())
