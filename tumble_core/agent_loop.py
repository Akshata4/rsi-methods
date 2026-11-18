"""The propose -> execute -> evaluate -> keep/revert agent loop (from tumble/agent_loop.py).

Changes from Tumble: the global daily budget is removed (single-user research
tool), and the trajectory is closed by the caller.
"""

import json
import os
import time

from tumble_core.llm_client import get_client_and_model
from tumble_core.trajectory import Trajectory

MAX_TURNS = 40
MAX_TOOL_RESULT_CHARS = 40_000  # ~10k tokens: small CSVs fit whole, huge ones get truncated


def _safe_parse_args(raw: str) -> tuple[dict | None, str | None]:
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, str(e)


def run_agent(
    prompt: str,
    system_prompt: str,
    tool_schemas: list[dict],
    tool_functions: dict,
    trajectory: Trajectory,
    max_turns: int = MAX_TURNS,
    max_wall_clock_seconds: float | None = None,
    usage: dict | None = None,
) -> str:
    """Returns "completed", "max_turns_exceeded", or "time_limit_exceeded".
    If `usage` is given, LLM token counts are added to it (usage["total_tokens"], usage["llm_calls"])."""
    client, model = get_client_and_model()
    trajectory.log_meta(os.environ.get("LLM_PROVIDER", "wandb"), model)
    trajectory.log_system(system_prompt)
    trajectory.log_user(prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    start = time.monotonic()

    for turn in range(1, max_turns + 1):
        if max_wall_clock_seconds is not None and time.monotonic() - start > max_wall_clock_seconds:
            trajectory.log_notice(f"stopped after {max_wall_clock_seconds:.0f}s wall-clock limit")
            return "time_limit_exceeded"

        response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
        if usage is not None:
            usage["llm_calls"] = usage.get("llm_calls", 0) + 1
            if getattr(response, "usage", None) is not None:
                usage["total_tokens"] = usage.get("total_tokens", 0) + (response.usage.total_tokens or 0)
        assistant_message = response.choices[0].message
        parsed_calls = [(c, *_safe_parse_args(c.function.arguments)) for c in (assistant_message.tool_calls or [])]

        # Never resend malformed tool-call JSON to the API (it rejects the whole request).
        dumped = assistant_message.model_dump(exclude_none=True)
        # A tool-call-only turn has content=None, which exclude_none drops. W&B's
        # DeepSeek chat template then fails with a 400 ("+ operator on undefined
        # and string") when this message is resent, so always send a string.
        dumped.setdefault("content", "")
        if dumped.get("tool_calls"):
            for tc, (_, _, err) in zip(dumped["tool_calls"], parsed_calls):
                if err is not None:
                    tc["function"]["arguments"] = "{}"
        messages.append(dumped)

        tool_calls_log = None
        if parsed_calls:
            tool_calls_log = [
                {"id": c.id, "name": c.function.name, "args": args if args is not None else {"_parse_error": err}}
                for c, args, err in parsed_calls
            ]
        trajectory.log_assistant(assistant_message.content, tool_calls_log, turn=turn)

        if not assistant_message.tool_calls:
            return "completed"

        for call, args, parse_error in parsed_calls:
            fn_name = call.function.name
            if parse_error is not None:
                result = f"Error: your arguments for {fn_name} were not valid JSON ({parse_error}). Please retry with valid JSON."
            elif fn_name not in tool_functions:
                result = f"Error: unknown tool {fn_name}."
            else:
                try:
                    result = tool_functions[fn_name](**args)
                except TypeError as e:
                    result = f"Error: bad arguments for {fn_name}: {e}"
            if len(result) > MAX_TOOL_RESULT_CHARS:
                # e.g. read_file("dataset.csv") on an 8k-row CSV is ~1.5M chars and
                # overflows the model's context window, killing the whole session.
                result = (
                    result[:MAX_TOOL_RESULT_CHARS]
                    + f"\n\n[truncated: output was {len(result):,} characters, showing the first "
                    f"{MAX_TOOL_RESULT_CHARS:,}. To inspect data, load it with pandas inside train.py.]"
                )
            trajectory.log_tool_result(fn_name, args or {}, result, turn=turn)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    trajectory.log_notice(f"stopped after {max_turns} turns without a final answer")
    return "max_turns_exceeded"
