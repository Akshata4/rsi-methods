"""Append-only JSONL event log for one agent run (copied from tumble/trajectory.py).

Change from Tumble: the caller chooses the output path, so each RSI
generation/dataset run gets its own file next to its other artifacts.
"""

import json
import os
import time
from typing import Callable


class Trajectory:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "a")
        self._subscribers: list[Callable[[dict], None]] = []

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subscribers.append(fn)

    def _log(self, event_type: str, data: dict) -> None:
        record = {"ts": time.time(), "type": event_type, **data}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        for sub in self._subscribers:
            try:
                sub(record)
            except Exception:
                pass

    def log_meta(self, provider: str, model: str) -> None:
        self._log("meta", {"provider": provider, "model": model})

    def log_system(self, content: str) -> None:
        self._log("system", {"content": content})

    def log_user(self, content: str) -> None:
        self._log("user", {"content": content})

    def log_assistant(self, content: str | None, tool_calls: list[dict] | None, turn: int | None = None) -> None:
        self._log("assistant", {"content": content, "tool_calls": tool_calls, "turn": turn})

    def log_tool_result(self, name: str, args: dict, result: str, turn: int | None = None) -> None:
        self._log("tool", {"name": name, "args": args, "result": result, "turn": turn})

    def log_notice(self, content: str) -> None:
        self._log("notice", {"content": content})

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
