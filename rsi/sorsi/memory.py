"""Inquiry memory: causal evidence that persists across sessions and rounds.

Every diagnosis the agent records is tested by its NEXT experiment and only
then stored, with `confirmed` set:
  - anomaly triggered by an error: confirmed if the next experiment ran cleanly
  - anomaly about scores (no error): confirmed if the next experiment was kept
Only confirmed diagnoses become lessons shown to future sessions. Growing,
verified memory is what lets later rounds avoid earlier rounds' thrashing --
the recursive part of SO-RSI here.

Each session writes its own memory.jsonl; the shared memory is the union over
FINISHED sessions (those with a result.json), so an interrupted session can't
leave half-written or duplicate lessons.
"""

import json
import os

MAX_LESSONS = 8


class InquiryMemory:
    def __init__(self, entries: list[dict] | None = None):
        self.entries = list(entries or [])

    @classmethod
    def from_runs(cls, root: str) -> "InquiryMemory":
        entries = []
        if os.path.isdir(root):
            for dirpath, _, files in sorted(os.walk(root)):
                if "memory.jsonl" in files and os.path.exists(os.path.join(dirpath, "result.json")):
                    with open(os.path.join(dirpath, "memory.jsonl")) as f:
                        entries += [json.loads(line) for line in f if line.strip()]
        return cls(entries)

    def confirmed(self) -> list[dict]:
        return [e for e in self.entries if e.get("confirmed")]

    def lessons(self) -> list[dict]:
        """Most recent confirmed diagnoses, one per (anomaly, root cause)."""
        seen, out = set(), []
        for e in reversed(self.confirmed()):
            key = (e["anomaly"], e["root_cause"].strip().lower()[:80])
            if key not in seen:
                seen.add(key)
                out.append(e)
            if len(out) >= MAX_LESSONS:
                break
        return out

    def render(self) -> str:
        lessons = self.lessons()
        if not lessons:
            return ""
        lines = ["", "- Lessons from earlier inquiries (diagnoses confirmed by the next experiment in earlier",
                 "  sessions on other clinical datasets; general patterns, not facts about this dataset):"]
        for e in lessons:
            lines.append(f"  * [{e['anomaly']}] When {e['trigger']}: root cause was {e['root_cause'].strip()}; "
                         f"what worked: {e['next_change'].strip()}")
        return "\n".join(lines)
