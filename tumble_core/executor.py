"""Runs an agent's train.py as a local subprocess.

Tumble runs experiments in an E2B sandbox (tumble/sandbox_exec.py). RSI needs
many more runs than one Tumble session, so this version runs locally instead:
it's free and fast. The interface matches Tumble's SandboxSession -- run(code, timeout)
returns an ExecResult -- so you can swap the E2B version back in without touching
anything else.

WARNING: this executes LLM-written code on your machine. Use only toy/public
datasets, and switch back to the sandbox for anything you don't trust.
"""

import os
import subprocess
import sys
from dataclasses import dataclass

TRAIN_SCRIPT_NAME = "train.py"


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class LocalExecutor:
    def __init__(self, workdir: str):
        self.workdir = workdir

    def run(self, code: str, timeout: int, script: str = TRAIN_SCRIPT_NAME) -> ExecResult:
        """Write `code` to `script` in the workspace and run it there."""
        with open(os.path.join(self.workdir, script), "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(proc.stdout, proc.stderr, proc.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            return ExecResult("", "", -1, timed_out=True)

    def close(self) -> None:
        pass
