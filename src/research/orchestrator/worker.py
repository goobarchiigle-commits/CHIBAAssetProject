"""Worker — process-isolated experiment execution.

Each experiment runs in a clean subprocess.
No shared mutable state between workers.
Retry capped at retry_max. Fail closed: any non-zero exit is failure.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class WorkerResult:
    experiment_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    attempt: int


def run_experiment_isolated(
    script_path: Path,
    experiment_id: str,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
    retry_max: int = 1,
) -> WorkerResult:
    """Run experiment script in a clean subprocess.

    The script is expected to accept --experiment-id <id>.
    Returns WorkerResult with success=True only on exit code 0.
    On failure, retries up to retry_max times total (so retry_max=1 means 1 attempt, no retry).
    """
    cmd = [sys.executable, str(Path(script_path).resolve()), "--experiment-id", experiment_id]
    worker_env = {**os.environ, **(env or {})}

    last: Optional[subprocess.CompletedProcess] = None
    for attempt in range(1, retry_max + 1):
        try:
            last = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=worker_env,
            )
        except subprocess.TimeoutExpired:
            return WorkerResult(
                experiment_id=experiment_id,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"timeout after {timeout}s on attempt {attempt}",
                attempt=attempt,
            )
        if last.returncode == 0:
            return WorkerResult(
                experiment_id=experiment_id,
                success=True,
                exit_code=0,
                stdout=last.stdout,
                stderr=last.stderr,
                attempt=attempt,
            )

    return WorkerResult(
        experiment_id=experiment_id,
        success=False,
        exit_code=last.returncode if last else -1,
        stdout=last.stdout if last else "",
        stderr=last.stderr if last else "",
        attempt=retry_max,
    )
