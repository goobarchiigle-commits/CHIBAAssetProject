"""BatchRunner — executes a batch through the existing orchestrator.

Design:
  - Sequential process isolation only (no threads)
  - Deterministic execution order via schedule_experiments()
  - Non-fatal failures: continue after any single experiment fails
  - Immutable execution log written atomically after completion
  - Append-only: log is never overwritten
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.research.contracts.experiment import ExperimentContract
from src.research.batch.batch import ExperimentBatch, BatchArtifacts
from src.research.orchestrator.scheduler import schedule_experiments
from src.research.orchestrator.worker import run_experiment_isolated


@dataclass
class BatchExecutionLog:
    batch_id: str
    started_at: str
    completed_at: str
    succeeded: List[str]
    failed: List[str]
    total: int
    worker_log: List[dict]   # per-experiment result records

    def to_dict(self) -> dict:
        total_run = len(self.succeeded) + len(self.failed)
        return {
            "batch_id": self.batch_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total": self.total,
            "succeeded": len(self.succeeded),
            "failed": len(self.failed),
            "success_rate": len(self.succeeded) / max(total_run, 1),
            "succeeded_ids": self.succeeded,
            "failed_ids": self.failed,
            "worker_log": self.worker_log,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


class BatchRunner:
    """Execute experiments in deterministic order with process isolation.

    Each experiment runs in a clean subprocess via run_experiment_isolated().
    A non-zero exit code is a non-fatal failure: execution continues.
    The execution log is written once, atomically, after all experiments complete.
    """

    def __init__(
        self,
        batch: ExperimentBatch,
        experiments: List[ExperimentContract],
        artifacts_base: Optional[Path] = None,
    ) -> None:
        self.batch = batch
        self.experiments = list(experiments)
        self.artifacts_base = Path(artifacts_base) if artifacts_base else Path("artifacts/batches")

    def run(
        self,
        script_path: Path,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 3600,
        retry_max: int = 1,
    ) -> BatchExecutionLog:
        """Execute all experiments sequentially in deterministic order.

        Returns BatchExecutionLog. Writes execution_log.json to batch artifacts.
        Does NOT overwrite an existing log (append-only policy).
        """
        started_at = datetime.now(timezone.utc).isoformat()
        ordered = schedule_experiments(self.experiments)

        succeeded: List[str] = []
        failed: List[str] = []
        worker_log: List[dict] = []

        for exp in ordered:
            result = run_experiment_isolated(
                script_path=Path(script_path),
                experiment_id=exp.experiment_id,
                env=env,
                timeout=timeout,
                retry_max=retry_max,
            )
            record = {
                "experiment_id": result.experiment_id,
                "success": result.success,
                "exit_code": result.exit_code,
                "attempt": result.attempt,
                "stderr_snippet": result.stderr[:500] if result.stderr else "",
            }
            worker_log.append(record)

            if result.success:
                succeeded.append(exp.experiment_id)
            else:
                failed.append(exp.experiment_id)
                # Non-fatal: continue with remaining experiments

        completed_at = datetime.now(timezone.utc).isoformat()
        log = BatchExecutionLog(
            batch_id=self.batch.batch_id,
            started_at=started_at,
            completed_at=completed_at,
            succeeded=succeeded,
            failed=failed,
            total=len(ordered),
            worker_log=worker_log,
        )
        self._write_log(log)
        return log

    def _write_log(self, log: BatchExecutionLog) -> None:
        arts = BatchArtifacts(self.batch.batch_id, base_dir=self.artifacts_base)
        if not arts.exists("execution_log.json"):
            arts.write("execution_log.json", log.to_json())
