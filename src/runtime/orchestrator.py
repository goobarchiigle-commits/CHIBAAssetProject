import json, os, time, hashlib, tempfile, shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .artifact_validator import ArtifactCompletionValidator
from .decision_ledger import RuntimeDecisionLedger
from .failure_recovery import FailureRecoveryManager, RecoveryConfig
from .state_machine import RuntimeStateMachine, RuntimeState

class PipelineOrchestrator:
    """
    Execute pipeline stages in deterministic order.
    Transactional: temp write → validate → atomic rename.
    Partial-write visibility is forbidden.
    """
    def __init__(self,
                 artifact_dir: str = "artifacts/runtime",
                 ledger: Optional[RuntimeDecisionLedger] = None,
                 recovery: Optional[FailureRecoveryManager] = None):
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._validator = ArtifactCompletionValidator()
        self._ledger = ledger or RuntimeDecisionLedger()
        self._recovery = recovery or FailureRecoveryManager()

    def run_stage(self, run_id: str, epoch_id: str, stage: str,
                  fn: Callable[[], Any], output_path: Optional[str] = None) -> bool:
        """
        Execute a single pipeline stage transactionally.
        fn() must return bytes or str to write, or None (no artifact).
        Returns True on success, False on failure (fail-closed).
        """
        sm = RuntimeStateMachine()
        sm.transition(RuntimeState.READY, f"stage:{stage}")
        sm.transition(RuntimeState.RUNNING, "executing")
        self._ledger.record("stage_started", run_id, epoch_id, f"stage:{stage}")
        try:
            result = fn()
            sm.transition(RuntimeState.VALIDATING, "fn_complete")
            if output_path and result is not None:
                # Transactional write: temp → validate → atomic rename
                tmp = str(output_path) + ".tmp"
                data = result if isinstance(result, bytes) else str(result).encode()
                Path(tmp).write_bytes(data)
                vr = self._validator.validate(tmp, min_bytes=1)
                if not vr.valid:
                    sm.transition(RuntimeState.FAILED, f"validation:{vr.reason}")
                    self._ledger.record("stage_failed", run_id, epoch_id,
                                       f"stage:{stage} reason:{vr.reason}")
                    Path(tmp).unlink(missing_ok=True)
                    return False
                shutil.move(tmp, output_path)  # atomic on same filesystem
            sm.transition(RuntimeState.COMMITTED, "artifact_committed")
            self._ledger.record("stage_committed", run_id, epoch_id, f"stage:{stage}")
            return True
        except Exception as e:
            sm.transition(RuntimeState.FAILED, str(e))
            self._ledger.record("stage_failed", run_id, epoch_id,
                               f"stage:{stage} error:{e}")
            self._recovery.record_attempt(run_id, stage, str(e))
            return False

    def run_pipeline(self, run_id: str, epoch_id: str,
                     stages: List[Dict[str, Any]]) -> bool:
        """
        stages: list of {"name": str, "fn": callable, "output_path": optional str}
        Deterministic ordering. Fails closed on first stage failure.
        """
        for stage_def in stages:
            name = stage_def["name"]
            fn = stage_def["fn"]
            output_path = stage_def.get("output_path")
            success = self.run_stage(run_id, epoch_id, name, fn, output_path)
            if not success:
                if self._recovery.should_retry(run_id, name):
                    # one bounded retry
                    success = self.run_stage(run_id, epoch_id, name, fn, output_path)
                    if success:
                        self._recovery.record_recovery(run_id, name)
                if not success:
                    self._ledger.record("pipeline_aborted", run_id, epoch_id,
                                       f"stage:{name} failed after retry")
                    return False
        self._ledger.record("pipeline_completed", run_id, epoch_id, "all_stages_ok")
        return True
