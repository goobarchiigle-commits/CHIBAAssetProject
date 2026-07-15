import json, time, threading
from pathlib import Path
from typing import Optional

from .idempotency import IdempotencyGuard
from .decision_ledger import RuntimeDecisionLedger

class RunCoordinator:
    """Prevent duplicate/overlapping runs. Lock-safe. Append-only manifests."""
    def __init__(self,
                 manifest_path: str = "artifacts/runtime/run_manifest.jsonl",
                 ledger: Optional[RuntimeDecisionLedger] = None):
        self._manifest = Path(manifest_path)
        self._manifest.parent.mkdir(parents=True, exist_ok=True)
        self._guard = IdempotencyGuard()
        self._ledger = ledger or RuntimeDecisionLedger()
        self._lock = threading.Lock()
        self._active_run: Optional[str] = None

    def try_acquire(self, run_id: str, epoch_id: str, schedule_id: str) -> bool:
        with self._lock:
            if self._active_run is not None:
                self._ledger.record("run_skipped", run_id, epoch_id,
                                    f"overlapping_run:{self._active_run}")
                return False
            if self._guard.is_duplicate(run_id):
                self._ledger.record("run_skipped", run_id, epoch_id, "duplicate_run_id")
                return False
            self._active_run = run_id
            with self._manifest.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "run_id": run_id,
                                    "epoch_id": epoch_id, "schedule_id": schedule_id,
                                    "event": "acquired"}) + "\n")
            self._ledger.record("run_started", run_id, epoch_id, f"schedule:{schedule_id}")
            return True

    def release(self, run_id: str, epoch_id: str, success: bool) -> None:
        with self._lock:
            event = "completed" if success else "failed"
            with self._manifest.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "run_id": run_id,
                                    "epoch_id": epoch_id, "event": event}) + "\n")
            if success:
                self._guard.mark_complete(run_id, epoch_id)
                self._ledger.record("run_completed", run_id, epoch_id, "success")
            else:
                self._ledger.record("run_failed", run_id, epoch_id, "stage_failure")
            self._active_run = None
