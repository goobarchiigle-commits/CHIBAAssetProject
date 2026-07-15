import time
from datetime import date
from typing import Any, Dict, List, Optional

from .circuit_breaker import RuntimeCircuitBreaker, CircuitBreakerConfig
from .decision_ledger import RuntimeDecisionLedger
from .deployment_gate import DeploymentGateInterface
from .drift_monitor import RuntimeDriftMonitor
from .epoch import ExecutionEpoch
from .failure_recovery import FailureRecoveryManager
from .health_governance import RuntimeHealthGovernance
from .idempotency import IdempotencyGuard
from .orchestrator import PipelineOrchestrator
from .quarantine import RuntimeQuarantineZone
from .run_coordinator import RunCoordinator
from .schedule_registry import ScheduleRegistry
from .snapshot import RuntimeSnapshot
from .state_machine import RuntimeStateMachine, RuntimeState

class RuntimeSupervisor:
    """
    Entry point for all autonomous research execution.
    Enforces: CircuitBreaker → IdempotencyGuard → Epoch → RunCoordinator → Pipeline.
    Deterministic. Fail-closed. No adaptive behavior.
    """
    def __init__(self,
                 circuit_config: CircuitBreakerConfig = None,
                 artifact_dir: str = "artifacts/runtime"):
        self._ledger = RuntimeDecisionLedger(f"{artifact_dir}/decision_ledger.jsonl")
        self._breaker = RuntimeCircuitBreaker(circuit_config,
                                              f"{artifact_dir}/circuit_breaker.jsonl")
        self._schedule_reg = ScheduleRegistry(f"{artifact_dir}/schedule_history.jsonl")
        self._coordinator = RunCoordinator(f"{artifact_dir}/run_manifest.jsonl",
                                           self._ledger)
        self._orchestrator = PipelineOrchestrator(artifact_dir, self._ledger)
        self._snapshot = RuntimeSnapshot(f"{artifact_dir}/snapshots")
        self._drift = RuntimeDriftMonitor(f"{artifact_dir}/drift_metrics.jsonl")
        self._health = RuntimeHealthGovernance(f"{artifact_dir}/health_reports.jsonl")
        self._gate = DeploymentGateInterface(f"{artifact_dir}/deployment_recommendations.jsonl")
        self._quarantine = RuntimeQuarantineZone(f"{artifact_dir}/quarantine",
                                                 f"{artifact_dir}/quarantine_manifest.jsonl")

    def execute(self, schedule_id: str,
                epoch_inputs: Dict[str, str],
                stages: List[Dict[str, Any]],
                date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Main execution entry point.
        epoch_inputs: dict with keys dataset_manifest, feature_manifest, etc.
        stages: list passed to PipelineOrchestrator.run_pipeline()
        Returns result dict with run_id, epoch_id, success.
        """
        if date_str is None:
            date_str = date.today().isoformat()

        # 1. Circuit breaker check
        if self._breaker.is_tripped:
            self._ledger.record("run_blocked", "N/A", "N/A",
                               "circuit_breaker_tripped")
            return {"success": False, "reason": "circuit_breaker_tripped"}

        # 2. Schedule lookup
        sched = self._schedule_reg.get(schedule_id)
        if sched is None:
            return {"success": False, "reason": f"unknown_schedule:{schedule_id}"}

        # 3. Build epoch (deterministic)
        epoch = ExecutionEpoch.create(
            dataset_manifest=epoch_inputs.get("dataset_manifest", ""),
            feature_manifest=epoch_inputs.get("feature_manifest", ""),
            governance_state=epoch_inputs.get("governance_state", ""),
            schema_version=epoch_inputs.get("schema_version", "1.0"),
            template_state=epoch_inputs.get("template_state", ""),
            reality_schema_version=epoch_inputs.get("reality_schema_version", "1.0"),
        )

        # 4. Build run_id (deterministic)
        run_id = IdempotencyGuard.make_run_id(epoch.epoch_id, schedule_id, date_str)

        # 5. Acquire run lock
        if not self._coordinator.try_acquire(run_id, epoch.epoch_id, schedule_id):
            return {"success": False, "reason": "run_skipped", "run_id": run_id}

        # 6. Snapshot initial state
        self._snapshot.capture(run_id, epoch.epoch_id,
                                {"stage": "supervisor_start", "schedule_id": schedule_id})
        self._schedule_reg.record_trigger(schedule_id, run_id, "supervisor_execute")

        # 7. Run pipeline
        t0 = time.time()
        success = self._orchestrator.run_pipeline(run_id, epoch.epoch_id, stages)
        duration = time.time() - t0

        # 8. Record drift
        self._drift.record(run_id, epoch.epoch_id, schedule_id, duration, 0, 0)

        # 9. Snapshot final state
        self._snapshot.capture(run_id, epoch.epoch_id,
                                {"stage": "supervisor_end", "success": success})

        # 10. Release coordinator
        self._coordinator.release(run_id, epoch.epoch_id, success)

        # 11. Deployment recommendation (recommend only, never authorize)
        if success:
            self._gate.recommend(run_id, epoch.epoch_id, eligible=True,
                                 reasons=["pipeline_completed"])
        else:
            self._gate.recommend(run_id, epoch.epoch_id, eligible=False,
                                 reasons=["pipeline_failed"])
            self._breaker.record_event("validation_failure", run_id,
                                       f"pipeline_failed:{schedule_id}")

        return {"success": success, "run_id": run_id, "epoch_id": epoch.epoch_id,
                "duration_s": duration}
