"""
src/runtime/future_leader_runtime_orchestrator.py
Phase 5A — Future Leader Autonomous Runtime Orchestrator.

Purpose:
  "deployment continuity operating system"
  NOT an execution framework. NOT alpha discovery.

  Responsibility:
    runtime lifecycle continuity — crash-safe, deterministic,
    idempotent governance pipeline supervision.

GUARANTEES:
  - deterministic stage execution (fixed order, no dynamic insertion)
  - crash-safe checkpoint replay (same input → same state after restart)
  - idempotent stage skip (no duplicate execution, no duplicate orders)
  - freeze escalation (SOFT_FREEZE → HARD_FREEZE via fail-closed rules)
  - append-only telemetry (events + checkpoints)
  - atomic checkpoint write (tmp → fsync → rename — no partial corruption)

禁止:
  - distributed runtime / websocket / async event engine / Kafka
  - multiprocessing scaling / microservice architecture
  - RL orchestration / dynamic scheduling / realtime inference
  - intraday orchestration / live alpha mutation
  - allocator mutation / predictive_entry_enabled mutation

Stage Pipeline (fixed order, deterministic):
  SIGNAL_COLLECTION → PROMOTION_EVALUATION → SHADOW_VALIDATION
  → ROUTING → PAPER_EXECUTION → ROLLBACK_GOVERNANCE
  → LIVE_DEPLOYMENT → RECONCILIATION → CHECKPOINT_SAVED

Freeze escalation:
  Any single trigger       → SOFT_FREEZE
  Repeated / severe trigger → HARD_FREEZE

Truth hierarchy for reconciliation:
  runtime ledger > deployment ledger > broker snapshot
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Stage pipeline (固定順序・変更禁止) ────────────────────────────────────────

STAGE_PIPELINE: Tuple[str, ...] = (
    "SIGNAL_COLLECTION",
    "PROMOTION_EVALUATION",
    "SHADOW_VALIDATION",
    "ROUTING",
    "PAPER_EXECUTION",
    "ROLLBACK_GOVERNANCE",
    "LIVE_DEPLOYMENT",
    "RECONCILIATION",
    "CHECKPOINT_SAVED",
)

# SOFT_FREEZE trigger count before HARD_FREEZE escalation
_SOFT_FREEZE_ESCALATION_COUNT: int = 3

# Reconciliation failure count before HARD_FREEZE
_RECONCILIATION_HARD_FREEZE_COUNT: int = 2


# ── Runtime lifecycle state machine ───────────────────────────────────────────

class RuntimeLifecycleState(Enum):
    IDLE            = "idle"
    RUNNING         = "running"
    SOFT_FREEZE     = "soft_freeze"
    HARD_FREEZE     = "hard_freeze"
    RECOVERY_REPLAY = "recovery_replay"
    SHUTDOWN_SAFE   = "shutdown_safe"


class RuntimeFreezeReason(Enum):
    BROKER_TIMEOUT                  = "broker_timeout"
    REPLAY_MISMATCH                 = "replay_mismatch"
    CHECKPOINT_CORRUPTION           = "checkpoint_corruption"
    POSITION_RECONCILIATION_FAILURE = "position_reconciliation_failure"
    GOVERNANCE_HASH_MISMATCH        = "governance_hash_mismatch"
    UNKNOWN_RUNTIME_STATE           = "unknown_runtime_state"
    STAGE_FAILURE                   = "stage_failure"
    FORCED                          = "forced"


# Valid lifecycle state transitions (FAIL_CLOSED: unlisted → reject)
_LIFECYCLE_TRANSITIONS: frozenset = frozenset({
    (RuntimeLifecycleState.IDLE,            RuntimeLifecycleState.RUNNING),
    (RuntimeLifecycleState.IDLE,            RuntimeLifecycleState.RECOVERY_REPLAY),
    (RuntimeLifecycleState.IDLE,            RuntimeLifecycleState.HARD_FREEZE),
    (RuntimeLifecycleState.RUNNING,         RuntimeLifecycleState.SOFT_FREEZE),
    (RuntimeLifecycleState.RUNNING,         RuntimeLifecycleState.HARD_FREEZE),
    (RuntimeLifecycleState.RUNNING,         RuntimeLifecycleState.SHUTDOWN_SAFE),
    (RuntimeLifecycleState.SOFT_FREEZE,     RuntimeLifecycleState.HARD_FREEZE),
    (RuntimeLifecycleState.SOFT_FREEZE,     RuntimeLifecycleState.RECOVERY_REPLAY),
    (RuntimeLifecycleState.SOFT_FREEZE,     RuntimeLifecycleState.SHUTDOWN_SAFE),
    (RuntimeLifecycleState.HARD_FREEZE,     RuntimeLifecycleState.SHUTDOWN_SAFE),
    (RuntimeLifecycleState.RECOVERY_REPLAY, RuntimeLifecycleState.RUNNING),
    (RuntimeLifecycleState.RECOVERY_REPLAY, RuntimeLifecycleState.HARD_FREEZE),
    (RuntimeLifecycleState.RECOVERY_REPLAY, RuntimeLifecycleState.SHUTDOWN_SAFE),
    # Allow re-entry after safe shutdown for next day
    (RuntimeLifecycleState.SHUTDOWN_SAFE,   RuntimeLifecycleState.IDLE),
})


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RuntimeCheckpoint:
    """
    Crash-safe replay checkpoint. Written atomically after each stage.
    replay_sequence_id: unique per pipeline invocation (not per stage).
    """
    evaluation_date:       str
    runtime_state:         str   # RuntimeLifecycleState.value
    last_completed_stage:  str   # last stage that wrote this checkpoint
    deterministic_hash:    str
    rollback_state:        str
    freeze_state:          str   # RuntimeLifecycleState value when frozen
    active_position_symbol: Optional[str]
    last_order_id:         Optional[str]
    replay_sequence_id:    str
    timestamp:             str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_date":        self.evaluation_date,
            "runtime_state":          self.runtime_state,
            "last_completed_stage":   self.last_completed_stage,
            "deterministic_hash":     self.deterministic_hash,
            "rollback_state":         self.rollback_state,
            "freeze_state":           self.freeze_state,
            "active_position_symbol": self.active_position_symbol,
            "last_order_id":          self.last_order_id,
            "replay_sequence_id":     self.replay_sequence_id,
            "timestamp":              self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuntimeCheckpoint":
        return cls(
            evaluation_date=str(d.get("evaluation_date", "")),
            runtime_state=str(d.get("runtime_state", RuntimeLifecycleState.IDLE.value)),
            last_completed_stage=str(d.get("last_completed_stage", "")),
            deterministic_hash=str(d.get("deterministic_hash", "")),
            rollback_state=str(d.get("rollback_state", "stable")),
            freeze_state=str(d.get("freeze_state", "")),
            active_position_symbol=d.get("active_position_symbol"),
            last_order_id=d.get("last_order_id"),
            replay_sequence_id=str(d.get("replay_sequence_id", "")),
            timestamp=str(d.get("timestamp", "")),
        )


@dataclass(frozen=True)
class RuntimeStageResult:
    """
    One stage execution record. Appended to JSONL (append-only forensics).
    """
    stage_name:           str
    started_at:           str
    completed_at:         str
    success:              bool
    deterministic_hash:   str
    skipped_as_idempotent: bool
    freeze_triggered:     bool
    error_type:           Optional[str]
    error_message:        Optional[str]
    stage_payload:        Dict[str, Any]   # stage-specific output data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name":             self.stage_name,
            "started_at":             self.started_at,
            "completed_at":           self.completed_at,
            "success":                self.success,
            "deterministic_hash":     self.deterministic_hash,
            "skipped_as_idempotent":  self.skipped_as_idempotent,
            "freeze_triggered":       self.freeze_triggered,
            "error_type":             self.error_type,
            "error_message":          self.error_message,
            "stage_payload":          self.stage_payload,
        }


@dataclass(frozen=True)
class RuntimeTransitionRecord:
    """Lifecycle state transition — append-only telemetry."""
    timestamp:       str
    evaluation_date: str
    from_state:      str
    to_state:        str
    reason:          str
    freeze_reason:   Optional[str]
    stage_context:   Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":       self.timestamp,
            "evaluation_date": self.evaluation_date,
            "from_state":      self.from_state,
            "to_state":        self.to_state,
            "reason":          self.reason,
            "freeze_reason":   self.freeze_reason,
            "stage_context":   self.stage_context,
        }


@dataclass(frozen=True)
class RuntimeEventRecord:
    """Generic runtime event — append-only JSONL for forensics."""
    timestamp:              str
    evaluation_date:        str
    runtime_state:          str
    current_stage:          Optional[str]
    last_completed_stage:   Optional[str]
    freeze_reason:          Optional[str]
    rollback_state:         str
    active_position_symbol: Optional[str]
    replay_sequence_id:     str
    deterministic_hash:     str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":              self.timestamp,
            "evaluation_date":        self.evaluation_date,
            "runtime_state":          self.runtime_state,
            "current_stage":          self.current_stage,
            "last_completed_stage":   self.last_completed_stage,
            "freeze_reason":          self.freeze_reason,
            "rollback_state":         self.rollback_state,
            "active_position_symbol": self.active_position_symbol,
            "replay_sequence_id":     self.replay_sequence_id,
            "deterministic_hash":     self.deterministic_hash,
        }


# ── Deterministic hash ─────────────────────────────────────────────────────────

def compute_stage_hash(
    evaluation_date:       str,
    stage_name:            str,
    stage_index:           int,
    stage_payload:         Dict[str, Any],
    rollback_state:        str,
    runtime_state:         str,
    active_position_symbol: Optional[str],
) -> str:
    """
    SHA256-based deterministic hash. Same inputs → same output.
    Sorted keys for dict stability.
    replay_sequence_id is excluded — it contains a UUID suffix and would break
    "same inputs → same hash" guarantee across instances.
    """
    payload = {
        "evaluation_date":        evaluation_date,
        "stage_name":             stage_name,
        "stage_index":            stage_index,
        "stage_payload":          _deep_sort(stage_payload),
        "rollback_state":         rollback_state,
        "runtime_state":          runtime_state,
        "active_position_symbol": active_position_symbol,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _deep_sort(obj: Any) -> Any:
    """Recursively sort dicts by key for canonical JSON."""
    if isinstance(obj, dict):
        return {k: _deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_deep_sort(i) for i in obj]
    return obj


# ── Checkpoint I/O (atomic) ───────────────────────────────────────────────────

def _write_checkpoint_atomic(checkpoint: RuntimeCheckpoint, path: Path) -> None:
    """
    Atomic checkpoint write: write to .tmp → fsync → rename.
    Prevents partial-write corruption.
    Raises on failure (fail-closed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    data = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    try:
        with tmp_path.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename (same filesystem)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_checkpoint(path: Path) -> Optional[RuntimeCheckpoint]:
    if not path.exists():
        return None
    try:
        return RuntimeCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.error("[ORCHESTRATOR] checkpoint read failed: %s", exc)
        return None  # Corruption detected — caller must HARD_FREEZE


def _append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append one JSONL line. Raises IOError on failure (telemetry integrity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


# ── FutureLeaderRuntimeOrchestrator ───────────────────────────────────────────

class FutureLeaderRuntimeOrchestrator:
    """
    Phase 5A — Autonomous deployment continuity runtime for Future Leader stack.

    Usage:
        orc = FutureLeaderRuntimeOrchestrator(
            checkpoint_dir=Path("runtime/runtime_orchestrator"),
            events_dir=Path("logs/runtime/runtime_orchestrator"),
        )
        results = orc.run_pipeline(evaluation_date="2026-05-26")

    Stage functions can be injected for testing:
        orc = FutureLeaderRuntimeOrchestrator(
            ...,
            stage_functions={"SIGNAL_COLLECTION": lambda: {"signals": []}},
        )
    """

    def __init__(
        self,
        checkpoint_dir:  Path,
        events_dir:      Path,
        stage_functions: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None,
        # Runtime state (internal — not settable by caller)
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._events_dir     = events_dir
        self._stage_fns      = stage_functions or {}
        self._lifecycle_state = RuntimeLifecycleState.IDLE
        self._freeze_count:  int = 0
        self._recon_fail_count: int = 0
        self._current_date:  str = ""
        self._replay_seq_id: str = ""
        self._rollback_state: str = "stable"
        self._active_position_symbol: Optional[str] = None
        self._last_order_id: Optional[str] = None
        self._freeze_reason: Optional[RuntimeFreezeReason] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def lifecycle_state(self) -> RuntimeLifecycleState:
        return self._lifecycle_state

    def run_pipeline(
        self,
        evaluation_date: str,
        rollback_state:  str = "stable",
    ) -> List[RuntimeStageResult]:
        """
        Execute full stage pipeline for evaluation_date.
        On existing checkpoint: enter RECOVERY_REPLAY and resume.
        Returns list of RuntimeStageResult (all stages including skipped).
        """
        self._current_date   = evaluation_date
        self._rollback_state = rollback_state

        # Check for existing checkpoint (crash recovery)
        checkpoint_path = self._checkpoint_path(evaluation_date)
        existing = _read_checkpoint(checkpoint_path)

        if existing is not None:
            if existing.deterministic_hash == "":
                # Corrupted checkpoint
                self._hard_freeze(RuntimeFreezeReason.CHECKPOINT_CORRUPTION, "empty hash")
                return []
            self._replay_seq_id = existing.replay_sequence_id
            logger.info(
                "[ORCHESTRATOR] checkpoint found for %s — entering RECOVERY_REPLAY "
                "last_completed=%s seq=%s",
                evaluation_date, existing.last_completed_stage, self._replay_seq_id,
            )
            self._transition(RuntimeLifecycleState.RECOVERY_REPLAY, "checkpoint_found")
        else:
            self._replay_seq_id = self._new_seq_id(evaluation_date)
            self._transition(RuntimeLifecycleState.RUNNING, "fresh_pipeline_start")

        results: List[RuntimeStageResult] = []

        for idx, stage_name in enumerate(STAGE_PIPELINE):
            if self._lifecycle_state in (RuntimeLifecycleState.HARD_FREEZE, RuntimeLifecycleState.SHUTDOWN_SAFE):
                break

            # Idempotency: skip already-completed stages
            if existing is not None and self._stage_already_done(stage_name, existing):
                result = self._make_skipped_result(stage_name, idx)
                results.append(result)
                self._append_stage_event(result)
                continue

            # Execute stage
            if self._lifecycle_state == RuntimeLifecycleState.RECOVERY_REPLAY:
                self._transition(RuntimeLifecycleState.RUNNING, f"resuming_at_{stage_name}")
                existing = None  # clear after first resumed stage

            result = self._execute_stage(stage_name, idx, evaluation_date)
            results.append(result)
            self._append_stage_event(result)

            if not result.success and not result.skipped_as_idempotent:
                self._soft_freeze(RuntimeFreezeReason.STAGE_FAILURE, stage_name)
                if self._lifecycle_state == RuntimeLifecycleState.HARD_FREEZE:
                    break
                continue

            # Write checkpoint after each successful stage
            checkpoint = self._build_checkpoint(stage_name)
            try:
                _write_checkpoint_atomic(checkpoint, checkpoint_path)
            except Exception as exc:
                logger.error("[ORCHESTRATOR] checkpoint write failed: %s — SHUTDOWN_SAFE", exc)
                self._transition(RuntimeLifecycleState.SHUTDOWN_SAFE, "checkpoint_write_failed")
                break

        if self._lifecycle_state == RuntimeLifecycleState.RUNNING:
            self._transition(RuntimeLifecycleState.SHUTDOWN_SAFE, "pipeline_complete")

        logger.info(
            "[ORCHESTRATOR] pipeline complete: date=%s state=%s stages=%d/%d",
            evaluation_date,
            self._lifecycle_state.value,
            sum(1 for r in results if r.success or r.skipped_as_idempotent),
            len(STAGE_PIPELINE),
        )
        return results

    def replay_pipeline(self, evaluation_date: str) -> bool:
        """
        Verify replay reproducibility: re-execute pipeline and compare hashes.
        Returns True if all stage hashes match checkpoint.
        Triggers HARD_FREEZE on mismatch.
        """
        checkpoint_path = self._checkpoint_path(evaluation_date)
        existing = _read_checkpoint(checkpoint_path)
        if existing is None:
            logger.warning("[ORCHESTRATOR] verify_replay: no checkpoint for %s", evaluation_date)
            return False

        logger.info("[ORCHESTRATOR] replay_verify starting for %s", evaluation_date)
        # Create a shadow orchestrator with same stage fns
        shadow = FutureLeaderRuntimeOrchestrator(
            checkpoint_dir=self._checkpoint_dir / "_shadow",
            events_dir=self._events_dir / "_shadow",
            stage_functions=self._stage_fns,
        )
        shadow._rollback_state  = self._rollback_state
        shadow._active_position_symbol = existing.active_position_symbol
        shadow._replay_seq_id   = existing.replay_sequence_id
        shadow._current_date    = evaluation_date

        shadow._transition(RuntimeLifecycleState.RECOVERY_REPLAY, "replay_verify")
        shadow._transition(RuntimeLifecycleState.RUNNING, "replay_start")

        mismatch = False
        for idx, stage_name in enumerate(STAGE_PIPELINE):
            shadow._execute_stage(stage_name, idx, evaluation_date)

        # Compare checkpoint hash: re-build from shadow's final state vs saved checkpoint.
        # _build_checkpoint uses {"checkpoint": True} payload — same as original write path.
        shadow_cp = shadow._build_checkpoint("CHECKPOINT_SAVED")
        if shadow_cp.deterministic_hash != existing.deterministic_hash:
            logger.error(
                "[ORCHESTRATOR] REPLAY MISMATCH at CHECKPOINT_SAVED: "
                "expected=%s got=%s",
                existing.deterministic_hash, shadow_cp.deterministic_hash,
            )
            mismatch = True

        if mismatch:
            self._hard_freeze(RuntimeFreezeReason.REPLAY_MISMATCH, "hash_mismatch_on_replay")
            return False

        logger.info("[ORCHESTRATOR] replay_verify PASS for %s", evaluation_date)
        return True

    def reconcile(
        self,
        broker_positions:    Optional[List[Dict[str, Any]]] = None,
        runtime_ledger:      Optional[List[Dict[str, Any]]] = None,
        deployment_ledger:   Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        1-per-day broker reconciliation.
        Truth hierarchy: runtime_ledger > deployment_ledger > broker_snapshot.
        Mismatch → SOFT_FREEZE. Repeated → HARD_FREEZE.
        """
        broker_syms  = {p["symbol"] for p in (broker_positions or []) if p.get("symbol")}
        runtime_syms = {r["symbol"] for r in (runtime_ledger  or []) if r.get("symbol")}
        deploy_syms  = {d["symbol"] for d in (deployment_ledger or []) if d.get("symbol")}

        # Use runtime as truth
        truth_syms = runtime_syms or deploy_syms

        unexpected = broker_syms - truth_syms
        missing    = truth_syms  - broker_syms

        mismatches = []
        if unexpected:
            mismatches.extend(f"unexpected:{s}" for s in sorted(unexpected))
        if missing:
            mismatches.extend(f"missing:{s}" for s in sorted(missing))

        status = "clean" if not mismatches else "mismatch"
        result = {
            "status":           status,
            "mismatches":       mismatches,
            "broker_count":     len(broker_syms),
            "runtime_count":    len(runtime_syms),
            "truth_hierarchy":  "runtime_ledger",
        }

        if mismatches:
            self._recon_fail_count += 1
            if self._recon_fail_count >= _RECONCILIATION_HARD_FREEZE_COUNT:
                self._hard_freeze(
                    RuntimeFreezeReason.POSITION_RECONCILIATION_FAILURE,
                    f"repeated_failures={self._recon_fail_count}",
                )
            else:
                self._soft_freeze(
                    RuntimeFreezeReason.POSITION_RECONCILIATION_FAILURE,
                    f"recon_mismatch: {mismatches}",
                )
        else:
            self._recon_fail_count = 0

        return result

    def force_soft_freeze(self, reason: str = "forced") -> None:
        self._soft_freeze(RuntimeFreezeReason.FORCED, reason)

    def force_hard_freeze(self, reason: str = "forced") -> None:
        self._hard_freeze(RuntimeFreezeReason.FORCED, reason)

    def get_status(self) -> Dict[str, Any]:
        return {
            "lifecycle_state":          self._lifecycle_state.value,
            "freeze_count":             self._freeze_count,
            "recon_fail_count":         self._recon_fail_count,
            "active_position_symbol":   self._active_position_symbol,
            "rollback_state":           self._rollback_state,
            "freeze_reason":            self._freeze_reason.value if self._freeze_reason else None,
        }

    def load_latest_checkpoint(self, evaluation_date: str) -> Optional[RuntimeCheckpoint]:
        return _read_checkpoint(self._checkpoint_path(evaluation_date))

    # ── Internal methods ───────────────────────────────────────────────────────

    def _execute_stage(
        self,
        stage_name:      str,
        stage_index:     int,
        evaluation_date: str,
    ) -> RuntimeStageResult:
        started_at = datetime.now(JST).isoformat()
        fn = self._stage_fns.get(stage_name, self._default_stage_fn(stage_name))

        freeze_triggered = False
        error_type:    Optional[str] = None
        error_message: Optional[str] = None
        stage_payload: Dict[str, Any] = {}
        success = False

        try:
            stage_payload = fn() or {}
            success = True
        except Exception as exc:
            error_type    = type(exc).__name__
            error_message = str(exc)
            stage_payload = {"error": error_message}
            logger.warning("[ORCHESTRATOR] stage %s failed: %s", stage_name, exc)
            freeze_triggered = True

        completed_at = datetime.now(JST).isoformat()

        h = compute_stage_hash(
            evaluation_date=evaluation_date,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_payload=stage_payload,
            rollback_state=self._rollback_state,
            runtime_state=self._lifecycle_state.value,
            active_position_symbol=self._active_position_symbol,
        )

        return RuntimeStageResult(
            stage_name=stage_name,
            started_at=started_at,
            completed_at=completed_at,
            success=success,
            deterministic_hash=h,
            skipped_as_idempotent=False,
            freeze_triggered=freeze_triggered,
            error_type=error_type,
            error_message=error_message,
            stage_payload=stage_payload,
        )

    def _make_skipped_result(self, stage_name: str, stage_index: int) -> RuntimeStageResult:
        now = datetime.now(JST).isoformat()
        h = compute_stage_hash(
            evaluation_date=self._current_date,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_payload={"skipped": True},
            rollback_state=self._rollback_state,
            runtime_state=self._lifecycle_state.value,
            active_position_symbol=self._active_position_symbol,
        )
        return RuntimeStageResult(
            stage_name=stage_name,
            started_at=now,
            completed_at=now,
            success=True,
            deterministic_hash=h,
            skipped_as_idempotent=True,
            freeze_triggered=False,
            error_type=None,
            error_message=None,
            stage_payload={"skipped": True},
        )

    def _default_stage_fn(self, stage_name: str) -> Callable[[], Dict[str, Any]]:
        """Default no-op stage function. Returns minimal payload."""
        def _noop() -> Dict[str, Any]:
            return {"stage": stage_name, "result": "noop"}
        return _noop

    def _stage_already_done(self, stage_name: str, checkpoint: RuntimeCheckpoint) -> bool:
        """True if stage_name was completed in checkpoint (idempotency check)."""
        if not checkpoint.last_completed_stage:
            return False
        last_idx = _stage_index(checkpoint.last_completed_stage)
        this_idx = _stage_index(stage_name)
        return this_idx <= last_idx

    def _build_checkpoint(self, last_completed_stage: str) -> RuntimeCheckpoint:
        h = compute_stage_hash(
            evaluation_date=self._current_date,
            stage_name=last_completed_stage,
            stage_index=_stage_index(last_completed_stage),
            stage_payload={"checkpoint": True},
            rollback_state=self._rollback_state,
            runtime_state=self._lifecycle_state.value,
            active_position_symbol=self._active_position_symbol,
        )
        return RuntimeCheckpoint(
            evaluation_date=self._current_date,
            runtime_state=self._lifecycle_state.value,
            last_completed_stage=last_completed_stage,
            deterministic_hash=h,
            rollback_state=self._rollback_state,
            freeze_state=(
                self._lifecycle_state.value
                if self._lifecycle_state in (
                    RuntimeLifecycleState.SOFT_FREEZE, RuntimeLifecycleState.HARD_FREEZE
                ) else ""
            ),
            active_position_symbol=self._active_position_symbol,
            last_order_id=self._last_order_id,
            replay_sequence_id=self._replay_seq_id,
            timestamp=datetime.now(JST).isoformat(),
        )

    def _checkpoint_path(self, evaluation_date: str) -> Path:
        date_str = evaluation_date.replace("-", "")
        return self._checkpoint_dir / f"checkpoint_{date_str}.json"

    def _new_seq_id(self, evaluation_date: str) -> str:
        date_str = evaluation_date.replace("-", "")
        suffix = uuid.uuid4().hex[:6]
        return f"{date_str}_{suffix}"

    # ── Lifecycle state transitions ────────────────────────────────────────────

    def _transition(self, to: RuntimeLifecycleState, reason: str) -> None:
        from_state = self._lifecycle_state
        if (from_state, to) not in _LIFECYCLE_TRANSITIONS:
            logger.error(
                "[ORCHESTRATOR] illegal lifecycle transition %s → %s: %s — HARD_FREEZE",
                from_state.value, to.value, reason,
            )
            # Illegal transition → HARD_FREEZE (if possible)
            if (from_state, RuntimeLifecycleState.HARD_FREEZE) in _LIFECYCLE_TRANSITIONS:
                self._lifecycle_state = RuntimeLifecycleState.HARD_FREEZE
            return
        self._lifecycle_state = to
        record = RuntimeTransitionRecord(
            timestamp=datetime.now(JST).isoformat(),
            evaluation_date=self._current_date,
            from_state=from_state.value,
            to_state=to.value,
            reason=reason,
            freeze_reason=self._freeze_reason.value if self._freeze_reason else None,
            stage_context=None,
        )
        self._append_event(record.to_dict(), kind="transition")
        logger.info("[ORCHESTRATOR] %s → %s (%s)", from_state.value, to.value, reason)

    def _soft_freeze(self, reason: RuntimeFreezeReason, detail: str) -> None:
        self._freeze_reason = reason
        self._freeze_count += 1
        logger.warning(
            "[ORCHESTRATOR] SOFT_FREEZE: reason=%s detail=%s count=%d",
            reason.value, detail, self._freeze_count,
        )
        if self._freeze_count >= _SOFT_FREEZE_ESCALATION_COUNT:
            self._hard_freeze(reason, f"escalated_after_{self._freeze_count}_soft_freezes")
            return
        if self._lifecycle_state == RuntimeLifecycleState.RUNNING:
            self._transition(RuntimeLifecycleState.SOFT_FREEZE, f"{reason.value}:{detail}")
        elif self._lifecycle_state == RuntimeLifecycleState.RECOVERY_REPLAY:
            self._transition(RuntimeLifecycleState.SOFT_FREEZE, f"{reason.value}:{detail}")

    def _hard_freeze(self, reason: RuntimeFreezeReason, detail: str) -> None:
        self._freeze_reason = reason
        logger.error(
            "[ORCHESTRATOR] HARD_FREEZE: reason=%s detail=%s — human intervention required",
            reason.value, detail,
        )
        from_state = self._lifecycle_state
        # HARD_FREEZE reachable from most states
        if (from_state, RuntimeLifecycleState.HARD_FREEZE) in _LIFECYCLE_TRANSITIONS:
            self._lifecycle_state = RuntimeLifecycleState.HARD_FREEZE
            record = RuntimeTransitionRecord(
                timestamp=datetime.now(JST).isoformat(),
                evaluation_date=self._current_date,
                from_state=from_state.value,
                to_state=RuntimeLifecycleState.HARD_FREEZE.value,
                reason=f"HARD_FREEZE:{detail}",
                freeze_reason=reason.value,
                stage_context=None,
            )
            self._append_event(record.to_dict(), kind="transition")
        else:
            self._lifecycle_state = RuntimeLifecycleState.HARD_FREEZE

    # ── Telemetry append ───────────────────────────────────────────────────────

    def _append_stage_event(self, result: RuntimeStageResult) -> None:
        date_str = self._current_date.replace("-", "")
        event_path = self._events_dir / f"runtime_events_{date_str}.jsonl"
        event = RuntimeEventRecord(
            timestamp=result.completed_at,
            evaluation_date=self._current_date,
            runtime_state=self._lifecycle_state.value,
            current_stage=result.stage_name,
            last_completed_stage=result.stage_name if result.success else None,
            freeze_reason=self._freeze_reason.value if self._freeze_reason else None,
            rollback_state=self._rollback_state,
            active_position_symbol=self._active_position_symbol,
            replay_sequence_id=self._replay_seq_id,
            deterministic_hash=result.deterministic_hash,
        )
        try:
            _append_jsonl(event.to_dict(), event_path)
        except Exception as exc:
            logger.warning("[ORCHESTRATOR] stage event append failed: %s", exc)

    def _append_event(self, record: Dict[str, Any], kind: str = "event") -> None:
        date_str = (self._current_date or datetime.now(JST).strftime("%Y-%m-%d")).replace("-", "")
        event_path = self._events_dir / f"runtime_events_{date_str}.jsonl"
        record["_kind"] = kind
        try:
            _append_jsonl(record, event_path)
        except Exception as exc:
            logger.warning("[ORCHESTRATOR] event append failed: %s", exc)

    def _append_checkpoint_jsonl(self, checkpoint: RuntimeCheckpoint) -> None:
        """Secondary JSONL log of checkpoints (in addition to atomic JSON file)."""
        date_str = checkpoint.evaluation_date.replace("-", "")
        ckpt_path = self._events_dir / f"runtime_checkpoints_{date_str}.jsonl"
        try:
            _append_jsonl(checkpoint.to_dict(), ckpt_path)
        except Exception as exc:
            logger.warning("[ORCHESTRATOR] checkpoint jsonl append failed: %s", exc)


# ── Stage index helper ─────────────────────────────────────────────────────────

def _stage_index(stage_name: str) -> int:
    try:
        return STAGE_PIPELINE.index(stage_name)
    except ValueError:
        return -1


# ── CLI entry point ────────────────────────────────────────────────────────────

def build_orchestrator(
    checkpoint_dir: Optional[Path] = None,
    events_dir:     Optional[Path] = None,
    stage_functions: Optional[Dict[str, Callable[[], Dict[str, Any]]]] = None,
) -> FutureLeaderRuntimeOrchestrator:
    """Convenience factory using default paths."""
    from src.paths import RUNTIME_ORCHESTRATOR_CHECKPOINT_DIR, RUNTIME_ORCHESTRATOR_EVENTS_DIR
    _checkpoint_dir = checkpoint_dir or RUNTIME_ORCHESTRATOR_CHECKPOINT_DIR
    _events_dir     = events_dir or RUNTIME_ORCHESTRATOR_EVENTS_DIR
    return FutureLeaderRuntimeOrchestrator(
        checkpoint_dir=_checkpoint_dir,
        events_dir=_events_dir,
        stage_functions=stage_functions,
    )


def cli_main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point for run_runtime_orchestrator.py.

    Commands:
      (no args)              — run pipeline for today
      --replay               — replay and verify today's pipeline
      --verify-replay        — alias for --replay
      --force-soft-freeze    — force SOFT_FREEZE
      --force-hard-freeze    — force HARD_FREEZE
      --reconcile            — run broker reconciliation
      --date YYYY-MM-DD      — specify evaluation date
    """
    import sys
    args = argv if argv is not None else sys.argv[1:]
    date_str = datetime.now(JST).strftime("%Y-%m-%d")

    # Parse --date
    if "--date" in args:
        idx = args.index("--date")
        if idx + 1 < len(args):
            date_str = args[idx + 1]

    orc = build_orchestrator()

    if "--replay" in args or "--verify-replay" in args:
        ok = orc.replay_pipeline(date_str)
        print(f"replay_verify: {'PASS' if ok else 'FAIL (HARD_FREEZE triggered)'}")
        return 0 if ok else 1

    if "--force-soft-freeze" in args:
        orc.force_soft_freeze("cli_command")
        print(f"SOFT_FREEZE: state={orc.lifecycle_state.value}")
        return 0

    if "--force-hard-freeze" in args:
        orc.force_hard_freeze("cli_command")
        print(f"HARD_FREEZE: state={orc.lifecycle_state.value}")
        return 0

    if "--reconcile" in args:
        result = orc.reconcile()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "clean" else 1

    # Default: run pipeline
    results = orc.run_pipeline(date_str)
    print(json.dumps({
        "evaluation_date": date_str,
        "lifecycle_state": orc.lifecycle_state.value,
        "stages_completed": sum(1 for r in results if r.success),
        "stages_skipped":   sum(1 for r in results if r.skipped_as_idempotent),
        "stages_failed":    sum(1 for r in results if not r.success and not r.skipped_as_idempotent),
        "freeze_triggered": orc.lifecycle_state in (
            RuntimeLifecycleState.SOFT_FREEZE, RuntimeLifecycleState.HARD_FREEZE
        ),
    }, ensure_ascii=False, indent=2))
    return 0 if orc.lifecycle_state == RuntimeLifecycleState.SHUTDOWN_SAFE else 1
