"""
execution_integrity_validator.py — Fail-closed execution gate

Blocks execution (live mode) when reconciliation reveals integrity violations
that cannot be safely ignored:
  - Blocking reconciliation mismatches (orphan fills, duplicate pending, missing fills)
  - Stale broker snapshot (data too old to trust)
  - Invalid snapshot checksum (data tampered / corrupted)
  - Unresolved SUBMITTED_UNKNOWN orders from a prior day

For dry-run: validation runs, result is logged, execution NOT blocked.
For live mode: execution_allowed=False → caller must abort order submission.

Telemetry (append-only JSONL) is always written regardless of mode.
Never raises. Fail-open for the telemetry path itself.
"""
from __future__ import annotations

import json
import logging
import time as _time_mod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.live.broker_truth_snapshot import (
    BrokerTruthSnapshot, is_snapshot_stale, validate_snapshot,
)
from src.live.reconciliation_engine import ReconciliationResult

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

VIOLATION_BLOCKING_MISMATCH = "blocking_reconciliation_mismatch"
VIOLATION_STALE_SNAPSHOT    = "stale_broker_snapshot"
VIOLATION_BAD_CHECKSUM      = "invalid_snapshot_checksum"
VIOLATION_NO_SNAPSHOT       = "missing_broker_snapshot"


@dataclass
class IntegrityViolation:
    violation_type: str
    detail: str
    blocking: bool = True


@dataclass
class ExecutionIntegrityResult:
    timestamp: str
    run_id: str
    execution_allowed: bool
    is_live: bool
    integrity_violations: list   # list[dict] — IntegrityViolation serialised
    fail_closed_reason: str      # first blocking violation detail; "" if allowed
    reconciliation_ok: bool
    snapshot_valid: bool
    blocking_count: int
    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_execution_integrity(
    reconciliation_result: ReconciliationResult,
    snapshot: Optional[BrokerTruthSnapshot],
    is_live: bool,
    now_sec: Optional[float] = None,
    snapshot_stale_sec: float = 300.0,
    timestamp: str = "",
) -> ExecutionIntegrityResult:
    """
    Validate execution integrity. Returns ExecutionIntegrityResult.

    execution_allowed is False when:
      - reconciliation_result has blocking mismatches AND is_live=True
      - snapshot is None AND is_live=True
      - snapshot checksum is invalid AND is_live=True
      - snapshot is stale AND is_live=True

    For dry-run (is_live=False): violations are recorded but execution_allowed=True.
    Never raises.
    """
    if now_sec is None:
        now_sec = _time_mod.time()
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()

    run_id = reconciliation_result.run_id if reconciliation_result else ""
    violations: List[IntegrityViolation] = []

    # Check 1: blocking reconciliation mismatches
    if reconciliation_result is not None:
        if reconciliation_result.blocking_mismatches > 0:
            details = [
                m.get("detail", m.get("mismatch_type", "?"))
                for m in reconciliation_result.mismatches
                if m.get("severity") == "blocking"
            ]
            violations.append(IntegrityViolation(
                violation_type=VIOLATION_BLOCKING_MISMATCH,
                detail=(
                    f"{reconciliation_result.blocking_mismatches} blocking mismatch(es): "
                    + "; ".join(details[:3])
                ),
                blocking=True,
            ))

    # Check 2: snapshot presence
    if snapshot is None:
        violations.append(IntegrityViolation(
            violation_type=VIOLATION_NO_SNAPSHOT,
            detail="broker_truth_snapshot is None — broker state unverified",
            blocking=True,
        ))
    else:
        # Check 3: snapshot checksum
        if not validate_snapshot(snapshot):
            violations.append(IntegrityViolation(
                violation_type=VIOLATION_BAD_CHECKSUM,
                detail=f"snapshot checksum mismatch run_id={snapshot.run_id}",
                blocking=True,
            ))

        # Check 4: snapshot staleness
        if is_snapshot_stale(snapshot, now_sec, snapshot_stale_sec):
            violations.append(IntegrityViolation(
                violation_type=VIOLATION_STALE_SNAPSHOT,
                detail=(
                    f"snapshot older than {snapshot_stale_sec:.0f}s"
                    f" ts={snapshot.timestamp}"
                ),
                blocking=True,
            ))

    blocking_violations = [v for v in violations if v.blocking]
    blocking_count = len(blocking_violations)
    reconciliation_ok = (
        reconciliation_result is not None and reconciliation_result.reconciliation_ok
    )
    snapshot_valid = (
        snapshot is not None and validate_snapshot(snapshot)
    )

    # execution_allowed: False only when there are blocking violations AND live mode
    execution_allowed = blocking_count == 0 or not is_live
    fail_closed_reason = ""
    if not execution_allowed:
        fail_closed_reason = blocking_violations[0].detail if blocking_violations else "unknown"

    if violations:
        for v in violations:
            log_fn = logger.error if v.blocking and is_live else logger.warning
            log_fn(
                "[INTEGRITY] %s %s: %s",
                "BLOCKING" if v.blocking else "WARNING",
                v.violation_type,
                v.detail,
            )
    else:
        logger.info("[INTEGRITY] ok — execution_allowed=True run_id=%s", run_id)

    return ExecutionIntegrityResult(
        timestamp=timestamp,
        run_id=run_id,
        execution_allowed=execution_allowed,
        is_live=is_live,
        integrity_violations=[asdict(v) for v in violations],
        fail_closed_reason=fail_closed_reason,
        reconciliation_ok=reconciliation_ok,
        snapshot_valid=snapshot_valid,
        blocking_count=blocking_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_integrity_result(result: ExecutionIntegrityResult, path: Path) -> None:
    """Fail-safe append. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(result), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[INTEGRITY] log write failed: %s", exc)


def load_integrity_results(path: Path) -> List[ExecutionIntegrityResult]:
    """Load all results. Corrupted lines skipped."""
    if not path.exists():
        return []
    results: List[ExecutionIntegrityResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            results.append(ExecutionIntegrityResult(
                timestamp=d.get("timestamp", ""),
                run_id=d.get("run_id", ""),
                execution_allowed=bool(d.get("execution_allowed", True)),
                is_live=bool(d.get("is_live", False)),
                integrity_violations=d.get("integrity_violations", []),
                fail_closed_reason=d.get("fail_closed_reason", ""),
                reconciliation_ok=bool(d.get("reconciliation_ok", True)),
                snapshot_valid=bool(d.get("snapshot_valid", True)),
                blocking_count=int(d.get("blocking_count", 0)),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[INTEGRITY] parse error: %s", exc)
    return results
