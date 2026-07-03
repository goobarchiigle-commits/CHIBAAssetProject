"""
quarantine_governance.py — Immutable quarantine attribution and governance

Quarantine is entered when:
  - integrity_violation: checksum failure, stale snapshot, blocking mismatch
  - reconciliation_mismatch: blocking mismatches from reconciliation engine
  - authority_failure: pending order authority divergence blocks deployment
  - replay_divergence: lineage chain fails verification
  - epoch_inconsistency: deployment epoch inconsistent with reconciliation epoch

Quarantine exit requires all three prerequisites:
  1. reconciliation_ok   — no blocking mismatches
  2. epoch_normalized    — deployment epoch consistent with current reconciliation epoch
  3. integrity_valid     — integrity validation passes

Entries are immutable. Recovery creates a separate exit attempt record.
Append-only JSONL. Never auto-exits without explicit prerequisite verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

# Entry reasons
QUARANTINE_INTEGRITY_VIOLATION    = "integrity_violation"
QUARANTINE_RECONCILIATION_MISMATCH = "reconciliation_mismatch"
QUARANTINE_AUTHORITY_FAILURE      = "authority_failure"
QUARANTINE_REPLAY_DIVERGENCE      = "replay_divergence"
QUARANTINE_EPOCH_INCONSISTENCY    = "epoch_inconsistency"

# Quarantine states (informational — state is derived from entry + exit log)
QSTATE_QUARANTINED      = "QUARANTINED"
QSTATE_PENDING_RECOVERY = "PENDING_RECOVERY"
QSTATE_RECOVERED        = "RECOVERED"

# Recovery prerequisite keys
RECOVERY_PREREQ_RECONCILIATION_OK = "reconciliation_ok"
RECOVERY_PREREQ_EPOCH_NORMALIZED  = "epoch_normalized"
RECOVERY_PREREQ_INTEGRITY_VALID   = "integrity_valid"


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# QuarantineEntry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineEntry:
    """
    Immutable attribution record for a quarantine event.
    Written once on quarantine entry. Never mutated.
    Recovery creates a QuarantineExitAttempt in a separate log.
    """
    quarantine_id: str
    run_id: str
    entry_reason: str
    entry_epoch_id: str
    broker_snapshot_hash: str
    reconciliation_epoch_id: str
    integrity_violation_detail: str
    reconciliation_mismatch_detail: str
    authority_failure_detail: str
    replay_divergence_detail: str
    quarantine_state: str
    created_at: str
    requires_reconciliation_ok: bool = True
    requires_epoch_normalized: bool = True
    requires_integrity_valid: bool = True
    recovery_blocked_reason: str = ""
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        run_id: str,
        entry_reason: str,
        entry_epoch_id: str,
        broker_snapshot_hash: str,
        reconciliation_epoch_id: str,
        integrity_violation_detail: str = "",
        reconciliation_mismatch_detail: str = "",
        authority_failure_detail: str = "",
        replay_divergence_detail: str = "",
        created_at: str = "",
    ) -> "QuarantineEntry":
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()
        quarantine_id = _sha256(_canonical_json({
            "created_at": created_at,
            "entry_epoch_id": entry_epoch_id,
            "entry_reason": entry_reason,
            "run_id": run_id,
        }))
        return QuarantineEntry(
            quarantine_id=quarantine_id,
            run_id=run_id,
            entry_reason=entry_reason,
            entry_epoch_id=entry_epoch_id,
            broker_snapshot_hash=broker_snapshot_hash,
            reconciliation_epoch_id=reconciliation_epoch_id,
            integrity_violation_detail=integrity_violation_detail,
            reconciliation_mismatch_detail=reconciliation_mismatch_detail,
            authority_failure_detail=authority_failure_detail,
            replay_divergence_detail=replay_divergence_detail,
            quarantine_state=QSTATE_QUARANTINED,
            created_at=created_at,
        )

    def validate_id(self) -> bool:
        try:
            expected = _sha256(_canonical_json({
                "created_at": self.created_at,
                "entry_epoch_id": self.entry_epoch_id,
                "entry_reason": self.entry_reason,
                "run_id": self.run_id,
            }))
            return expected == self.quarantine_id
        except Exception:
            return False

    def unmet_prerequisites(
        self,
        reconciliation_ok: bool,
        epoch_normalized: bool,
        integrity_valid: bool,
    ) -> List[str]:
        unmet = []
        if self.requires_reconciliation_ok and not reconciliation_ok:
            unmet.append(RECOVERY_PREREQ_RECONCILIATION_OK)
        if self.requires_epoch_normalized and not epoch_normalized:
            unmet.append(RECOVERY_PREREQ_EPOCH_NORMALIZED)
        if self.requires_integrity_valid and not integrity_valid:
            unmet.append(RECOVERY_PREREQ_INTEGRITY_VALID)
        return unmet


# ─────────────────────────────────────────────────────────────────────────────
# QuarantineExitAttempt
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineExitAttempt:
    """
    Records a quarantine exit attempt with all prerequisite results.
    exit_granted=True iff all three prerequisites pass.
    """
    attempt_id: str
    quarantine_id: str
    run_id: str
    reconciliation_ok: bool
    epoch_normalized: bool
    integrity_valid: bool
    exit_granted: bool
    exit_reason: str
    new_epoch_id: str
    attempted_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        quarantine_id: str,
        run_id: str,
        reconciliation_ok: bool,
        epoch_normalized: bool,
        integrity_valid: bool,
        exit_reason: str = "",
        new_epoch_id: str = "",
        attempted_at: str = "",
    ) -> "QuarantineExitAttempt":
        if not attempted_at:
            attempted_at = datetime.now(timezone.utc).isoformat()
        exit_granted = reconciliation_ok and epoch_normalized and integrity_valid
        if not exit_reason:
            if exit_granted:
                exit_reason = "all prerequisites satisfied"
            else:
                parts = []
                if not reconciliation_ok:
                    parts.append("reconciliation still has blocking mismatches")
                if not epoch_normalized:
                    parts.append("epoch not normalized")
                if not integrity_valid:
                    parts.append("integrity validation failed")
                exit_reason = "; ".join(parts)
        attempt_id = _sha256(_canonical_json({
            "attempted_at": attempted_at,
            "quarantine_id": quarantine_id,
            "run_id": run_id,
        }))
        return QuarantineExitAttempt(
            attempt_id=attempt_id,
            quarantine_id=quarantine_id,
            run_id=run_id,
            reconciliation_ok=reconciliation_ok,
            epoch_normalized=epoch_normalized,
            integrity_valid=integrity_valid,
            exit_granted=exit_granted,
            exit_reason=exit_reason,
            new_epoch_id=new_epoch_id,
            attempted_at=attempted_at,
        )


# ─────────────────────────────────────────────────────────────────────────────
# QuarantineStatus
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineStatus:
    """Current quarantine status derived from immutable entry + exit logs."""
    run_id: str
    is_quarantined: bool
    active_quarantine_id: Optional[str]
    entry_reason: Optional[str]
    recovery_prerequisites: List[str]
    checked_at: str


def evaluate_quarantine_exit(
    entry: QuarantineEntry,
    reconciliation_ok: bool,
    epoch_normalized: bool,
    integrity_valid: bool,
    exit_reason: str = "",
    new_epoch_id: str = "",
) -> QuarantineExitAttempt:
    """Evaluate exit prerequisites for the given quarantine entry."""
    return QuarantineExitAttempt.create(
        quarantine_id=entry.quarantine_id,
        run_id=entry.run_id,
        reconciliation_ok=reconciliation_ok,
        epoch_normalized=epoch_normalized,
        integrity_valid=integrity_valid,
        exit_reason=exit_reason,
        new_epoch_id=new_epoch_id,
    )


def get_quarantine_status(
    run_id: str,
    entries: List[QuarantineEntry],
    exit_attempts: List[QuarantineExitAttempt],
) -> QuarantineStatus:
    """
    Compute current quarantine status from immutable logs.
    A run is quarantined if its earliest active entry has no successful exit.
    """
    run_entries = sorted(
        [e for e in entries if e.run_id == run_id],
        key=lambda e: e.created_at,
    )
    successful_exits: set = {
        a.quarantine_id
        for a in exit_attempts
        if a.run_id == run_id and a.exit_granted
    }

    active_entry: Optional[QuarantineEntry] = None
    for entry in run_entries:
        if entry.quarantine_id not in successful_exits:
            active_entry = entry
            break

    if active_entry is None:
        return QuarantineStatus(
            run_id=run_id,
            is_quarantined=False,
            active_quarantine_id=None,
            entry_reason=None,
            recovery_prerequisites=[],
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    prereqs: List[str] = []
    if active_entry.requires_reconciliation_ok:
        prereqs.append(RECOVERY_PREREQ_RECONCILIATION_OK)
    if active_entry.requires_epoch_normalized:
        prereqs.append(RECOVERY_PREREQ_EPOCH_NORMALIZED)
    if active_entry.requires_integrity_valid:
        prereqs.append(RECOVERY_PREREQ_INTEGRITY_VALID)

    return QuarantineStatus(
        run_id=run_id,
        is_quarantined=True,
        active_quarantine_id=active_entry.quarantine_id,
        entry_reason=active_entry.entry_reason,
        recovery_prerequisites=prereqs,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def _safe_append(path: Path, record: dict, tag: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.warning("[%s] write failed: %s", tag, exc)


def append_quarantine_entry(entry: QuarantineEntry, path: Path) -> None:
    _safe_append(path, asdict(entry), "QUARANTINE")


def append_exit_attempt(attempt: QuarantineExitAttempt, path: Path) -> None:
    _safe_append(path, asdict(attempt), "QUARANTINE_EXIT")


def load_quarantine_entries(path: Path) -> List[QuarantineEntry]:
    if not path.exists():
        return []
    result: List[QuarantineEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(QuarantineEntry(
                quarantine_id=d.get("quarantine_id", ""),
                run_id=d.get("run_id", ""),
                entry_reason=d.get("entry_reason", ""),
                entry_epoch_id=d.get("entry_epoch_id", ""),
                broker_snapshot_hash=d.get("broker_snapshot_hash", ""),
                reconciliation_epoch_id=d.get("reconciliation_epoch_id", ""),
                integrity_violation_detail=d.get("integrity_violation_detail", ""),
                reconciliation_mismatch_detail=d.get("reconciliation_mismatch_detail", ""),
                authority_failure_detail=d.get("authority_failure_detail", ""),
                replay_divergence_detail=d.get("replay_divergence_detail", ""),
                quarantine_state=d.get("quarantine_state", QSTATE_QUARANTINED),
                created_at=d.get("created_at", ""),
                requires_reconciliation_ok=bool(d.get("requires_reconciliation_ok", True)),
                requires_epoch_normalized=bool(d.get("requires_epoch_normalized", True)),
                requires_integrity_valid=bool(d.get("requires_integrity_valid", True)),
                recovery_blocked_reason=d.get("recovery_blocked_reason", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[QUARANTINE] parse error: %s", exc)
    return result


def load_exit_attempts(path: Path) -> List[QuarantineExitAttempt]:
    if not path.exists():
        return []
    result: List[QuarantineExitAttempt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(QuarantineExitAttempt(
                attempt_id=d.get("attempt_id", ""),
                quarantine_id=d.get("quarantine_id", ""),
                run_id=d.get("run_id", ""),
                reconciliation_ok=bool(d.get("reconciliation_ok", False)),
                epoch_normalized=bool(d.get("epoch_normalized", False)),
                integrity_valid=bool(d.get("integrity_valid", False)),
                exit_granted=bool(d.get("exit_granted", False)),
                exit_reason=d.get("exit_reason", ""),
                new_epoch_id=d.get("new_epoch_id", ""),
                attempted_at=d.get("attempted_at", ""),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[QUARANTINE_EXIT] parse error: %s", exc)
    return result
