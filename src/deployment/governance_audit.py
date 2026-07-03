"""
src/deployment/governance_audit.py
GovernanceAuditRecord — deterministic, append-only governance telemetry.

Purpose:
  Record every governance evaluation cycle with deterministic hash.
  Enables replay mismatch detection, rollback traceability, and
  governance continuity verification across restarts.

INVARIANTS:
  - append-only JSONL: records are never mutated or deleted
  - deterministic hash: same inputs → same hash
  - freeze persistence: MIN_FREEZE_DAYS = 3
  - unfreeze requires: consecutive_clean_reconciliations >= 3
                        + no_api_failures + no_replay_mismatch
                        + rollback_pressure_score < UNFREEZE_PRESSURE_THRESHOLD

Freeze state file: runtime/deployment/governance_audit/freeze_state.json
Audit JSONL dir:   runtime/deployment/governance_audit/
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Freeze governance constants (変更禁止) ─────────────────────────────────────
MIN_FREEZE_DAYS: int            = 3
MIN_CLEAN_RECONCILIATIONS: int  = 3
UNFREEZE_PRESSURE_THRESHOLD: float = 0.50   # rollback_pressure_score must be below this


# ── Reconciliation status ──────────────────────────────────────────────────────

class ReconciliationStatus:
    CLEAN:    str = "clean"
    MISMATCH: str = "mismatch"
    UNKNOWN:  str = "unknown"
    SKIPPED:  str = "skipped"


# ── Core dataclass ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class GovernanceAuditRecord:
    """
    One governance evaluation cycle record.
    deterministic_hash computed from all upstream inputs — same inputs → same hash.
    """
    evaluation_date:       str
    upstream_hashes:       Dict[str, str]
    eligibility_gate_passed: bool
    governance_decisions:  List[str]
    rollback_decisions:    List[str]
    freeze_decisions:      List[str]
    reconciliation_status: str
    deterministic_hash:    str
    # metadata
    live_mode:             str = "human_confirmed"
    freeze_active:         bool = False
    freeze_day_count:      int  = 0
    consecutive_clean:     int  = 0
    timestamp:             str  = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_date":        self.evaluation_date,
            "upstream_hashes":        self.upstream_hashes,
            "eligibility_gate_passed": self.eligibility_gate_passed,
            "governance_decisions":   self.governance_decisions,
            "rollback_decisions":     self.rollback_decisions,
            "freeze_decisions":       self.freeze_decisions,
            "reconciliation_status":  self.reconciliation_status,
            "deterministic_hash":     self.deterministic_hash,
            "live_mode":              self.live_mode,
            "freeze_active":          self.freeze_active,
            "freeze_day_count":       self.freeze_day_count,
            "consecutive_clean":      self.consecutive_clean,
            "timestamp":              self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GovernanceAuditRecord":
        return cls(
            evaluation_date=d["evaluation_date"],
            upstream_hashes=d.get("upstream_hashes", {}),
            eligibility_gate_passed=bool(d.get("eligibility_gate_passed", False)),
            governance_decisions=d.get("governance_decisions", []),
            rollback_decisions=d.get("rollback_decisions", []),
            freeze_decisions=d.get("freeze_decisions", []),
            reconciliation_status=d.get("reconciliation_status", "unknown"),
            deterministic_hash=d.get("deterministic_hash", ""),
            live_mode=d.get("live_mode", "human_confirmed"),
            freeze_active=bool(d.get("freeze_active", False)),
            freeze_day_count=int(d.get("freeze_day_count", 0)),
            consecutive_clean=int(d.get("consecutive_clean", 0)),
            timestamp=d.get("timestamp", ""),
        )


# ── Freeze state ───────────────────────────────────────────────────────────────

@dataclass
class GovernanceFreezeState:
    """
    Persisted freeze governance state.
    Mutable only through FreezeEvaluator — never patched directly.
    """
    frozen:                      bool  = False
    freeze_start_date:           Optional[str] = None
    freeze_day_count:            int   = 0
    consecutive_clean_reconciliations: int = 0
    last_evaluation_date:        Optional[str] = None
    last_unfreeze_date:          Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frozen":                       self.frozen,
            "freeze_start_date":            self.freeze_start_date,
            "freeze_day_count":             self.freeze_day_count,
            "consecutive_clean_reconciliations": self.consecutive_clean_reconciliations,
            "last_evaluation_date":         self.last_evaluation_date,
            "last_unfreeze_date":           self.last_unfreeze_date,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GovernanceFreezeState":
        return cls(
            frozen=bool(d.get("frozen", False)),
            freeze_start_date=d.get("freeze_start_date"),
            freeze_day_count=int(d.get("freeze_day_count", 0)),
            consecutive_clean_reconciliations=int(
                d.get("consecutive_clean_reconciliations", 0)
            ),
            last_evaluation_date=d.get("last_evaluation_date"),
            last_unfreeze_date=d.get("last_unfreeze_date"),
        )


def load_freeze_state(path: Path) -> GovernanceFreezeState:
    if not path.exists():
        return GovernanceFreezeState()
    try:
        return GovernanceFreezeState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("[FREEZE_STATE] load failed (%s) — returning default", exc)
        return GovernanceFreezeState()


def save_freeze_state(state: GovernanceFreezeState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ── FreezeEvaluator ────────────────────────────────────────────────────────────

class FreezeEvaluator:
    """
    Evaluates and transitions governance freeze state.

    Freeze triggers (any one is sufficient):
      - broker_timeout=True
      - api_integrity_unknown=True
      - replay_mismatch=True
      - reconciliation_status == "mismatch"

    Unfreeze conditions (ALL must hold):
      - frozen for at least MIN_FREEZE_DAYS days
      - consecutive_clean_reconciliations >= MIN_CLEAN_RECONCILIATIONS
      - no_api_failures (api_reachable=True)
      - no_replay_mismatch
      - rollback_pressure_score < UNFREEZE_PRESSURE_THRESHOLD

    Freeze oscillation prevention:
      - immediate freeze recovery forbidden
      - auto-unfreeze without persistence validation forbidden
    """

    def evaluate(
        self,
        state: GovernanceFreezeState,
        evaluation_date: str,
        *,
        broker_timeout:          bool  = False,
        api_integrity_unknown:   bool  = False,
        replay_mismatch:         bool  = False,
        reconciliation_status:   str   = ReconciliationStatus.UNKNOWN,
        api_reachable:           bool  = True,
        rollback_pressure_score: float = 0.0,
    ) -> tuple[GovernanceFreezeState, List[str]]:
        """
        Returns updated freeze state + list of freeze_decisions taken this cycle.
        Never mutates input state.
        """
        new = GovernanceFreezeState(
            frozen=state.frozen,
            freeze_start_date=state.freeze_start_date,
            freeze_day_count=state.freeze_day_count,
            consecutive_clean_reconciliations=state.consecutive_clean_reconciliations,
            last_evaluation_date=evaluation_date,
            last_unfreeze_date=state.last_unfreeze_date,
        )
        decisions: List[str] = []

        # Trigger conditions
        should_freeze = (
            broker_timeout
            or api_integrity_unknown
            or replay_mismatch
            or reconciliation_status == ReconciliationStatus.MISMATCH
        )

        if should_freeze and not new.frozen:
            new.frozen = True
            new.freeze_start_date = evaluation_date
            new.freeze_day_count = 1
            new.consecutive_clean_reconciliations = 0
            reasons = []
            if broker_timeout:          reasons.append("broker_timeout")
            if api_integrity_unknown:   reasons.append("api_integrity_unknown")
            if replay_mismatch:         reasons.append("replay_mismatch")
            if reconciliation_status == ReconciliationStatus.MISMATCH:
                reasons.append("reconciliation_mismatch")
            decisions.append(f"FREEZE_TRIGGERED: {','.join(reasons)}")
            logger.warning("[FREEZE] governance frozen: %s", decisions[-1])
            return new, decisions

        if new.frozen:
            new.freeze_day_count += 1
            # Clean reconciliation tracking
            if (
                reconciliation_status == ReconciliationStatus.CLEAN
                and api_reachable
                and not replay_mismatch
            ):
                new.consecutive_clean_reconciliations += 1
            else:
                # Reset clean counter on any impurity
                new.consecutive_clean_reconciliations = 0
                decisions.append(
                    f"FREEZE_CLEAN_RESET: recon={reconciliation_status} "
                    f"api_reachable={api_reachable} replay_mismatch={replay_mismatch}"
                )

            # Re-trigger: reset counter if new freeze condition during frozen period
            if should_freeze:
                new.consecutive_clean_reconciliations = 0
                decisions.append("FREEZE_RETRIGGERED: new freeze condition during frozen period")
                return new, decisions

            # Unfreeze conditions (ALL required)
            min_days_met   = new.freeze_day_count >= MIN_FREEZE_DAYS
            clean_met      = new.consecutive_clean_reconciliations >= MIN_CLEAN_RECONCILIATIONS
            api_ok         = api_reachable and not api_integrity_unknown
            pressure_ok    = rollback_pressure_score < UNFREEZE_PRESSURE_THRESHOLD

            if min_days_met and clean_met and api_ok and pressure_ok:
                decisions.append(
                    f"FREEZE_RELEASED: days={new.freeze_day_count} "
                    f"clean={new.consecutive_clean_reconciliations} "
                    f"pressure={rollback_pressure_score:.3f}"
                )
                logger.info("[FREEZE] governance unfrozen after %d days", new.freeze_day_count)
                new.frozen = False
                new.freeze_start_date = None
                new.freeze_day_count = 0
                new.consecutive_clean_reconciliations = 0
                new.last_unfreeze_date = evaluation_date
            else:
                pending = []
                if not min_days_met:
                    pending.append(f"need {MIN_FREEZE_DAYS - new.freeze_day_count}d more")
                if not clean_met:
                    pending.append(
                        f"need {MIN_CLEAN_RECONCILIATIONS - new.consecutive_clean_reconciliations} more clean recons"
                    )
                if not api_ok:
                    pending.append("api_not_healthy")
                if not pressure_ok:
                    pending.append(f"pressure={rollback_pressure_score:.3f} >= {UNFREEZE_PRESSURE_THRESHOLD}")
                decisions.append(f"FREEZE_MAINTAINED: {'; '.join(pending)}")
        else:
            # Not frozen: track clean reconciliations for historical context
            if reconciliation_status == ReconciliationStatus.CLEAN:
                new.consecutive_clean_reconciliations = min(
                    new.consecutive_clean_reconciliations + 1, MIN_CLEAN_RECONCILIATIONS
                )

        return new, decisions


# ── Deterministic hash ─────────────────────────────────────────────────────────

def compute_governance_hash(
    universe_version:       str,
    deployment_manifest_hash: str,
    rollback_state:         str,
    freeze_state_frozen:    bool,
    governance_decisions:   List[str],
    live_mode:              str,
    upstream_hashes:        Dict[str, str],
) -> str:
    """
    Deterministic hash of governance inputs.
    Same inputs always produce the same hash.
    Used for replay mismatch detection.
    """
    payload = {
        "universe_version":        universe_version,
        "deployment_manifest_hash": deployment_manifest_hash,
        "rollback_state":          rollback_state,
        "freeze_state_frozen":     freeze_state_frozen,
        "governance_decisions":    sorted(governance_decisions),
        "live_mode":               live_mode,
        "upstream_hashes":         dict(sorted(upstream_hashes.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ── Append-only JSONL writer ───────────────────────────────────────────────────

def append_governance_audit(record: GovernanceAuditRecord, audit_dir: Path) -> None:
    """
    Append GovernanceAuditRecord to dated JSONL file.
    Creates directory if needed.
    Raises IOError on write failure (fail-closed).
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    date_str = record.evaluation_date.replace("-", "")
    out_path = audit_dir / f"governance_audit_{date_str}.jsonl"
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_governance_audits(audit_dir: Path, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load governance audit records from JSONL.
    date_str: YYYYMMDD format to load specific day; None loads all.
    """
    if not audit_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    if date_str:
        paths = [audit_dir / f"governance_audit_{date_str}.jsonl"]
    else:
        paths = sorted(audit_dir.glob("governance_audit_*.jsonl"))
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_last_governance_audit(audit_dir: Path) -> Optional[GovernanceAuditRecord]:
    """Return the most recent GovernanceAuditRecord for diff comparison."""
    if not audit_dir.exists():
        return None
    paths = sorted(audit_dir.glob("governance_audit_*.jsonl"))
    for p in reversed(paths):
        try:
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            continue
        for line in reversed(lines):
            try:
                return GovernanceAuditRecord.from_dict(json.loads(line))
            except Exception:
                pass
    return None


# ── Phase 4B.1: Governance Decision Diff ──────────────────────────────────────

@dataclass(slots=True)
class GovernanceDecisionDiff:
    """
    Lightweight diff between two consecutive governance audit records.
    Set-based only — O(n) — no deep comparison.
    One record per pipeline run (append-only JSONL).
    """
    ts:                    str
    previous_hash:         Optional[str]
    current_hash:          str
    changed:               bool
    added_decisions:       List[str]
    removed_decisions:     List[str]
    freeze_state_changed:  bool
    rollback_state_changed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts":                     self.ts,
            "previous_hash":          self.previous_hash,
            "current_hash":           self.current_hash,
            "changed":                self.changed,
            "added_decisions":        self.added_decisions,
            "removed_decisions":      self.removed_decisions,
            "freeze_state_changed":   self.freeze_state_changed,
            "rollback_state_changed": self.rollback_state_changed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GovernanceDecisionDiff":
        return cls(
            ts=d["ts"],
            previous_hash=d.get("previous_hash"),
            current_hash=d.get("current_hash", ""),
            changed=bool(d.get("changed", False)),
            added_decisions=d.get("added_decisions", []),
            removed_decisions=d.get("removed_decisions", []),
            freeze_state_changed=bool(d.get("freeze_state_changed", False)),
            rollback_state_changed=bool(d.get("rollback_state_changed", False)),
        )


def compute_governance_diff(
    previous: Optional[GovernanceAuditRecord],
    current:  GovernanceAuditRecord,
) -> GovernanceDecisionDiff:
    """
    Compute set-based diff between two consecutive audit records.
    previous=None means first-ever run (no prior record).
    """
    ts = datetime.now(JST).isoformat()
    prev_hash = previous.deterministic_hash if previous is not None else None

    if previous is not None:
        prev_all = (
            set(previous.governance_decisions)
            | set(previous.rollback_decisions)
            | set(previous.freeze_decisions)
        )
        freeze_changed    = previous.freeze_active != current.freeze_active
        rollback_changed  = set(previous.rollback_decisions) != set(current.rollback_decisions)
    else:
        prev_all         = set()
        freeze_changed   = False
        rollback_changed = False

    curr_all = (
        set(current.governance_decisions)
        | set(current.rollback_decisions)
        | set(current.freeze_decisions)
    )

    added   = sorted(curr_all - prev_all)
    removed = sorted(prev_all - curr_all)

    changed = (
        prev_hash != current.deterministic_hash
        or bool(added)
        or bool(removed)
        or freeze_changed
        or rollback_changed
    )

    return GovernanceDecisionDiff(
        ts=ts,
        previous_hash=prev_hash,
        current_hash=current.deterministic_hash,
        changed=changed,
        added_decisions=added,
        removed_decisions=removed,
        freeze_state_changed=freeze_changed,
        rollback_state_changed=rollback_changed,
    )


def append_governance_diff(diff: GovernanceDecisionDiff, diff_dir: Path) -> None:
    """
    Append GovernanceDecisionDiff to dated JSONL.
    Max 1 record per pipeline run.
    Raises IOError on failure.
    """
    diff_dir.mkdir(parents=True, exist_ok=True)
    try:
        date_str = diff.ts[:10].replace("-", "")
    except Exception:
        date_str = datetime.now(JST).strftime("%Y%m%d")
    out_path = diff_dir / f"governance_diff_{date_str}.jsonl"
    line = json.dumps(diff.to_dict(), ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)


def load_governance_diffs(diff_dir: Path, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load governance diff records from JSONL.
    date_str: YYYYMMDD format for specific day; None loads all.
    """
    if not diff_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    if date_str:
        paths = [diff_dir / f"governance_diff_{date_str}.jsonl"]
    else:
        paths = sorted(diff_dir.glob("governance_diff_*.jsonl"))
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
