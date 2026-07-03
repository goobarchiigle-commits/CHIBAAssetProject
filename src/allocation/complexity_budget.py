"""
complexity_budget.py — Allocation complexity governance

New allocation signals/factors may ONLY be admitted if:
  - out-of-sample persistence confirmed
  - post-cost persistence confirmed
  - marginal CAGR improvement verified
  - replay determinism preserved
  - telemetry explainability preserved
  - operational complexity justified

Hard cap: MAX_FACTORS prevents replay-safe spaghetti allocators.
All decisions are append-only to a governance log.
Operational simplicity is a core invariant.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FACTORS: int = 12                     # hard cap on active factor count
MIN_MARGINAL_CAGR_BPS: float = 10.0      # minimum marginal improvement to admit
SCHEMA_VERSION: int = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AllocatorFactor:
    """Registered allocation factor. Immutable after admission."""
    factor_id: str
    description: str
    admitted_date: str                    # YYYY-MM-DD
    out_of_sample_confirmed: bool
    post_cost_confirmed: bool
    marginal_cagr_bps: float             # observed marginal CAGR improvement
    replay_deterministic: bool
    explainability_preserved: bool
    active: bool = True


@dataclass
class ComplexityBudget:
    """Current allocator complexity state."""
    active_factor_count: int = 0
    max_factors: int = MAX_FACTORS
    factors: list = field(default_factory=list)   # list[AllocatorFactor] as dicts
    schema_version: int = SCHEMA_VERSION


@dataclass
class ComplexityAuditRecord:
    """Append-only governance log record."""
    timestamp: str
    action: str                           # "ADMITTED" | "REJECTED" | "DEACTIVATED"
    factor_id: str
    reason: str
    active_count_before: int
    active_count_after: int
    schema_version: int = SCHEMA_VERSION


# ── Admission logic ────────────────────────────────────────────────────────────

def can_admit_factor(
    budget: ComplexityBudget,
    factor: AllocatorFactor,
) -> tuple[bool, str]:
    """
    Check all admission criteria for a new factor.
    Returns (can_admit: bool, reason: str).
    All checks must pass. Fail-closed: any failure → reject.
    """
    active_count = sum(
        1 for f in budget.factors
        if (f.get("active", True) if isinstance(f, dict) else f.active)
    )

    if active_count >= budget.max_factors:
        return False, (
            f"complexity cap reached ({active_count}/{budget.max_factors}); "
            "deactivate an existing factor first"
        )

    if not factor.out_of_sample_confirmed:
        return False, "out-of-sample persistence not confirmed"

    if not factor.post_cost_confirmed:
        return False, "post-cost persistence not confirmed"

    if factor.marginal_cagr_bps < MIN_MARGINAL_CAGR_BPS:
        return False, (
            f"marginal CAGR {factor.marginal_cagr_bps:.1f}bps < "
            f"minimum {MIN_MARGINAL_CAGR_BPS:.1f}bps"
        )

    if not factor.replay_deterministic:
        return False, "replay determinism not verified"

    if not factor.explainability_preserved:
        return False, "explainability not preserved"

    return True, f"all criteria met; marginal_cagr={factor.marginal_cagr_bps:.1f}bps"


def admit_factor(
    budget: ComplexityBudget,
    factor: AllocatorFactor,
    timestamp: str,
    audit_path: Path,
) -> tuple[ComplexityBudget, bool, str]:
    """
    Attempt to admit a factor. Updates budget if successful.
    Returns (updated_budget, admitted, reason).
    Appends to audit log regardless of outcome.
    """
    active_before = sum(
        1 for f in budget.factors
        if (f.get("active", True) if isinstance(f, dict) else f.active)
    )
    ok, reason = can_admit_factor(budget, factor)

    if ok:
        new_factors = list(budget.factors) + [asdict(factor)]
        active_after = active_before + 1
        new_budget = ComplexityBudget(
            active_factor_count=active_after,
            max_factors=budget.max_factors,
            factors=new_factors,
        )
        action = "ADMITTED"
    else:
        new_budget = budget
        active_after = active_before
        action = "REJECTED"

    audit = ComplexityAuditRecord(
        timestamp=timestamp,
        action=action,
        factor_id=factor.factor_id,
        reason=reason,
        active_count_before=active_before,
        active_count_after=active_after,
    )
    append_audit_record(audit, audit_path)

    return new_budget, ok, reason


def deactivate_factor(
    budget: ComplexityBudget,
    factor_id: str,
    timestamp: str,
    reason: str,
    audit_path: Path,
) -> ComplexityBudget:
    """
    Mark a factor as inactive. Decrements active count.
    Appends deactivation record to audit log.
    """
    active_before = budget.active_factor_count
    new_factors = []
    deactivated = False

    for f in budget.factors:
        if isinstance(f, dict):
            if f.get("factor_id") == factor_id and f.get("active", True):
                f = dict(f)
                f["active"] = False
                deactivated = True
        new_factors.append(f)

    active_after = sum(
        1 for f in new_factors
        if (f.get("active", True) if isinstance(f, dict) else f.active)
    )

    audit = ComplexityAuditRecord(
        timestamp=timestamp,
        action="DEACTIVATED",
        factor_id=factor_id,
        reason=reason,
        active_count_before=active_before,
        active_count_after=active_after,
    )
    append_audit_record(audit, audit_path)

    if not deactivated:
        logger.warning("deactivate_factor: factor_id '%s' not found or already inactive", factor_id)

    return ComplexityBudget(
        active_factor_count=active_after,
        max_factors=budget.max_factors,
        factors=new_factors,
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def append_audit_record(record: ComplexityAuditRecord, path: Path) -> None:
    """Append governance audit record to append-only JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_complexity_budget(path: Path) -> ComplexityBudget:
    if not path.exists():
        return ComplexityBudget()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        factors = data.get("factors", [])
        active_count = sum(
            1 for f in factors if f.get("active", True)
        )
        return ComplexityBudget(
            active_factor_count=active_count,
            max_factors=int(data.get("max_factors", MAX_FACTORS)),
            factors=factors,
        )
    except Exception as exc:
        logger.warning("complexity_budget load failed (%s) — default", exc)
        return ComplexityBudget()


def save_complexity_budget(budget: ComplexityBudget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(budget), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_audit_records(path: Path) -> list[ComplexityAuditRecord]:
    """Replay full governance audit log."""
    if not path.exists():
        return []
    records: list[ComplexityAuditRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(ComplexityAuditRecord(
                timestamp=data.get("timestamp", ""),
                action=data.get("action", ""),
                factor_id=data.get("factor_id", ""),
                reason=data.get("reason", ""),
                active_count_before=int(data.get("active_count_before", 0)),
                active_count_after=int(data.get("active_count_after", 0)),
            ))
    except Exception as exc:
        logger.warning("audit_records load failed (%s)", exc)
    return records
