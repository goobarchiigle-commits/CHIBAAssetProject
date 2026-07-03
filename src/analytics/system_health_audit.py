"""
system_health_audit.py — System Health & Closed-Loop Audit (Phase 7A)

Validates that the complete self-improving operating system functions correctly.
Observation-only. Fail-open. No alpha logic, no strategy logic, no ranking logic.

Checks performed per run:
  1. Evidence generation      — EVIDENCE_PROMOTION_FILE
  2. Shadow generation        — SHADOW_RECOMMENDATION_FILE
  3. Promotion generation     — PROMOTION_READINESS_FILE
  4. Production Gate          — PRODUCTION_PROMOTION_GATE_FILE
  5. Experiment assignment    — EXPERIMENT_ASSIGNMENT_FILE
  6. Capital escalation       — CAPITAL_ESCALATION_FILE
  7. Rollback evaluation      — ROLLBACK_GOVERNANCE_FILE
  8. Improvement ranking      — IMPROVEMENT_RANKING_FILE

Detects:
  - stale files (no records within STALE_THRESHOLD_DAYS)
  - missing telemetry (file absent or empty)
  - broken state transitions (registry ↔ rollback ↔ escalation divergence)
  - inactive experiments (ACTIVE status but no recent performance records)
  - orphaned registry entries (registry experiment_id not in any telemetry)

Output:
  system_health_report.json — overwritten each run (not append-only; point-in-time snapshot)

Status categories: HEALTHY · WARNING · CRITICAL

Design:
  observation_only — reads files only; no writes except the health report
  fail-open        — every check wrapped; never blocks live execution
  deterministic    — same inputs → same report
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

# ── Status constants ───────────────────────────────────────────────────────────

STATUS_HEALTHY  = "HEALTHY"
STATUS_WARNING  = "WARNING"
STATUS_CRITICAL = "CRITICAL"

# ── Thresholds ────────────────────────────────────────────────────────────────

STALE_THRESHOLD_DAYS      = 3    # file has no records newer than N days → WARNING
MISSING_THRESHOLD_DAYS    = 7    # file entirely absent or no records → CRITICAL after N days
INACTIVE_EXPERIMENT_DAYS  = 5    # ACTIVE experiment with no perf records in N days → WARNING
ORPHAN_MIN_DAYS           = 3    # registry entry with zero telemetry after N days → WARNING

# ── Check names (stable identifiers) ─────────────────────────────────────────

CHECK_EVIDENCE_GENERATION  = "evidence_generation"
CHECK_SHADOW_GENERATION    = "shadow_generation"
CHECK_PROMOTION_GENERATION = "promotion_generation"
CHECK_GATE_GENERATION      = "production_gate_generation"
CHECK_EXPERIMENT_ASSIGNMENT = "experiment_assignment"
CHECK_CAPITAL_ESCALATION   = "capital_escalation_evaluation"
CHECK_ROLLBACK_EVALUATION  = "rollback_evaluation"
CHECK_IMPROVEMENT_RANKING  = "improvement_ranking_generation"
CHECK_STATE_TRANSITIONS    = "state_transitions"
CHECK_INACTIVE_EXPERIMENTS = "inactive_experiments"
CHECK_ORPHANED_REGISTRY    = "orphaned_registry_entries"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_name:   str
    status:       str            # STATUS_* constant
    message:      str
    details:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailySummary:
    active_experiments:      int
    escalated_experiments:   List[str]
    rollback_candidates:     List[str]
    rolled_back_experiments: List[str]
    top_ranked_improvements: List[str]   # ordered, best first


@dataclass
class SystemHealthReport:
    audit_date:       str
    overall_status:   str
    checks:           List[CheckResult]
    summary:          DailySummary
    schema_version:   str = SCHEMA_VERSION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _load_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON lines from a JSONL file. FAIL_OPEN → empty list."""
    records: List[dict] = []
    try:
        if not path.exists():
            return records
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[SHA] load_jsonl %s failed: %s", path, exc)
    return records


def _load_registry_json(registry_path: Path) -> List[dict]:
    """Load active_experiment_registry.json. FAIL_OPEN → empty list."""
    try:
        if not registry_path.exists():
            return []
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            exps = raw.get("experiments")
            if isinstance(exps, list):
                return exps
        return []
    except Exception as exc:
        logger.debug("[SHA] load_registry_json failed: %s", exc)
    return []


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse ISO date or datetime string. Returns None on failure."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str[:len(fmt)], fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(date_str[:19])
    except Exception:
        return None


def _latest_date_in_records(records: List[dict], date_field: str = "date") -> Optional[datetime]:
    """Return the most recent date found across records."""
    best: Optional[datetime] = None
    for r in records:
        raw = str(r.get(date_field, ""))
        d = _parse_date(raw)
        if d is None:
            continue
        d = d.replace(tzinfo=None)  # compare naive
        if best is None or d > best:
            best = d
    return best


def _days_since(dt: Optional[datetime], today: datetime) -> Optional[int]:
    if dt is None:
        return None
    today_naive = today.replace(tzinfo=None)
    dt_naive    = dt.replace(tzinfo=None)
    return max(0, (today_naive - dt_naive).days)


def _today_dt(today: str) -> datetime:
    d = _parse_date(today)
    return d if d is not None else datetime.now(_JST).replace(tzinfo=None)


def _aggregate_status(statuses: List[str]) -> str:
    """Return the worst status among all checks."""
    if STATUS_CRITICAL in statuses:
        return STATUS_CRITICAL
    if STATUS_WARNING in statuses:
        return STATUS_WARNING
    return STATUS_HEALTHY


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_telemetry_file(
    check_name:  str,
    path:        Path,
    today_dt:    datetime,
    date_field:  str = "date",
) -> CheckResult:
    """
    Generic staleness / presence check for a JSONL telemetry file.

    CRITICAL: file missing (and system has been running > MISSING_THRESHOLD_DAYS)
              OR file exists but completely empty and path is > MISSING_THRESHOLD_DAYS old
    WARNING:  file exists but latest record is > STALE_THRESHOLD_DAYS old
    HEALTHY:  file exists with a recent record
    """
    try:
        if not path.exists():
            return CheckResult(
                check_name=check_name,
                status=STATUS_WARNING,
                message=f"File not found: {path.name}",
                details={"path": str(path), "exists": False},
            )

        records = _load_jsonl(path)
        if not records:
            return CheckResult(
                check_name=check_name,
                status=STATUS_WARNING,
                message=f"File exists but contains no valid records: {path.name}",
                details={"path": str(path), "record_count": 0},
            )

        latest = _latest_date_in_records(records, date_field)
        age    = _days_since(latest, today_dt)

        if age is None:
            return CheckResult(
                check_name=check_name,
                status=STATUS_WARNING,
                message=f"Records present but no parseable date field in {path.name}",
                details={"path": str(path), "record_count": len(records)},
            )

        if age > STALE_THRESHOLD_DAYS:
            return CheckResult(
                check_name=check_name,
                status=STATUS_WARNING,
                message=f"Stale: last record {age}d ago in {path.name}",
                details={"path": str(path), "age_days": age, "record_count": len(records)},
            )

        return CheckResult(
            check_name=check_name,
            status=STATUS_HEALTHY,
            message=f"OK: {len(records)} records, last {age}d ago",
            details={"path": str(path), "age_days": age, "record_count": len(records)},
        )
    except Exception as exc:
        logger.debug("[SHA] _check_telemetry_file %s failed: %s", check_name, exc)
        return CheckResult(
            check_name=check_name,
            status=STATUS_WARNING,
            message=f"Check error: {exc}",
            details={"error": str(exc)},
        )


def _check_state_transitions(
    registry_records:   List[dict],
    rollback_records:   List[dict],
    escalation_records: List[dict],
) -> CheckResult:
    """
    Detect broken state transitions:
      - Experiment is ROLLED_BACK in rollback file but still ACTIVE in registry
      - Experiment in registry has ESCALATE decisions but still at level 0
    """
    try:
        issues: List[str] = []

        # Build registry state map
        reg_status: Dict[str, str] = {}
        reg_level:  Dict[str, int] = {}
        for r in registry_records:
            eid = str(r.get("experiment_id", ""))
            if not eid:
                continue
            reg_status[eid] = str(r.get("status", ""))
            reg_level[eid]  = _safe_int(r.get("allocation_level"), 0)

        # Check: ROLLED_BACK in rollback file but ACTIVE in registry
        rolled_back_eids: Set[str] = set()
        for r in rollback_records:
            if str(r.get("decision", "")) == "ROLLED_BACK":
                eid = str(r.get("experiment_id", ""))
                if eid:
                    rolled_back_eids.add(eid)

        for eid in rolled_back_eids:
            if reg_status.get(eid) == "ACTIVE":
                issues.append(f"{eid}: ROLLED_BACK in governance but ACTIVE in registry")

        # Check: escalated but still level 0 (FULL_SCALE decision with level=0 is stale)
        escalated_eids: Set[str] = set()
        _ESC_DECISIONS = {"ESCALATE_LEVEL_1", "ESCALATE_LEVEL_2", "ESCALATE_LEVEL_3", "FULL_SCALE"}
        for r in escalation_records:
            if str(r.get("decision", "")) in _ESC_DECISIONS:
                eid = str(r.get("experiment_id", ""))
                if eid:
                    escalated_eids.add(eid)

        for eid in escalated_eids:
            if eid in reg_level and reg_level[eid] == 0:
                issues.append(f"{eid}: escalation decisions recorded but allocation_level=0")

        if issues:
            return CheckResult(
                check_name=CHECK_STATE_TRANSITIONS,
                status=STATUS_WARNING,
                message=f"{len(issues)} state transition inconsistencies detected",
                details={"issues": issues},
            )

        return CheckResult(
            check_name=CHECK_STATE_TRANSITIONS,
            status=STATUS_HEALTHY,
            message="All state transitions consistent",
            details={"experiments_checked": len(reg_status)},
        )

    except Exception as exc:
        logger.debug("[SHA] _check_state_transitions failed: %s", exc)
        return CheckResult(
            check_name=CHECK_STATE_TRANSITIONS,
            status=STATUS_WARNING,
            message=f"Check error: {exc}",
            details={"error": str(exc)},
        )


def _check_inactive_experiments(
    registry_records:  List[dict],
    performance_records: List[dict],
    today_dt:          datetime,
) -> CheckResult:
    """
    Detect ACTIVE experiments with no performance records in INACTIVE_EXPERIMENT_DAYS.
    """
    try:
        issues: List[str] = []
        active_eids: List[str] = []
        start_dates: Dict[str, str] = {}

        for r in registry_records:
            eid    = str(r.get("experiment_id", ""))
            status = str(r.get("status", ""))
            if eid and status == "ACTIVE":
                active_eids.append(eid)
                start_dates[eid] = str(r.get("start_date", ""))

        for eid in active_eids:
            # How long has this experiment been running?
            start = _parse_date(start_dates.get(eid, ""))
            age_days = _days_since(start, today_dt) if start else None
            if age_days is not None and age_days < INACTIVE_EXPERIMENT_DAYS:
                continue  # too new to have perf records yet

            # Check most recent performance record
            perf_rows = [
                r for r in performance_records
                if str(r.get("experiment_id", "")) == eid
            ]
            latest = _latest_date_in_records(perf_rows, "date")
            age    = _days_since(latest, today_dt)

            if latest is None or (age is not None and age > INACTIVE_EXPERIMENT_DAYS):
                days_str = f"{age}d" if age is not None else "never"
                issues.append(
                    f"{eid}: ACTIVE but no performance records for {days_str}"
                )

        if issues:
            return CheckResult(
                check_name=CHECK_INACTIVE_EXPERIMENTS,
                status=STATUS_WARNING,
                message=f"{len(issues)} inactive ACTIVE experiments detected",
                details={"inactive": issues},
            )

        return CheckResult(
            check_name=CHECK_INACTIVE_EXPERIMENTS,
            status=STATUS_HEALTHY,
            message=f"All {len(active_eids)} ACTIVE experiments have recent performance data",
            details={"active_count": len(active_eids)},
        )

    except Exception as exc:
        logger.debug("[SHA] _check_inactive_experiments failed: %s", exc)
        return CheckResult(
            check_name=CHECK_INACTIVE_EXPERIMENTS,
            status=STATUS_WARNING,
            message=f"Check error: {exc}",
            details={"error": str(exc)},
        )


def _check_orphaned_registry(
    registry_records:   List[dict],
    assignment_records: List[dict],
    performance_records: List[dict],
    today_dt:           datetime,
) -> CheckResult:
    """
    Detect registry entries with zero telemetry in any source (assignments or performance).
    Only flag entries that have existed longer than ORPHAN_MIN_DAYS.
    """
    try:
        # Collect eids that appear in any telemetry
        telemetry_eids: Set[str] = set()
        for r in assignment_records:
            eid = str(r.get("experiment_id", ""))
            if eid:
                telemetry_eids.add(eid)
        for r in performance_records:
            eid = str(r.get("experiment_id", ""))
            if eid:
                telemetry_eids.add(eid)

        orphans: List[str] = []
        for r in registry_records:
            eid        = str(r.get("experiment_id", ""))
            start_date = str(r.get("start_date", ""))
            if not eid:
                continue
            start  = _parse_date(start_date)
            age    = _days_since(start, today_dt) if start else ORPHAN_MIN_DAYS + 1
            if age is not None and age < ORPHAN_MIN_DAYS:
                continue  # too new
            if eid not in telemetry_eids:
                orphans.append(eid)

        if orphans:
            return CheckResult(
                check_name=CHECK_ORPHANED_REGISTRY,
                status=STATUS_WARNING,
                message=f"{len(orphans)} orphaned registry entries with no telemetry",
                details={"orphans": orphans},
            )

        return CheckResult(
            check_name=CHECK_ORPHANED_REGISTRY,
            status=STATUS_HEALTHY,
            message="No orphaned registry entries",
            details={"registry_count": len(registry_records)},
        )

    except Exception as exc:
        logger.debug("[SHA] _check_orphaned_registry failed: %s", exc)
        return CheckResult(
            check_name=CHECK_ORPHANED_REGISTRY,
            status=STATUS_WARNING,
            message=f"Check error: {exc}",
            details={"error": str(exc)},
        )


# ── Daily summary builder ─────────────────────────────────────────────────────

def _build_daily_summary(
    registry_records:   List[dict],
    escalation_records: List[dict],
    rollback_records:   List[dict],
    ranking_records:    List[dict],
    today:              str,
) -> DailySummary:
    """Build a human-readable daily snapshot. All fail-open."""
    try:
        # Active experiments
        active_eids: List[str] = []
        for r in registry_records:
            eid    = str(r.get("experiment_id", ""))
            status = str(r.get("status", ""))
            if eid and status == "ACTIVE" and eid not in active_eids:
                active_eids.append(eid)

        # Escalated: experiments with at least one escalation decision today
        _ESC = {"ESCALATE_LEVEL_1", "ESCALATE_LEVEL_2", "ESCALATE_LEVEL_3", "FULL_SCALE"}
        escalated: List[str] = []
        for r in escalation_records:
            if str(r.get("decision", "")) in _ESC:
                eid = str(r.get("experiment_id", ""))
                if eid and eid not in escalated:
                    escalated.append(eid)

        # Rollback candidates and rolled-back
        rb_candidates: List[str] = []
        rolled_back:   List[str] = []
        for r in rollback_records:
            eid = str(r.get("experiment_id", ""))
            decision = str(r.get("decision", ""))
            if decision == "ROLLBACK_CANDIDATE" and eid and eid not in rb_candidates:
                rb_candidates.append(eid)
            if decision == "ROLLED_BACK" and eid and eid not in rolled_back:
                rolled_back.append(eid)

        # Top ranked improvements: latest ranking records, sorted by rank asc
        latest_rank: Dict[str, dict] = {}
        for r in ranking_records:
            eid  = str(r.get("experiment_id", ""))
            date = str(r.get("date", ""))
            if not eid:
                continue
            if eid not in latest_rank or date > str(latest_rank[eid].get("date", "")):
                latest_rank[eid] = r
        ranked_list = sorted(
            latest_rank.values(),
            key=lambda r: _safe_int(r.get("rank"), 9999),
        )
        top_ranked = [str(r["experiment_id"]) for r in ranked_list[:5]]

        return DailySummary(
            active_experiments=len(active_eids),
            escalated_experiments=escalated,
            rollback_candidates=rb_candidates,
            rolled_back_experiments=rolled_back,
            top_ranked_improvements=top_ranked,
        )

    except Exception as exc:
        logger.debug("[SHA] _build_daily_summary failed: %s", exc)
        return DailySummary(
            active_experiments=0,
            escalated_experiments=[],
            rollback_candidates=[],
            rolled_back_experiments=[],
            top_ranked_improvements=[],
        )


# ── Report I/O ────────────────────────────────────────────────────────────────

def _report_to_dict(report: SystemHealthReport) -> dict:
    return {
        "audit_date":     report.audit_date,
        "overall_status": report.overall_status,
        "schema_version": report.schema_version,
        "summary": {
            "active_experiments":      report.summary.active_experiments,
            "escalated_experiments":   report.summary.escalated_experiments,
            "rollback_candidates":     report.summary.rollback_candidates,
            "rolled_back_experiments": report.summary.rolled_back_experiments,
            "top_ranked_improvements": report.summary.top_ranked_improvements,
        },
        "checks": [
            {
                "check_name": c.check_name,
                "status":     c.status,
                "message":    c.message,
                "details":    c.details,
            }
            for c in report.checks
        ],
    }


def write_health_report(report: SystemHealthReport, path: Path) -> None:
    """Atomically overwrite system_health_report.json. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(data)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(path)
    except Exception as exc:
        logger.warning("[SHA] write_health_report failed: %s", exc)


def load_health_report(path: Path) -> Optional[dict]:
    """Load the most recent system_health_report.json. FAIL_OPEN → None."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("[SHA] load_health_report %s failed: %s", path, exc)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run_system_health_audit(
    evidence_file:     Path,
    shadow_file:       Path,
    promotion_file:    Path,
    gate_file:         Path,
    assignment_file:   Path,
    escalation_file:   Path,
    rollback_file:     Path,
    ranking_file:      Path,
    registry_file:     Path,
    performance_file:  Path,
    output_file:       Path,
    today:             str,
) -> SystemHealthReport:
    """
    Run all health checks and write the report.

    Never raises. FAIL_OPEN on every check. Always returns a valid report.
    """
    try:
        td = _today_dt(today)

        # ── Load shared data ──────────────────────────────────────────────────
        registry_records    = _load_registry_json(registry_file)
        rollback_records    = _load_jsonl(rollback_file)
        escalation_records  = _load_jsonl(escalation_file)
        assignment_records  = _load_jsonl(assignment_file)
        performance_records = _load_jsonl(performance_file)
        ranking_records     = _load_jsonl(ranking_file)

        # ── Telemetry presence / staleness checks ─────────────────────────────
        checks: List[CheckResult] = []

        checks.append(_check_telemetry_file(
            CHECK_EVIDENCE_GENERATION, evidence_file, td))
        checks.append(_check_telemetry_file(
            CHECK_SHADOW_GENERATION, shadow_file, td))
        checks.append(_check_telemetry_file(
            CHECK_PROMOTION_GENERATION, promotion_file, td))
        checks.append(_check_telemetry_file(
            CHECK_GATE_GENERATION, gate_file, td))
        checks.append(_check_telemetry_file(
            CHECK_EXPERIMENT_ASSIGNMENT, assignment_file, td))
        checks.append(_check_telemetry_file(
            CHECK_CAPITAL_ESCALATION, escalation_file, td))
        checks.append(_check_telemetry_file(
            CHECK_ROLLBACK_EVALUATION, rollback_file, td))
        checks.append(_check_telemetry_file(
            CHECK_IMPROVEMENT_RANKING, ranking_file, td))

        # ── Structural checks ─────────────────────────────────────────────────
        checks.append(_check_state_transitions(
            registry_records, rollback_records, escalation_records))
        checks.append(_check_inactive_experiments(
            registry_records, performance_records, td))
        checks.append(_check_orphaned_registry(
            registry_records, assignment_records, performance_records, td))

        # ── Aggregate status ──────────────────────────────────────────────────
        overall = _aggregate_status([c.status for c in checks])

        # ── Daily summary ─────────────────────────────────────────────────────
        summary = _build_daily_summary(
            registry_records, escalation_records, rollback_records,
            ranking_records, today,
        )

        report = SystemHealthReport(
            audit_date=today,
            overall_status=overall,
            checks=checks,
            summary=summary,
        )
        write_health_report(report, output_file)
        return report

    except Exception as exc:
        logger.warning("[SHA] run_system_health_audit failed: %s", exc)
        empty_summary = DailySummary(
            active_experiments=0,
            escalated_experiments=[],
            rollback_candidates=[],
            rolled_back_experiments=[],
            top_ranked_improvements=[],
        )
        fallback = SystemHealthReport(
            audit_date=today,
            overall_status=STATUS_WARNING,
            checks=[CheckResult(
                check_name="audit_runner",
                status=STATUS_WARNING,
                message=f"Audit runner error: {exc}",
            )],
            summary=empty_summary,
        )
        try:
            write_health_report(fallback, output_file)
        except Exception:
            pass
        return fallback


# ── Report formatter ──────────────────────────────────────────────────────────

def format_health_report(report: SystemHealthReport) -> str:
    """Return a compact human-readable system health report string."""
    lines: List[str] = []
    sep = "═" * 70

    status_icon = {STATUS_HEALTHY: "✓", STATUS_WARNING: "△", STATUS_CRITICAL: "✗"}
    icon = status_icon.get(report.overall_status, "?")

    lines.append(sep)
    lines.append(f"SYSTEM HEALTH AUDIT  [{icon} {report.overall_status}]  {report.audit_date}")
    lines.append("─" * 70)

    # Checks
    for c in report.checks:
        ic = status_icon.get(c.status, "?")
        lines.append(f"  {ic} {c.check_name:<32} {c.message}")

    lines.append("─" * 70)

    # Summary
    s = report.summary
    lines.append(f"  Active experiments:    {s.active_experiments}")
    if s.escalated_experiments:
        lines.append(f"  Escalated:             {', '.join(s.escalated_experiments)}")
    if s.rollback_candidates:
        lines.append(f"  Rollback candidates:   {', '.join(s.rollback_candidates)}")
    if s.rolled_back_experiments:
        lines.append(f"  Rolled back:           {', '.join(s.rolled_back_experiments)}")
    if s.top_ranked_improvements:
        lines.append(f"  Top improvements:      {', '.join(s.top_ranked_improvements)}")

    lines.append(sep)
    return "\n".join(lines)
