"""
promotion_deployment_planner.py — Promotion Deployment Planner (Phase 6D-Lite)

read-only planning layer: documents what change is planned per ACTIVE candidate.
No parameters are actually modified.

Input (read-only):
  PROMOTION_CANDIDATE_REGISTRY_FILE — promotion_candidate_registry.jsonl

Output:
  PROMOTION_DEPLOYMENT_PLAN_FILE — promotion_deployment_plan.jsonl
    append-only JSONL; 1 record per newly-planned candidate

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  idempotent       — a candidate already PLANNED is never re-added
  O(n)             — linear in JSONL lines
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

DEPLOYMENT_STATUS_PLANNED = "PLANNED"
_REGISTRY_STATUS_ACTIVE   = "ACTIVE"

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]

# ── Fixed candidate → planned_change mapping ──────────────────────────────────
_PLANNED_CHANGE_MAP: Dict[str, str] = {
    "EXIT_EXTENSION":        "increase_exit_extension_shadow",
    "CAPITAL_CONCENTRATION": "priority_weighted_capital_shadow",
    "PRIORITY":              "priority_threshold_shadow",
    "SIGNAL_LEADER":         "signal_weight_shadow",
    "CONTINUATION_SIGNAL":   "continuation_boost_shadow",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DeploymentPlanRecord:
    date:              str
    candidate:         str
    planned_change:    str
    readiness_score:   float
    deployment_status: str = DEPLOYMENT_STATUS_PLANNED
    schema_version:    str = SCHEMA_VERSION


@dataclass
class DeploymentPlanSummary:
    planned_count:              int
    highest_priority_candidate: str    # highest readiness_score; "" if none
    avg_readiness_score:        float


@dataclass
class DeploymentPlanRunResult:
    date:            str
    new_records:     List[DeploymentPlanRecord]   # written this run
    planned_records: List[DeploymentPlanRecord]   # all PLANNED, in _CANDIDATE_ORDER
    summary:         DeploymentPlanSummary


# ── I/O helpers ───────────────────────────────────────────────────────────────

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


def _load_all_jsonl(path: Path) -> List[dict]:
    """Load all valid JSON lines. FAIL_OPEN → empty list."""
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
        logger.debug("[PDP] load_all_jsonl %s failed: %s", path, exc)
    return records


def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic append-only JSONL write. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
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
        logger.warning("[PDP] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _load_planned_set(plan_file: Path) -> Set[str]:
    """Return set of candidate names already in PLANNED status."""
    planned: Set[str] = set()
    for r in _load_all_jsonl(plan_file):
        if str(r.get("deployment_status", "")) == DEPLOYMENT_STATUS_PLANNED:
            ct = str(r.get("candidate", ""))
            if ct:
                planned.add(ct)
    return planned


def _load_planned_records(plan_file: Path) -> List[DeploymentPlanRecord]:
    """Return PLANNED records, latest per candidate, in _CANDIDATE_ORDER."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(plan_file):
        if str(r.get("deployment_status", "")) != DEPLOYMENT_STATUS_PLANNED:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r

    result: List[DeploymentPlanRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = latest.get(ct)
        if r is None:
            continue
        result.append(DeploymentPlanRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            planned_change=str(r.get("planned_change", "")),
            readiness_score=_safe_float(r.get("readiness_score"), 0.0),
        ))
    return result


def _load_active_from_registry(registry_file: Path) -> List[dict]:
    """Return ACTIVE registry records, latest per candidate."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(registry_file):
        if str(r.get("registry_status", "")) != _REGISTRY_STATUS_ACTIVE:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r
    # preserve _CANDIDATE_ORDER
    return [latest[ct] for ct in _CANDIDATE_ORDER if ct in latest]


def _compute_summary(planned_records: List[DeploymentPlanRecord]) -> DeploymentPlanSummary:
    """Compute aggregated KPIs from all PLANNED records."""
    if not planned_records:
        return DeploymentPlanSummary(
            planned_count=0,
            highest_priority_candidate="",
            avg_readiness_score=0.0,
        )
    scores = [r.readiness_score for r in planned_records]
    avg    = sum(scores) / len(scores)
    top    = max(planned_records, key=lambda r: r.readiness_score).candidate
    return DeploymentPlanSummary(
        planned_count=len(planned_records),
        highest_priority_candidate=top,
        avg_readiness_score=round(avg, 1),
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_plan_record(record: DeploymentPlanRecord, path: Path) -> None:
    """Serialize one DeploymentPlanRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":              record.date,
            "candidate":         record.candidate,
            "planned_change":    record.planned_change,
            "readiness_score":   record.readiness_score,
            "deployment_status": record.deployment_status,
            "schema_version":    record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PDP] append_plan_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_deployment_planner(
    registry_file: Path,
    output_file:   Path,
    today:         str,
) -> DeploymentPlanRunResult:
    """
    Create PLANNED deployment records for ACTIVE registry candidates.

    For each ACTIVE candidate in _CANDIDATE_ORDER:
      - Skip if already PLANNED in output JSONL (idempotent)
      - Append DeploymentPlanRecord with planned_change from _PLANNED_CHANGE_MAP

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        active_records = _load_active_from_registry(registry_file)
        planned_set    = _load_planned_set(output_file)

        new_records: List[DeploymentPlanRecord] = []

        for r in active_records:
            ct = str(r.get("candidate", ""))
            if ct in planned_set:
                continue
            planned_change = _PLANNED_CHANGE_MAP.get(ct)
            if planned_change is None:
                continue

            record = DeploymentPlanRecord(
                date=today,
                candidate=ct,
                planned_change=planned_change,
                readiness_score=_safe_float(r.get("readiness_score"), 0.0),
            )
            append_plan_record(record, output_file)
            new_records.append(record)
            planned_set.add(ct)

        planned_records = _load_planned_records(output_file)
        summary         = _compute_summary(planned_records)

        return DeploymentPlanRunResult(
            date=today,
            new_records=new_records,
            planned_records=planned_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PDP] run_promotion_deployment_planner failed: %s", exc)
        return DeploymentPlanRunResult(
            date=today,
            new_records=[],
            planned_records=[],
            summary=DeploymentPlanSummary(
                planned_count=0,
                highest_priority_candidate="",
                avg_readiness_score=0.0,
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_deployment_report(
    planned_records: List[DeploymentPlanRecord],
    summary:         DeploymentPlanSummary,
) -> str:
    """Return PROMOTION DEPLOYMENT PLAN report section string."""
    if not planned_records:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION DEPLOYMENT PLAN")
    lines.append("─" * 60)
    lines.append(f"Planned: {summary.planned_count}")
    if summary.highest_priority_candidate:
        lines.append(f"Highest Priority: {summary.highest_priority_candidate}")
    lines.append(f"Average Readiness: {summary.avg_readiness_score:.1f}")
    lines.append("")
    lines.append("PLANNED:")
    for r in planned_records:
        lines.append(f"  {r.candidate}")
        lines.append(f"    {r.planned_change}")
        lines.append(f"    score={r.readiness_score:.1f}  ({r.date})")
    lines.append(sep)
    return "\n".join(lines)
