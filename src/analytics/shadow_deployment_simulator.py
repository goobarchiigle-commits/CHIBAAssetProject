"""
shadow_deployment_simulator.py — Shadow Deployment Simulator (Phase 6E-Lite)

read-only simulation layer: records PLANNED candidates as DEPLOYED in shadow
without modifying any live trading logic or parameters.

Input (read-only):
  PROMOTION_DEPLOYMENT_PLAN_FILE — promotion_deployment_plan.jsonl

Output:
  SHADOW_DEPLOYMENT_FILE — shadow_deployment.jsonl
    append-only JSONL; 1 record per newly-deployed candidate

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  idempotent       — a candidate already DEPLOYED is never re-added
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

SHADOW_STATUS_DEPLOYED    = "DEPLOYED"
_PLAN_STATUS_PLANNED      = "PLANNED"
SHADOW_DURATION_DAYS: int = 30

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShadowDeploymentRecord:
    date:                 str
    candidate:            str
    planned_change:       str
    shadow_status:        str = SHADOW_STATUS_DEPLOYED
    shadow_duration_days: int = SHADOW_DURATION_DAYS
    schema_version:       str = SCHEMA_VERSION


@dataclass
class ShadowDeploymentSummary:
    active_shadow_deployments: int
    deployment_count:          int          # total DEPLOYED records in file
    candidate_counts:          Dict[str, int]  # candidate → count of DEPLOYED records


@dataclass
class ShadowDeploymentRunResult:
    date:            str
    new_records:     List[ShadowDeploymentRecord]   # written this run
    deployed_records: List[ShadowDeploymentRecord]  # all DEPLOYED, in _CANDIDATE_ORDER
    summary:         ShadowDeploymentSummary


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
        logger.debug("[SDS] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[SDS] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _load_deployed_set(shadow_file: Path) -> Set[str]:
    """Return set of candidate names already DEPLOYED."""
    deployed: Set[str] = set()
    for r in _load_all_jsonl(shadow_file):
        if str(r.get("shadow_status", "")) == SHADOW_STATUS_DEPLOYED:
            ct = str(r.get("candidate", ""))
            if ct:
                deployed.add(ct)
    return deployed


def _load_deployed_records(shadow_file: Path) -> List[ShadowDeploymentRecord]:
    """Return DEPLOYED records, latest per candidate, in _CANDIDATE_ORDER."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(shadow_file):
        if str(r.get("shadow_status", "")) != SHADOW_STATUS_DEPLOYED:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r

    result: List[ShadowDeploymentRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = latest.get(ct)
        if r is None:
            continue
        result.append(ShadowDeploymentRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            planned_change=str(r.get("planned_change", "")),
            shadow_status=str(r.get("shadow_status", SHADOW_STATUS_DEPLOYED)),
            shadow_duration_days=_safe_int(r.get("shadow_duration_days"), SHADOW_DURATION_DAYS),
        ))
    return result


def _load_planned_from_deployment_plan(plan_file: Path) -> List[dict]:
    """Return PLANNED records from deployment plan, latest per candidate."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(plan_file):
        if str(r.get("deployment_status", "")) != _PLAN_STATUS_PLANNED:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r
    return [latest[ct] for ct in _CANDIDATE_ORDER if ct in latest]


def _compute_summary(deployed_records: List[ShadowDeploymentRecord]) -> ShadowDeploymentSummary:
    """Compute aggregated KPIs from all DEPLOYED records."""
    candidate_counts: Dict[str, int] = {}
    for r in deployed_records:
        candidate_counts[r.candidate] = candidate_counts.get(r.candidate, 0) + 1

    return ShadowDeploymentSummary(
        active_shadow_deployments=len(deployed_records),
        deployment_count=len(deployed_records),
        candidate_counts=candidate_counts,
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_shadow_record(record: ShadowDeploymentRecord, path: Path) -> None:
    """Serialize one ShadowDeploymentRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":                 record.date,
            "candidate":            record.candidate,
            "planned_change":       record.planned_change,
            "shadow_status":        record.shadow_status,
            "shadow_duration_days": record.shadow_duration_days,
            "schema_version":       record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[SDS] append_shadow_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_shadow_deployment_simulator(
    plan_file:   Path,
    output_file: Path,
    today:       str,
) -> ShadowDeploymentRunResult:
    """
    Create DEPLOYED shadow records for PLANNED deployment plan candidates.

    For each PLANNED candidate in _CANDIDATE_ORDER:
      - Skip if already DEPLOYED in output JSONL (idempotent)
      - Append ShadowDeploymentRecord with shadow_status=DEPLOYED

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        planned_records = _load_planned_from_deployment_plan(plan_file)
        deployed_set    = _load_deployed_set(output_file)

        new_records: List[ShadowDeploymentRecord] = []

        for r in planned_records:
            ct = str(r.get("candidate", ""))
            if ct in deployed_set:
                continue

            record = ShadowDeploymentRecord(
                date=today,
                candidate=ct,
                planned_change=str(r.get("planned_change", "")),
            )
            append_shadow_record(record, output_file)
            new_records.append(record)
            deployed_set.add(ct)

        deployed_records = _load_deployed_records(output_file)
        summary          = _compute_summary(deployed_records)

        return ShadowDeploymentRunResult(
            date=today,
            new_records=new_records,
            deployed_records=deployed_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[SDS] run_shadow_deployment_simulator failed: %s", exc)
        return ShadowDeploymentRunResult(
            date=today,
            new_records=[],
            deployed_records=[],
            summary=ShadowDeploymentSummary(
                active_shadow_deployments=0,
                deployment_count=0,
                candidate_counts={},
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_shadow_deployment_report(
    deployed_records: List[ShadowDeploymentRecord],
    summary:          ShadowDeploymentSummary,
) -> str:
    """Return SHADOW DEPLOYMENTS report section string."""
    if not deployed_records:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("SHADOW DEPLOYMENTS")
    lines.append("─" * 60)
    lines.append(f"Active: {summary.active_shadow_deployments}")
    lines.append(f"Total Deployed: {summary.deployment_count}")
    lines.append("")
    lines.append("ACTIVE:")
    for r in deployed_records:
        lines.append(f"  {r.candidate}")
        lines.append(f"    {r.planned_change}")
        lines.append(f"    duration={r.shadow_duration_days}d  ({r.date})")
    lines.append(sep)
    return "\n".join(lines)
