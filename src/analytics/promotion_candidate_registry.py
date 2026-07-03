"""
promotion_candidate_registry.py — Promotion Candidate Registry (Phase 6C-Lite)

Maintains a ledger of PROMOTION_READY candidates for weekly governance review.

Input (read-only):
  PROMOTION_READINESS_FILE — promotion_readiness.jsonl

Output:
  PROMOTION_CANDIDATE_REGISTRY_FILE — promotion_candidate_registry.jsonl
    append-only JSONL; 1 record per newly-activated candidate

Registration rules:
  - candidate.status == "PROMOTION_READY" in latest PR snapshot
  - not already ACTIVE in registry (idempotent, once per candidate)

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — never blocks live execution
  idempotent       — a candidate already ACTIVE is never re-registered
  O(n)             — linear in JSONL lines; no simulation, no backtest, no ML
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

_STATUS_PROMOTION_READY = "PROMOTION_READY"
REGISTRY_STATUS_ACTIVE  = "ACTIVE"
DEFAULT_LOOKBACK_DAYS   = 7

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RegistryRecord:
    date:             str
    candidate:        str
    readiness_score:  float   # from PR record
    success_rate:     float   # percentage 0-100
    n_completed:      int
    registry_status:  str = REGISTRY_STATUS_ACTIVE
    schema_version:   str = SCHEMA_VERSION


@dataclass
class RegistrySummary:
    active_candidates:   int
    new_candidates_7d:   int
    avg_readiness_score: float
    top_candidate:       str    # highest readiness_score; "" if none


@dataclass
class RegistryRunResult:
    date:            str
    new_records:     List[RegistryRecord]   # written this run
    active_registry: List[RegistryRecord]   # all ACTIVE, ordered by _CANDIDATE_ORDER
    summary:         RegistrySummary


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _safe_int(v: Any, fallback: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _safe_float(v: Any, fallback: float = 0.0) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
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
        logger.debug("[PCR] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PCR] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _load_active_set(registry_file: Path) -> Set[str]:
    """Return set of candidate names with at least one ACTIVE entry in the registry."""
    active: Set[str] = set()
    for r in _load_all_jsonl(registry_file):
        if str(r.get("registry_status", "")) == REGISTRY_STATUS_ACTIVE:
            ct = str(r.get("candidate", ""))
            if ct:
                active.add(ct)
    return active


def _load_active_registry(registry_file: Path) -> List[RegistryRecord]:
    """
    Return one RegistryRecord per ACTIVE candidate (latest date wins),
    ordered by _CANDIDATE_ORDER.
    """
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(registry_file):
        if str(r.get("registry_status", "")) != REGISTRY_STATUS_ACTIVE:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r

    result: List[RegistryRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = latest.get(ct)
        if r is None:
            continue
        result.append(RegistryRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            readiness_score=_safe_float(r.get("readiness_score"), 0.0),
            success_rate=_safe_float(r.get("success_rate"), 0.0),
            n_completed=_safe_int(r.get("n_completed"), 0),
            registry_status=REGISTRY_STATUS_ACTIVE,
        ))
    return result


def _cutoff_date(today: str, lookback_days: int) -> str:
    """Return ISO date string `lookback_days` before today. Returns "" on error."""
    try:
        dt = datetime.strptime(today, "%Y-%m-%d") - timedelta(days=lookback_days)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _compute_summary(
    active_registry: List[RegistryRecord],
    today: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> RegistrySummary:
    """Compute aggregated KPIs from the active registry."""
    if not active_registry:
        return RegistrySummary(
            active_candidates=0,
            new_candidates_7d=0,
            avg_readiness_score=0.0,
            top_candidate="",
        )

    cutoff = _cutoff_date(today, lookback_days)
    active_candidates = len(active_registry)
    new_7d = sum(1 for r in active_registry if cutoff and r.date >= cutoff)
    scores = [r.readiness_score for r in active_registry]
    avg    = sum(scores) / len(scores) if scores else 0.0
    top    = max(active_registry, key=lambda r: r.readiness_score).candidate

    return RegistrySummary(
        active_candidates=active_candidates,
        new_candidates_7d=new_7d,
        avg_readiness_score=round(avg, 1),
        top_candidate=top,
    )


def _latest_pr_per_candidate(pr_records: List[dict]) -> Dict[str, dict]:
    """Build candidate → latest PR record mapping (by date)."""
    latest: Dict[str, dict] = {}
    for r in pr_records:
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct and date:
            if ct not in latest or date > latest[ct].get("date", ""):
                latest[ct] = r
    return latest


# ── Serialization ─────────────────────────────────────────────────────────────

def append_registry_record(record: RegistryRecord, path: Path) -> None:
    """Serialize one RegistryRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":            record.date,
            "candidate":       record.candidate,
            "readiness_score": record.readiness_score,
            "success_rate":    record.success_rate,
            "n_completed":     record.n_completed,
            "registry_status": record.registry_status,
            "schema_version":  record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PCR] append_registry_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_candidate_registry(
    pr_file:       Path,
    output_file:   Path,
    today:         str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> RegistryRunResult:
    """
    Register new PROMOTION_READY candidates into the candidate registry.

    For each candidate in _CANDIDATE_ORDER:
      - Skip if latest PR record is not PROMOTION_READY
      - Skip if candidate already ACTIVE in registry (idempotent)
      - Append ACTIVE registry record

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        pr_records = _load_all_jsonl(pr_file)
        latest_pr  = _latest_pr_per_candidate(pr_records)
        active_set = _load_active_set(output_file)

        new_records: List[RegistryRecord] = []

        for ct in _CANDIDATE_ORDER:
            pr = latest_pr.get(ct)
            if pr is None:
                continue
            if str(pr.get("status", "")) != _STATUS_PROMOTION_READY:
                continue
            if ct in active_set:
                continue

            record = RegistryRecord(
                date=today,
                candidate=ct,
                readiness_score=_safe_float(pr.get("readiness_score"), 0.0),
                success_rate=_safe_float(pr.get("success_rate"), 0.0),
                n_completed=_safe_int(pr.get("n_completed"), 0),
            )
            append_registry_record(record, output_file)
            new_records.append(record)
            active_set.add(ct)

        active_registry = _load_active_registry(output_file)
        summary         = _compute_summary(active_registry, today, lookback_days)

        return RegistryRunResult(
            date=today,
            new_records=new_records,
            active_registry=active_registry,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PCR] run_promotion_candidate_registry failed: %s", exc)
        return RegistryRunResult(
            date=today,
            new_records=[],
            active_registry=[],
            summary=RegistrySummary(
                active_candidates=0,
                new_candidates_7d=0,
                avg_readiness_score=0.0,
                top_candidate="",
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_registry_report(
    active_registry: List[RegistryRecord],
    summary:         RegistrySummary,
    today:           str,
    lookback_days:   int = DEFAULT_LOOKBACK_DAYS,
) -> str:
    """Return PROMOTION CANDIDATE REGISTRY report section string."""
    if not active_registry:
        return ""

    cutoff = _cutoff_date(today, lookback_days)
    new_this_week = [r for r in active_registry if cutoff and r.date >= cutoff]

    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION CANDIDATE REGISTRY")
    lines.append("─" * 60)
    lines.append(
        f"Active: {summary.active_candidates}"
        f"  │  New (7d): {summary.new_candidates_7d}"
    )
    lines.append("")
    lines.append("ACTIVE:")
    for r in active_registry:
        lines.append(
            f"  ★ {r.candidate.ljust(24)} score={r.readiness_score:.0f}"
            f"  ({r.date})"
        )

    if new_this_week:
        lines.append("")
        lines.append("NEW THIS WEEK:")
        for r in new_this_week:
            lines.append(f"  ★ {r.candidate}")

    lines.append("")
    lines.append(f"Average Readiness: {summary.avg_readiness_score:.1f}")
    if summary.top_candidate:
        lines.append(f"Top Candidate: {summary.top_candidate}")
    lines.append(sep)
    return "\n".join(lines)
