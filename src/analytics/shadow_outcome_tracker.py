"""
shadow_outcome_tracker.py — Shadow Outcome Tracker (Phase 6A-Lite)

Minimal closed-loop verification: did shadow-recommended improvements
actually materialize?

Inputs (read-only):
  SHADOW_RECOMMENDATION_FILE — shadow_recommendation.jsonl
  EVIDENCE_PROMOTION_FILE    — evidence_promotion.jsonl

Output:
  SHADOW_OUTCOME_FILE — shadow_outcome.jsonl
    append-only JSONL; 1 record per newly-matured evaluation

Design:
  observation_only — no signals, no orders, no execution mutation
  fail-open        — every stage wrapped; never blocks live execution
  idempotent       — (shadow_start_date, candidate) pairs are never re-evaluated
  O(n)             — linear in JSONL lines; no simulation, no backtest, no ML
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

SHADOW_READY = "SHADOW_READY"

# ── KPI field per candidate type (from supporting_kpis in EP record) ──────────
_KPI_FIELD: Dict[str, str] = {
    "EXIT_EXTENSION":        "suppression_return_delta",
    "CAPITAL_CONCENTRATION": "mean_concentration_alpha",
    "PRIORITY":              "priority_monotonicity",
    "SIGNAL_LEADER":         "top_signal_ir",
    "CONTINUATION_SIGNAL":   "top_signal_ir",
}

# ── Candidate ordering ────────────────────────────────────────────────────────
_CANDIDATE_ORDER = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShadowOutcomeRecord:
    date:                str    # evaluation date (today)
    candidate:           str    # candidate_type
    shadow_start_date:   str    # date of the originating SR record
    shadow_duration_days: int   # from SR record
    before_kpi:          float  # KPI value at shadow_start_date
    after_kpi:           float  # KPI value today (latest EP)
    delta:               float  # improvement magnitude
    success:             bool   # True if KPI improved
    schema_version:      str = SCHEMA_VERSION


@dataclass
class ShadowOutcomeSummary:
    n_completed:    int
    success_rate:   float   # 0.0-1.0
    avg_delta:      float
    best_candidate: str     # highest per-candidate success rate; "" if none
    worst_candidate: str    # lowest per-candidate success rate; "" if none


@dataclass
class ShadowOutcomeRunResult:
    date:        str
    new_records: List[ShadowOutcomeRecord]
    summary:     ShadowOutcomeSummary


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
        logger.debug("[SOT] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[SOT] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _days_elapsed(start_date: str, today: str) -> int:
    """Calendar days from start_date to today. Returns 0 on parse error."""
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d")
        d1 = datetime.strptime(today, "%Y-%m-%d")
        return (d1 - d0).days
    except Exception:
        return 0


def _extract_kpi(ep_record: dict, candidate_type: str) -> Optional[float]:
    """Extract the relevant KPI value from an EP record for a candidate type."""
    field = _KPI_FIELD.get(candidate_type)
    if field is None:
        return None
    for c in ep_record.get("candidates", []):
        if c.get("candidate_type") == candidate_type:
            v = c.get("supporting_kpis", {}).get(field)
            if v is None:
                return None
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None
    return None


def _compute_delta(candidate_type: str, before: float, after: float) -> float:
    """Compute improvement magnitude."""
    if candidate_type == "SIGNAL_LEADER":
        return abs(after) - abs(before)
    return after - before


def _compute_success(candidate_type: str, before: float, after: float) -> bool:
    """True if the KPI improved relative to before."""
    if candidate_type == "SIGNAL_LEADER":
        return abs(after) > abs(before)
    return after > before


def _build_ep_index(ep_records: List[dict]) -> Dict[str, dict]:
    """Build date → EP record index. Last record per date wins."""
    index: Dict[str, dict] = {}
    for r in ep_records:
        date = str(r.get("date", ""))
        if date:
            index[date] = r
    return index


def _load_evaluated_keys(outcome_file: Path) -> Set[Tuple[str, str]]:
    """Return set of (shadow_start_date, candidate) already written to output JSONL."""
    keys: Set[Tuple[str, str]] = set()
    for r in _load_all_jsonl(outcome_file):
        sd = str(r.get("shadow_start_date", ""))
        ct = str(r.get("candidate", ""))
        if sd and ct:
            keys.add((sd, ct))
    return keys


def _compute_summary(all_outcome_records: List[dict]) -> ShadowOutcomeSummary:
    """Aggregate all outcome records into summary KPIs."""
    if not all_outcome_records:
        return ShadowOutcomeSummary(
            n_completed=0,
            success_rate=0.0,
            avg_delta=0.0,
            best_candidate="",
            worst_candidate="",
        )

    n_completed = len(all_outcome_records)
    successes   = sum(1 for r in all_outcome_records if r.get("success", False))
    success_rate = successes / n_completed

    deltas: List[float] = []
    candidate_outcomes: Dict[str, List[bool]] = defaultdict(list)

    for r in all_outcome_records:
        try:
            deltas.append(float(r.get("delta", 0.0)))
        except (TypeError, ValueError):
            pass
        ct = str(r.get("candidate", ""))
        if ct:
            candidate_outcomes[ct].append(bool(r.get("success", False)))

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    best_candidate  = ""
    worst_candidate = ""
    if candidate_outcomes:
        rates = {ct: sum(v) / len(v) for ct, v in candidate_outcomes.items()}
        best_candidate  = max(rates, key=lambda k: rates[k])
        worst_candidate = min(rates, key=lambda k: rates[k])

    return ShadowOutcomeSummary(
        n_completed=n_completed,
        success_rate=round(success_rate, 4),
        avg_delta=round(avg_delta, 6),
        best_candidate=best_candidate,
        worst_candidate=worst_candidate,
    )


# ── Append helper ─────────────────────────────────────────────────────────────

def append_outcome_record(record: ShadowOutcomeRecord, path: Path) -> None:
    """Serialize one ShadowOutcomeRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":                 record.date,
            "candidate":            record.candidate,
            "shadow_start_date":    record.shadow_start_date,
            "shadow_duration_days": record.shadow_duration_days,
            "before_kpi":           record.before_kpi,
            "after_kpi":            record.after_kpi,
            "delta":                record.delta,
            "success":              record.success,
            "schema_version":       record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[SOT] append_outcome_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_shadow_outcome_tracker(
    sr_file:     Path,
    ep_file:     Path,
    output_file: Path,
    today:       str,
) -> ShadowOutcomeRunResult:
    """
    Evaluate matured SHADOW_READY recommendations against current KPIs.

    For each SR record with shadow_status==SHADOW_READY:
      - Skip if shadow_duration_days have not yet elapsed
      - Skip if (shadow_start_date, candidate) already in output JSONL
      - Extract before_kpi from EP record at shadow_start_date
      - Extract after_kpi from latest EP record
      - Append outcome record

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        sr_records = _load_all_jsonl(sr_file)
        ep_records = _load_all_jsonl(ep_file)
        ep_index   = _build_ep_index(ep_records)
        latest_ep  = ep_records[-1] if ep_records else {}
        already    = _load_evaluated_keys(output_file)

        new_records: List[ShadowOutcomeRecord] = []

        for sr_rec in sr_records:
            sr_date = str(sr_rec.get("date", ""))
            if not sr_date:
                continue

            for rec in sr_rec.get("recommendations", []):
                if str(rec.get("shadow_status", "")) != SHADOW_READY:
                    continue

                ct       = str(rec.get("candidate_type", ""))
                duration = _safe_int(rec.get("shadow_duration_days"), 0)

                if _days_elapsed(sr_date, today) < duration:
                    continue

                if (sr_date, ct) in already:
                    continue

                ep_before = ep_index.get(sr_date, {})
                before_kpi = _extract_kpi(ep_before, ct)
                if before_kpi is None:
                    continue

                after_kpi = _extract_kpi(latest_ep, ct)
                if after_kpi is None:
                    continue

                delta   = _compute_delta(ct, before_kpi, after_kpi)
                success = _compute_success(ct, before_kpi, after_kpi)

                outcome = ShadowOutcomeRecord(
                    date=today,
                    candidate=ct,
                    shadow_start_date=sr_date,
                    shadow_duration_days=duration,
                    before_kpi=round(before_kpi, 6),
                    after_kpi=round(after_kpi, 6),
                    delta=round(delta, 6),
                    success=success,
                )
                append_outcome_record(outcome, output_file)
                new_records.append(outcome)
                already.add((sr_date, ct))

        all_records = _load_all_jsonl(output_file)
        summary     = _compute_summary(all_records)

        return ShadowOutcomeRunResult(
            date=today,
            new_records=new_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[SOT] run_shadow_outcome_tracker failed: %s", exc)
        return ShadowOutcomeRunResult(
            date=today,
            new_records=[],
            summary=ShadowOutcomeSummary(
                n_completed=0,
                success_rate=0.0,
                avg_delta=0.0,
                best_candidate="",
                worst_candidate="",
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_outcome_report(summary: ShadowOutcomeSummary) -> str:
    """Return SHADOW OUTCOME VALIDATION report section string."""
    if summary.n_completed == 0:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("SHADOW OUTCOME VALIDATION")
    lines.append("─" * 60)
    lines.append(f"Completed: {summary.n_completed}")
    lines.append(f"Success Rate: {summary.success_rate:.1%}")
    lines.append(f"Average Delta: {summary.avg_delta:+.1%}")
    if summary.best_candidate:
        lines.append(f"\nBest: {summary.best_candidate}")
    if summary.worst_candidate and summary.worst_candidate != summary.best_candidate:
        lines.append(f"Worst: {summary.worst_candidate}")
    lines.append(sep)
    return "\n".join(lines)
