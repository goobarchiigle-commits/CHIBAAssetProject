"""
promotion_impact_tracker.py — Promotion Impact Tracker (Phase 6F-Lite)

Measures whether shadow-deployed candidates actually improved system outcomes.

Inputs (all read-only):
  SHADOW_DEPLOYMENT_FILE            — deployed candidates + deployment date
  HOLD_DURATION_CALIBRATION_FILE    — EXIT_EXTENSION metric (suppression_return_delta)
  CAPITAL_CONCENTRATION_SHADOW_FILE — CAPITAL_CONCENTRATION metric (mean_concentration_alpha)
  PRIORITY_CALIBRATION_FILE         — PRIORITY metric (priority_monotonicity)
  SIGNAL_ATTRIBUTION_FILE           — SIGNAL_LEADER metric (top_signal_ir)
  FORWARD_CONTINUATION_OUTCOME_FILE — CONTINUATION_SIGNAL metric (top_signal_ir)

Output:
  PROMOTION_IMPACT_TRACKER_FILE — promotion_impact_tracker.jsonl
    append-only JSONL; 1 record per (date, candidate) per run

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  idempotent       — (date, candidate) pair never re-added
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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

IMPACT_POSITIVE = "POSITIVE"
IMPACT_NEGATIVE = "NEGATIVE"
IMPACT_NEUTRAL  = "NEUTRAL"

_SHADOW_STATUS_DEPLOYED = "DEPLOYED"

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]

# candidate → (metric_field_name, source_file_key)
# source_file_key maps to the runtime path passed into run_
_METRIC_FIELD: Dict[str, str] = {
    "EXIT_EXTENSION":        "suppression_return_delta",
    "CAPITAL_CONCENTRATION": "mean_concentration_alpha",
    "PRIORITY":              "priority_monotonicity",
    "SIGNAL_LEADER":         "top_signal_ir",
    "CONTINUATION_SIGNAL":   "top_signal_ir",
}

_METRIC_SOURCE: Dict[str, str] = {
    "EXIT_EXTENSION":        "hold_duration",
    "CAPITAL_CONCENTRATION": "capital_concentration",
    "PRIORITY":              "priority_calibration",
    "SIGNAL_LEADER":         "signal_attribution",
    "CONTINUATION_SIGNAL":   "forward_continuation",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ImpactRecord:
    date:              str
    candidate:         str
    metric_field:      str
    deployment_date:   str
    before_metric:     float
    after_metric:      float
    impact_delta:      float
    impact_status:     str
    schema_version:    str = SCHEMA_VERSION


@dataclass
class ImpactSummary:
    n_deployed:        int
    n_positive:        int
    n_negative:        int
    success_rate:      float   # n_positive / n_deployed (0-100)
    best_candidate:    str     # highest impact_delta; "" if none
    best_impact_delta: float


@dataclass
class ImpactRunResult:
    date:            str
    new_records:     List[ImpactRecord]
    impact_records:  List[ImpactRecord]   # latest per candidate, in _CANDIDATE_ORDER
    summary:         ImpactSummary


# ── I/O helpers ───────────────────────────────────────────────────────────────

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
        logger.debug("[PIT] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PIT] append_jsonl failed: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def _load_deployed_candidates(shadow_file: Path) -> Dict[str, str]:
    """Return {candidate: deployment_date} for DEPLOYED records (latest date per candidate)."""
    latest: Dict[str, str] = {}
    for r in _load_all_jsonl(shadow_file):
        if str(r.get("shadow_status", "")) != _SHADOW_STATUS_DEPLOYED:
            continue
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct]:
            latest[ct] = date
    return {ct: latest[ct] for ct in _CANDIDATE_ORDER if ct in latest}


def _extract_metric_at_or_before(records: List[dict], field: str, cutoff_date: str) -> Optional[float]:
    """Return metric value from the latest record with date <= cutoff_date."""
    best_date  = ""
    best_value: Optional[float] = None
    for r in records:
        date = str(r.get("date", ""))
        if not date or date > cutoff_date:
            continue
        if field not in r:
            continue
        v = _safe_float(r.get(field), 0.0)
        if date >= best_date:
            best_date  = date
            best_value = v
    return best_value


def _extract_latest_metric(records: List[dict], field: str) -> Optional[float]:
    """Return metric value from the most recent record."""
    best_date  = ""
    best_value: Optional[float] = None
    for r in records:
        date = str(r.get("date", ""))
        if not date or field not in r:
            continue
        if date >= best_date:
            best_date  = date
            best_value = _safe_float(r.get(field), 0.0)
    return best_value


def _classify_impact(delta: float) -> str:
    if delta > 0:
        return IMPACT_POSITIVE
    if delta < 0:
        return IMPACT_NEGATIVE
    return IMPACT_NEUTRAL


def _load_evaluated_keys(output_file: Path) -> Set[Tuple[str, str]]:
    """Return set of (date, candidate) already recorded."""
    keys: Set[Tuple[str, str]] = set()
    for r in _load_all_jsonl(output_file):
        date = str(r.get("date", ""))
        ct   = str(r.get("candidate", ""))
        if date and ct:
            keys.add((date, ct))
    return keys


def _load_latest_impact_records(output_file: Path) -> List[ImpactRecord]:
    """Return latest ImpactRecord per candidate, in _CANDIDATE_ORDER."""
    latest: Dict[str, dict] = {}
    for r in _load_all_jsonl(output_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if not ct or not date:
            continue
        if ct not in latest or date > latest[ct].get("date", ""):
            latest[ct] = r

    result: List[ImpactRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = latest.get(ct)
        if r is None:
            continue
        result.append(ImpactRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            metric_field=str(r.get("metric_field", "")),
            deployment_date=str(r.get("deployment_date", "")),
            before_metric=_safe_float(r.get("before_metric"), 0.0),
            after_metric=_safe_float(r.get("after_metric"), 0.0),
            impact_delta=_safe_float(r.get("impact_delta"), 0.0),
            impact_status=str(r.get("impact_status", IMPACT_NEUTRAL)),
        ))
    return result


def _compute_summary(impact_records: List[ImpactRecord]) -> ImpactSummary:
    if not impact_records:
        return ImpactSummary(
            n_deployed=0,
            n_positive=0,
            n_negative=0,
            success_rate=0.0,
            best_candidate="",
            best_impact_delta=0.0,
        )
    n_pos = sum(1 for r in impact_records if r.impact_status == IMPACT_POSITIVE)
    n_neg = sum(1 for r in impact_records if r.impact_status == IMPACT_NEGATIVE)
    sr    = round(100.0 * n_pos / len(impact_records), 1)
    best  = max(impact_records, key=lambda r: r.impact_delta)
    return ImpactSummary(
        n_deployed=len(impact_records),
        n_positive=n_pos,
        n_negative=n_neg,
        success_rate=sr,
        best_candidate=best.candidate,
        best_impact_delta=round(best.impact_delta, 6),
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_impact_record(record: ImpactRecord, path: Path) -> None:
    """Serialize one ImpactRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":            record.date,
            "candidate":       record.candidate,
            "metric_field":    record.metric_field,
            "deployment_date": record.deployment_date,
            "before_metric":   record.before_metric,
            "after_metric":    record.after_metric,
            "impact_delta":    record.impact_delta,
            "impact_status":   record.impact_status,
            "schema_version":  record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PIT] append_impact_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_impact_tracker(
    shadow_file:          Path,
    hold_duration_file:   Path,
    capital_conc_file:    Path,
    priority_cal_file:    Path,
    signal_attr_file:     Path,
    fco_file:             Path,
    output_file:          Path,
    today:                str,
) -> ImpactRunResult:
    """
    For each DEPLOYED candidate compute before/after metric delta.

    before_metric = latest metric value with date <= deployment_date
    after_metric  = latest metric value overall

    Never raises. FAIL_OPEN on all errors.
    """
    try:
        deployed = _load_deployed_candidates(shadow_file)
        evaluated = _load_evaluated_keys(output_file)

        # source file map
        source_records: Dict[str, List[dict]] = {
            "hold_duration":       _load_all_jsonl(hold_duration_file),
            "capital_concentration": _load_all_jsonl(capital_conc_file),
            "priority_calibration": _load_all_jsonl(priority_cal_file),
            "signal_attribution":  _load_all_jsonl(signal_attr_file),
            "forward_continuation": _load_all_jsonl(fco_file),
        }

        new_records: List[ImpactRecord] = []

        for ct in _CANDIDATE_ORDER:
            if ct not in deployed:
                continue
            if (today, ct) in evaluated:
                continue

            deployment_date = deployed[ct]
            field           = _METRIC_FIELD[ct]
            src_key         = _METRIC_SOURCE[ct]
            recs            = source_records.get(src_key, [])

            before = _extract_metric_at_or_before(recs, field, deployment_date)
            after  = _extract_latest_metric(recs, field)

            if before is None or after is None:
                continue

            delta  = round(after - before, 6)
            status = _classify_impact(delta)

            record = ImpactRecord(
                date=today,
                candidate=ct,
                metric_field=field,
                deployment_date=deployment_date,
                before_metric=round(before, 6),
                after_metric=round(after, 6),
                impact_delta=delta,
                impact_status=status,
            )
            append_impact_record(record, output_file)
            new_records.append(record)
            evaluated.add((today, ct))

        impact_records = _load_latest_impact_records(output_file)
        summary        = _compute_summary(impact_records)

        return ImpactRunResult(
            date=today,
            new_records=new_records,
            impact_records=impact_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PIT] run_promotion_impact_tracker failed: %s", exc)
        return ImpactRunResult(
            date=today,
            new_records=[],
            impact_records=[],
            summary=ImpactSummary(
                n_deployed=0,
                n_positive=0,
                n_negative=0,
                success_rate=0.0,
                best_candidate="",
                best_impact_delta=0.0,
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_impact_report(
    impact_records: List[ImpactRecord],
    summary:        ImpactSummary,
) -> str:
    """Return PROMOTION IMPACT TRACKER report section string."""
    if not impact_records:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION IMPACT TRACKER")
    lines.append("─" * 60)
    # table header
    lines.append(f"{'Candidate':<22} {'Before':>8} {'After':>8} {'Delta':>8}  Status")
    lines.append("─" * 60)
    for r in impact_records:
        sign = "+" if r.impact_delta >= 0 else ""
        lines.append(
            f"  {r.candidate:<20} {r.before_metric:>+8.4f} {r.after_metric:>+8.4f}"
            f" {sign}{r.impact_delta:>7.4f}  {r.impact_status}"
        )
    lines.append("─" * 60)
    lines.append(f"Success Rate: {summary.success_rate:.1f}%"
                 f"  ({summary.n_positive}/{summary.n_deployed} positive)")
    if summary.best_candidate:
        lines.append(f"Best: {summary.best_candidate}"
                     f"  delta={summary.best_impact_delta:+.4f}")
    lines.append(sep)
    return "\n".join(lines)
