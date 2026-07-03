"""
production_promotion_gate.py — Production Promotion Gate (Phase 6I-Lite)

Final governance gate before any production activation.
Evaluates PRODUCTION_READY candidates against three approval conditions
and classifies each as APPROVED / REVIEW_REQUIRED / REJECTED.

Inputs (all read-only):
  PROMOTION_ROLLOUT_MANAGER_FILE
  PROMOTION_GOVERNANCE_SCOREBOARD_FILE
  PROMOTION_IMPACT_TRACKER_FILE

Output:
  PRODUCTION_PROMOTION_GATE_FILE — append-only JSONL
    one aggregate record per run (keyed by date)

Approval rules:
  APPROVED:         governance_score >= 80 AND impact == POSITIVE AND days_in_state >= 30
  REVIEW_REQUIRED:  governance_score >= 80 AND impact != NEGATIVE AND NOT all of above
  REJECTED:         governance_score < 80 OR impact == NEGATIVE

Risk score:
  risk_score = clamp(100 - governance_score, 0, 100)

Design:
  observation_only — no signals, no orders, no parameter changes
  fail-open        — never blocks live execution
  idempotent       — date key never re-appended
  O(n)             — linear in JSONL lines
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"

DECISION_APPROVED        = "APPROVED"
DECISION_REVIEW_REQUIRED = "REVIEW_REQUIRED"
DECISION_REJECTED        = "REJECTED"

_STATE_PRODUCTION_READY = "PRODUCTION_READY"
_IMPACT_POSITIVE        = "POSITIVE"
_IMPACT_NEGATIVE        = "NEGATIVE"

_APPROVAL_SCORE_MIN = 80.0
_APPROVAL_DAYS_MIN  = 30

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class GateRecord:
    candidate:        str
    state:            str
    governance_score: float
    risk_score:       float
    days_in_state:    int
    impact_status:    str
    decision:         str


@dataclass
class GateSummary:
    n_approved:            int
    n_review_required:     int
    n_rejected:            int
    lowest_risk_candidate: str  # PRODUCTION_READY candidate with lowest risk_score
    top_candidate:         str  # APPROVED candidate with highest governance_score


@dataclass
class GateRunResult:
    date:         str
    gate_records: List[GateRecord]
    summary:      GateSummary
    skipped:      bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

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
        logger.debug("[PPG] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PPG] append_jsonl failed: %s", exc)


# ── Pure classification functions ──────────────────────────────────────────────

def compute_risk_score(governance_score: float) -> float:
    """risk_score = clamp(100 - governance_score, 0, 100), 2 decimal places."""
    return round(max(0.0, min(100.0, 100.0 - governance_score)), 2)


def classify_candidate(
    governance_score: float,
    impact_status:    str,
    days_in_state:    int,
) -> str:
    """
    Classify a PRODUCTION_READY candidate.

    REJECTED:         score < 80 OR impact == NEGATIVE
    APPROVED:         score >= 80 AND impact == POSITIVE AND days >= 30
    REVIEW_REQUIRED:  score >= 80 AND impact != NEGATIVE AND NOT fully approved
    """
    if governance_score < _APPROVAL_SCORE_MIN or impact_status == _IMPACT_NEGATIVE:
        return DECISION_REJECTED
    if impact_status == _IMPACT_POSITIVE and days_in_state >= _APPROVAL_DAYS_MIN:
        return DECISION_APPROVED
    return DECISION_REVIEW_REQUIRED


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_evaluated_dates(output_file: Path) -> Set[str]:
    """Return set of dates already recorded in output_file."""
    dates: Set[str] = set()
    for r in _load_all_jsonl(output_file):
        d = str(r.get("date", ""))
        if d:
            dates.add(d)
    return dates


def _latest_rollout_states(rollout_file: Path) -> Dict[str, dict]:
    """Return latest rollout record per candidate (keyed by candidate)."""
    latest_date: Dict[str, str] = {}
    result: Dict[str, dict] = {}
    for r in _load_all_jsonl(rollout_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            result[ct] = r
    return result


def _latest_governance_scores(scoreboard_file: Path) -> Dict[str, float]:
    """Return latest governance_score per candidate."""
    latest_date: Dict[str, str] = {}
    result: Dict[str, float] = {}
    for r in _load_all_jsonl(scoreboard_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            result[ct] = _safe_float(r.get("governance_score"), 0.0)
    return result


def _latest_impact_statuses(impact_file: Path) -> Dict[str, str]:
    """Return latest impact_status per candidate from impact tracker."""
    latest_date: Dict[str, str] = {}
    result: Dict[str, str] = {}
    for r in _load_all_jsonl(impact_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            result[ct] = str(r.get("impact_status", ""))
    return result


# ── Gate computation ───────────────────────────────────────────────────────────

def compute_gate_records(
    rollout_states:    Dict[str, dict],
    gov_scores:        Dict[str, float],
    impact_statuses:   Dict[str, str],
) -> List[GateRecord]:
    """
    Build GateRecord list for PRODUCTION_READY candidates only.
    Ordered by _CANDIDATE_ORDER.
    """
    records: List[GateRecord] = []
    for ct in _CANDIDATE_ORDER:
        rollout = rollout_states.get(ct)
        if rollout is None:
            continue
        state = str(rollout.get("state", ""))
        if state != _STATE_PRODUCTION_READY:
            continue
        gov_score     = gov_scores.get(ct, _safe_float(rollout.get("governance_score"), 0.0))
        impact_status = impact_statuses.get(ct, str(rollout.get("latest_impact_status", "")))
        days_in_state = _safe_int(rollout.get("days_in_state"), 0)
        risk_score    = compute_risk_score(gov_score)
        decision      = classify_candidate(gov_score, impact_status, days_in_state)
        records.append(GateRecord(
            candidate=ct,
            state=state,
            governance_score=gov_score,
            risk_score=risk_score,
            days_in_state=days_in_state,
            impact_status=impact_status,
            decision=decision,
        ))
    return records


def compute_gate_summary(gate_records: List[GateRecord]) -> GateSummary:
    """Compute aggregate KPIs from gate records."""
    n_approved        = sum(1 for r in gate_records if r.decision == DECISION_APPROVED)
    n_review_required = sum(1 for r in gate_records if r.decision == DECISION_REVIEW_REQUIRED)
    n_rejected        = sum(1 for r in gate_records if r.decision == DECISION_REJECTED)

    # lowest_risk_candidate: min risk_score across all PRODUCTION_READY
    lowest_risk_candidate = ""
    if gate_records:
        best = min(gate_records, key=lambda r: (r.risk_score, r.candidate))
        lowest_risk_candidate = best.candidate

    # top_candidate: APPROVED with highest governance_score
    approved = [r for r in gate_records if r.decision == DECISION_APPROVED]
    top_candidate = ""
    if approved:
        top = max(approved, key=lambda r: r.governance_score)
        top_candidate = top.candidate

    return GateSummary(
        n_approved=n_approved,
        n_review_required=n_review_required,
        n_rejected=n_rejected,
        lowest_risk_candidate=lowest_risk_candidate,
        top_candidate=top_candidate,
    )


# ── Serialization ──────────────────────────────────────────────────────────────

def append_gate_result(
    date:         str,
    gate_records: List[GateRecord],
    summary:      GateSummary,
    path:         Path,
) -> None:
    """Serialize one run result to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":         date,
            "gate_records": [
                {
                    "candidate":        r.candidate,
                    "state":            r.state,
                    "governance_score": r.governance_score,
                    "risk_score":       r.risk_score,
                    "days_in_state":    r.days_in_state,
                    "impact_status":    r.impact_status,
                    "decision":         r.decision,
                }
                for r in gate_records
            ],
            "n_approved":            summary.n_approved,
            "n_review_required":     summary.n_review_required,
            "n_rejected":            summary.n_rejected,
            "lowest_risk_candidate": summary.lowest_risk_candidate,
            "top_candidate":         summary.top_candidate,
            "schema_version":        SCHEMA_VERSION,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PPG] append_gate_result failed: %s", exc)


def _reconstruct_from_record(r: dict) -> GateRunResult:
    """Reconstruct GateRunResult from a stored JSONL record."""
    gate_records: List[GateRecord] = []
    for gr in r.get("gate_records", []):
        gate_records.append(GateRecord(
            candidate=str(gr.get("candidate", "")),
            state=str(gr.get("state", "")),
            governance_score=_safe_float(gr.get("governance_score"), 0.0),
            risk_score=_safe_float(gr.get("risk_score"), 0.0),
            days_in_state=_safe_int(gr.get("days_in_state"), 0),
            impact_status=str(gr.get("impact_status", "")),
            decision=str(gr.get("decision", "")),
        ))
    summary = GateSummary(
        n_approved=_safe_int(r.get("n_approved"), 0),
        n_review_required=_safe_int(r.get("n_review_required"), 0),
        n_rejected=_safe_int(r.get("n_rejected"), 0),
        lowest_risk_candidate=str(r.get("lowest_risk_candidate", "")),
        top_candidate=str(r.get("top_candidate", "")),
    )
    return GateRunResult(
        date=str(r.get("date", "")),
        gate_records=gate_records,
        summary=summary,
        skipped=True,
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def run_production_promotion_gate(
    rollout_file:    Path,
    scoreboard_file: Path,
    impact_file:     Path,
    output_file:     Path,
    today:           str,
) -> GateRunResult:
    """
    Evaluate PRODUCTION_READY candidates against approval rules.
    Never raises. FAIL_OPEN on all errors.

    Idempotent: date never re-written.
    """
    _empty_summary = GateSummary(
        n_approved=0,
        n_review_required=0,
        n_rejected=0,
        lowest_risk_candidate="",
        top_candidate="",
    )
    try:
        evaluated = _load_evaluated_dates(output_file)

        # Idempotent: return existing record if already run today
        if today in evaluated:
            for r in reversed(_load_all_jsonl(output_file)):
                if str(r.get("date", "")) == today:
                    return _reconstruct_from_record(r)
            return GateRunResult(date=today, gate_records=[], summary=_empty_summary, skipped=True)

        rollout_states  = _latest_rollout_states(rollout_file)
        gov_scores      = _latest_governance_scores(scoreboard_file)
        impact_statuses = _latest_impact_statuses(impact_file)

        gate_records = compute_gate_records(rollout_states, gov_scores, impact_statuses)
        summary      = compute_gate_summary(gate_records)

        append_gate_result(today, gate_records, summary, output_file)

        return GateRunResult(
            date=today,
            gate_records=gate_records,
            summary=summary,
            skipped=False,
        )

    except Exception as exc:
        logger.warning("[PPG] run_production_promotion_gate failed: %s", exc)
        return GateRunResult(date=today, gate_records=[], summary=_empty_summary)


# ── Report formatter ───────────────────────────────────────────────────────────

def format_gate_report(result: GateRunResult) -> str:
    """Return PRODUCTION PROMOTION GATE report section string."""
    if not result.gate_records:
        return ""
    lines: List[str] = []
    sep = "═" * 70
    lines.append(sep)
    lines.append("PRODUCTION PROMOTION GATE")
    lines.append("─" * 70)
    lines.append(
        f"{'Candidate':<22} {'State':<18} {'GovScore':>8}  {'RiskScore':>9}  Decision"
    )
    lines.append("─" * 70)
    for r in result.gate_records:
        lines.append(
            f"  {r.candidate:<20} {r.state:<18} {r.governance_score:>8.1f}"
            f"  {r.risk_score:>9.1f}  {r.decision}"
        )
    lines.append("─" * 70)
    lines.append(
        f"Approved: {result.summary.n_approved}"
        f"  Review: {result.summary.n_review_required}"
        f"  Rejected: {result.summary.n_rejected}"
    )
    if result.summary.top_candidate:
        lines.append(f"Top Candidate: {result.summary.top_candidate}")
    if result.summary.lowest_risk_candidate:
        lines.append(f"Lowest Risk:   {result.summary.lowest_risk_candidate}")
    lines.append(sep)
    return "\n".join(lines)
