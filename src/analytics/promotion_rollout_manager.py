"""
promotion_rollout_manager.py — Promotion Rollout Manager (Phase 6H-Lite)

Manages lifecycle state for promotion candidates. Pure analytics / governance.
No strategy execution changes. No order routing changes.

Lifecycle:
  OBSERVE → PAPER_SHADOW → MICRO_CAPITAL → PRODUCTION_READY
  Any state → ROLLBACK_REQUIRED (when latest impact is NEGATIVE)

Transition rules:
  OBSERVE      → PAPER_SHADOW:     classification == PROMOTE_FIRST AND governance_score >= 80
  PAPER_SHADOW → MICRO_CAPITAL:    days_in_state >= 14 AND impact NOT NEGATIVE
  MICRO_CAPITAL→ PRODUCTION_READY: days_in_state >= 30 AND impact == POSITIVE
  Any          → ROLLBACK_REQUIRED: latest impact_status == NEGATIVE (takes priority)

Inputs (all read-only):
  PROMOTION_GOVERNANCE_SCOREBOARD_FILE
  PROMOTION_CANDIDATE_REGISTRY_FILE
  PROMOTION_IMPACT_TRACKER_FILE

Output:
  PROMOTION_ROLLOUT_MANAGER_FILE — append-only JSONL
    one record per (date, candidate) per run

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

STATE_OBSERVE           = "OBSERVE"
STATE_PAPER_SHADOW      = "PAPER_SHADOW"
STATE_MICRO_CAPITAL     = "MICRO_CAPITAL"
STATE_PRODUCTION_READY  = "PRODUCTION_READY"
STATE_ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"

_PAPER_SHADOW_MIN_DAYS  = 14
_MICRO_CAPITAL_MIN_DAYS = 30

_PROMOTE_FIRST      = "PROMOTE_FIRST"
_SCORE_PROMOTE_MIN  = 80.0
_IMPACT_NEGATIVE    = "NEGATIVE"
_IMPACT_POSITIVE    = "POSITIVE"
_REGISTRY_ACTIVE    = "ACTIVE"

_CANDIDATE_ORDER: List[str] = [
    "EXIT_EXTENSION",
    "CAPITAL_CONCENTRATION",
    "PRIORITY",
    "SIGNAL_LEADER",
    "CONTINUATION_SIGNAL",
]

# State priority for top_rollout_candidate ranking (higher = better)
_STATE_PRIORITY: Dict[str, int] = {
    STATE_PRODUCTION_READY:  4,
    STATE_MICRO_CAPITAL:     3,
    STATE_PAPER_SHADOW:      2,
    STATE_OBSERVE:           1,
    STATE_ROLLBACK_REQUIRED: 0,
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RolloutRecord:
    date:                 str
    candidate:            str
    state:                str
    days_in_state:        int
    state_entry_date:     str
    latest_impact_status: str
    governance_score:     float
    schema_version:       str = SCHEMA_VERSION


@dataclass
class RolloutSummary:
    n_paper_shadow:        int
    n_micro_capital:       int
    n_production_ready:    int
    n_rollback_required:   int
    top_rollout_candidate: str   # "" if none beyond OBSERVE


@dataclass
class RolloutRunResult:
    date:           str
    new_records:    List[RolloutRecord]
    state_records:  List[RolloutRecord]   # latest per candidate, in _CANDIDATE_ORDER
    summary:        RolloutSummary


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
        logger.debug("[PRM] load_all_jsonl %s failed: %s", path, exc)
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
        logger.warning("[PRM] append_jsonl failed: %s", exc)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _compute_days(from_date: str, to_date: str) -> int:
    """Return (to_date - from_date).days. Returns 0 on any error."""
    try:
        d1 = datetime.fromisoformat(from_date)
        d2 = datetime.fromisoformat(to_date)
        return max(0, (d2 - d1).days)
    except Exception:
        return 0


# ── Metric loaders ────────────────────────────────────────────────────────────

def _latest_governance(scoreboard_file: Path) -> Dict[str, Tuple[float, str]]:
    """Return {candidate: (governance_score, classification)} latest per candidate."""
    latest_date: Dict[str, str] = {}
    result: Dict[str, Tuple[float, str]] = {}
    for r in _load_all_jsonl(scoreboard_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            result[ct] = (
                _safe_float(r.get("governance_score"), 0.0),
                str(r.get("classification", "")),
            )
    return result


def _latest_impact_status(impact_file: Path) -> Dict[str, str]:
    """Return {candidate: latest impact_status} from impact tracker."""
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


def _load_current_states(output_file: Path) -> Dict[str, dict]:
    """Return latest rollout record dict per candidate."""
    latest_date: Dict[str, str] = {}
    result: Dict[str, dict] = {}
    for r in _load_all_jsonl(output_file):
        ct   = str(r.get("candidate", ""))
        date = str(r.get("date", ""))
        if ct not in _CANDIDATE_ORDER or not date:
            continue
        if ct not in latest_date or date > latest_date[ct]:
            latest_date[ct] = date
            result[ct] = r
    return result


# ── Transition logic ──────────────────────────────────────────────────────────

def _next_state(
    current_state:        Optional[str],
    days_in_state:        int,
    latest_impact_status: str,
    governance_score:     float,
    classification:       str,
) -> str:
    """
    Determine next lifecycle state.

    Rollback takes priority over all other transitions.
    """
    # Rollback: negative impact in any non-observe state
    if (latest_impact_status == _IMPACT_NEGATIVE
            and current_state not in (None, STATE_OBSERVE, STATE_ROLLBACK_REQUIRED)):
        return STATE_ROLLBACK_REQUIRED

    # Once rolled back, stays rolled back
    if current_state == STATE_ROLLBACK_REQUIRED:
        return STATE_ROLLBACK_REQUIRED

    # MICRO_CAPITAL → PRODUCTION_READY
    if current_state == STATE_MICRO_CAPITAL:
        if days_in_state >= _MICRO_CAPITAL_MIN_DAYS and latest_impact_status == _IMPACT_POSITIVE:
            return STATE_PRODUCTION_READY
        return STATE_MICRO_CAPITAL

    # PAPER_SHADOW → MICRO_CAPITAL
    if current_state == STATE_PAPER_SHADOW:
        if days_in_state >= _PAPER_SHADOW_MIN_DAYS and latest_impact_status != _IMPACT_NEGATIVE:
            return STATE_MICRO_CAPITAL
        return STATE_PAPER_SHADOW

    # PRODUCTION_READY → stays (or rolls back if negative, handled above)
    if current_state == STATE_PRODUCTION_READY:
        return STATE_PRODUCTION_READY

    # OBSERVE (or None) → check entry condition
    if classification == _PROMOTE_FIRST and governance_score >= _SCORE_PROMOTE_MIN:
        return STATE_PAPER_SHADOW
    return STATE_OBSERVE


# ── Idempotency ───────────────────────────────────────────────────────────────

def _load_evaluated_keys(output_file: Path) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for r in _load_all_jsonl(output_file):
        date = str(r.get("date", ""))
        ct   = str(r.get("candidate", ""))
        if date and ct:
            keys.add((date, ct))
    return keys


# ── State record loader ───────────────────────────────────────────────────────

def _load_state_records(output_file: Path) -> List[RolloutRecord]:
    """Return latest RolloutRecord per candidate, in _CANDIDATE_ORDER."""
    current = _load_current_states(output_file)
    result: List[RolloutRecord] = []
    for ct in _CANDIDATE_ORDER:
        r = current.get(ct)
        if r is None:
            continue
        result.append(RolloutRecord(
            date=str(r.get("date", "")),
            candidate=ct,
            state=str(r.get("state", STATE_OBSERVE)),
            days_in_state=_safe_int(r.get("days_in_state"), 0),
            state_entry_date=str(r.get("state_entry_date", "")),
            latest_impact_status=str(r.get("latest_impact_status", "")),
            governance_score=_safe_float(r.get("governance_score"), 0.0),
        ))
    return result


# ── Summary ───────────────────────────────────────────────────────────────────

def _compute_summary(state_records: List[RolloutRecord]) -> RolloutSummary:
    n_ps  = sum(1 for r in state_records if r.state == STATE_PAPER_SHADOW)
    n_mc  = sum(1 for r in state_records if r.state == STATE_MICRO_CAPITAL)
    n_pr  = sum(1 for r in state_records if r.state == STATE_PRODUCTION_READY)
    n_rb  = sum(1 for r in state_records if r.state == STATE_ROLLBACK_REQUIRED)

    # top_rollout_candidate: highest state_priority, break ties by governance_score
    active = [r for r in state_records if r.state not in (STATE_OBSERVE, STATE_ROLLBACK_REQUIRED)]
    if active:
        top = max(active, key=lambda r: (_STATE_PRIORITY.get(r.state, 0), r.governance_score))
        top_name = top.candidate
    else:
        top_name = ""

    return RolloutSummary(
        n_paper_shadow=n_ps,
        n_micro_capital=n_mc,
        n_production_ready=n_pr,
        n_rollback_required=n_rb,
        top_rollout_candidate=top_name,
    )


# ── Serialization ─────────────────────────────────────────────────────────────

def append_rollout_record(record: RolloutRecord, path: Path) -> None:
    """Serialize one RolloutRecord to JSONL (append-only). FAIL_OPEN."""
    try:
        data = {
            "date":                 record.date,
            "candidate":            record.candidate,
            "state":                record.state,
            "days_in_state":        record.days_in_state,
            "state_entry_date":     record.state_entry_date,
            "latest_impact_status": record.latest_impact_status,
            "governance_score":     record.governance_score,
            "schema_version":       record.schema_version,
        }
        _append_jsonl(data, path)
    except Exception as exc:
        logger.warning("[PRM] append_rollout_record failed: %s", exc)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_promotion_rollout_manager(
    scoreboard_file: Path,
    registry_file:   Path,
    impact_file:     Path,
    output_file:     Path,
    today:           str,
) -> RolloutRunResult:
    """
    Advance lifecycle state for each candidate based on governance score and
    impact status. Never raises. FAIL_OPEN on all errors.

    Idempotent: (date, candidate) pair never re-written.
    """
    try:
        evaluated     = _load_evaluated_keys(output_file)
        current_states = _load_current_states(output_file)
        gov_data       = _latest_governance(scoreboard_file)
        impact_data    = _latest_impact_status(impact_file)

        new_records: List[RolloutRecord] = []

        for ct in _CANDIDATE_ORDER:
            if (today, ct) in evaluated:
                continue

            gov_score, classification = gov_data.get(ct, (0.0, ""))
            impact_status             = impact_data.get(ct, "")

            # load current state
            prev = current_states.get(ct)
            if prev is None:
                prev_state      = None
                state_entry_date = today
                days_in_state   = 0
            else:
                prev_state       = str(prev.get("state", STATE_OBSERVE))
                state_entry_date = str(prev.get("state_entry_date", today))
                days_in_state    = _compute_days(state_entry_date, today)

            new_state = _next_state(
                current_state=prev_state,
                days_in_state=days_in_state,
                latest_impact_status=impact_status,
                governance_score=gov_score,
                classification=classification,
            )

            # reset state_entry_date on state change
            if prev_state != new_state:
                state_entry_date = today
                days_in_state    = 0

            record = RolloutRecord(
                date=today,
                candidate=ct,
                state=new_state,
                days_in_state=days_in_state,
                state_entry_date=state_entry_date,
                latest_impact_status=impact_status,
                governance_score=gov_score,
            )
            append_rollout_record(record, output_file)
            new_records.append(record)
            evaluated.add((today, ct))

        state_records = _load_state_records(output_file)
        summary       = _compute_summary(state_records)

        return RolloutRunResult(
            date=today,
            new_records=new_records,
            state_records=state_records,
            summary=summary,
        )

    except Exception as exc:
        logger.warning("[PRM] run_promotion_rollout_manager failed: %s", exc)
        return RolloutRunResult(
            date=today,
            new_records=[],
            state_records=[],
            summary=RolloutSummary(
                n_paper_shadow=0,
                n_micro_capital=0,
                n_production_ready=0,
                n_rollback_required=0,
                top_rollout_candidate="",
            ),
        )


# ── Report formatter ──────────────────────────────────────────────────────────

def format_rollout_report(
    state_records: List[RolloutRecord],
    summary:       RolloutSummary,
) -> str:
    """Return PROMOTION ROLLOUT MANAGER report section string."""
    if not state_records:
        return ""
    lines: List[str] = []
    sep = "═" * 60
    lines.append(sep)
    lines.append("PROMOTION ROLLOUT MANAGER")
    lines.append("─" * 60)
    lines.append(f"{'Candidate':<22} {'State':<18} {'Days':>4}  Impact")
    lines.append("─" * 60)
    for r in state_records:
        impact_str = r.latest_impact_status if r.latest_impact_status else "—"
        lines.append(
            f"  {r.candidate:<20} {r.state:<18} {r.days_in_state:>4}  {impact_str}"
        )
    lines.append("─" * 60)
    lines.append(f"Production Ready: {summary.n_production_ready}"
                 f"  Rollback: {summary.n_rollback_required}")
    if summary.top_rollout_candidate:
        lines.append(f"Top Rollout Candidate: {summary.top_rollout_candidate}")
    lines.append(f"PAPER_SHADOW={summary.n_paper_shadow}"
                 f"  MICRO_CAPITAL={summary.n_micro_capital}"
                 f"  PRODUCTION_READY={summary.n_production_ready}")
    lines.append(sep)
    return "\n".join(lines)
