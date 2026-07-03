"""
slot_pressure.py — Slot Pressure Forecasting (Phase 5E)

Lightweight observation-only telemetry that measures portfolio congestion risk
BEFORE opportunity cost events occur.

Detects: weak holdings + high-quality candidate overflow → future slot starvation.

Observation-only. Append-only. Fail-open. O(n). No execution mutation.

Telemetry: runtime/analytics/slot_pressure/
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.analytics.opportunity_cost import (
    SEVERITY_SEVERE,
    MAT_BUFFER_DAYS,
    _add_business_days,
    _append_jsonl,
    _due_for_materialization,
    _load_jsonl,
    _safe,
)

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = "v1"
FORWARD_BUSINESS_DAYS = 5

# ── Priority tier labels (mirror ContinuationPriority) ────────────────────────
TIER_CORE    = "core_continuation"
TIER_HIGH    = "high_quality"
TIER_NEUTRAL = "neutral"
TIER_EXIT    = "exit_candidate"

# ── Scoring constants ──────────────────────────────────────────────────────────
WEAK_SLOT_THRESHOLD:           float = 45.0
FULL_PORTFOLIO_BONUS:          float = 20.0
WEAK_SLOT_BONUS:               float = 20.0
REJECTED_THRESHOLD:            int   = 3
REJECTED_BONUS:                float = 15.0
SEVERE_THRESHOLD:              int   = 2
SEVERE_BONUS:                  float = 20.0
CANDIDATE_QUALITY_THRESHOLD:   float = 80.0
CANDIDATE_QUALITY_BONUS:       float = 15.0
ALL_CORE_REDUCTION:            float = 20.0
OPEN_SLOT_REDUCTION:           float = 25.0

# ── Pressure regime labels ─────────────────────────────────────────────────────
PRESSURE_CRITICAL = "critical_pressure"   # score >= 80
PRESSURE_HIGH     = "high_pressure"       # score >= 60
PRESSURE_MODERATE = "moderate_pressure"   # score >= 40
PRESSURE_LOW      = "low_pressure"        # score <  40

PRESSURE_CRITICAL_THRESHOLD = 80.0
PRESSURE_HIGH_THRESHOLD     = 60.0
PRESSURE_MODERATE_THRESHOLD = 40.0

# ── Congestion reason labels ───────────────────────────────────────────────────
REASON_WEAK_SLOT = "weak_slot_congestion"
REASON_OVERFLOW  = "candidate_overflow"
REASON_SEVERE    = "severe_priority_gap"
REASON_HEALTHY   = "healthy_capacity"

# ── OC lookback for adapter ────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlotPressureInput:
    """Snapshot of portfolio state for pressure evaluation."""
    max_positions: int
    current_positions: int
    holding_priorities: List[float]      # one per held position; 0-100
    holding_tiers: List[str]             # core_continuation|high_quality|neutral|exit_candidate
    avg_candidate_priority_5d: float     # rolling avg of blocked candidate priorities (5d)
    recent_rejected_candidates: int      # OC events in last DEFAULT_LOOKBACK_DAYS days
    recent_severe_opportunity_cost_events: int  # SEVERITY_SEVERE events in same window


@dataclass
class SlotPressureResult:
    """Computed pressure assessment for one evaluation."""
    pressure_score: float           # 0-100
    pressure_regime: str            # low|moderate|high|critical
    reasons: List[str]              # active congestion signals
    lowest_priority: Optional[float]
    open_slots: int


@dataclass
class SlotPressureRecord:
    """Telemetry record written to JSONL."""
    date: str
    pressure_score: float
    pressure_regime: str
    holding_priorities: List[float]
    lowest_priority: float
    recent_rejected_candidates: int
    recent_severe_opportunity_cost_events: int
    avg_candidate_priority_5d: float
    reasons: List[str]
    max_positions: int
    current_positions: int
    open_slots: int
    schema_version: str = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Slot Pressure Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_slot_pressure(inp: SlotPressureInput) -> SlotPressureResult:
    """
    Task 1: Compute 0-100 portfolio congestion pressure score.

    Positive signals (add pressure):
      +20  portfolio full
      +20  lowest holding priority < 45
      +15  recent rejected candidates >= 3
      +20  recent severe OC events >= 2
      +15  avg candidate priority (5d) > 80

    Negative signals (reduce pressure):
      -20  all holdings are core_continuation
      -25  open portfolio slots exist

    Score is clamped to [0, 100]. Pure function; O(n).
    """
    score = 0.0
    reasons: List[str] = []

    open_slots = max(0, inp.max_positions - inp.current_positions)
    lowest = min(inp.holding_priorities) if inp.holding_priorities else None

    # Portfolio full: +20
    if inp.current_positions >= inp.max_positions:
        score += FULL_PORTFOLIO_BONUS

    # Weak slot: +20
    if lowest is not None and lowest < WEAK_SLOT_THRESHOLD:
        score += WEAK_SLOT_BONUS
        reasons.append(REASON_WEAK_SLOT)

    # Candidate overflow: +15
    if inp.recent_rejected_candidates >= REJECTED_THRESHOLD:
        score += REJECTED_BONUS
        reasons.append(REASON_OVERFLOW)

    # Severe OC events: +20
    if inp.recent_severe_opportunity_cost_events >= SEVERE_THRESHOLD:
        score += SEVERE_BONUS
        reasons.append(REASON_SEVERE)

    # High candidate quality pressure: +15
    if inp.avg_candidate_priority_5d > CANDIDATE_QUALITY_THRESHOLD:
        score += CANDIDATE_QUALITY_BONUS

    # All core continuation: -20
    if inp.holding_tiers and all(t == TIER_CORE for t in inp.holding_tiers):
        score -= ALL_CORE_REDUCTION

    # Open slots exist: -25
    if open_slots > 0:
        score -= OPEN_SLOT_REDUCTION
        reasons.append(REASON_HEALTHY)

    score = max(0.0, min(100.0, round(score, 2)))
    regime = classify_pressure_regime(score)

    return SlotPressureResult(
        pressure_score=score,
        pressure_regime=regime,
        reasons=reasons,
        lowest_priority=lowest,
        open_slots=open_slots,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Pressure Regime Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_pressure_regime(score: float) -> str:
    """
    Task 2: Map pressure score → regime label.
      >= 80 → critical_pressure
      >= 60 → high_pressure
      >= 40 → moderate_pressure
       < 40 → low_pressure
    """
    s = _safe(score)
    if s >= PRESSURE_CRITICAL_THRESHOLD:
        return PRESSURE_CRITICAL
    if s >= PRESSURE_HIGH_THRESHOLD:
        return PRESSURE_HIGH
    if s >= PRESSURE_MODERATE_THRESHOLD:
        return PRESSURE_MODERATE
    return PRESSURE_LOW


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Congestion Reason Attribution (via compute_slot_pressure)
# ─────────────────────────────────────────────────────────────────────────────
# Reasons are populated inline in compute_slot_pressure.
# Exposed separately for direct inspection if needed.

def extract_congestion_reasons(inp: SlotPressureInput) -> List[str]:
    """
    Task 3: Return active congestion reason labels for a SlotPressureInput.
    Delegates to compute_slot_pressure to avoid duplication.
    """
    return compute_slot_pressure(inp).reasons


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Telemetry Persistence
# ─────────────────────────────────────────────────────────────────────────────

def record_slot_pressure(
    inp: SlotPressureInput,
    date: str,
    pressure_file: Path,
) -> Optional[SlotPressureRecord]:
    """
    Task 5: Compute and append one SlotPressureRecord to pressure_file.
    Fail-open: returns None on any error, never raises.
    """
    try:
        result = compute_slot_pressure(inp)
        lowest = result.lowest_priority if result.lowest_priority is not None else 0.0

        rec = SlotPressureRecord(
            date=date,
            pressure_score=result.pressure_score,
            pressure_regime=result.pressure_regime,
            holding_priorities=list(inp.holding_priorities),
            lowest_priority=lowest,
            recent_rejected_candidates=inp.recent_rejected_candidates,
            recent_severe_opportunity_cost_events=inp.recent_severe_opportunity_cost_events,
            avg_candidate_priority_5d=round(_safe(inp.avg_candidate_priority_5d), 2),
            reasons=list(result.reasons),
            max_positions=inp.max_positions,
            current_positions=inp.current_positions,
            open_slots=result.open_slots,
        )
        ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        row = asdict(rec)
        row["ts"] = ts
        _append_jsonl(row, pressure_file)
        return rec
    except Exception as exc:
        logger.warning("record_slot_pressure failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Forward Validation (Materialization)
# ─────────────────────────────────────────────────────────────────────────────

def materialize_slot_pressure(
    pressure_file: Path,
    enriched_file: Path,
    oc_events_file: Path,
    today: str,
    forward_business_days: int = FORWARD_BUSINESS_DAYS,
    buffer_days: int = MAT_BUFFER_DAYS,
) -> int:
    """
    Task 6: Forward validate pressure scores against actual future OC events.

    For each un-validated pressure record ≥ buffer_days old:
      - Count OC events in the 5-business-day forward window
      - Write enrichment companion record to enriched_file

    Idempotent (dedup by date). Fail-open. Append-only.

    Returns: count of records newly enriched.
    """
    pressure_records = _load_jsonl(pressure_file)
    oc_events        = _load_jsonl(oc_events_file)

    done_dates = {r.get("date", "") for r in _load_jsonl(enriched_file)}

    count = 0
    for rec in pressure_records:
        try:
            date = rec.get("date", "")
            if not date or date in done_dates:
                continue
            if not _due_for_materialization(date, today, buffer_days):
                continue

            target_date = _add_business_days(date, forward_business_days)

            future_oc = sum(
                1 for e in oc_events
                if date < e.get("date", "") <= target_date
            )
            future_rejected = future_oc  # one OC event = one rejected candidate

            ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
            enriched = {
                "date":                              date,
                "pressure_score":                    rec.get("pressure_score", 0.0),
                "pressure_regime":                   rec.get("pressure_regime", PRESSURE_LOW),
                "reasons":                           rec.get("reasons", []),
                "future_opportunity_cost_events_5d": future_oc,
                "future_rejected_candidates_5d":     future_rejected,
                "materialized":                      True,
                "materialized_at":                   ts,
                "schema_version":                    SCHEMA_VERSION,
            }
            _append_jsonl(enriched, enriched_file)
            done_dates.add(date)
            count += 1

        except Exception as exc:
            logger.debug("materialize_slot_pressure: skipped date=%s: %s", rec.get("date"), exc)

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — KPI Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_pressure_kpis(
    pressure_file: Path,
    enriched_file: Optional[Path] = None,
) -> dict:
    """
    Task 7: Aggregate pressure KPIs from telemetry + enriched records.

    Most important: high_vs_low_pressure_future_oc_diff
        = avg OC events after high_pressure - avg OC events after low_pressure

    Returns dict with all KPIs. Fail-open.
    """
    try:
        records  = _load_jsonl(pressure_file)
        enriched = _load_jsonl(enriched_file) if (
            enriched_file is not None and enriched_file.exists()
        ) else []

        total = len(records)
        critical_count = sum(1 for r in records if r.get("pressure_regime") == PRESSURE_CRITICAL)
        critical_frequency = round(critical_count / total, 4) if total > 0 else 0.0

        # avg_future_opportunity_cost_by_pressure
        by_regime_oc: Dict[str, List[int]] = {}
        by_regime_rej: Dict[str, List[int]] = {}
        for e in enriched:
            regime = e.get("pressure_regime", "")
            future_oc  = int(e.get("future_opportunity_cost_events_5d", 0) or 0)
            future_rej = int(e.get("future_rejected_candidates_5d",     0) or 0)
            by_regime_oc.setdefault(regime, []).append(future_oc)
            by_regime_rej.setdefault(regime, []).append(future_rej)

        def _avg_int(lst: List[int]) -> Optional[float]:
            return round(sum(lst) / len(lst), 4) if lst else None

        avg_future_oc   = {k: _avg_int(v) for k, v in by_regime_oc.items()}
        avg_future_rej  = {k: _avg_int(v) for k, v in by_regime_rej.items()}

        # future_congestion_prediction_accuracy
        high_enriched = [
            e for e in enriched
            if e.get("pressure_regime") in (PRESSURE_HIGH, PRESSURE_CRITICAL)
        ]
        if high_enriched:
            correct = sum(
                1 for e in high_enriched
                if int(e.get("future_opportunity_cost_events_5d", 0) or 0) > 0
            )
            prediction_accuracy = round(correct / len(high_enriched), 4)
        else:
            prediction_accuracy = 0.0

        # KEY KPI: high vs low pressure future OC difference
        high_oc = [
            int(e.get("future_opportunity_cost_events_5d", 0) or 0)
            for e in enriched
            if e.get("pressure_regime") in (PRESSURE_HIGH, PRESSURE_CRITICAL)
        ]
        low_oc = [
            int(e.get("future_opportunity_cost_events_5d", 0) or 0)
            for e in enriched
            if e.get("pressure_regime") in (PRESSURE_LOW, PRESSURE_MODERATE)
        ]

        high_avg = round(sum(high_oc) / len(high_oc), 4) if high_oc else None
        low_avg  = round(sum(low_oc)  / len(low_oc),  4) if low_oc  else None
        diff     = round(high_avg - low_avg, 4) if (high_avg is not None and low_avg is not None) else None

        return {
            "total_pressure_records":                    total,
            "critical_pressure_frequency":               critical_frequency,
            "avg_future_opportunity_cost_by_pressure":   avg_future_oc,
            "avg_rejected_candidates_by_regime":         avg_future_rej,
            "future_congestion_prediction_accuracy":     prediction_accuracy,
            "avg_future_oc_high_pressure":               high_avg,
            "avg_future_oc_low_pressure":                low_avg,
            "high_vs_low_pressure_future_oc_diff":       diff,
            "schema_version":                            SCHEMA_VERSION,
        }

    except Exception as exc:
        logger.warning("aggregate_pressure_kpis failed: %s", exc)
        return _empty_pressure_kpis()


def _empty_pressure_kpis() -> dict:
    return {
        "total_pressure_records":                0,
        "critical_pressure_frequency":           0.0,
        "avg_future_opportunity_cost_by_pressure": {},
        "avg_rejected_candidates_by_regime":     {},
        "future_congestion_prediction_accuracy": 0.0,
        "avg_future_oc_high_pressure":           None,
        "avg_future_oc_low_pressure":            None,
        "high_vs_low_pressure_future_oc_diff":   None,
        "schema_version":                        SCHEMA_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Dry-run Visualization
# ─────────────────────────────────────────────────────────────────────────────

def format_slot_pressure_report(
    inp: SlotPressureInput,
    result: SlotPressureResult,
    date: str,
) -> str:
    """
    Task 4: Format SLOT PRESSURE FORECAST section for DRY/LIVE report.
    Fails open (returns "" on any error).
    """
    try:
        sep    = "─" * 58
        regime = result.pressure_regime.upper().replace("_", " ")
        score  = result.pressure_score

        lines = [
            sep,
            f"  SLOT PRESSURE FORECAST  ({date})",
            sep,
            f"  pressure={score:.0f}  {regime}",
            "",
        ]

        if result.reasons:
            lines.append("  Reasons:")
            for r in result.reasons:
                lines.append(f"    - {r}")
            lines.append("")

        lp = result.lowest_priority
        lines.append(
            f"  Lowest holding priority  : "
            f"{lp:.1f}" if lp is not None else "  Lowest holding priority  : N/A"
        )
        lines.append(f"  Recent rejected cands    : {inp.recent_rejected_candidates}")
        lines.append(f"  Recent severe OC events  : {inp.recent_severe_opportunity_cost_events}")
        if inp.avg_candidate_priority_5d > 0:
            lines.append(f"  Avg candidate priority 5d: {inp.avg_candidate_priority_5d:.1f}")
        lines.append(f"  Open slots               : {result.open_slots}/{inp.max_positions}")
        lines.append(sep)

        return "\n".join(lines)

    except Exception as exc:
        logger.debug("format_slot_pressure_report failed: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: build from existing run_live_signal.py data
# ─────────────────────────────────────────────────────────────────────────────

def build_slot_pressure_input_from_oc_data(
    current_positions: int,
    max_positions: int,
    cp_results: List[Any],
    oc_events_file: Path,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    today: str = "",
) -> SlotPressureInput:
    """
    Build SlotPressureInput from existing system data.

    cp_results: list of ContinuationPriorityResult objects (or dicts) with
                priority_score + priority_tier attributes.
    oc_events_file: path to opportunity_cost events JSONL.

    Fail-open: returns minimal-default input on any error.
    """
    try:
        # Extract holding priorities + tiers from cp_results
        priorities: List[float] = []
        tiers: List[str] = []
        for cp in cp_results:
            try:
                if isinstance(cp, dict):
                    priorities.append(_safe(cp.get("priority_score", 0.0)))
                    tiers.append(str(cp.get("priority_tier", TIER_NEUTRAL)))
                else:
                    priorities.append(_safe(getattr(cp, "priority_score", 0.0)))
                    tiers.append(str(getattr(cp, "priority_tier", TIER_NEUTRAL)))
            except Exception:
                pass

        # Read recent OC events
        events = _load_jsonl(oc_events_file)

        # Filter to lookback window
        if today:
            try:
                today_d = datetime.strptime(today, "%Y-%m-%d").date()
                cutoff  = (today_d - timedelta(days=lookback_days)).isoformat()
                recent  = [e for e in events if e.get("date", "") >= cutoff]
            except Exception:
                recent = events[-lookback_days * 3:]
        else:
            recent = events[-lookback_days * 3:]

        recent_rejected = len(recent)
        recent_severe   = sum(
            1 for e in recent
            if e.get("opportunity_cost_severity") == SEVERITY_SEVERE
        )

        # Avg candidate priority from recent events
        cand_priorities = [
            _safe(e.get("candidate_priority_estimate", 0.0))
            for e in recent
            if e.get("candidate_priority_estimate") is not None
        ]
        avg_cand = round(sum(cand_priorities) / len(cand_priorities), 2) if cand_priorities else 0.0

        return SlotPressureInput(
            max_positions=max(1, int(max_positions)),
            current_positions=max(0, int(current_positions)),
            holding_priorities=priorities,
            holding_tiers=tiers,
            avg_candidate_priority_5d=avg_cand,
            recent_rejected_candidates=recent_rejected,
            recent_severe_opportunity_cost_events=recent_severe,
        )

    except Exception as exc:
        logger.warning("build_slot_pressure_input_from_oc_data failed: %s", exc)
        return SlotPressureInput(
            max_positions=max_positions,
            current_positions=current_positions,
            holding_priorities=[],
            holding_tiers=[],
            avg_candidate_priority_5d=0.0,
            recent_rejected_candidates=0,
            recent_severe_opportunity_cost_events=0,
        )
