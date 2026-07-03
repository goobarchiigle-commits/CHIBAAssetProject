"""
research_priority_summary.py — Deterministic daily research-priority summaries

Converts runtime analytics into actionable research diagnostics for daily sessions.

Diagnostics only — no auto-tuning, no parameter changes, no execution changes,
no strategy mutation, no deployment automation, no autonomous recommendations.

Generates deterministic daily research-priority summaries from:
  - alpha_diagnostic snapshots
  - daily leakage summaries
  - skipped opportunity records
  - execution integrity summaries

Identifies for each runtime day:
  - dominant bottleneck (9 supported categories)
  - estimated deployable-alpha leakage (JPY)
  - largest missed PnL contributor
  - persistence vs trailing_5d and trailing_20d
  - confidence level (HIGH / MEDIUM / LOW)
  - affected symbols and sectors
  - likely root-cause category

Supported bottleneck categories:
  alloc_cap_pressure, idle_cash_persistence, high_priced_exclusion,
  concentration_throttle, sector_suppression, premature_exit_behavior,
  stale_signal_decay, insufficient_buying_power, execution_integrity_degradation

Deterministic recommendation categories only (no auto-fixes, no parameters):
  allocation_research, exit_research, deployability_research,
  concentration_governance_review, capital_efficiency_review,
  execution_integrity_review

Outputs:
  1. runtime/reports/daily_research_summary_YYYYMMDD.md (human-readable)
  2. Append-only JSONL artifact (structured, replay-compatible)

Requirements:
  - append-only artifacts
  - deterministic markdown (same inputs → byte-identical output)
  - replay reproducibility
  - fail-open only (never raises to caller)
  - read-only analytics — no live-state mutation
  - Windows compatible
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.analytics.alpha_diagnostic import AlphaDiagnosticSnapshot
from src.analytics.daily_leakage_summary import DailyLeakageSummary
from src.analytics.skipped_opportunity_analytics import (
    SkippedOpportunityRecord,
    ENRICHMENT_ENRICHED,
    REJECTION_ALLOC_CAP,
    REJECTION_HIGH_PRICE,
    REJECTION_SECTOR_SUPPRESSION,
    REJECTION_CONCENTRATION_THROTTLE,
    REJECTION_INSUFFICIENT_CASH,
    REJECTION_STALE_SIGNAL,
    REJECTION_TRUTH_CONFIDENCE,
    REJECTION_RATE_LIMIT,
    REJECTION_DUPLICATE,
    REJECTION_EPOCH_BLOCKED,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck category constants
# ─────────────────────────────────────────────────────────────────────────────

BOTTLENECK_ALLOC_CAP_PRESSURE        = "alloc_cap_pressure"
BOTTLENECK_IDLE_CASH_PERSISTENCE     = "idle_cash_persistence"
BOTTLENECK_HIGH_PRICED_EXCLUSION     = "high_priced_exclusion"
BOTTLENECK_CONCENTRATION_THROTTLE    = "concentration_throttle"
BOTTLENECK_SECTOR_SUPPRESSION        = "sector_suppression"
BOTTLENECK_PREMATURE_EXIT            = "premature_exit_behavior"
BOTTLENECK_STALE_SIGNAL_DECAY        = "stale_signal_decay"
BOTTLENECK_INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
BOTTLENECK_EXECUTION_INTEGRITY       = "execution_integrity_degradation"
BOTTLENECK_NONE                      = "none"

ALL_BOTTLENECK_CATEGORIES = (
    BOTTLENECK_ALLOC_CAP_PRESSURE,
    BOTTLENECK_IDLE_CASH_PERSISTENCE,
    BOTTLENECK_HIGH_PRICED_EXCLUSION,
    BOTTLENECK_CONCENTRATION_THROTTLE,
    BOTTLENECK_SECTOR_SUPPRESSION,
    BOTTLENECK_PREMATURE_EXIT,
    BOTTLENECK_STALE_SIGNAL_DECAY,
    BOTTLENECK_INSUFFICIENT_BUYING_POWER,
    BOTTLENECK_EXECUTION_INTEGRITY,
    BOTTLENECK_NONE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Research category constants
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_ALLOCATION         = "allocation_research"
RESEARCH_EXIT               = "exit_research"
RESEARCH_DEPLOYABILITY      = "deployability_research"
RESEARCH_CONCENTRATION      = "concentration_governance_review"
RESEARCH_CAPITAL_EFFICIENCY = "capital_efficiency_review"
RESEARCH_INTEGRITY          = "execution_integrity_review"

ALL_RESEARCH_CATEGORIES = (
    RESEARCH_ALLOCATION,
    RESEARCH_EXIT,
    RESEARCH_DEPLOYABILITY,
    RESEARCH_CONCENTRATION,
    RESEARCH_CAPITAL_EFFICIENCY,
    RESEARCH_INTEGRITY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Confidence levels
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_HIGH   = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW    = "LOW"

# Thresholds for bottleneck classification
_IDLE_CASH_PERSISTENCE_THRESHOLD  = 0.25   # idle_cash_ratio > this → idle_cash_persistence secondary
_PREMATURE_EXIT_THRESHOLD          = 0.40   # idle_cash_ratio > this AND alloc_cap → premature_exit
_IDLE_CASH_DOMINANT_THRESHOLD      = 0.35   # idle_cash_ratio > this → dominates over other bottlenecks

# ─────────────────────────────────────────────────────────────────────────────
# Rejection reason → bottleneck category
# ─────────────────────────────────────────────────────────────────────────────

_REASON_TO_BOTTLENECK: Dict[str, str] = {
    REJECTION_ALLOC_CAP:              BOTTLENECK_ALLOC_CAP_PRESSURE,
    REJECTION_HIGH_PRICE:             BOTTLENECK_HIGH_PRICED_EXCLUSION,
    REJECTION_SECTOR_SUPPRESSION:     BOTTLENECK_SECTOR_SUPPRESSION,
    REJECTION_CONCENTRATION_THROTTLE: BOTTLENECK_CONCENTRATION_THROTTLE,
    REJECTION_INSUFFICIENT_CASH:      BOTTLENECK_INSUFFICIENT_BUYING_POWER,
    REJECTION_STALE_SIGNAL:           BOTTLENECK_STALE_SIGNAL_DECAY,
    REJECTION_TRUTH_CONFIDENCE:       BOTTLENECK_EXECUTION_INTEGRITY,
    REJECTION_RATE_LIMIT:             BOTTLENECK_EXECUTION_INTEGRITY,
    REJECTION_DUPLICATE:              BOTTLENECK_EXECUTION_INTEGRITY,
    REJECTION_EPOCH_BLOCKED:          BOTTLENECK_EXECUTION_INTEGRITY,
}

# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck → research category
# ─────────────────────────────────────────────────────────────────────────────

_BOTTLENECK_TO_RESEARCH: Dict[str, str] = {
    BOTTLENECK_ALLOC_CAP_PRESSURE:        RESEARCH_ALLOCATION,
    BOTTLENECK_IDLE_CASH_PERSISTENCE:     RESEARCH_CAPITAL_EFFICIENCY,
    BOTTLENECK_HIGH_PRICED_EXCLUSION:     RESEARCH_DEPLOYABILITY,
    BOTTLENECK_CONCENTRATION_THROTTLE:    RESEARCH_CONCENTRATION,
    BOTTLENECK_SECTOR_SUPPRESSION:        RESEARCH_ALLOCATION,
    BOTTLENECK_PREMATURE_EXIT:            RESEARCH_EXIT,
    BOTTLENECK_STALE_SIGNAL_DECAY:        RESEARCH_DEPLOYABILITY,
    BOTTLENECK_INSUFFICIENT_BUYING_POWER: RESEARCH_CAPITAL_EFFICIENCY,
    BOTTLENECK_EXECUTION_INTEGRITY:       RESEARCH_INTEGRITY,
    BOTTLENECK_NONE:                      RESEARCH_CAPITAL_EFFICIENCY,
}

# Trend direction constants
_TREND_IMPROVING         = "improving"
_TREND_DEGRADING         = "degrading"
_TREND_STABLE            = "stable"
_TREND_INSUFFICIENT_DATA = "insufficient_data"

# Secondary category signal threshold (fraction of total leakage)
_SECONDARY_CATEGORY_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakageSourceEntry:
    bottleneck_category: str
    source_reason: str
    estimated_missed_pnl: float
    skip_count: int
    fraction_of_skipped: float


@dataclass
class ResearchPrioritySummary:
    """
    Deterministic daily research-priority diagnostic.

    Byte-identical for the same set of inputs (excluding generated_at).
    """
    summary_id: str               # sha256(date + run_id)
    date: str                     # YYYY-MM-DD
    run_id: str
    generated_at: str             # ISO-8601 UTC (metadata only — not in hash)
    is_live: bool

    # ── Bottleneck identification ─────────────────────────────────────────────
    dominant_bottleneck: str          # one of BOTTLENECK_* constants
    dominant_bottleneck_source: str   # raw rejection_reason or derived indicator
    estimated_cagr_leakage_jpy: float # estimated missed PnL for dominant bottleneck

    # ── Signal / execution metrics ────────────────────────────────────────────
    total_signals: int
    executed_count: int
    skipped_count: int
    deployable_alpha_loss_rate: float  # skipped_pnl / (skipped + realized)
    capital_efficiency_score: float    # composite [0, 1]
    idle_cash_ratio: float             # mean(idle_cash / capital)

    # ── Top missed opportunity ────────────────────────────────────────────────
    largest_missed_pnl_symbol: str
    largest_missed_pnl_jpy: float
    largest_missed_pnl_reason: str

    # ── Trailing window comparison ────────────────────────────────────────────
    persistence_vs_5d: dict    # TrailingWindowMetrics dict (or empty {})
    persistence_vs_20d: dict   # TrailingWindowMetrics dict (or empty {})
    capital_efficiency_trend: str  # improving | degrading | stable | insufficient_data

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_level: str     # HIGH | MEDIUM | LOW
    confidence_basis: str     # human-readable rationale (deterministic)
    enrichment_coverage: float

    # ── Affected symbols / sectors ────────────────────────────────────────────
    affected_symbols: List[str]
    affected_sectors: List[str]

    # ── Research categories ───────────────────────────────────────────────────
    primary_research_category: str
    secondary_research_categories: List[str]

    # ── Leakage sources ───────────────────────────────────────────────────────
    leakage_sources: List[dict]    # List[LeakageSourceEntry as dict]

    # ── Execution integrity ───────────────────────────────────────────────────
    integrity_ok: bool
    integrity_warnings: List[str]

    # ── Executive summary (deterministic text) ────────────────────────────────
    executive_summary: str

    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def generate_research_priority_summary(
    date: str,
    run_id: str,
    is_live: bool,
    skipped_records: Optional[List[SkippedOpportunityRecord]] = None,
    alpha_snap: Optional[AlphaDiagnosticSnapshot] = None,
    leakage_summary: Optional[DailyLeakageSummary] = None,
    historical_leakage: Optional[List[DailyLeakageSummary]] = None,
    integrity_result=None,   # Optional[ExecutionIntegrityResult] — avoid circular import in type hint
    capital: float = 3_000_000.0,
) -> ResearchPrioritySummary:
    """
    Generate a deterministic daily research-priority summary.

    Inputs are consumed in priority order:
      alpha_snap > leakage_summary > skipped_records > empty/defaults

    All inputs are optional. Missing data → LOW confidence summary. Never raises.

    Args:
        date:             YYYY-MM-DD trading day.
        run_id:           Runtime run identifier.
        is_live:          True if live execution, False if dry-run.
        skipped_records:  Current-session SkippedOpportunityRecord list.
        alpha_snap:       AlphaDiagnosticSnapshot for the day (richest source).
        leakage_summary:  DailyLeakageSummary (fallback if no alpha_snap).
        historical_leakage: Past DailyLeakageSummary records (trailing windows).
        integrity_result: ExecutionIntegrityResult (for integrity warnings).
        capital:          Total capital (JPY) — denominates idle ratio.

    Returns:
        ResearchPrioritySummary (deterministic for fixed inputs).
    """
    summary_id = _make_id(date, run_id)
    generated_at = datetime.now(timezone.utc).isoformat()

    skipped = skipped_records or []

    # ── Extract core metrics (source priority: alpha_snap > leakage_summary > records) ──
    (
        total_signals, executed, skipped_count,
        alpha_loss_rate, eff_score, idle_cash_ratio,
        idle_cash_mean, idle_cash_peak,
        enrichment_cov, skipped_pnl, realized_pnl,
        raw_bottleneck, bottleneck_pnl,
        t5, t20,
        top_missed_list, excl_costs, restrictive_periods,
    ) = _extract_metrics(
        alpha_snap, leakage_summary, skipped, historical_leakage, capital
    )

    # ── Bottleneck classification ─────────────────────────────────────────────
    bottleneck_category = _classify_bottleneck(
        raw_bottleneck, idle_cash_ratio, idle_cash_peak,
        executed, skipped_count, capital, restrictive_periods,
    )

    # ── Leakage sources (structured breakdown) ────────────────────────────────
    leakage_sources = _build_leakage_sources(excl_costs, skipped_count)

    # ── Top missed opportunity ────────────────────────────────────────────────
    largest_missed = _get_largest_missed(top_missed_list)

    # ── Capital efficiency trend vs trailing window ────────────────────────────
    capital_efficiency_trend = _compute_trend(t5, eff_score)

    # ── Confidence level ──────────────────────────────────────────────────────
    confidence_level, confidence_basis = _compute_confidence(
        enrichment_cov, total_signals, skipped_count
    )

    # ── Affected symbols and sectors ─────────────────────────────────────────
    affected_symbols = _extract_affected_symbols(top_missed_list)
    affected_sectors = _extract_affected_sectors(skipped, top_missed_list)

    # ── Research categories ───────────────────────────────────────────────────
    primary_category = _BOTTLENECK_TO_RESEARCH.get(bottleneck_category, RESEARCH_CAPITAL_EFFICIENCY)
    secondary_categories = _build_secondary_categories(
        leakage_sources, primary_category,
        idle_cash_ratio, integrity_result,
    )

    # ── Integrity ─────────────────────────────────────────────────────────────
    integrity_ok, integrity_warnings = _extract_integrity(integrity_result)

    # ── Executive summary (deterministic text from data fields) ───────────────
    executive_summary = _build_executive_summary(
        bottleneck_category=bottleneck_category,
        bottleneck_pnl=bottleneck_pnl,
        eff_score=eff_score,
        t5=t5,
        total_signals=total_signals,
        skipped_count=skipped_count,
        executed=executed,
        largest_missed=largest_missed,
        confidence_level=confidence_level,
        primary_category=primary_category,
        is_live=is_live,
        integrity_ok=integrity_ok,
    )

    return ResearchPrioritySummary(
        summary_id=summary_id,
        date=date,
        run_id=run_id,
        generated_at=generated_at,
        is_live=is_live,
        dominant_bottleneck=bottleneck_category,
        dominant_bottleneck_source=raw_bottleneck,
        estimated_cagr_leakage_jpy=round(bottleneck_pnl, 2),
        total_signals=total_signals,
        executed_count=executed,
        skipped_count=skipped_count,
        deployable_alpha_loss_rate=round(alpha_loss_rate, 6),
        capital_efficiency_score=round(eff_score, 6),
        idle_cash_ratio=round(idle_cash_ratio, 6),
        largest_missed_pnl_symbol=largest_missed.get("symbol", ""),
        largest_missed_pnl_jpy=round(largest_missed.get("estimated_missed_pnl", 0.0) or 0.0, 2),
        largest_missed_pnl_reason=largest_missed.get("rejection_reason", ""),
        persistence_vs_5d=t5,
        persistence_vs_20d=t20,
        capital_efficiency_trend=capital_efficiency_trend,
        confidence_level=confidence_level,
        confidence_basis=confidence_basis,
        enrichment_coverage=round(enrichment_cov, 6),
        affected_symbols=affected_symbols,
        affected_sectors=affected_sectors,
        primary_research_category=primary_category,
        secondary_research_categories=secondary_categories,
        leakage_sources=leakage_sources,
        integrity_ok=integrity_ok,
        integrity_warnings=integrity_warnings,
        executive_summary=executive_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics extraction (source priority fan-out)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metrics(
    alpha_snap: Optional[AlphaDiagnosticSnapshot],
    leakage_summary: Optional[DailyLeakageSummary],
    skipped: List[SkippedOpportunityRecord],
    historical_leakage: Optional[List[DailyLeakageSummary]],
    capital: float,
) -> tuple:
    """Extract unified core metrics from best available source. Returns 18-tuple."""
    hist = historical_leakage or []

    if alpha_snap is not None:
        t5  = alpha_snap.trailing_5d  if isinstance(alpha_snap.trailing_5d, dict)  else {}
        t20 = alpha_snap.trailing_20d if isinstance(alpha_snap.trailing_20d, dict) else {}
        return (
            alpha_snap.total_signals_seen,
            alpha_snap.executed_count,
            alpha_snap.skipped_count,
            alpha_snap.deployable_alpha_loss_rate,
            alpha_snap.capital_efficiency_score,
            alpha_snap.idle_cash_ratio,
            alpha_snap.idle_capital_mean_jpy,
            alpha_snap.idle_capital_peak_jpy,
            alpha_snap.enrichment_coverage,
            alpha_snap.skipped_pnl_estimate,
            alpha_snap.realized_pnl_estimate,
            alpha_snap.dominant_bottleneck,     # raw_bottleneck
            alpha_snap.dominant_bottleneck_pnl_jpy,
            t5,
            t20,
            alpha_snap.top_missed_winners,
            alpha_snap.largest_exclusion_costs,
            alpha_snap.most_restrictive_allocator_periods,
        )

    if leakage_summary is not None:
        skipped_pnl  = leakage_summary.skipped_pnl_estimate
        realized_pnl = leakage_summary.realized_pnl_estimate
        total_pnl    = skipped_pnl + realized_pnl
        loss_rate    = skipped_pnl / total_pnl if total_pnl > 0 else 0.0

        total = leakage_summary.total_signals_seen
        executed = leakage_summary.executed_count
        skipped_cnt = leakage_summary.skipped_count
        util = executed / total if total > 0 else 0.0
        conv = leakage_summary.deployable_alpha_conversion_rate
        idle_ratio = min(1.0, leakage_summary.idle_capital_mean_jpy / capital) if capital > 0 else 0.0
        eff = round(util * 0.40 + conv * 0.40 + max(0.0, 1.0 - idle_ratio) * 0.20, 6)

        raw_bottleneck, bottleneck_pnl = _bottleneck_from_breakdown(
            leakage_summary.exclusion_breakdown
        )
        t5  = _compute_trailing_dict_from_history(5,  hist, util, conv, eff, loss_rate, capital)
        t20 = _compute_trailing_dict_from_history(20, hist, util, conv, eff, loss_rate, capital)

        # Convert leakage exclusion breakdown to excl_costs format
        excl_costs = [
            {
                "reason": e.get("reason", ""),
                "total_missed_pnl": e.get("estimated_missed_pnl", 0.0),
                "count": e.get("count", 0),
                "fraction": e.get("fraction", 0.0),
            }
            for e in leakage_summary.exclusion_breakdown
        ]

        return (
            total, executed, skipped_cnt,
            loss_rate, eff, idle_ratio,
            leakage_summary.idle_capital_mean_jpy,
            leakage_summary.idle_capital_peak_jpy,
            leakage_summary.enrichment_coverage,
            skipped_pnl, realized_pnl,
            raw_bottleneck, bottleneck_pnl,
            t5, t20,
            leakage_summary.top_missed_by_pnl,
            excl_costs,
            [],  # no restrictive_periods from leakage_summary
        )

    # Fallback: compute directly from skipped_records
    skipped_cnt = len(skipped)
    enriched    = [r for r in skipped if r.enrichment_status == ENRICHMENT_ENRICHED]
    skipped_pnl = sum(r.estimated_missed_pnl for r in enriched if r.estimated_missed_pnl is not None)
    total_pnl   = skipped_pnl   # realized unknown
    loss_rate   = 1.0 if total_pnl > 0 else 0.0
    enrich_cov  = len(enriched) / skipped_cnt if skipped_cnt > 0 else 0.0

    cash_vals   = [r.available_cash for r in skipped if r.available_cash > 0]
    idle_mean   = sum(cash_vals) / len(cash_vals) if cash_vals else 0.0
    idle_peak   = max(cash_vals) if cash_vals else 0.0
    idle_ratio  = min(1.0, idle_mean / capital) if capital > 0 else 0.0

    raw_bottleneck, bottleneck_pnl = _bottleneck_from_records(skipped, enriched)
    excl_costs = _excl_costs_from_records(skipped, skipped_cnt)

    top_missed = sorted(
        (r for r in enriched if r.estimated_missed_pnl is not None),
        key=lambda r: abs(r.estimated_missed_pnl or 0),
        reverse=True,
    )[:5]
    top_missed_list = [
        {
            "symbol": r.symbol,
            "estimated_missed_pnl": r.estimated_missed_pnl or 0.0,
            "rejection_reason": r.rejection_reason,
            "forward_return_5d": r.forward_return_5d,
            "signal_strength": r.signal_strength,
            "timestamp": r.timestamp,
        }
        for r in top_missed
    ]

    t5  = _compute_trailing_dict_from_history(5,  hist, 0.0, 0.0, 0.0, loss_rate, capital)
    t20 = _compute_trailing_dict_from_history(20, hist, 0.0, 0.0, 0.0, loss_rate, capital)

    return (
        skipped_cnt, 0, skipped_cnt,
        loss_rate, 0.0, idle_ratio,
        idle_mean, idle_peak,
        enrich_cov, skipped_pnl, 0.0,
        raw_bottleneck, bottleneck_pnl,
        t5, t20,
        top_missed_list, excl_costs,
        [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_bottleneck(
    raw_bottleneck: str,
    idle_cash_ratio: float,
    idle_cash_peak: float,
    executed_count: int,
    skipped_count: int,
    capital: float,
    restrictive_periods: List[dict],
) -> str:
    """
    Map raw rejection-based bottleneck to supported bottleneck category.

    Derived bottlenecks:
      premature_exit_behavior — high idle cash + capital-blocked signals + restrictor periods
      idle_cash_persistence   — high idle cash ratio when another bottleneck dominates
    """
    base = _REASON_TO_BOTTLENECK.get(raw_bottleneck, BOTTLENECK_NONE)

    # Premature exit: alloc/concentration blocking while cash is structurally idle
    if (
        base in (BOTTLENECK_ALLOC_CAP_PRESSURE, BOTTLENECK_CONCENTRATION_THROTTLE)
        and idle_cash_ratio > _PREMATURE_EXIT_THRESHOLD
        and len(restrictive_periods) > 0
        and executed_count > 0
    ):
        return BOTTLENECK_PREMATURE_EXIT

    # Idle cash persistence: cash not being deployed despite capacity
    if (
        idle_cash_ratio > _IDLE_CASH_DOMINANT_THRESHOLD
        and base not in (BOTTLENECK_INSUFFICIENT_BUYING_POWER, BOTTLENECK_NONE)
        and skipped_count > executed_count
    ):
        return BOTTLENECK_IDLE_CASH_PERSISTENCE

    return base


def _bottleneck_from_breakdown(breakdown: List[dict]) -> Tuple[str, float]:
    """Derive dominant bottleneck from exclusion_breakdown list (DailyLeakageSummary format)."""
    if not breakdown:
        return ("none", 0.0)
    # Sort by estimated_missed_pnl desc, then count desc, then reason asc for stability
    sorted_bd = sorted(
        breakdown,
        key=lambda e: (-abs(e.get("estimated_missed_pnl", 0.0)), -e.get("count", 0), e.get("reason", "")),
    )
    top = sorted_bd[0]
    return (top.get("reason", "none"), abs(top.get("estimated_missed_pnl", 0.0)))


def _bottleneck_from_records(
    records: List[SkippedOpportunityRecord],
    enriched: List[SkippedOpportunityRecord],
) -> Tuple[str, float]:
    """Derive dominant bottleneck directly from skipped records."""
    if not records:
        return ("none", 0.0)
    reason_pnl: Dict[str, float] = {}
    for r in enriched:
        if r.estimated_missed_pnl is not None:
            reason_pnl[r.rejection_reason] = (
                reason_pnl.get(r.rejection_reason, 0.0) + r.estimated_missed_pnl
            )
    if reason_pnl:
        dominant = max(reason_pnl, key=lambda k: (abs(reason_pnl[k]), k))
        return (dominant, abs(reason_pnl[dominant]))
    count: Dict[str, int] = {}
    for r in records:
        count[r.rejection_reason] = count.get(r.rejection_reason, 0) + 1
    dominant = max(count, key=lambda k: (count[k], k))
    return (dominant, 0.0)


def _excl_costs_from_records(
    records: List[SkippedOpportunityRecord],
    skipped_count: int,
) -> List[dict]:
    """Build exclusion costs list from raw skipped records."""
    reason_pnl: Dict[str, float] = {}
    reason_cnt: Dict[str, int]   = {}
    for r in records:
        reason_cnt[r.rejection_reason] = reason_cnt.get(r.rejection_reason, 0) + 1
        if r.estimated_missed_pnl is not None:
            reason_pnl[r.rejection_reason] = (
                reason_pnl.get(r.rejection_reason, 0.0) + r.estimated_missed_pnl
            )
    all_reasons = sorted(set(list(reason_cnt) + list(reason_pnl)))
    return [
        {
            "reason": reason,
            "total_missed_pnl": round(reason_pnl.get(reason, 0.0), 2),
            "count": reason_cnt.get(reason, 0),
            "fraction": round(reason_cnt.get(reason, 0) / skipped_count, 6)
            if skipped_count > 0 else 0.0,
        }
        for reason in all_reasons
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Trailing window helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trailing_dict_from_history(
    window: int,
    hist: List[DailyLeakageSummary],
    current_util: float,
    current_conv: float,
    current_eff: float,
    current_loss: float,
    capital: float,
) -> dict:
    """Compute TrailingWindowMetrics-compatible dict from historical DailyLeakageSummary list."""
    sorted_hist = sorted(hist, key=lambda s: s.date, reverse=True)[:window]
    n = len(sorted_hist)
    if n == 0:
        return {
            "window_days": window,
            "sample_count": 0,
            "mean_util_ratio": 0.0,
            "mean_conv_rate": 0.0,
            "mean_idle_cash_jpy": 0.0,
            "mean_skipped_pnl": 0.0,
            "mean_capital_efficiency_score": 0.0,
            "mean_deployable_alpha_loss_rate": 0.0,
            "delta_util_ratio": round(current_util, 6),
            "delta_conv_rate": round(current_conv, 6),
            "delta_capital_efficiency_score": round(current_eff, 6),
            "delta_deployable_alpha_loss_rate": round(current_loss, 6),
        }
    mean_util = sum(s.opportunity_utilization_ratio for s in sorted_hist) / n
    mean_conv = sum(s.deployable_alpha_conversion_rate for s in sorted_hist) / n
    mean_idle = sum(s.idle_capital_mean_jpy for s in sorted_hist) / n
    mean_pnl  = sum(s.skipped_pnl_estimate for s in sorted_hist) / n

    def _s_eff(s: DailyLeakageSummary) -> float:
        total = s.executed_count + s.skipped_count
        u = s.executed_count / total if total > 0 else 0.0
        ir = min(1.0, s.idle_capital_mean_jpy / capital) if capital > 0 else 0.0
        return u * 0.40 + s.deployable_alpha_conversion_rate * 0.40 + max(0.0, 1.0 - ir) * 0.20

    def _s_loss(s: DailyLeakageSummary) -> float:
        tot = s.skipped_pnl_estimate + s.realized_pnl_estimate
        return s.skipped_pnl_estimate / tot if tot > 0 else 0.0

    mean_eff  = sum(_s_eff(s)  for s in sorted_hist) / n
    mean_loss = sum(_s_loss(s) for s in sorted_hist) / n

    return {
        "window_days": window,
        "sample_count": n,
        "mean_util_ratio": round(mean_util, 6),
        "mean_conv_rate": round(mean_conv, 6),
        "mean_idle_cash_jpy": round(mean_idle, 2),
        "mean_skipped_pnl": round(mean_pnl, 2),
        "mean_capital_efficiency_score": round(mean_eff, 6),
        "mean_deployable_alpha_loss_rate": round(mean_loss, 6),
        "delta_util_ratio": round(current_util - mean_util, 6),
        "delta_conv_rate": round(current_conv - mean_conv, 6),
        "delta_capital_efficiency_score": round(current_eff - mean_eff, 6),
        "delta_deployable_alpha_loss_rate": round(current_loss - mean_loss, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trend / confidence / classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trend(t5: dict, current_eff: float) -> str:
    """Classify capital efficiency trend vs trailing_5d."""
    if not t5 or t5.get("sample_count", 0) == 0:
        return _TREND_INSUFFICIENT_DATA
    delta = t5.get("delta_capital_efficiency_score", 0.0)
    if delta > 0.02:
        return _TREND_IMPROVING
    if delta < -0.02:
        return _TREND_DEGRADING
    return _TREND_STABLE


def _compute_confidence(
    enrichment_coverage: float,
    total_signals: int,
    skipped_count: int,
) -> Tuple[str, str]:
    """Return (confidence_level, confidence_basis) deterministically."""
    if enrichment_coverage >= 0.70 and total_signals >= 5 and skipped_count >= 3:
        return (
            CONFIDENCE_HIGH,
            f"enrichment={enrichment_coverage:.0%} signals={total_signals} skipped={skipped_count}",
        )
    if enrichment_coverage >= 0.40 or (total_signals >= 3 and skipped_count >= 1):
        return (
            CONFIDENCE_MEDIUM,
            f"enrichment={enrichment_coverage:.0%} signals={total_signals} skipped={skipped_count}",
        )
    return (
        CONFIDENCE_LOW,
        f"enrichment={enrichment_coverage:.0%} signals={total_signals} skipped={skipped_count} — insufficient data",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Leakage source / symbol / sector extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_leakage_sources(excl_costs: List[dict], skipped_count: int) -> List[dict]:
    """Convert exclusion cost entries to LeakageSourceEntry dicts, sorted by PnL desc."""
    if not excl_costs:
        return []
    sorted_costs = sorted(
        excl_costs,
        key=lambda e: (-abs(e.get("total_missed_pnl", 0.0)), e.get("reason", "")),
    )
    return [
        asdict(LeakageSourceEntry(
            bottleneck_category=_REASON_TO_BOTTLENECK.get(e.get("reason", ""), BOTTLENECK_NONE),
            source_reason=e.get("reason", ""),
            estimated_missed_pnl=round(e.get("total_missed_pnl", 0.0), 2),
            skip_count=e.get("count", 0),
            fraction_of_skipped=round(e.get("fraction", 0.0), 6),
        ))
        for e in sorted_costs
    ]


def _get_largest_missed(top_missed_list: List[dict]) -> dict:
    """Return the record with the largest absolute estimated_missed_pnl."""
    if not top_missed_list:
        return {}
    return max(
        top_missed_list,
        key=lambda e: abs(e.get("estimated_missed_pnl", 0.0) or 0),
    )


def _extract_affected_symbols(top_missed_list: List[dict]) -> List[str]:
    """Deterministic sorted list of symbols from top missed opportunities."""
    seen = []
    for e in top_missed_list:
        sym = e.get("symbol", "")
        if sym and sym not in seen:
            seen.append(sym)
    return sorted(seen)


def _extract_affected_sectors(
    skipped: List[SkippedOpportunityRecord],
    top_missed_list: List[dict],
) -> List[str]:
    """Deterministic sorted list of affected sectors from skipped records."""
    sectors: set = set()
    for r in skipped:
        if r.sector_state and r.sector_state.strip():
            sectors.add(r.sector_state.strip())
    return sorted(sectors)


def _build_secondary_categories(
    leakage_sources: List[dict],
    primary_category: str,
    idle_cash_ratio: float = 0.0,
    integrity_result=None,
) -> List[str]:
    """
    Build ordered list of secondary research categories.

    A secondary category is added when:
      - A leakage source accounts for >= 15% of skipped signals
      - Its research category differs from the primary
    Additionally:
      - RESEARCH_CAPITAL_EFFICIENCY if idle_cash_ratio > 0.25 and not already primary/secondary
      - RESEARCH_INTEGRITY if integrity has violations and not already included
    """
    seen: List[str] = [primary_category]
    result: List[str] = []

    for src in leakage_sources:
        frac = src.get("fraction_of_skipped", 0.0)
        if frac < _SECONDARY_CATEGORY_THRESHOLD:
            continue
        cat = _BOTTLENECK_TO_RESEARCH.get(
            src.get("bottleneck_category", BOTTLENECK_NONE), RESEARCH_CAPITAL_EFFICIENCY
        )
        if cat not in seen:
            seen.append(cat)
            result.append(cat)

    if idle_cash_ratio > _IDLE_CASH_PERSISTENCE_THRESHOLD:
        if RESEARCH_CAPITAL_EFFICIENCY not in seen:
            seen.append(RESEARCH_CAPITAL_EFFICIENCY)
            result.append(RESEARCH_CAPITAL_EFFICIENCY)

    if integrity_result is not None:
        violations = getattr(integrity_result, "integrity_violations", []) or []
        if violations and RESEARCH_INTEGRITY not in seen:
            seen.append(RESEARCH_INTEGRITY)
            result.append(RESEARCH_INTEGRITY)

    return result


def _extract_integrity(integrity_result) -> Tuple[bool, List[str]]:
    """Extract (integrity_ok, warning_strings) from ExecutionIntegrityResult or None."""
    if integrity_result is None:
        return (True, [])
    ok = bool(getattr(integrity_result, "execution_allowed", True))
    violations = getattr(integrity_result, "integrity_violations", []) or []
    warnings = [
        v.get("detail", str(v)) if isinstance(v, dict) else str(v)
        for v in violations
    ]
    return (ok, warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Executive summary (deterministic text generation)
# ─────────────────────────────────────────────────────────────────────────────

def _build_executive_summary(
    bottleneck_category: str,
    bottleneck_pnl: float,
    eff_score: float,
    t5: dict,
    total_signals: int,
    skipped_count: int,
    executed: int,
    largest_missed: dict,
    confidence_level: str,
    primary_category: str,
    is_live: bool,
    integrity_ok: bool,
) -> str:
    """Build deterministic one-paragraph executive summary from data fields."""
    mode_str = "live" if is_live else "dry-run"
    delta5 = t5.get("delta_capital_efficiency_score", 0.0)
    trend_str = (
        f"Δ5d {delta5:+.3f}"
        if t5.get("sample_count", 0) > 0
        else "no trailing data"
    )

    missed_str = ""
    if largest_missed:
        sym = largest_missed.get("symbol", "")
        pnl = largest_missed.get("estimated_missed_pnl", 0.0) or 0.0
        reason = largest_missed.get("rejection_reason", "")
        if sym:
            missed_str = f" Largest missed opportunity: {sym} (¥{pnl:,.0f}, {reason})."

    integrity_str = "" if integrity_ok else " INTEGRITY VIOLATION detected."

    return (
        f"[{mode_str}] Dominant bottleneck: {bottleneck_category}"
        f" (est. ¥{bottleneck_pnl:,.0f} leakage)."
        f" Capital efficiency: {eff_score:.3f} ({trend_str})."
        f" {executed}/{total_signals} signals executed, {skipped_count} skipped."
        f"{missed_str}"
        f" Confidence: {confidence_level}."
        f" Priority research domain: {primary_category}."
        f"{integrity_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report (deterministic: same snapshot → byte-identical output)
# ─────────────────────────────────────────────────────────────────────────────

def generate_research_priority_markdown(summary: ResearchPrioritySummary) -> str:
    """
    Generate a deterministic markdown research-priority report.

    Content is derived solely from summary fields — byte-identical for
    the same ResearchPrioritySummary input (generated_at excluded from content).
    """
    lines: List[str] = []
    a = lines.append

    a(f"# Daily Research Priority Summary — {summary.date}")
    a(f"")
    a(f"| Field | Value |")
    a(f"|---|---|")
    a(f"| summary_id | `{summary.summary_id}` |")
    a(f"| run_id | `{summary.run_id}` |")
    a(f"| generated_at | {summary.generated_at} |")
    a(f"| mode | {'LIVE' if summary.is_live else 'DRY-RUN'} |")
    a(f"")

    # ── Executive Summary ─────────────────────────────────────────────────────
    a(f"## Executive Summary")
    a(f"")
    a(summary.executive_summary)
    a(f"")

    # ── Dominant Bottleneck ───────────────────────────────────────────────────
    a(f"## Dominant Bottleneck")
    a(f"")
    a(f"| Field | Value |")
    a(f"|---|---|")
    a(f"| dominant_bottleneck | **{summary.dominant_bottleneck}** |")
    a(f"| source_reason | `{summary.dominant_bottleneck_source}` |")
    a(f"| estimated_leakage_jpy | ¥{summary.estimated_cagr_leakage_jpy:,.0f} |")
    a(f"| idle_cash_ratio | {summary.idle_cash_ratio:.3f} |")
    a(f"| capital_efficiency_score | {summary.capital_efficiency_score:.4f} |")
    a(f"| deployable_alpha_loss_rate | {summary.deployable_alpha_loss_rate:.4f} |")
    a(f"| confidence | **{summary.confidence_level}** |")
    a(f"| confidence_basis | {summary.confidence_basis} |")
    a(f"")

    # ── Estimated CAGR Leakage Sources ────────────────────────────────────────
    a(f"## Estimated CAGR Leakage Sources")
    a(f"")
    if summary.leakage_sources:
        a(f"| Bottleneck Category | Source Reason | Skip Count | Fraction | Missed PnL (JPY) |")
        a(f"|---|---|---|---|---|")
        for src in summary.leakage_sources:
            a(
                f"| {src['bottleneck_category']}"
                f" | {src['source_reason']}"
                f" | {src['skip_count']}"
                f" | {src['fraction_of_skipped']:.1%}"
                f" | ¥{src['estimated_missed_pnl']:,.0f} |"
            )
    else:
        a(f"_No leakage records._")
    a(f"")

    # ── Top Missed Opportunities ──────────────────────────────────────────────
    a(f"## Top Missed Opportunities")
    a(f"")
    a(f"| Field | Value |")
    a(f"|---|---|")
    if summary.largest_missed_pnl_symbol:
        a(f"| largest_missed_symbol | **{summary.largest_missed_pnl_symbol}** |")
        a(f"| largest_missed_pnl_jpy | ¥{summary.largest_missed_pnl_jpy:,.0f} |")
        a(f"| rejection_reason | {summary.largest_missed_pnl_reason} |")
    else:
        a(f"| largest_missed_symbol | _none_ |")
    a(f"")
    if summary.affected_symbols:
        a(f"**Affected symbols:** {', '.join(summary.affected_symbols)}")
        a(f"")
    if summary.affected_sectors:
        a(f"**Affected sectors:** {', '.join(summary.affected_sectors)}")
        a(f"")

    # ── Capital Efficiency Trends ─────────────────────────────────────────────
    a(f"## Capital Efficiency Trends")
    a(f"")
    a(f"| Metric | Today | Δ vs 5d | Δ vs 20d |")
    a(f"|---|---|---|---|")

    t5  = summary.persistence_vs_5d  or {}
    t20 = summary.persistence_vs_20d or {}

    def _d5(key: str) -> str:
        v = t5.get(key, 0.0)
        return f"{v:+.4f}" if (t5.get("sample_count", 0) > 0 and v != 0.0) else "—"

    def _d20(key: str) -> str:
        v = t20.get(key, 0.0)
        return f"{v:+.4f}" if (t20.get("sample_count", 0) > 0 and v != 0.0) else "—"

    a(f"| capital_efficiency_score | {summary.capital_efficiency_score:.4f}"
      f" | {_d5('delta_capital_efficiency_score')}"
      f" | {_d20('delta_capital_efficiency_score')} |")
    a(f"| deployable_alpha_loss_rate | {summary.deployable_alpha_loss_rate:.4f}"
      f" | {_d5('delta_deployable_alpha_loss_rate')}"
      f" | {_d20('delta_deployable_alpha_loss_rate')} |")
    a(f"| opportunity_utilization | {(summary.executed_count / summary.total_signals):.4f}"
      if summary.total_signals > 0 else
      f"| opportunity_utilization | — | — | — |")
    a(f"")
    a(f"**Trend vs trailing_5d:** {summary.capital_efficiency_trend}")
    a(f"")
    a(f"| Window | Sample N | Mean Efficiency | Mean Loss Rate |")
    a(f"|---|---|---|---|")
    a(f"| 5d  | {t5.get('sample_count', 0)}"
      f" | {t5.get('mean_capital_efficiency_score', 0.0):.4f}"
      f" | {t5.get('mean_deployable_alpha_loss_rate', 0.0):.4f} |")
    a(f"| 20d | {t20.get('sample_count', 0)}"
      f" | {t20.get('mean_capital_efficiency_score', 0.0):.4f}"
      f" | {t20.get('mean_deployable_alpha_loss_rate', 0.0):.4f} |")
    a(f"")

    # ── Signal Counts ─────────────────────────────────────────────────────────
    a(f"## Signal Counts")
    a(f"")
    a(f"| | Count |")
    a(f"|---|---|")
    a(f"| total_signals | {summary.total_signals} |")
    a(f"| executed | {summary.executed_count} |")
    a(f"| skipped | {summary.skipped_count} |")
    a(f"| enrichment_coverage | {summary.enrichment_coverage:.1%} |")
    a(f"")

    # ── Risk / Integrity Warnings ─────────────────────────────────────────────
    a(f"## Risk / Integrity Warnings")
    a(f"")
    if not summary.integrity_ok:
        a(f"> **EXECUTION INTEGRITY VIOLATION**")
        a(f"")
    if summary.integrity_warnings:
        for w in summary.integrity_warnings:
            a(f"- {w}")
    else:
        a(f"_No integrity warnings._")
    a(f"")

    # ── Research Priority For Next Session ────────────────────────────────────
    a(f"## Research Priority For Next Session")
    a(f"")
    a(f"**Primary:** `{summary.primary_research_category}`")
    a(f"")
    if summary.secondary_research_categories:
        a(f"**Secondary:**")
        for cat in summary.secondary_research_categories:
            a(f"- `{cat}`")
    else:
        a(f"**Secondary:** _none_")
    a(f"")

    # ── Bottleneck context ────────────────────────────────────────────────────
    a(f"### Bottleneck Rationale")
    a(f"")
    a(_bottleneck_rationale(summary.dominant_bottleneck))
    a(f"")

    return "\n".join(lines)


def _bottleneck_rationale(bottleneck: str) -> str:
    """Return deterministic human-readable rationale for the bottleneck category."""
    rationales = {
        BOTTLENECK_ALLOC_CAP_PRESSURE: (
            "Allocation cap (effective-N / concentration cap) is zeroing order quantities. "
            "Research: investigate whether position limits are overly restrictive relative to available capital."
        ),
        BOTTLENECK_IDLE_CASH_PERSISTENCE: (
            "Capital is structurally idle — signals are being skipped despite available cash. "
            "Research: examine whether allocation policy is consistently blocking deployment."
        ),
        BOTTLENECK_HIGH_PRICED_EXCLUSION: (
            "High-priced symbols are being excluded because single-unit cost exceeds allocation limits. "
            "Research: evaluate whether universe composition or position sizing needs adjustment."
        ),
        BOTTLENECK_CONCENTRATION_THROTTLE: (
            "Portfolio concentration throttle is suppressing order quantities. "
            "Research: assess whether concentration governance is calibrated correctly for current capital."
        ),
        BOTTLENECK_SECTOR_SUPPRESSION: (
            "Sector filter is blocking signals. "
            "Research: review sector suppression logic for potential over-restriction."
        ),
        BOTTLENECK_PREMATURE_EXIT: (
            "Existing positions appear to be holding capital hostage — alloc_cap is zero "
            "while cash is idle, suggesting exit criteria may be too permissive. "
            "Research: review exit logic and Turtle 55d exit calibration."
        ),
        BOTTLENECK_STALE_SIGNAL_DECAY: (
            "Stale signals are being rejected at the data freshness gate. "
            "Research: investigate signal pipeline latency and data delivery timing."
        ),
        BOTTLENECK_INSUFFICIENT_BUYING_POWER: (
            "Insufficient available cash is blocking execution. "
            "Research: review capital deployment pace and position sizing relative to available capital."
        ),
        BOTTLENECK_EXECUTION_INTEGRITY: (
            "Execution integrity violations (duplicate, epoch-blocked, truth-confidence, rate-limit). "
            "Research: review execution governance, rate limiting, and truth-layer confidence calibration."
        ),
        BOTTLENECK_NONE: (
            "No dominant bottleneck identified. Low signal volume or no skipped records. "
            "Research: ensure signal tracking and analytics pipelines are fully wired."
        ),
    }
    return rationales.get(bottleneck, f"Unknown bottleneck category: {bottleneck}")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_research_priority_summary(summary: ResearchPrioritySummary) -> None:
    """Emit structured log for research priority summary."""
    logger.info(
        "[RESEARCH_PRI] date=%s bottleneck=%s leakage=¥%.0f eff=%.3f loss=%.3f conf=%s",
        summary.date,
        summary.dominant_bottleneck,
        summary.estimated_cagr_leakage_jpy,
        summary.capital_efficiency_score,
        summary.deployable_alpha_loss_rate,
        summary.confidence_level,
    )
    logger.info(
        "[RESEARCH_PRI] signals=%d executed=%d skipped=%d trend=%s primary=%s",
        summary.total_signals,
        summary.executed_count,
        summary.skipped_count,
        summary.capital_efficiency_trend,
        summary.primary_research_category,
    )
    if summary.largest_missed_pnl_symbol:
        logger.info(
            "[RESEARCH_PRI] largest_missed=%s ¥%.0f reason=%s",
            summary.largest_missed_pnl_symbol,
            summary.largest_missed_pnl_jpy,
            summary.largest_missed_pnl_reason,
        )
    if not summary.integrity_ok:
        for w in summary.integrity_warnings:
            logger.error("[RESEARCH_PRI] INTEGRITY: %s", w)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_research_priority_summary(summary: ResearchPrioritySummary, path: Path) -> None:
    """Append-only JSONL persistence. Fail-open — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[RESEARCH_PRI] JSONL append failed (%s)", exc)


def write_research_priority_report(summary: ResearchPrioritySummary, report_dir: Path) -> Optional[Path]:
    """
    Write deterministic markdown report to report_dir/daily_research_summary_YYYYMMDD.md.

    Appends if file exists (multiple runs per day). Fail-open — returns None on error.
    """
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        date_compact = summary.date.replace("-", "")
        report_path = report_dir / f"daily_research_summary_{date_compact}.md"
        md = generate_research_priority_markdown(summary)
        separator = "\n\n---\n\n" if report_path.exists() else ""
        with report_path.open("a", encoding="utf-8") as f:
            f.write(separator + md)
        return report_path
    except Exception as exc:
        logger.warning("[RESEARCH_PRI] markdown write failed (%s)", exc)
        return None


def load_research_priority_summaries(path: Path) -> List[ResearchPrioritySummary]:
    """Load all summaries from JSONL. Corrupted lines are skipped."""
    if not path.exists():
        return []
    result: List[ResearchPrioritySummary] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result.append(_parse_summary(json.loads(line)))
        except Exception as exc:
            logger.warning("[RESEARCH_PRI] parse error: %s", exc)
    return result


def get_summary_for_date(path: Path, date: str) -> Optional[ResearchPrioritySummary]:
    summaries = [s for s in load_research_priority_summaries(path) if s.date == date]
    return summaries[-1] if summaries else None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for integration into run_live_signal.py
# ─────────────────────────────────────────────────────────────────────────────

def run_research_priority_summary(
    date: str,
    run_id: str,
    is_live: bool,
    summary_jsonl_path: Path,
    report_dir: Path,
    skipped_records: Optional[List[SkippedOpportunityRecord]] = None,
    alpha_snap: Optional[AlphaDiagnosticSnapshot] = None,
    leakage_summary: Optional[DailyLeakageSummary] = None,
    historical_leakage: Optional[List[DailyLeakageSummary]] = None,
    integrity_result=None,
    capital: float = 3_000_000.0,
) -> Tuple[Optional[ResearchPrioritySummary], Optional[Path]]:
    """
    Generate, log, persist, and report research priority summary.

    Fail-open: any error returns (None, None) without raising.

    Returns:
        (summary, report_path) — both None on failure.
    """
    try:
        summary = generate_research_priority_summary(
            date=date,
            run_id=run_id,
            is_live=is_live,
            skipped_records=skipped_records,
            alpha_snap=alpha_snap,
            leakage_summary=leakage_summary,
            historical_leakage=historical_leakage,
            integrity_result=integrity_result,
            capital=capital,
        )
        log_research_priority_summary(summary)
        append_research_priority_summary(summary, summary_jsonl_path)
        report_path = write_research_priority_report(summary, report_dir)
        return (summary, report_path)
    except Exception as exc:
        logger.warning("[RESEARCH_PRI] run failed (%s)", exc)
        return (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_id(date: str, run_id: str) -> str:
    payload = json.dumps({"date": date, "run_id": run_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_summary(d: dict) -> ResearchPrioritySummary:
    return ResearchPrioritySummary(
        summary_id=d.get("summary_id", ""),
        date=d.get("date", ""),
        run_id=d.get("run_id", ""),
        generated_at=d.get("generated_at", ""),
        is_live=bool(d.get("is_live", False)),
        dominant_bottleneck=d.get("dominant_bottleneck", BOTTLENECK_NONE),
        dominant_bottleneck_source=d.get("dominant_bottleneck_source", ""),
        estimated_cagr_leakage_jpy=float(d.get("estimated_cagr_leakage_jpy", 0.0)),
        total_signals=int(d.get("total_signals", 0)),
        executed_count=int(d.get("executed_count", 0)),
        skipped_count=int(d.get("skipped_count", 0)),
        deployable_alpha_loss_rate=float(d.get("deployable_alpha_loss_rate", 0.0)),
        capital_efficiency_score=float(d.get("capital_efficiency_score", 0.0)),
        idle_cash_ratio=float(d.get("idle_cash_ratio", 0.0)),
        largest_missed_pnl_symbol=d.get("largest_missed_pnl_symbol", ""),
        largest_missed_pnl_jpy=float(d.get("largest_missed_pnl_jpy", 0.0)),
        largest_missed_pnl_reason=d.get("largest_missed_pnl_reason", ""),
        persistence_vs_5d=d.get("persistence_vs_5d", {}),
        persistence_vs_20d=d.get("persistence_vs_20d", {}),
        capital_efficiency_trend=d.get("capital_efficiency_trend", _TREND_INSUFFICIENT_DATA),
        confidence_level=d.get("confidence_level", CONFIDENCE_LOW),
        confidence_basis=d.get("confidence_basis", ""),
        enrichment_coverage=float(d.get("enrichment_coverage", 0.0)),
        affected_symbols=list(d.get("affected_symbols", [])),
        affected_sectors=list(d.get("affected_sectors", [])),
        primary_research_category=d.get("primary_research_category", RESEARCH_CAPITAL_EFFICIENCY),
        secondary_research_categories=list(d.get("secondary_research_categories", [])),
        leakage_sources=list(d.get("leakage_sources", [])),
        integrity_ok=bool(d.get("integrity_ok", True)),
        integrity_warnings=list(d.get("integrity_warnings", [])),
        executive_summary=d.get("executive_summary", ""),
        schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
    )
