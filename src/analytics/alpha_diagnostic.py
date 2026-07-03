"""
alpha_diagnostic.py — Deterministic daily deployable-alpha diagnostic summaries

Compresses runtime analytics into actionable capital-efficiency diagnostics.

Diagnostics only — no allocator optimization, no auto-tuning, no recommendations,
no execution changes, no portfolio optimization.

Generates deterministic daily summaries for:
  - skipped opportunity attribution
  - alloc_cap pressure
  - idle cash persistence
  - high-priced exclusion impact
  - sector suppression impact
  - concentration throttle impact
  - top missed convexity events
  - opportunity utilization ratio
  - deployable vs realized pnl
  - capital deployment efficiency

Rankings:
  - top missed winners (by forward_return_5d)
  - largest exclusion costs (by summed missed PnL per reason)
  - highest idle-cash periods (by available_cash)
  - most restrictive allocator periods (alloc_cap=0 + highest cash)

Derived metrics:
  - deployable_alpha_loss_rate      = skipped_pnl / (skipped + realized)
  - capital_efficiency_score        = weighted composite of util/conv/idle
  - convexity_capture_rate          = 1 - missed_convexity / total_signals
  - missed_runner_ratio             = missed_runners / total_signals

Daily comparisons vs trailing_5d and trailing_20d.

Dominant bottleneck identification — largest deployable-alpha leakage source.

Outputs:
  - deterministic markdown reports (same snapshot → byte-identical)
  - structured summary tables
  - replay-compatible append-only JSONL
  - structured logging

Requirements:
  - append-only outputs
  - replay reproducibility (deterministic for fixed inputs)
  - read-only analytics — no live-state mutation
  - fail-open only
  - Windows compatible
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.analytics.skipped_opportunity_analytics import (
    ENRICHMENT_ENRICHED,
    REJECTION_ALLOC_CAP,
    REJECTION_HIGH_PRICE,
    REJECTION_SECTOR_SUPPRESSION,
    REJECTION_CONCENTRATION_THROTTLE,
    SkippedOpportunityRecord,
)
from src.analytics.daily_leakage_summary import DailyLeakageSummary

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

# Thresholds for classifying "convexity event" and "runner"
CONVEXITY_THRESHOLD: float = 0.03   # fr5d > 3%  → missed convexity event
RUNNER_THRESHOLD: float = 0.05      # fr5d > 5%  → missed runner

# capital_efficiency_score weights (must sum to 1.0)
_W_UTIL = 0.40
_W_CONV = 0.40
_W_IDLE = 0.20

TOP_N_DEFAULT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrailingWindowMetrics:
    window_days: int
    sample_count: int                        # actual days of data
    mean_util_ratio: float
    mean_conv_rate: float
    mean_idle_cash_jpy: float
    mean_skipped_pnl: float
    mean_capital_efficiency_score: float
    mean_deployable_alpha_loss_rate: float
    # deltas vs current day (current − trailing_mean)
    delta_util_ratio: float
    delta_conv_rate: float
    delta_capital_efficiency_score: float
    delta_deployable_alpha_loss_rate: float


@dataclass
class MissedWinnerEntry:
    symbol: str
    forward_return_5d: float
    estimated_missed_pnl: float
    rejection_reason: str
    signal_strength: float
    timestamp: str


@dataclass
class ExclusionCostEntry:
    reason: str
    total_missed_pnl: float
    count: int
    fraction: float


@dataclass
class IdleCashPeriodEntry:
    timestamp: str
    available_cash_jpy: float
    symbol: str
    rejection_reason: str


@dataclass
class RestrictiveAllocatorEntry:
    timestamp: str
    available_cash_jpy: float
    symbol: str
    alloc_cap: int
    rejection_reason: str


@dataclass
class AlphaDiagnosticSnapshot:
    """Full daily capital-efficiency diagnostic. Deterministic for fixed inputs."""
    diagnostic_id: str          # sha256(date + run_id)
    date: str                   # YYYY-MM-DD
    run_id: str
    generated_at: str           # ISO-8601 UTC (metadata only — not in hash)

    # ── Core metrics (from DailyLeakageSummary) ───────────────────────────────
    total_signals_seen: int
    executed_count: int
    skipped_count: int
    opportunity_utilization_ratio: float
    deployable_alpha_conversion_rate: float
    skipped_pnl_estimate: float
    realized_pnl_estimate: float
    idle_capital_mean_jpy: float
    idle_capital_peak_jpy: float
    enrichment_coverage: float

    # ── Derived metrics ───────────────────────────────────────────────────────
    deployable_alpha_loss_rate: float   # skipped_pnl / (skipped + realized)
    capital_efficiency_score: float     # weighted composite [0, 1]
    convexity_capture_rate: float       # 1 - missed_convexity / total_signals
    missed_runner_ratio: float          # missed_runners / total_signals

    # Raw counts for transparency
    convexity_missed_count: int         # enriched skips with fr5d > CONVEXITY_THRESHOLD
    runner_missed_count: int            # enriched skips with fr5d > RUNNER_THRESHOLD
    idle_cash_ratio: float              # mean(available_cash) / capital

    # ── Rankings ──────────────────────────────────────────────────────────────
    top_missed_winners: List[dict]              # top N by fr5d (enriched)
    largest_exclusion_costs: List[dict]         # top N reasons by summed missed PnL
    highest_idle_cash_periods: List[dict]       # top N skips by available_cash
    most_restrictive_allocator_periods: List[dict]  # top N alloc_cap=0 by cash
    top_missed_convexity_events: List[dict]     # enriched fr5d > CONVEXITY_THRESHOLD

    # ── Trailing comparisons ──────────────────────────────────────────────────
    trailing_5d: dict
    trailing_20d: dict

    # ── Bottleneck identification ─────────────────────────────────────────────
    dominant_bottleneck: str            # reason with highest total missed PnL
    dominant_bottleneck_pnl_jpy: float
    largest_leakage_source: str         # reason with highest skip count

    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def generate_diagnostic(
    date: str,
    run_id: str,
    skipped_records: List[SkippedOpportunityRecord],
    executed_count: int,
    realized_pnl: float = 0.0,
    capital: float = 3_000_000.0,
    historical_summaries: Optional[List[DailyLeakageSummary]] = None,
    top_n: int = TOP_N_DEFAULT,
) -> AlphaDiagnosticSnapshot:
    """
    Generate deterministic daily diagnostic snapshot.

    Args:
        date:                YYYY-MM-DD for this trading day.
        run_id:              Runtime run identifier.
        skipped_records:     All SkippedOpportunityRecord for this date.
        executed_count:      Number of successfully executed orders.
        realized_pnl:        Sum of realized PnL from executed orders (JPY).
        capital:             Total capital (JPY) for idle ratio denominator.
        historical_summaries: Past DailyLeakageSummary records (for trailing windows).
                              Should exclude the current date.
        top_n:               Number of top entries in each ranking.

    Returns:
        AlphaDiagnosticSnapshot (deterministic for fixed inputs, fixed ordering).
    """
    diagnostic_id = _make_id(date, run_id)
    generated_at = datetime.now(timezone.utc).isoformat()

    skipped_count = len(skipped_records)
    total_signals = executed_count + skipped_count

    # ── Core metrics ──────────────────────────────────────────────────────────
    util_ratio = executed_count / total_signals if total_signals > 0 else 0.0

    enriched = [
        r for r in skipped_records
        if r.enrichment_status == ENRICHMENT_ENRICHED
    ]
    enrichment_coverage = len(enriched) / skipped_count if skipped_count > 0 else 0.0

    # skipped_pnl_estimate from enriched records
    skipped_pnl = sum(
        r.estimated_missed_pnl for r in enriched
        if r.estimated_missed_pnl is not None
    )
    total_pnl = realized_pnl + skipped_pnl
    conv_rate = realized_pnl / total_pnl if total_pnl > 0 else 0.0

    cash_values = [r.available_cash for r in skipped_records if r.available_cash > 0]
    idle_mean_jpy = sum(cash_values) / len(cash_values) if cash_values else 0.0
    idle_peak_jpy = max(cash_values) if cash_values else 0.0

    # ── Derived metrics ───────────────────────────────────────────────────────
    deployable_alpha_loss_rate = (
        skipped_pnl / total_pnl if total_pnl > 0 else 0.0
    )

    idle_cash_ratio = idle_mean_jpy / capital if capital > 0 else 0.0
    idle_cash_ratio = min(1.0, max(0.0, idle_cash_ratio))

    capital_efficiency_score = round(
        util_ratio * _W_UTIL
        + conv_rate * _W_CONV
        + max(0.0, 1.0 - idle_cash_ratio) * _W_IDLE,
        6,
    )

    convexity_recs = [
        r for r in enriched
        if r.forward_return_5d is not None and r.forward_return_5d > CONVEXITY_THRESHOLD
    ]
    convexity_missed_count = len(convexity_recs)
    convexity_capture_rate = (
        1.0 - convexity_missed_count / total_signals
        if total_signals > 0
        else 1.0
    )

    runner_recs = [
        r for r in enriched
        if r.forward_return_5d is not None and r.forward_return_5d > RUNNER_THRESHOLD
    ]
    runner_missed_count = len(runner_recs)
    missed_runner_ratio = (
        runner_missed_count / total_signals
        if total_signals > 0
        else 0.0
    )

    # ── Rankings ──────────────────────────────────────────────────────────────
    top_missed_winners = _rank_missed_winners(enriched, top_n)
    largest_exclusion_costs = _rank_exclusion_costs(skipped_records, skipped_count, top_n)
    highest_idle_cash_periods = _rank_idle_cash(skipped_records, top_n)
    most_restrictive_allocator_periods = _rank_restrictive_allocator(skipped_records, top_n)
    top_missed_convexity_events = _rank_convexity_events(convexity_recs, top_n)

    # ── Bottleneck identification ─────────────────────────────────────────────
    dominant_bottleneck, dominant_bottleneck_pnl = _identify_dominant_bottleneck(
        skipped_records, enriched
    )
    largest_leakage_source = _identify_leakage_source(skipped_records)

    # ── Trailing comparisons ──────────────────────────────────────────────────
    past = _exclude_date(historical_summaries or [], date)
    trailing_5d = _compute_trailing(
        window=5,
        past_summaries=past,
        current_util=util_ratio,
        current_conv=conv_rate,
        current_efficiency=capital_efficiency_score,
        current_loss_rate=deployable_alpha_loss_rate,
    )
    trailing_20d = _compute_trailing(
        window=20,
        past_summaries=past,
        current_util=util_ratio,
        current_conv=conv_rate,
        current_efficiency=capital_efficiency_score,
        current_loss_rate=deployable_alpha_loss_rate,
    )

    return AlphaDiagnosticSnapshot(
        diagnostic_id=diagnostic_id,
        date=date,
        run_id=run_id,
        generated_at=generated_at,
        total_signals_seen=total_signals,
        executed_count=executed_count,
        skipped_count=skipped_count,
        opportunity_utilization_ratio=round(util_ratio, 6),
        deployable_alpha_conversion_rate=round(conv_rate, 6),
        skipped_pnl_estimate=round(skipped_pnl, 2),
        realized_pnl_estimate=round(realized_pnl, 2),
        idle_capital_mean_jpy=round(idle_mean_jpy, 2),
        idle_capital_peak_jpy=round(idle_peak_jpy, 2),
        enrichment_coverage=round(enrichment_coverage, 6),
        deployable_alpha_loss_rate=round(deployable_alpha_loss_rate, 6),
        capital_efficiency_score=capital_efficiency_score,
        convexity_capture_rate=round(convexity_capture_rate, 6),
        missed_runner_ratio=round(missed_runner_ratio, 6),
        convexity_missed_count=convexity_missed_count,
        runner_missed_count=runner_missed_count,
        idle_cash_ratio=round(idle_cash_ratio, 6),
        top_missed_winners=top_missed_winners,
        largest_exclusion_costs=largest_exclusion_costs,
        highest_idle_cash_periods=highest_idle_cash_periods,
        most_restrictive_allocator_periods=most_restrictive_allocator_periods,
        top_missed_convexity_events=top_missed_convexity_events,
        trailing_5d=asdict(trailing_5d),
        trailing_20d=asdict(trailing_20d),
        dominant_bottleneck=dominant_bottleneck,
        dominant_bottleneck_pnl_jpy=round(dominant_bottleneck_pnl, 2),
        largest_leakage_source=largest_leakage_source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ranking helpers (all deterministic: stable sort, fixed ordering)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_missed_winners(
    enriched: List[SkippedOpportunityRecord], top_n: int
) -> List[dict]:
    sorted_recs = sorted(
        enriched,
        key=lambda r: (r.forward_return_5d or 0.0, r.signal_strength),
        reverse=True,
    )[:top_n]
    return [
        asdict(MissedWinnerEntry(
            symbol=r.symbol,
            forward_return_5d=r.forward_return_5d or 0.0,
            estimated_missed_pnl=r.estimated_missed_pnl or 0.0,
            rejection_reason=r.rejection_reason,
            signal_strength=r.signal_strength,
            timestamp=r.timestamp,
        ))
        for r in sorted_recs
    ]


def _rank_exclusion_costs(
    records: List[SkippedOpportunityRecord],
    skipped_count: int,
    top_n: int,
) -> List[dict]:
    reason_pnl: Dict[str, float] = {}
    reason_count: Dict[str, int] = {}
    for r in records:
        reason_count[r.rejection_reason] = reason_count.get(r.rejection_reason, 0) + 1
        if r.estimated_missed_pnl is not None:
            reason_pnl[r.rejection_reason] = (
                reason_pnl.get(r.rejection_reason, 0.0) + r.estimated_missed_pnl
            )
    all_reasons = set(reason_count) | set(reason_pnl)
    # Sort by total_missed_pnl desc, then reason alphabetically for stability
    sorted_reasons = sorted(
        all_reasons,
        key=lambda r: (-abs(reason_pnl.get(r, 0.0)), r),
    )[:top_n]
    return [
        asdict(ExclusionCostEntry(
            reason=reason,
            total_missed_pnl=round(reason_pnl.get(reason, 0.0), 2),
            count=reason_count.get(reason, 0),
            fraction=round(
                reason_count.get(reason, 0) / skipped_count, 6
            ) if skipped_count > 0 else 0.0,
        ))
        for reason in sorted_reasons
    ]


def _rank_idle_cash(
    records: List[SkippedOpportunityRecord], top_n: int
) -> List[dict]:
    sorted_recs = sorted(
        records,
        key=lambda r: (-r.available_cash, r.timestamp, r.symbol),
    )[:top_n]
    return [
        asdict(IdleCashPeriodEntry(
            timestamp=r.timestamp,
            available_cash_jpy=r.available_cash,
            symbol=r.symbol,
            rejection_reason=r.rejection_reason,
        ))
        for r in sorted_recs
    ]


def _rank_restrictive_allocator(
    records: List[SkippedOpportunityRecord], top_n: int
) -> List[dict]:
    blocked = [r for r in records if r.alloc_cap == 0]
    sorted_recs = sorted(
        blocked,
        key=lambda r: (-r.available_cash, r.timestamp, r.symbol),
    )[:top_n]
    return [
        asdict(RestrictiveAllocatorEntry(
            timestamp=r.timestamp,
            available_cash_jpy=r.available_cash,
            symbol=r.symbol,
            alloc_cap=r.alloc_cap,
            rejection_reason=r.rejection_reason,
        ))
        for r in sorted_recs
    ]


def _rank_convexity_events(
    convexity_recs: List[SkippedOpportunityRecord], top_n: int
) -> List[dict]:
    sorted_recs = sorted(
        convexity_recs,
        key=lambda r: (r.forward_return_5d or 0.0, r.signal_strength),
        reverse=True,
    )[:top_n]
    return [
        asdict(MissedWinnerEntry(
            symbol=r.symbol,
            forward_return_5d=r.forward_return_5d or 0.0,
            estimated_missed_pnl=r.estimated_missed_pnl or 0.0,
            rejection_reason=r.rejection_reason,
            signal_strength=r.signal_strength,
            timestamp=r.timestamp,
        ))
        for r in sorted_recs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck identification
# ─────────────────────────────────────────────────────────────────────────────

def _identify_dominant_bottleneck(
    records: List[SkippedOpportunityRecord],
    enriched: List[SkippedOpportunityRecord],
) -> tuple[str, float]:
    """Return (reason_with_highest_missed_pnl, total_pnl). Falls back to count."""
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
        return (dominant, reason_pnl[dominant])
    # Fallback: reason with most skips
    count: Dict[str, int] = {}
    for r in records:
        count[r.rejection_reason] = count.get(r.rejection_reason, 0) + 1
    dominant = max(count, key=lambda k: (count[k], k))
    return (dominant, 0.0)


def _identify_leakage_source(records: List[SkippedOpportunityRecord]) -> str:
    """Reason with highest skip count."""
    if not records:
        return "none"
    count: Dict[str, int] = {}
    for r in records:
        count[r.rejection_reason] = count.get(r.rejection_reason, 0) + 1
    return max(count, key=lambda k: (count[k], k))


# ─────────────────────────────────────────────────────────────────────────────
# Trailing window computation
# ─────────────────────────────────────────────────────────────────────────────

def _exclude_date(
    summaries: List[DailyLeakageSummary], date: str
) -> List[DailyLeakageSummary]:
    return [s for s in summaries if s.date != date]


def _compute_trailing(
    window: int,
    past_summaries: List[DailyLeakageSummary],
    current_util: float,
    current_conv: float,
    current_efficiency: float,
    current_loss_rate: float,
) -> TrailingWindowMetrics:
    # Sort by date descending, take most recent N
    sorted_sums = sorted(past_summaries, key=lambda s: s.date, reverse=True)[:window]
    n = len(sorted_sums)

    if n == 0:
        return TrailingWindowMetrics(
            window_days=window,
            sample_count=0,
            mean_util_ratio=0.0,
            mean_conv_rate=0.0,
            mean_idle_cash_jpy=0.0,
            mean_skipped_pnl=0.0,
            mean_capital_efficiency_score=0.0,
            mean_deployable_alpha_loss_rate=0.0,
            delta_util_ratio=current_util,
            delta_conv_rate=current_conv,
            delta_capital_efficiency_score=current_efficiency,
            delta_deployable_alpha_loss_rate=current_loss_rate,
        )

    mean_util = sum(s.opportunity_utilization_ratio for s in sorted_sums) / n
    mean_conv = sum(s.deployable_alpha_conversion_rate for s in sorted_sums) / n
    mean_idle = sum(s.idle_capital_mean_jpy for s in sorted_sums) / n
    mean_pnl = sum(s.skipped_pnl_estimate for s in sorted_sums) / n

    # Recompute efficiency and loss_rate from summary fields
    mean_eff = sum(
        _summary_efficiency_score(s) for s in sorted_sums
    ) / n
    mean_loss = sum(
        _summary_loss_rate(s) for s in sorted_sums
    ) / n

    return TrailingWindowMetrics(
        window_days=window,
        sample_count=n,
        mean_util_ratio=round(mean_util, 6),
        mean_conv_rate=round(mean_conv, 6),
        mean_idle_cash_jpy=round(mean_idle, 2),
        mean_skipped_pnl=round(mean_pnl, 2),
        mean_capital_efficiency_score=round(mean_eff, 6),
        mean_deployable_alpha_loss_rate=round(mean_loss, 6),
        delta_util_ratio=round(current_util - mean_util, 6),
        delta_conv_rate=round(current_conv - mean_conv, 6),
        delta_capital_efficiency_score=round(current_efficiency - mean_eff, 6),
        delta_deployable_alpha_loss_rate=round(current_loss_rate - mean_loss, 6),
    )


def _summary_efficiency_score(s: DailyLeakageSummary) -> float:
    total = s.executed_count + s.skipped_count
    util = s.executed_count / total if total > 0 else 0.0
    conv = s.deployable_alpha_conversion_rate
    idle_ratio = 0.0  # not stored in DailyLeakageSummary directly; omit idle term
    return util * _W_UTIL + conv * _W_CONV + max(0.0, 1.0 - idle_ratio) * _W_IDLE


def _summary_loss_rate(s: DailyLeakageSummary) -> float:
    total = s.skipped_pnl_estimate + s.realized_pnl_estimate
    return s.skipped_pnl_estimate / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report (deterministic: same snapshot → byte-identical markdown)
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(snap: AlphaDiagnosticSnapshot) -> str:
    """
    Generate a deterministic markdown diagnostic report.

    The report content is derived solely from snap fields, so it is
    byte-identical for the same AlphaDiagnosticSnapshot input.
    generated_at is included as metadata, not in the hash.
    """
    lines: List[str] = []
    a = lines.append

    a(f"# Alpha Diagnostic — {snap.date}")
    a(f"")
    a(f"| Field | Value |")
    a(f"|---|---|")
    a(f"| diagnostic_id | `{snap.diagnostic_id}` |")
    a(f"| run_id | `{snap.run_id}` |")
    a(f"| generated_at | {snap.generated_at} |")
    a(f"")

    # ── Capital efficiency summary ────────────────────────────────────────────
    a(f"## Capital Efficiency Summary")
    a(f"")
    a(f"| Metric | Today | Δ vs 5d | Δ vs 20d |")
    a(f"|---|---|---|---|")

    t5 = snap.trailing_5d
    t20 = snap.trailing_20d

    def _d5(key: str) -> str:
        v = t5.get(key, 0.0)
        return f"{v:+.4f}" if v != 0.0 else "—"

    def _d20(key: str) -> str:
        v = t20.get(key, 0.0)
        return f"{v:+.4f}" if v != 0.0 else "—"

    a(f"| opportunity_utilization_ratio | {snap.opportunity_utilization_ratio:.4f} "
      f"| {_d5('delta_util_ratio')} | {_d20('delta_util_ratio')} |")
    a(f"| deployable_alpha_conversion_rate | {snap.deployable_alpha_conversion_rate:.4f} "
      f"| {_d5('delta_conv_rate')} | {_d20('delta_conv_rate')} |")
    a(f"| capital_efficiency_score | {snap.capital_efficiency_score:.4f} "
      f"| {_d5('delta_capital_efficiency_score')} | {_d20('delta_capital_efficiency_score')} |")
    a(f"| deployable_alpha_loss_rate | {snap.deployable_alpha_loss_rate:.4f} "
      f"| {_d5('delta_deployable_alpha_loss_rate')} | {_d20('delta_deployable_alpha_loss_rate')} |")
    a(f"| convexity_capture_rate | {snap.convexity_capture_rate:.4f} | — | — |")
    a(f"| missed_runner_ratio | {snap.missed_runner_ratio:.4f} | — | — |")
    a(f"| idle_cash_ratio | {snap.idle_cash_ratio:.4f} | — | — |")
    a(f"| idle_capital_mean_jpy | ¥{snap.idle_capital_mean_jpy:,.0f} | — | — |")
    a(f"| idle_capital_peak_jpy | ¥{snap.idle_capital_peak_jpy:,.0f} | — | — |")
    a(f"| skipped_pnl_estimate | ¥{snap.skipped_pnl_estimate:,.0f} | — | — |")
    a(f"| realized_pnl_estimate | ¥{snap.realized_pnl_estimate:,.0f} | — | — |")
    a(f"| enrichment_coverage | {snap.enrichment_coverage:.2%} | — | — |")
    a(f"")

    # ── Signal counts ─────────────────────────────────────────────────────────
    a(f"## Signal Counts")
    a(f"")
    a(f"| | Count |")
    a(f"|---|---|")
    a(f"| total_signals_seen | {snap.total_signals_seen} |")
    a(f"| executed | {snap.executed_count} |")
    a(f"| skipped | {snap.skipped_count} |")
    a(f"| convexity_missed_count (fr5d>{CONVEXITY_THRESHOLD:.0%}) | {snap.convexity_missed_count} |")
    a(f"| runner_missed_count (fr5d>{RUNNER_THRESHOLD:.0%}) | {snap.runner_missed_count} |")
    a(f"")

    # ── Bottleneck identification ─────────────────────────────────────────────
    a(f"## Dominant Capital-Efficiency Bottleneck")
    a(f"")
    a(f"| | |")
    a(f"|---|---|")
    a(f"| dominant_bottleneck | **{snap.dominant_bottleneck}** |")
    a(f"| estimated_cost_jpy | ¥{snap.dominant_bottleneck_pnl_jpy:,.0f} |")
    a(f"| largest_leakage_source (by count) | **{snap.largest_leakage_source}** |")
    a(f"")

    # ── Exclusion breakdown ───────────────────────────────────────────────────
    a(f"## Exclusion Breakdown")
    a(f"")
    if snap.largest_exclusion_costs:
        a(f"| Reason | Count | Fraction | Missed PnL (JPY) |")
        a(f"|---|---|---|---|")
        for entry in snap.largest_exclusion_costs:
            a(
                f"| {entry['reason']} "
                f"| {entry['count']} "
                f"| {entry['fraction']:.1%} "
                f"| ¥{entry['total_missed_pnl']:,.0f} |"
            )
    else:
        a(f"_No skipped records._")
    a(f"")

    # ── Top missed winners ────────────────────────────────────────────────────
    a(f"## Top Missed Winners (by 5d forward return)")
    a(f"")
    if snap.top_missed_winners:
        a(f"| Symbol | fr5d | Signal | Reason | Missed PnL |")
        a(f"|---|---|---|---|---|")
        for entry in snap.top_missed_winners:
            a(
                f"| {entry['symbol']} "
                f"| {entry['forward_return_5d']:+.2%} "
                f"| {entry['signal_strength']:.1f} "
                f"| {entry['rejection_reason']} "
                f"| ¥{entry['estimated_missed_pnl']:,.0f} |"
            )
    else:
        a(f"_No enriched skipped records._")
    a(f"")

    # ── Top missed convexity events ───────────────────────────────────────────
    a(f"## Top Missed Convexity Events (fr5d > {CONVEXITY_THRESHOLD:.0%})")
    a(f"")
    if snap.top_missed_convexity_events:
        a(f"| Symbol | fr5d | Signal | Reason | Missed PnL |")
        a(f"|---|---|---|---|---|")
        for entry in snap.top_missed_convexity_events:
            a(
                f"| {entry['symbol']} "
                f"| {entry['forward_return_5d']:+.2%} "
                f"| {entry['signal_strength']:.1f} "
                f"| {entry['rejection_reason']} "
                f"| ¥{entry['estimated_missed_pnl']:,.0f} |"
            )
    else:
        a(f"_No convexity events above threshold._")
    a(f"")

    # ── Highest idle-cash periods ─────────────────────────────────────────────
    a(f"## Highest Idle-Cash Periods")
    a(f"")
    if snap.highest_idle_cash_periods:
        a(f"| Timestamp | Available Cash | Symbol | Reason |")
        a(f"|---|---|---|---|")
        for entry in snap.highest_idle_cash_periods:
            a(
                f"| {entry['timestamp']} "
                f"| ¥{entry['available_cash_jpy']:,.0f} "
                f"| {entry['symbol']} "
                f"| {entry['rejection_reason']} |"
            )
    else:
        a(f"_No idle-cash records._")
    a(f"")

    # ── Most restrictive allocator periods ────────────────────────────────────
    a(f"## Most Restrictive Allocator Periods (alloc_cap = 0)")
    a(f"")
    if snap.most_restrictive_allocator_periods:
        a(f"| Timestamp | Cash Available | Symbol | Reason |")
        a(f"|---|---|---|---|")
        for entry in snap.most_restrictive_allocator_periods:
            a(
                f"| {entry['timestamp']} "
                f"| ¥{entry['available_cash_jpy']:,.0f} "
                f"| {entry['symbol']} "
                f"| {entry['rejection_reason']} |"
            )
    else:
        a(f"_No alloc_cap=0 records._")
    a(f"")

    # ── Trailing windows ──────────────────────────────────────────────────────
    a(f"## Trailing Window Comparison")
    a(f"")
    a(f"| Metric | 5d mean | 20d mean |")
    a(f"|---|---|---|")
    a(f"| sample_count | {t5.get('sample_count', 0)} | {t20.get('sample_count', 0)} |")
    a(f"| mean_util_ratio | {t5.get('mean_util_ratio', 0.0):.4f} | {t20.get('mean_util_ratio', 0.0):.4f} |")
    a(f"| mean_conv_rate | {t5.get('mean_conv_rate', 0.0):.4f} | {t20.get('mean_conv_rate', 0.0):.4f} |")
    a(f"| mean_capital_efficiency_score | {t5.get('mean_capital_efficiency_score', 0.0):.4f} "
      f"| {t20.get('mean_capital_efficiency_score', 0.0):.4f} |")
    a(f"| mean_skipped_pnl | ¥{t5.get('mean_skipped_pnl', 0.0):,.0f} "
      f"| ¥{t20.get('mean_skipped_pnl', 0.0):,.0f} |")
    a(f"")

    return "\n".join(lines)


def log_diagnostic(snap: AlphaDiagnosticSnapshot) -> None:
    """Emit diagnostic to structured log."""
    logger.info(
        "[ALPHA_DIAG] date=%s eff=%.3f loss_rate=%.3f conv_miss=%.3f "
        "runner_miss=%.3f bottleneck=%s cost=¥%.0f",
        snap.date,
        snap.capital_efficiency_score,
        snap.deployable_alpha_loss_rate,
        1.0 - snap.convexity_capture_rate,
        snap.missed_runner_ratio,
        snap.dominant_bottleneck,
        snap.dominant_bottleneck_pnl_jpy,
    )
    logger.info(
        "[ALPHA_DIAG] util=%.1f%% conv=%.1f%% idle_cash=¥%.0f peak=¥%.0f "
        "skipped_pnl=¥%.0f realized_pnl=¥%.0f",
        snap.opportunity_utilization_ratio * 100,
        snap.deployable_alpha_conversion_rate * 100,
        snap.idle_capital_mean_jpy,
        snap.idle_capital_peak_jpy,
        snap.skipped_pnl_estimate,
        snap.realized_pnl_estimate,
    )
    if snap.largest_exclusion_costs:
        logger.info("[ALPHA_DIAG] largest exclusion costs:")
        for e in snap.largest_exclusion_costs[:3]:
            logger.info(
                "[ALPHA_DIAG]   %-30s count=%d pnl=¥%.0f",
                e["reason"], e["count"], e["total_missed_pnl"],
            )
    if snap.top_missed_winners:
        top = snap.top_missed_winners[0]
        logger.info(
            "[ALPHA_DIAG] top missed winner: %s fr5d=%.2f%% pnl=¥%.0f reason=%s",
            top["symbol"],
            top["forward_return_5d"] * 100,
            top["estimated_missed_pnl"],
            top["rejection_reason"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_diagnostic(snap: AlphaDiagnosticSnapshot, path: Path) -> None:
    """Append-only JSONL. Fail-open — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(snap), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[ALPHA_DIAG] append failed (%s)", exc)


def append_markdown_report(
    snap: AlphaDiagnosticSnapshot, report_dir: Path
) -> None:
    """Write/append markdown report to report_dir/YYYY-MM-DD.md. Fail-open."""
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{snap.date}.md"
        md = generate_markdown_report(snap)
        separator = "\n\n---\n\n" if report_path.exists() else ""
        with report_path.open("a", encoding="utf-8") as f:
            f.write(separator + md)
    except Exception as exc:
        logger.warning("[ALPHA_DIAG] markdown write failed (%s)", exc)


def load_diagnostics(path: Path) -> List[AlphaDiagnosticSnapshot]:
    """Load all snapshots from JSONL. Corrupted lines are skipped."""
    if not path.exists():
        return []
    result: List[AlphaDiagnosticSnapshot] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result.append(_parse_snapshot(json.loads(line)))
        except Exception as exc:
            logger.warning("[ALPHA_DIAG] parse error: %s", exc)
    return result


def get_diagnostic_for_date(
    path: Path, date: str
) -> Optional[AlphaDiagnosticSnapshot]:
    snaps = [s for s in load_diagnostics(path) if s.date == date]
    return snaps[-1] if snaps else None


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _make_id(date: str, run_id: str) -> str:
    payload = json.dumps({"date": date, "run_id": run_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_snapshot(d: dict) -> AlphaDiagnosticSnapshot:
    return AlphaDiagnosticSnapshot(
        diagnostic_id=d.get("diagnostic_id", ""),
        date=d.get("date", ""),
        run_id=d.get("run_id", ""),
        generated_at=d.get("generated_at", ""),
        total_signals_seen=int(d.get("total_signals_seen", 0)),
        executed_count=int(d.get("executed_count", 0)),
        skipped_count=int(d.get("skipped_count", 0)),
        opportunity_utilization_ratio=float(d.get("opportunity_utilization_ratio", 0.0)),
        deployable_alpha_conversion_rate=float(d.get("deployable_alpha_conversion_rate", 0.0)),
        skipped_pnl_estimate=float(d.get("skipped_pnl_estimate", 0.0)),
        realized_pnl_estimate=float(d.get("realized_pnl_estimate", 0.0)),
        idle_capital_mean_jpy=float(d.get("idle_capital_mean_jpy", 0.0)),
        idle_capital_peak_jpy=float(d.get("idle_capital_peak_jpy", 0.0)),
        enrichment_coverage=float(d.get("enrichment_coverage", 0.0)),
        deployable_alpha_loss_rate=float(d.get("deployable_alpha_loss_rate", 0.0)),
        capital_efficiency_score=float(d.get("capital_efficiency_score", 0.0)),
        convexity_capture_rate=float(d.get("convexity_capture_rate", 1.0)),
        missed_runner_ratio=float(d.get("missed_runner_ratio", 0.0)),
        convexity_missed_count=int(d.get("convexity_missed_count", 0)),
        runner_missed_count=int(d.get("runner_missed_count", 0)),
        idle_cash_ratio=float(d.get("idle_cash_ratio", 0.0)),
        top_missed_winners=d.get("top_missed_winners", []),
        largest_exclusion_costs=d.get("largest_exclusion_costs", []),
        highest_idle_cash_periods=d.get("highest_idle_cash_periods", []),
        most_restrictive_allocator_periods=d.get("most_restrictive_allocator_periods", []),
        top_missed_convexity_events=d.get("top_missed_convexity_events", []),
        trailing_5d=d.get("trailing_5d", {}),
        trailing_20d=d.get("trailing_20d", {}),
        dominant_bottleneck=d.get("dominant_bottleneck", "none"),
        dominant_bottleneck_pnl_jpy=float(d.get("dominant_bottleneck_pnl_jpy", 0.0)),
        largest_leakage_source=d.get("largest_leakage_source", "none"),
        schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
    )
