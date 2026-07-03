"""
daily_leakage_summary.py — Deterministic daily alpha leakage summaries

Generates a structured daily summary capturing:
  - top missed opportunities (by signal_strength)
  - largest missed PnL (by estimated_missed_pnl)
  - exclusion reason breakdown (count + estimated_pnl per reason)
  - idle capital pressure (mean + peak available_cash at skip times)
  - deployable vs theoretical alpha comparison (utilization ratio + conversion rate)

Deterministic: same date + same records → byte-identical summary.
Append-only JSONL + structured log output.
Read-only analytics: no broker mutations, no execution-path writes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.analytics.skipped_opportunity_analytics import (
    SkippedOpportunityRecord,
    ENRICHMENT_ENRICHED,
)
from src.analytics.deployable_alpha_metrics import (
    DeployableAlphaMetrics,
    compute_alpha_metrics,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1

TOP_N_DEFAULT = 5   # top N records in each ranking


@dataclass
class ExclusionBreakdownEntry:
    reason: str
    count: int
    fraction: float
    estimated_missed_pnl: float   # sum across enriched records for this reason (JPY)


@dataclass
class TopMissedOpportunity:
    symbol: str
    strategy_id: str
    signal_strength: float
    predicted_rank: int
    rejection_reason: str
    price_at_signal: float
    intended_position_size: int
    estimated_missed_pnl: Optional[float]
    forward_return_5d: Optional[float]
    timestamp: str


@dataclass
class DailyLeakageSummary:
    """Complete daily leakage summary. Deterministic for fixed inputs."""
    summary_id: str              # sha256(date + run_id) — unique per day+run
    date: str                    # YYYY-MM-DD
    run_id: str
    generated_at: str            # ISO-8601 UTC
    # Counts
    total_signals_seen: int
    executed_count: int
    skipped_count: int
    # Top missed opportunities (by signal_strength, descending)
    top_missed_by_signal: List[dict]
    # Largest missed PnL (by estimated_missed_pnl, descending; enriched only)
    top_missed_by_pnl: List[dict]
    # Exclusion breakdown
    exclusion_breakdown: List[dict]
    # Capital pressure
    idle_capital_mean_jpy: float    # mean available_cash at skip time
    idle_capital_peak_jpy: float    # max available_cash at skip time
    # Deployable vs theoretical alpha
    opportunity_utilization_ratio: float
    deployable_alpha_conversion_rate: float
    skipped_pnl_estimate: float
    realized_pnl_estimate: float
    enrichment_coverage: float
    schema_version: int = SCHEMA_VERSION


def generate_daily_summary(
    date: str,
    run_id: str,
    skipped_records: List[SkippedOpportunityRecord],
    executed_count: int,
    realized_pnl: float = 0.0,
    capital: float = 3_000_000.0,
    top_n: int = TOP_N_DEFAULT,
) -> DailyLeakageSummary:
    """
    Generate deterministic daily summary.

    Args:
        date:            YYYY-MM-DD string for this trading day.
        run_id:          Runtime run identifier.
        skipped_records: All SkippedOpportunityRecord for this date.
        executed_count:  Number of successfully executed orders.
        realized_pnl:    Sum of realized PnL from executed orders (JPY).
        capital:         Total capital for idle ratio denominator (JPY).
        top_n:           Number of top records to include in each list.

    Returns:
        DailyLeakageSummary (deterministic for fixed inputs).
    """
    import hashlib, json as _json
    summary_id = hashlib.sha256(
        _json.dumps({"date": date, "run_id": run_id}, sort_keys=True).encode()
    ).hexdigest()

    metrics = compute_alpha_metrics(
        skipped_records=skipped_records,
        executed_count=executed_count,
        realized_pnl=realized_pnl,
        capital=capital,
        date_range_start=date,
        date_range_end=date,
    )

    # Top missed by signal_strength (descending)
    sorted_by_signal = sorted(
        skipped_records,
        key=lambda r: r.signal_strength,
        reverse=True,
    )[:top_n]

    top_by_signal = [
        asdict(TopMissedOpportunity(
            symbol=r.symbol,
            strategy_id=r.strategy_id,
            signal_strength=r.signal_strength,
            predicted_rank=r.predicted_rank,
            rejection_reason=r.rejection_reason,
            price_at_signal=r.price_at_signal,
            intended_position_size=r.intended_position_size,
            estimated_missed_pnl=r.estimated_missed_pnl,
            forward_return_5d=r.forward_return_5d,
            timestamp=r.timestamp,
        ))
        for r in sorted_by_signal
    ]

    # Top missed by estimated_missed_pnl (enriched only, descending)
    enriched_records = [
        r for r in skipped_records
        if r.enrichment_status == ENRICHMENT_ENRICHED
        and r.estimated_missed_pnl is not None
    ]
    sorted_by_pnl = sorted(
        enriched_records,
        key=lambda r: abs(r.estimated_missed_pnl or 0),
        reverse=True,
    )[:top_n]

    top_by_pnl = [
        asdict(TopMissedOpportunity(
            symbol=r.symbol,
            strategy_id=r.strategy_id,
            signal_strength=r.signal_strength,
            predicted_rank=r.predicted_rank,
            rejection_reason=r.rejection_reason,
            price_at_signal=r.price_at_signal,
            intended_position_size=r.intended_position_size,
            estimated_missed_pnl=r.estimated_missed_pnl,
            forward_return_5d=r.forward_return_5d,
            timestamp=r.timestamp,
        ))
        for r in sorted_by_pnl
    ]

    # Exclusion breakdown
    reason_pnl: Dict[str, float] = {}
    for rec in skipped_records:
        if rec.enrichment_status == ENRICHMENT_ENRICHED and rec.estimated_missed_pnl is not None:
            reason_pnl[rec.rejection_reason] = (
                reason_pnl.get(rec.rejection_reason, 0.0) + rec.estimated_missed_pnl
            )

    breakdown = [
        asdict(ExclusionBreakdownEntry(
            reason=reason,
            count=count,
            fraction=metrics.exclusion_reason_distribution.get(reason, 0.0),
            estimated_missed_pnl=round(reason_pnl.get(reason, 0.0), 2),
        ))
        for reason, count in sorted(metrics.exclusion_reason_counts.items())
    ]

    # Capital pressure
    cash_at_skips = [r.available_cash for r in skipped_records if r.available_cash > 0]
    idle_mean = sum(cash_at_skips) / len(cash_at_skips) if cash_at_skips else 0.0
    idle_peak = max(cash_at_skips) if cash_at_skips else 0.0

    return DailyLeakageSummary(
        summary_id=summary_id,
        date=date,
        run_id=run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_signals_seen=metrics.total_signals,
        executed_count=metrics.executed_count,
        skipped_count=metrics.skipped_count,
        top_missed_by_signal=top_by_signal,
        top_missed_by_pnl=top_by_pnl,
        exclusion_breakdown=breakdown,
        idle_capital_mean_jpy=round(idle_mean, 2),
        idle_capital_peak_jpy=round(idle_peak, 2),
        opportunity_utilization_ratio=metrics.opportunity_utilization_ratio,
        deployable_alpha_conversion_rate=metrics.deployable_alpha_conversion_rate,
        skipped_pnl_estimate=metrics.skipped_pnl_estimate,
        realized_pnl_estimate=metrics.realized_pnl_estimate,
        enrichment_coverage=metrics.enrichment_coverage,
    )


def log_daily_summary(summary: DailyLeakageSummary) -> None:
    """Emit daily summary to structured log. Observable in runtime logs."""
    logger.info(
        "[LEAKAGE] date=%s signals=%d executed=%d skipped=%d "
        "util=%.1f%% conv=%.1f%% skipped_pnl=¥%.0f idle_cash=¥%.0f",
        summary.date,
        summary.total_signals_seen,
        summary.executed_count,
        summary.skipped_count,
        summary.opportunity_utilization_ratio * 100,
        summary.deployable_alpha_conversion_rate * 100,
        summary.skipped_pnl_estimate,
        summary.idle_capital_mean_jpy,
    )
    for entry in summary.exclusion_breakdown:
        logger.info(
            "[LEAKAGE] exclusion: reason=%-30s count=%d (%.0f%%) pnl=¥%.0f",
            entry["reason"],
            entry["count"],
            entry["fraction"] * 100,
            entry["estimated_missed_pnl"],
        )
    if summary.top_missed_by_signal:
        logger.info("[LEAKAGE] top missed by signal strength:")
        for opp in summary.top_missed_by_signal:
            logger.info(
                "[LEAKAGE]   %s rank=%d rsr=%.1f reason=%s pnl=¥%.0f",
                opp["symbol"],
                opp["predicted_rank"],
                opp["signal_strength"],
                opp["rejection_reason"],
                opp.get("estimated_missed_pnl") or 0.0,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_daily_summary(summary: DailyLeakageSummary, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[LEAKAGE_SUMMARY] write failed (%s)", exc)


def load_daily_summaries(path: Path) -> List[DailyLeakageSummary]:
    if not path.exists():
        return []
    result: List[DailyLeakageSummary] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(DailyLeakageSummary(
                summary_id=d.get("summary_id", ""),
                date=d.get("date", ""),
                run_id=d.get("run_id", ""),
                generated_at=d.get("generated_at", ""),
                total_signals_seen=int(d.get("total_signals_seen", 0)),
                executed_count=int(d.get("executed_count", 0)),
                skipped_count=int(d.get("skipped_count", 0)),
                top_missed_by_signal=d.get("top_missed_by_signal", []),
                top_missed_by_pnl=d.get("top_missed_by_pnl", []),
                exclusion_breakdown=d.get("exclusion_breakdown", []),
                idle_capital_mean_jpy=float(d.get("idle_capital_mean_jpy", 0.0)),
                idle_capital_peak_jpy=float(d.get("idle_capital_peak_jpy", 0.0)),
                opportunity_utilization_ratio=float(d.get("opportunity_utilization_ratio", 0.0)),
                deployable_alpha_conversion_rate=float(d.get("deployable_alpha_conversion_rate", 0.0)),
                skipped_pnl_estimate=float(d.get("skipped_pnl_estimate", 0.0)),
                realized_pnl_estimate=float(d.get("realized_pnl_estimate", 0.0)),
                enrichment_coverage=float(d.get("enrichment_coverage", 0.0)),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[LEAKAGE_SUMMARY] parse error: %s", exc)
    return result


def get_summary_for_date(path: Path, date: str) -> Optional[DailyLeakageSummary]:
    summaries = [s for s in load_daily_summaries(path) if s.date == date]
    return summaries[-1] if summaries else None
