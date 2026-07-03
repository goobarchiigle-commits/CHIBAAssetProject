"""
deployable_alpha_metrics.py — Aggregate alpha leakage metrics

Computes from accumulated records:
  opportunity_utilization_ratio    — executed / (executed + skipped)
  deployable_alpha_conversion_rate — realized_pnl / (realized + missed)
  skipped_pnl_estimate             — sum of estimated_missed_pnl (enriched records only)
  idle_cash_ratio                  — mean(available_cash_at_skip / capital)
  exclusion_reason_distribution    — {reason: fraction_of_total}

All computations are deterministic: same input records → same output metrics.
Read-only analytics. No execution-path mutation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.analytics.skipped_opportunity_analytics import (
    SkippedOpportunityRecord,
    ENRICHMENT_ENRICHED,
)

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


@dataclass
class DeployableAlphaMetrics:
    """
    Aggregate leakage metrics for a time window.
    All floats are in [0, 1] unless noted; pnl values are in JPY.
    """
    computed_at: str
    date_range_start: str
    date_range_end: str
    # Counts
    total_signals: int           # executed + skipped
    executed_count: int
    skipped_count: int
    # Ratios
    opportunity_utilization_ratio: float    # executed / total (0 if total=0)
    deployable_alpha_conversion_rate: float # realized_pnl / (realized + missed); NaN-safe → 0
    # PnL estimates (JPY)
    skipped_pnl_estimate: float    # sum of estimated_missed_pnl from enriched records
    realized_pnl_estimate: float   # passed in from outcome records
    # Capital efficiency
    idle_cash_ratio: float         # mean(available_cash / capital) at skip times
    mean_available_cash_at_skip: float  # raw mean (JPY)
    # Breakdown
    exclusion_reason_distribution: Dict[str, float]  # reason → fraction
    exclusion_reason_counts: Dict[str, int]          # reason → count
    enrichment_coverage: float     # fraction of skipped records with enrichment_status=enriched
    schema_version: int = SCHEMA_VERSION


def compute_alpha_metrics(
    skipped_records: List[SkippedOpportunityRecord],
    executed_count: int,
    realized_pnl: float = 0.0,
    capital: float = 3_000_000.0,
    date_range_start: str = "",
    date_range_end: str = "",
) -> DeployableAlphaMetrics:
    """
    Compute aggregate leakage metrics.

    Args:
        skipped_records: All SkippedOpportunityRecord for the window.
        executed_count:  Number of orders that passed all filters and were submitted.
        realized_pnl:    Sum of realized PnL from executed orders (JPY). Use 0 if unavailable.
        capital:         Total capital (JPY) for idle_cash_ratio denominator.
        date_range_start/end: ISO date strings for the window.

    Returns:
        DeployableAlphaMetrics (deterministic for given inputs).
    """
    now_str = datetime.now(timezone.utc).isoformat()
    skipped_count = len(skipped_records)
    total = executed_count + skipped_count

    # opportunity_utilization_ratio
    util_ratio = executed_count / total if total > 0 else 0.0

    # skipped_pnl_estimate (enriched records only; None treated as 0)
    enriched = [r for r in skipped_records if r.enrichment_status == ENRICHMENT_ENRICHED]
    missed_pnl = sum(
        r.estimated_missed_pnl for r in enriched if r.estimated_missed_pnl is not None
    )
    enrichment_coverage = len(enriched) / skipped_count if skipped_count > 0 else 0.0

    # deployable_alpha_conversion_rate
    total_pnl = realized_pnl + missed_pnl
    conv_rate = realized_pnl / total_pnl if total_pnl > 0 else 0.0

    # idle_cash_ratio: mean(available_cash / capital) at skip times
    cash_values = [r.available_cash for r in skipped_records if r.available_cash > 0]
    mean_cash = sum(cash_values) / len(cash_values) if cash_values else 0.0
    idle_ratio = mean_cash / capital if capital > 0 else 0.0

    # exclusion_reason_distribution
    reason_counts: Dict[str, int] = {}
    for rec in skipped_records:
        reason_counts[rec.rejection_reason] = reason_counts.get(rec.rejection_reason, 0) + 1

    reason_dist: Dict[str, float] = {
        reason: count / skipped_count
        for reason, count in sorted(reason_counts.items())
    } if skipped_count > 0 else {}

    return DeployableAlphaMetrics(
        computed_at=now_str,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        total_signals=total,
        executed_count=executed_count,
        skipped_count=skipped_count,
        opportunity_utilization_ratio=round(util_ratio, 6),
        deployable_alpha_conversion_rate=round(conv_rate, 6),
        skipped_pnl_estimate=round(missed_pnl, 2),
        realized_pnl_estimate=round(realized_pnl, 2),
        idle_cash_ratio=round(idle_ratio, 6),
        mean_available_cash_at_skip=round(mean_cash, 2),
        exclusion_reason_distribution=reason_dist,
        exclusion_reason_counts=dict(sorted(reason_counts.items())),
        enrichment_coverage=round(enrichment_coverage, 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_alpha_metrics(metrics: DeployableAlphaMetrics, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(metrics), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[ALPHA_METRICS] write failed (%s)", exc)


def load_alpha_metrics(path: Path) -> List[DeployableAlphaMetrics]:
    if not path.exists():
        return []
    result: List[DeployableAlphaMetrics] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(DeployableAlphaMetrics(
                computed_at=d.get("computed_at", ""),
                date_range_start=d.get("date_range_start", ""),
                date_range_end=d.get("date_range_end", ""),
                total_signals=int(d.get("total_signals", 0)),
                executed_count=int(d.get("executed_count", 0)),
                skipped_count=int(d.get("skipped_count", 0)),
                opportunity_utilization_ratio=float(d.get("opportunity_utilization_ratio", 0.0)),
                deployable_alpha_conversion_rate=float(d.get("deployable_alpha_conversion_rate", 0.0)),
                skipped_pnl_estimate=float(d.get("skipped_pnl_estimate", 0.0)),
                realized_pnl_estimate=float(d.get("realized_pnl_estimate", 0.0)),
                idle_cash_ratio=float(d.get("idle_cash_ratio", 0.0)),
                mean_available_cash_at_skip=float(d.get("mean_available_cash_at_skip", 0.0)),
                exclusion_reason_distribution=d.get("exclusion_reason_distribution", {}),
                exclusion_reason_counts=d.get("exclusion_reason_counts", {}),
                enrichment_coverage=float(d.get("enrichment_coverage", 0.0)),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            ))
        except Exception as exc:
            logger.warning("[ALPHA_METRICS] parse error: %s", exc)
    return result
