"""
src/execution/deferred_metrics.py
Observability metrics for the deferred entry queue.

Emits structured metrics to runtime/deferred_metrics.jsonl (append-only).
Tracks forward returns for missed entries to measure alpha destruction.

Metrics emitted:
  deferred_queue_size       current queue depth
  recovered_entries         entries retried successfully today
  expired_entries           entries dropped today (TTL / max retry)
  retry_success_rate        recovered / retry_attempts
  fractional_demand_total   sum of accumulated_demand across all symbols
  fractional_execution_count  total successful fractional → whole lot executions
  missed_entries_before     rejected entries before deferred system
  missed_entries_after      rejected entries after deferred retry
  rejected_by_reason        {sector_cap: N, lot_rounding: N, ...}
  alpha_capture_estimate    (missed_entries_before - missed_entries_after) / missed_entries_before
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


@dataclass
class DeferredMetrics:
    timestamp:                  str
    deferred_queue_size:        int   = 0
    recovered_entries:          int   = 0
    expired_entries:            int   = 0
    retry_attempts:             int   = 0
    retry_successes:            int   = 0
    fractional_demand_total:    float = 0.0
    fractional_execution_count: int   = 0
    missed_entries_before:      int   = 0
    missed_entries_after:       int   = 0
    rejected_by_reason:         dict  = field(default_factory=dict)
    alpha_capture_estimate:     float = 0.0
    # Storm protection counters
    deferred_replay_skipped:    int   = 0   # blocked by open auction window
    deferred_replay_throttled:  int   = 0   # blocked by cycle quota
    deferred_sector_blocked:    int   = 0   # blocked by per-sector cycle quota
    # Regime drift counters
    deferred_regime_invalidations: int = 0  # dropped: market_regime changed
    deferred_rank_decay_drops:     int = 0  # dropped: rsr_rank or sector_rank decayed

    @property
    def retry_success_rate(self) -> float:
        if self.retry_attempts == 0:
            return 0.0
        return round(self.retry_successes / self.retry_attempts, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retry_success_rate"] = self.retry_success_rate
        return d


def build_metrics(
    *,
    deferred_entries:           list,
    recovered_entries:          int,
    expired_entries:            int,
    retry_attempts:             int,
    retry_successes:            int,
    fractional_demand_total:    float,
    fractional_execution_count: int,
    missed_entries_before:      int,
    missed_entries_after:       int,
    deferred_replay_skipped:       int = 0,
    deferred_replay_throttled:     int = 0,
    deferred_sector_blocked:       int = 0,
    deferred_regime_invalidations: int = 0,
    deferred_rank_decay_drops:     int = 0,
) -> DeferredMetrics:
    """
    Construct a DeferredMetrics snapshot.

    alpha_capture_estimate: fraction of missed entries recovered.
    rejected_by_reason: aggregated from deferred_entries.
    Storm counters (optional, default 0 for backward compatibility):
      deferred_replay_skipped        — from StormProtectionResult.deferred_replay_skipped
      deferred_replay_throttled      — from StormProtectionResult.deferred_replay_throttled
      deferred_sector_blocked        — from StormProtectionResult.deferred_sector_blocked
    Regime drift counters (optional, default 0 for backward compatibility):
      deferred_regime_invalidations  — entries dropped: regime changed
      deferred_rank_decay_drops      — entries dropped: rsr or sector rank decayed
    """
    by_reason: dict[str, int] = {}
    for e in deferred_entries:
        by_reason[e.reject_reason] = by_reason.get(e.reject_reason, 0) + 1

    alpha_capture = 0.0
    if missed_entries_before > 0:
        recovered = missed_entries_before - missed_entries_after
        alpha_capture = round(max(0.0, recovered / missed_entries_before), 4)

    return DeferredMetrics(
        timestamp                  = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        deferred_queue_size        = len(deferred_entries),
        recovered_entries          = recovered_entries,
        expired_entries            = expired_entries,
        retry_attempts             = retry_attempts,
        retry_successes            = retry_successes,
        fractional_demand_total    = round(fractional_demand_total, 4),
        fractional_execution_count = fractional_execution_count,
        missed_entries_before      = missed_entries_before,
        missed_entries_after       = missed_entries_after,
        rejected_by_reason         = by_reason,
        alpha_capture_estimate     = alpha_capture,
        deferred_replay_skipped        = deferred_replay_skipped,
        deferred_replay_throttled      = deferred_replay_throttled,
        deferred_sector_blocked        = deferred_sector_blocked,
        deferred_regime_invalidations  = deferred_regime_invalidations,
        deferred_rank_decay_drops      = deferred_rank_decay_drops,
    )


def emit_metrics(
    metrics:      DeferredMetrics,
    metrics_path: Path,
) -> None:
    """Append one metrics record to JSONL file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + "\n")
    logger.info(
        "[DEFERRED_METRICS] queue=%d recovered=%d expired=%d "
        "retry_rate=%.3f alpha_capture=%.4f fractional_total=%.3f "
        "replay_skipped=%d replay_throttled=%d sector_blocked=%d "
        "regime_invalidations=%d rank_decay_drops=%d",
        metrics.deferred_queue_size, metrics.recovered_entries,
        metrics.expired_entries, metrics.retry_success_rate,
        metrics.alpha_capture_estimate, metrics.fractional_demand_total,
        metrics.deferred_replay_skipped, metrics.deferred_replay_throttled,
        metrics.deferred_sector_blocked,
        metrics.deferred_regime_invalidations, metrics.deferred_rank_decay_drops,
    )


def track_missed_entry(
    *,
    entry_id:      str,
    symbol:        str,
    reject_reason: str,
    signal_score:  float,
    miss_date:     str,
    miss_price:    float,
    tracking_path: Path,
) -> None:
    """
    Record a missed entry for forward return tracking (append-only JSONL).

    Purpose: measure whether constraints destroy alpha or protect capital.
    """
    record = {
        "entry_id":          entry_id,
        "symbol":            symbol,
        "reject_reason":     reject_reason,
        "signal_score":      round(signal_score, 6),
        "miss_date":         miss_date,
        "miss_price":        miss_price,
        "forward_return_5d": None,
        "forward_return_20d": None,
        "recorded_at":       datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    with tracking_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "[DEFERRED_METRICS] missed_entry tracked: symbol=%s reason=%s "
        "score=%.4f price=%.1f date=%s",
        symbol, reject_reason, signal_score, miss_price, miss_date,
    )


def update_forward_returns(
    tracking_path:  Path,
    price_lookup:   dict[str, float],
    days_since_miss: dict[str, int],
) -> None:
    """
    Update forward_return_5d / forward_return_20d for tracked entries.

    Args:
        tracking_path:   path to missed_entry_tracking.jsonl
        price_lookup:    symbol → current price
        days_since_miss: entry_id → calendar days elapsed since miss_date

    Rewrites tracking file atomically.
    """
    if not tracking_path.exists():
        return

    records: list[dict] = []
    try:
        with tracking_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as exc:
        logger.error("[DEFERRED_METRICS] forward_return load failed: %s", exc)
        return

    for rec in records:
        eid        = rec.get("entry_id", "")
        sym        = rec.get("symbol", "")
        miss_price = float(rec.get("miss_price", 0) or 0)
        days       = days_since_miss.get(eid, 0)
        cur_price  = price_lookup.get(sym)

        if not cur_price or cur_price <= 0 or miss_price <= 0:
            continue

        fwd = (cur_price - miss_price) / miss_price

        if days >= 5 and rec.get("forward_return_5d") is None:
            rec["forward_return_5d"] = round(fwd, 6)
            logger.info(
                "[DEFERRED_METRICS] fwd_5d: symbol=%s entry_id=%s return=%.4f",
                sym, eid, fwd,
            )
        if days >= 20 and rec.get("forward_return_20d") is None:
            rec["forward_return_20d"] = round(fwd, 6)
            logger.info(
                "[DEFERRED_METRICS] fwd_20d: symbol=%s entry_id=%s return=%.4f",
                sym, eid, fwd,
            )

    tmp = tracking_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(tracking_path)
