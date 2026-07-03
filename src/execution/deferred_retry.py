"""
src/execution/deferred_retry.py
Safe retry engine for the deferred entry queue.

Called once per morning before fresh signal processing:
  1. expire_stale() — remove TTL-expired / max-retry entries
  2. evaluate_retry_candidates() — check all active entries
  3. apply_retry_result() — update queue state after execution attempt

SAFETY INVARIANTS (must NEVER be violated):
  - sector_cap is always enforced (never bypassed)
  - max_open / max_positions enforced
  - max_alloc_pct enforced
  - no synthetic leverage created
  - account state not mutated before fill confirmation
  - deferred logic is advisory only until actual broker execution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.execution.deferred_queue import DeferredEntry, DeferredQueue, MAX_RETRY_COUNT
from src.execution.regime_drift import check_regime_drift

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

MIN_RETRY_SIGNAL_SCORE:   float = 0.0
MIN_SECTOR_HEADROOM:      float = 0.01   # must have >1% remaining cap to retry
CASH_SUFFICIENCY_MARGIN:  float = 1.05   # require 5% cash slack


@dataclass
class RetryEvaluation:
    """Result of evaluating one deferred entry for retry."""
    entry:        DeferredEntry
    should_retry: bool
    drop_reason:  Optional[str]   # None when should_retry=True
    new_score:    Optional[float] # current signal score (None if unknown)


def evaluate_retry_candidates(
    queue:                DeferredQueue,
    signal_scores:        dict[str, float],
    tradeable_symbols:    set[str],
    sector_weights:       dict[str, float],
    sector_caps:          dict[str, float],
    current_positions:    dict[str, int],
    max_open:             int,
    portfolio_equity:     float,
    available_cash:       float,
    current_rsr_ranks:    dict[str, int] | None = None,
    current_sector_ranks: dict[str, int] | None = None,
    current_regime:       str | None            = None,
) -> list[RetryEvaluation]:
    """
    Evaluate all active deferred entries for retry eligibility.

    All risk filters are ENFORCED. No filter bypass is permitted.

    Args:
        queue:                DeferredQueue to query
        signal_scores:        symbol → current signal score (0.0 = invalid)
        tradeable_symbols:    symbols valid for trading today
        sector_weights:       sector → currently allocated weight fraction
        sector_caps:          sector → maximum weight fraction (from config)
        current_positions:    symbol → broker qty (sole source of truth)
        max_open:             maximum concurrent open positions (from CLAUDE.md)
        portfolio_equity:     total equity from broker (not estimated)
        available_cash:       available cash from broker (not estimated)
        current_rsr_ranks:    symbol → current universe RSR rank (optional; None skips check)
        current_sector_ranks: symbol → current sector rank (optional; None skips check)
        current_regime:       current MarketRegime value string (optional; None skips check)

    Returns:
        List of RetryEvaluation, ordered: eligible first, then by signal score desc.
    """
    active  = queue.get_active()
    results = [
        _evaluate_single(
            entry                = e,
            signal_scores        = signal_scores,
            tradeable_symbols    = tradeable_symbols,
            sector_weights       = sector_weights,
            sector_caps          = sector_caps,
            current_positions    = current_positions,
            max_open             = max_open,
            portfolio_equity     = portfolio_equity,
            available_cash       = available_cash,
            current_rsr_ranks    = current_rsr_ranks,
            current_sector_ranks = current_sector_ranks,
            current_regime       = current_regime,
        )
        for e in active
    ]
    results.sort(key=lambda r: (0 if r.should_retry else 1, -(r.new_score or 0.0)))
    return results


def _evaluate_single(
    entry:                DeferredEntry,
    signal_scores:        dict[str, float],
    tradeable_symbols:    set[str],
    sector_weights:       dict[str, float],
    sector_caps:          dict[str, float],
    current_positions:    dict[str, int],
    max_open:             int,
    portfolio_equity:     float,
    available_cash:       float,
    current_rsr_ranks:    dict[str, int] | None = None,
    current_sector_ranks: dict[str, int] | None = None,
    current_regime:       str | None            = None,
) -> RetryEvaluation:
    sym = entry.symbol

    # 1. TTL / retry exhaustion
    if entry.is_expired:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="ttl_expired", new_score=None)
    if entry.retry_count >= MAX_RETRY_COUNT:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="max_retry_exceeded", new_score=None)

    # 2. Symbol tradeable today
    if sym not in tradeable_symbols:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="symbol_not_tradeable", new_score=None)

    # 3. Signal still valid
    new_score = signal_scores.get(sym)
    if new_score is None or new_score <= MIN_RETRY_SIGNAL_SCORE:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="signal_invalidated", new_score=new_score)

    # 3.5. Regime / rank drift invalidation
    drift = check_regime_drift(
        entry               = entry,
        current_rsr_rank    = current_rsr_ranks.get(sym) if current_rsr_ranks else None,
        current_sector_rank = current_sector_ranks.get(sym) if current_sector_ranks else None,
        current_regime      = current_regime,
    )
    if drift.should_drop:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason=drift.reason, new_score=new_score)

    # 4. Position limit — sector_cap / max_positions enforced
    open_count = sum(1 for q in current_positions.values() if q > 0)
    if open_count >= max_open and sym not in current_positions:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="max_positions_reached", new_score=new_score)

    # 5. Sector cap — headroom must exist (NEVER bypass)
    sector           = entry.sector
    sector_cap       = sector_caps.get(sector, 0.25)
    current_sec_wt   = sector_weights.get(sector, 0.0)
    sector_headroom  = sector_cap - current_sec_wt

    if sector_headroom <= MIN_SECTOR_HEADROOM:
        logger.info(
            "[DEFERRED_RETRY] entry_id=%s symbol=%s sector=%s "
            "headroom=%.4f cap=%.4f → sector_cap_still_binding",
            entry.entry_id, sym, sector, sector_headroom, sector_cap,
        )
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="sector_cap_still_binding",
                               new_score=new_score)

    # 6. Cash sufficiency — desired_weight × equity must fit in available cash
    desired_cash = entry.desired_weight * max(portfolio_equity, 0.0)
    if desired_cash > available_cash * CASH_SUFFICIENCY_MARGIN:
        return RetryEvaluation(entry=entry, should_retry=False,
                               drop_reason="insufficient_cash", new_score=new_score)

    logger.info(
        "[DEFERRED_RETRY] entry_id=%s symbol=%s sector=%s "
        "retry_count=%d new_score=%.4f sector_headroom=%.4f → ELIGIBLE",
        entry.entry_id, sym, sector,
        entry.retry_count, new_score, sector_headroom,
    )
    return RetryEvaluation(entry=entry, should_retry=True,
                           drop_reason=None, new_score=new_score)


def apply_retry_result(
    queue:      DeferredQueue,
    evaluation: RetryEvaluation,
    executed:   bool,
) -> None:
    """
    Update queue state after a retry evaluation.

    Args:
        queue:      DeferredQueue to update
        evaluation: result from evaluate_retry_candidates
        executed:   True if actual broker order was placed and confirmed

    Behaviour:
        should_retry=False              → remove from queue (drop_reason logged)
        should_retry=True, executed     → remove from queue (success)
        should_retry=True, not executed → increment retry_count, keep in queue
    """
    entry = evaluation.entry

    if not evaluation.should_retry:
        queue.remove(entry.entry_id, reason=evaluation.drop_reason or "drop")
        return

    if executed:
        queue.remove(entry.entry_id, reason="retry_executed")
        logger.info(
            "[DEFERRED_EXECUTE] entry_id=%s symbol=%s sector=%s retry_count=%d",
            entry.entry_id, entry.symbol, entry.sector, entry.retry_count,
        )
    else:
        now                = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        entry.retry_count  += 1
        entry.updated_at   = now
        entry.last_failed_at = now   # stamp for cooldown enforcement
        queue.update(entry)
        logger.info(
            "[DEFERRED_RETRY] entry_id=%s symbol=%s retry_count=%d "
            "ttl_days=%d is_expired=%s last_failed_at=%s",
            entry.entry_id, entry.symbol, entry.retry_count,
            entry.ttl_days, entry.is_expired, now,
        )
