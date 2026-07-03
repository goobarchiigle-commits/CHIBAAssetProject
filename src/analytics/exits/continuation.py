"""
src/analytics/exits/continuation.py — Post-exit continuation tracker + vol-adjusted analytics

Classes:
  PostExitContinuationTracker    — computes continuation metrics from post-exit price dicts
  VolatilityAdjustedExitAnalytics — enriches with vol-relative metrics

Analytics only. No execution mutation. Deterministic for fixed inputs.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from src.analytics.exits.models import (
    CompletedTrade,
    PostExitContinuation,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    return num / denom if abs(denom) > _EPS else fallback


def _price_return(entry: float, current: float) -> Optional[float]:
    """Fraction return from entry to current. None if entry invalid."""
    if entry <= 0 or current <= 0:
        return None
    return (current - entry) / entry


class PostExitContinuationTracker:
    """
    Computes post-exit continuation metrics from price dicts.

    All inputs are dicts of {days_offset: price} to avoid dependency on
    external market data loading — callers supply prices directly.
    This keeps computation pure and testable.
    """

    def compute(
        self,
        trade: CompletedTrade,
        post_exit_prices: Dict[int, float],
        benchmark_prices: Optional[Dict[int, float]] = None,
        sector_price_5d: Optional[float] = None,
    ) -> PostExitContinuation:
        """
        Compute post-exit continuation from prices supplied by caller.

        Args:
            trade:               Completed trade (for exit_price as base).
            post_exit_prices:    {days_offset: price} after exit, e.g. {1: 1010.0, 3: 1020.0, 5: 1030.0}.
            benchmark_prices:    {days_offset: price} of benchmark at same offsets.
            sector_price_5d:     Sector index price at 5d after exit (for sector-relative).

        Returns:
            PostExitContinuation (deterministic for fixed inputs).
        """
        ep = trade.exit_price

        p1d = _price_return(ep, post_exit_prices.get(1, 0) or 0)
        p3d = _price_return(ep, post_exit_prices.get(3, 0) or 0)
        p5d = _price_return(ep, post_exit_prices.get(5, 0) or 0)

        # Zero-value lookups → None (missing data)
        p1d = p1d if post_exit_prices.get(1) else None
        p3d = p3d if post_exit_prices.get(3) else None
        p5d = p5d if post_exit_prices.get(5) else None

        b1d = b3d = b5d = None
        if benchmark_prices:
            bp_ep = benchmark_prices.get(0) or ep  # benchmark at exit, fallback = trade price
            b1d_raw = _price_return(bp_ep, benchmark_prices.get(1, 0) or 0)
            b3d_raw = _price_return(bp_ep, benchmark_prices.get(3, 0) or 0)
            b5d_raw = _price_return(bp_ep, benchmark_prices.get(5, 0) or 0)

            if p1d is not None and b1d_raw is not None and benchmark_prices.get(1):
                b1d = p1d - b1d_raw
            if p3d is not None and b3d_raw is not None and benchmark_prices.get(3):
                b3d = p3d - b3d_raw
            if p5d is not None and b5d_raw is not None and benchmark_prices.get(5):
                b5d = p5d - b5d_raw

        sector_rel = None
        if sector_price_5d is not None and p5d is not None and ep > 0:
            sector_ret = _price_return(ep, sector_price_5d)
            if sector_ret is not None:
                sector_rel = p5d - sector_ret

        # relative_trend_persistence: how consistently the stock outperformed benchmark post-exit
        rel_trend: Optional[float] = None
        if b1d is not None and b3d is not None and b5d is not None:
            outperform_count = sum(1 for v in (b1d, b3d, b5d) if v > 0)
            rel_trend = outperform_count / 3.0

        return PostExitContinuation(
            record_id=trade.record_id,
            post_exit_return_1d=p1d,
            post_exit_return_3d=p3d,
            post_exit_return_5d=p5d,
            benchmark_relative_post_exit_return_1d=b1d,
            benchmark_relative_post_exit_return_3d=b3d,
            benchmark_relative_post_exit_return_5d=b5d,
            sector_relative_post_exit_return=sector_rel,
            relative_trend_persistence_score=rel_trend,
            schema_version=SCHEMA_VERSION,
        )


class VolatilityAdjustedExitAnalytics:
    """
    Enriches a PostExitContinuation with volatility-relative context.

    Returns a new PostExitContinuation with relative_trend_persistence_score
    recalculated after accounting for holding-period volatility scaling.
    """

    def compute(
        self,
        trade: CompletedTrade,
        continuation: PostExitContinuation,
        volatility_during_hold: float,
    ) -> PostExitContinuation:
        """
        Scale benchmark-relative returns by holding-period vol to produce
        a volatility-adjusted relative_trend_persistence_score.

        Vol scaling: divide post-exit returns by (1 + vol) to normalize
        across high/low volatility regimes.
        """
        if volatility_during_hold <= 0:
            return continuation

        scale = 1.0 + volatility_during_hold

        def _scale(val: Optional[float]) -> Optional[float]:
            return val / scale if val is not None else None

        b1d_s = _scale(continuation.benchmark_relative_post_exit_return_1d)
        b3d_s = _scale(continuation.benchmark_relative_post_exit_return_3d)
        b5d_s = _scale(continuation.benchmark_relative_post_exit_return_5d)

        # Recompute trend persistence with scaled values
        rel_trend: Optional[float] = None
        scaled = [v for v in (b1d_s, b3d_s, b5d_s) if v is not None]
        if len(scaled) == 3:
            outperform = sum(1 for v in scaled if v > 0)
            rel_trend = outperform / 3.0

        return PostExitContinuation(
            record_id=continuation.record_id,
            post_exit_return_1d=continuation.post_exit_return_1d,
            post_exit_return_3d=continuation.post_exit_return_3d,
            post_exit_return_5d=continuation.post_exit_return_5d,
            benchmark_relative_post_exit_return_1d=b1d_s,
            benchmark_relative_post_exit_return_3d=b3d_s,
            benchmark_relative_post_exit_return_5d=b5d_s,
            sector_relative_post_exit_return=continuation.sector_relative_post_exit_return,
            relative_trend_persistence_score=rel_trend if rel_trend is not None else continuation.relative_trend_persistence_score,
            schema_version=continuation.schema_version,
        )
