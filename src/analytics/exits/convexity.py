"""
src/analytics/exits/convexity.py — Exit convexity analyzer + runner retention diagnostics

Classes:
  ExitConvexityAnalyzer      — per-trade MFE/MAE/vol/convexity metrics
  RunnerRetentionDiagnostics — batch aggregation across trades
  ExitEfficiencyScorer       — composite exit efficiency score

Analytics only. No execution mutation. Deterministic for fixed inputs.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

from src.analytics.exits.models import (
    CompletedTrade,
    ExitConvexityRecord,
    HoldingPath,
    PostExitContinuation,
    RunnerRetentionSummary,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_PREMATURE_EXIT_THRESHOLD = 0.5        # post_exit > 50% of realized → premature
_CONVEXITY_TRUNCATION_THRESHOLD = 0.7  # capture_rate < 70% → truncated
_VOL_INSENSITIVE_THRESHOLD = 0.5       # var-adj return < 50% → vol-insensitive
_RUNNER_RETAINED_THRESHOLD = 0.8       # runner_retention > 80% → retained
_EPS = 1e-10                           # avoid div/0
_ANNUALIZE = math.sqrt(252)

# ── Efficiency score weights ──────────────────────────────────────────────────
_W_CONVEXITY = 0.40
_W_GIVEBACK = 0.30
_W_CONTINUATION = 0.30

# ── Runner retention summary top-N ───────────────────────────────────────────
_TOP_N = 10


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    return num / denom if abs(denom) > _EPS else fallback


def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _compute_excursions(
    entry_price: float,
    shares: int,
    prices: List[float],
) -> tuple[float, float, float, float, int]:
    """
    Returns (MFE, MAE, peak_unrealized_pnl, pnl_giveback_ratio, peak_idx).
    MFE >= 0, MAE <= 0.
    """
    if not prices or entry_price <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    peak_price = max(prices)
    trough_price = min(prices)
    peak_idx = prices.index(peak_price)

    mfe = _safe_div(peak_price - entry_price, entry_price, 0.0)
    mae = _safe_div(trough_price - entry_price, entry_price, 0.0)
    peak_pnl = max(0.0, (peak_price - entry_price) * shares)

    exit_price = prices[-1]
    realized_pnl = (exit_price - entry_price) * shares
    giveback = _safe_div(peak_pnl - realized_pnl, peak_pnl, 0.0) if peak_pnl > 0 else 0.0
    giveback = _clip(giveback)

    return mfe, mae, peak_pnl, giveback, peak_idx


def _compute_volatility(prices: List[float]) -> float:
    """Annualized daily close-to-close volatility. 0.0 if < 2 data points."""
    if len(prices) < 2:
        return 0.0
    returns = [
        _safe_div(prices[i] - prices[i - 1], prices[i - 1], 0.0)
        for i in range(1, len(prices))
    ]
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    return math.sqrt(variance) * _ANNUALIZE


def _compute_trend_stability(mfe: float, mae: float) -> float:
    """
    Measures trend consistency: 1.0 = no adverse excursion, 0.0 = MAE equals MFE.
    """
    return _clip(1.0 - _safe_div(abs(mae), mfe + _EPS, 0.0))


class ExitConvexityAnalyzer:
    """
    Computes exit convexity diagnostics for a single completed trade.

    All computations are pure functions of the inputs. Deterministic.
    """

    def analyze(
        self,
        trade: CompletedTrade,
        holding_path: HoldingPath,
        continuation: Optional[PostExitContinuation] = None,
    ) -> ExitConvexityRecord:
        """
        Produce a full ExitConvexityRecord for the given trade.

        Args:
            trade:        Completed trade record.
            holding_path: Price/equity path during hold. May be empty.
            continuation: Optional post-exit continuation metrics.

        Returns:
            ExitConvexityRecord (deterministic for fixed inputs).
        """
        prices = [p["price"] for p in holding_path.holding_price_path if "price" in p]
        days = [p["day"] for p in holding_path.holding_price_path if "day" in p]

        # ── Excursions ────────────────────────────────────────────────────────
        mfe, mae, peak_pnl, giveback, peak_idx = _compute_excursions(
            trade.entry_price, trade.shares, prices
        )

        # ── Volatility ────────────────────────────────────────────────────────
        vol = _compute_volatility(prices)
        vol_adj_return = (
            _safe_div(trade.realized_return, vol, trade.realized_return)
            if vol > 0
            else trade.realized_return
        )

        # ── Holding-path timing ───────────────────────────────────────────────
        time_to_peak = days[peak_idx] if (days and peak_idx < len(days)) else 0.0
        exit_distance = max(0.0, trade.holding_duration_days - time_to_peak)

        peak_price = prices[peak_idx] if prices else trade.entry_price
        exit_price = prices[-1] if prices else trade.exit_price
        post_peak_decay = (
            _safe_div(
                (peak_price - exit_price),
                peak_price * exit_distance,
                0.0,
            )
            if exit_distance > 0 and peak_price > 0
            else 0.0
        )

        trend_stability = _compute_trend_stability(mfe, mae)

        # ── Convexity diagnostics (no continuation yet) ───────────────────────
        convexity_capture = _clip(_safe_div(trade.realized_return, mfe + _EPS, 1.0))
        scorer = ExitEfficiencyScorer()

        # ── Apply continuation if available ───────────────────────────────────
        p1d = p3d = p5d = None
        b1d = b3d = b5d = None
        sect = None
        rel_trend = None

        runner_cont_score = 0.0
        runner_retention = 1.0
        premature_ratio = 0.0
        alpha_leakage: Optional[float] = None
        trend_pers_cap: Optional[float] = None

        if continuation is not None:
            p1d = continuation.post_exit_return_1d
            p3d = continuation.post_exit_return_3d
            p5d = continuation.post_exit_return_5d
            b1d = continuation.benchmark_relative_post_exit_return_1d
            b3d = continuation.benchmark_relative_post_exit_return_3d
            b5d = continuation.benchmark_relative_post_exit_return_5d
            sect = continuation.sector_relative_post_exit_return
            rel_trend = continuation.relative_trend_persistence_score

            pos5d = max(0.0, p5d) if p5d is not None else 0.0
            runner_cont_score = _safe_div(pos5d, mfe + _EPS, 0.0)
            runner_cont_score = max(0.0, runner_cont_score)

            total_move = trade.realized_return + pos5d
            runner_retention = _clip(
                _safe_div(trade.realized_return, total_move + _EPS, 1.0)
            )
            premature_ratio = _clip(
                _safe_div(pos5d, total_move + _EPS, 0.0)
            )
            alpha_leakage = pos5d
            trend_pers_cap = _clip(
                _safe_div(trade.realized_return, total_move + _EPS, 1.0)
            )

        efficiency = scorer.score_components(
            convexity_capture_rate=convexity_capture,
            pnl_giveback_ratio=giveback,
            premature_exit_ratio=premature_ratio,
        )

        # ── Flags ─────────────────────────────────────────────────────────────
        is_premature = (
            p5d is not None and p5d > 0 and p5d > trade.realized_return * _PREMATURE_EXIT_THRESHOLD
        )
        is_truncated = convexity_capture < _CONVEXITY_TRUNCATION_THRESHOLD
        is_vol_insensitive = (
            vol > 0 and
            trade.realized_return > 0 and
            _safe_div(trade.realized_return, vol, 1.0) < _VOL_INSENSITIVE_THRESHOLD
        )
        is_runner_kept = runner_retention > _RUNNER_RETAINED_THRESHOLD

        generated_at = datetime.now(timezone.utc).isoformat()

        return ExitConvexityRecord(
            record_id=trade.record_id,
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            entry_timestamp=trade.entry_timestamp,
            exit_timestamp=trade.exit_timestamp,
            holding_duration_days=trade.holding_duration_days,
            realized_return=trade.realized_return,
            exit_reason=trade.exit_reason,
            # Excursion
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            peak_unrealized_pnl=peak_pnl,
            pnl_giveback_ratio=giveback,
            # Volatility
            volatility_during_hold=vol,
            volatility_adjusted_return=vol_adj_return,
            # Post-exit continuation
            post_exit_return_1d=p1d,
            post_exit_return_3d=p3d,
            post_exit_return_5d=p5d,
            benchmark_relative_post_exit_return_1d=b1d,
            benchmark_relative_post_exit_return_3d=b3d,
            benchmark_relative_post_exit_return_5d=b5d,
            sector_relative_post_exit_return=sect,
            relative_trend_persistence_score=rel_trend,
            # Convexity diagnostics
            runner_continuation_score=runner_cont_score,
            convexity_capture_rate=convexity_capture,
            runner_retention_rate=runner_retention,
            premature_exit_ratio=premature_ratio,
            exit_efficiency_score=efficiency,
            post_exit_alpha_leakage=alpha_leakage,
            trend_persistence_capture=trend_pers_cap,
            # Holding-path analytics
            time_to_peak_pnl=time_to_peak,
            exit_distance_from_peak=exit_distance,
            post_peak_decay_rate=post_peak_decay,
            trend_stability_score=trend_stability,
            # Flags
            is_premature_exit=is_premature,
            is_convexity_truncated=is_truncated,
            is_volatility_insensitive=is_vol_insensitive,
            is_runner_retained=is_runner_kept,
            generated_at=generated_at,
            schema_version=SCHEMA_VERSION,
        )


class ExitEfficiencyScorer:
    """Computes composite exit efficiency score from component metrics."""

    def score_components(
        self,
        convexity_capture_rate: float,
        pnl_giveback_ratio: float,
        premature_exit_ratio: float,
    ) -> float:
        """
        Returns [0, 1] where 1.0 = ideal exit (captured MFE, no giveback, no continuation).
        """
        return round(
            _clip(convexity_capture_rate) * _W_CONVEXITY
            + _clip(1.0 - pnl_giveback_ratio) * _W_GIVEBACK
            + _clip(1.0 - premature_exit_ratio) * _W_CONTINUATION,
            6,
        )

    def score(self, record: ExitConvexityRecord) -> float:
        return self.score_components(
            convexity_capture_rate=record.convexity_capture_rate,
            pnl_giveback_ratio=record.pnl_giveback_ratio,
            premature_exit_ratio=record.premature_exit_ratio,
        )


class RunnerRetentionDiagnostics:
    """Aggregate runner retention diagnostics across a batch of records."""

    def compute(
        self,
        records: List[ExitConvexityRecord],
        top_n: int = _TOP_N,
    ) -> RunnerRetentionSummary:
        """
        Compute aggregate diagnostics. Deterministic for fixed inputs.
        """
        n = len(records)
        generated_at = datetime.now(timezone.utc).isoformat()

        if n == 0:
            return RunnerRetentionSummary(
                total_trades=0,
                premature_exit_count=0,
                convexity_truncated_count=0,
                volatility_insensitive_count=0,
                runner_retained_count=0,
                premature_exit_rate=0.0,
                convexity_truncation_rate=0.0,
                mean_convexity_capture_rate=0.0,
                mean_exit_efficiency_score=0.0,
                mean_pnl_giveback_ratio=0.0,
                mean_runner_retention_rate=0.0,
                mean_post_exit_alpha_leakage=None,
                top_premature_exits=[],
                top_convexity_truncations=[],
                generated_at=generated_at,
            )

        premature = [r for r in records if r.is_premature_exit]
        truncated = [r for r in records if r.is_convexity_truncated]
        vol_insensitive = [r for r in records if r.is_volatility_insensitive]
        runner_retained = [r for r in records if r.is_runner_retained]

        def _mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        leakage_vals = [
            r.post_exit_alpha_leakage
            for r in records
            if r.post_exit_alpha_leakage is not None
        ]
        mean_leakage: Optional[float] = _mean(leakage_vals) if leakage_vals else None

        # Top premature exits: highest post_exit_return_5d (continuation = most left on table)
        with_5d = [r for r in records if r.post_exit_return_5d is not None]
        top_premature = [
            r.record_id
            for r in sorted(with_5d, key=lambda x: -(x.post_exit_return_5d or 0.0))[:top_n]
        ]

        # Top convexity truncations: lowest convexity_capture_rate
        top_truncations = [
            r.record_id
            for r in sorted(records, key=lambda x: x.convexity_capture_rate)[:top_n]
        ]

        return RunnerRetentionSummary(
            total_trades=n,
            premature_exit_count=len(premature),
            convexity_truncated_count=len(truncated),
            volatility_insensitive_count=len(vol_insensitive),
            runner_retained_count=len(runner_retained),
            premature_exit_rate=_safe_div(len(premature), n, 0.0),
            convexity_truncation_rate=_safe_div(len(truncated), n, 0.0),
            mean_convexity_capture_rate=_mean([r.convexity_capture_rate for r in records]),
            mean_exit_efficiency_score=_mean([r.exit_efficiency_score for r in records]),
            mean_pnl_giveback_ratio=_mean([r.pnl_giveback_ratio for r in records]),
            mean_runner_retention_rate=_mean([r.runner_retention_rate for r in records]),
            mean_post_exit_alpha_leakage=mean_leakage,
            top_premature_exits=top_premature,
            top_convexity_truncations=top_truncations,
            generated_at=generated_at,
        )
