"""
src/analytics/exit_intelligence/deterioration.py — Deterioration Modeling Layer

5 pure analyzers. No side effects. No ML. Deterministic.
Each takes a list of ExitPathSnapshots (chronological) and returns metrics.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from .models import (
    ConvexityRetentionMetrics,
    MomentumDecayMetrics,
    PeakDecayVelocityMetrics,
    PersistenceMetrics,
    StructuralFailureMetrics,
)
from .observation import ExitPathSnapshot

JST = timezone(timedelta(hours=9))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _linear_slope(ys: List[float]) -> float:
    """Least-squares slope of y over evenly-spaced x=0..n-1. Returns 0 if <2 points."""
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


# ── 1. MomentumDecayAnalyzer ──────────────────────────────────────────────────

class MomentumDecayAnalyzer:
    """
    Observe momentum velocity decay, breakout exhaustion,
    volume deterioration, range contraction, upper-shadow expansion,
    failed continuation behavior.
    """

    WINDOW = 5  # observation window in days

    def analyze(
        self,
        snapshots: List[ExitPathSnapshot],
        as_of_date: Optional[str] = None,
    ) -> MomentumDecayMetrics:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        snapshots = sorted(snapshots, key=lambda s: s.day_number)
        symbol = snapshots[-1].symbol
        date = as_of_date or snapshots[-1].snapshot_date
        window = snapshots[-self.WINDOW:]

        # Momentum velocity: slope of momentum over window
        mom_series = [s.momentum for s in window]
        momentum_velocity = _linear_slope(mom_series)

        # Volume deterioration: fraction of days below 1.0 volume ratio
        vol_below = sum(1 for s in window if s.volume_ratio < 1.0)
        volume_deterioration = _clamp(vol_below / max(len(window), 1))

        # Range contraction: high upper_shadow = range is upper-heavy (exhaustion)
        # volatility_compression close to 1 = range tightening
        avg_vol_compression = sum(s.volatility_compression for s in window) / len(window)
        range_contraction = _clamp(avg_vol_compression)

        # Upper shadow expansion
        avg_upper_shadow = sum(s.upper_shadow_ratio for s in window) / len(window)
        upper_shadow_expansion = _clamp(avg_upper_shadow)

        # Breakout exhaustion: momentum declining after being positive
        pos_then_neg = sum(
            1 for i in range(1, len(window))
            if window[i - 1].momentum > 0 and window[i].momentum < window[i - 1].momentum
        )
        breakout_exhaustion_score = _clamp(pos_then_neg / max(len(window) - 1, 1))

        # Failed continuation: exit_signal_count > 0 in window
        failed_continuation_count = sum(1 for s in window if s.exit_signal_count > 0)

        return MomentumDecayMetrics(
            symbol=symbol,
            as_of_date=date,
            momentum_velocity=momentum_velocity,
            volume_deterioration=volume_deterioration,
            range_contraction=range_contraction,
            upper_shadow_expansion=upper_shadow_expansion,
            breakout_exhaustion_score=breakout_exhaustion_score,
            failed_continuation_count=failed_continuation_count,
        )


# ── 2. StructuralFailureDetector ──────────────────────────────────────────────

class StructuralFailureDetector:
    """
    Detect structural failure patterns:
    breakout failure, VWAP reclaim failure, RS collapse,
    breadth deterioration, failed trend continuation,
    support rejection, convexity collapse.
    """

    RS_COLLAPSE_THRESHOLD = 10.0     # RS rank drop in 3d considered collapse
    BREADTH_DETERIORATION_THRESHOLD = 0.4  # breadth < 0.4 = deteriorating
    CONVEXITY_COLLAPSE_THRESHOLD = 0.20  # >20% distance from peak

    def analyze(
        self,
        snapshots: List[ExitPathSnapshot],
        as_of_date: Optional[str] = None,
    ) -> StructuralFailureMetrics:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        snapshots = sorted(snapshots, key=lambda s: s.day_number)
        latest = snapshots[-1]
        symbol = latest.symbol
        date = as_of_date or latest.snapshot_date
        last_3 = snapshots[-3:]

        # Breakout failure: momentum was positive, now negative AND distance from peak > 5%
        was_positive = any(s.momentum > 0.02 for s in snapshots[:max(len(snapshots) - 3, 1)])
        now_negative = latest.momentum < 0
        breakout_failure = bool(was_positive and now_negative and latest.distance_from_peak_pct > 0.05)

        # VWAP reclaim failure: price below VWAP proxy (momentum < 0 after bounce)
        # Use momentum + upper_shadow as proxy: momentum<0 and high upper shadow = rejection
        vwap_reclaim_failure = bool(latest.momentum < 0 and latest.upper_shadow_ratio > 0.6)

        # RS collapse: RS drop over 3d
        if len(last_3) >= 2:
            rs_drop = last_3[0].rs_rank - latest.rs_rank
            rs_collapse = rs_drop > self.RS_COLLAPSE_THRESHOLD
        else:
            rs_collapse = False

        # Breadth deterioration
        breadth_deterioration = bool(latest.breadth_score < self.BREADTH_DETERIORATION_THRESHOLD)

        # Failed trend continuation: downward slope in unrealized PnL over 3d
        if len(last_3) >= 2:
            pnl_slope = _linear_slope([s.unrealized_pnl_fraction for s in last_3])
            failed_trend_continuation = pnl_slope < -0.01  # losing > 1% per day
        else:
            failed_trend_continuation = False

        # Support rejection: price below entry (unrealized_pnl < 0)
        support_rejection = bool(latest.unrealized_pnl_fraction < -0.02)

        # Convexity collapse: >20% distance from peak
        convexity_collapse = bool(
            latest.distance_from_peak_pct > self.CONVEXITY_COLLAPSE_THRESHOLD
        )

        # Composite structural failure score
        flags = [
            breakout_failure, vwap_reclaim_failure, rs_collapse,
            breadth_deterioration, failed_trend_continuation,
            support_rejection, convexity_collapse,
        ]
        structural_failure_score = _clamp(sum(flags) / len(flags))

        return StructuralFailureMetrics(
            symbol=symbol,
            as_of_date=date,
            breakout_failure=breakout_failure,
            vwap_reclaim_failure=vwap_reclaim_failure,
            rs_collapse=rs_collapse,
            breadth_deterioration=breadth_deterioration,
            failed_trend_continuation=failed_trend_continuation,
            support_rejection=support_rejection,
            convexity_collapse=convexity_collapse,
            structural_failure_score=structural_failure_score,
        )


# ── 3. PeakDecayVelocityAnalyzer ──────────────────────────────────────────────

class PeakDecayVelocityAnalyzer:
    """
    Distinguish slow deterioration vs orderly consolidation vs fast convex collapse.
    Based on distance-from-peak trajectory.
    """

    SLOW_THRESHOLD = 0.005     # < 0.5%/day = slow
    FAST_THRESHOLD = 0.020     # > 2.0%/day = fast collapse

    def analyze(
        self,
        snapshots: List[ExitPathSnapshot],
        as_of_date: Optional[str] = None,
    ) -> PeakDecayVelocityMetrics:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        snapshots = sorted(snapshots, key=lambda s: s.day_number)
        latest = snapshots[-1]
        symbol = latest.symbol
        date = as_of_date or latest.snapshot_date

        # Find when peak occurred
        peak_day_idx = max(
            range(len(snapshots)),
            key=lambda i: snapshots[i].peak_price_so_far,
        )
        days_from_peak = latest.day_number - snapshots[peak_day_idx].day_number

        # Peak decay velocity
        if days_from_peak > 0 and latest.distance_from_peak_pct > 0:
            peak_decay_velocity = latest.distance_from_peak_pct / days_from_peak
        else:
            peak_decay_velocity = 0.0

        # Deterioration slope (distance_from_peak over last 5 days)
        window = snapshots[-5:]
        dfp_series = [s.distance_from_peak_pct for s in window]
        deterioration_slope = _linear_slope(dfp_series)

        # Drawdown acceleration: Δslope (second derivative)
        if len(window) >= 3:
            first_half = window[: len(window) // 2]
            second_half = window[len(window) // 2 :]
            slope_first = _linear_slope([s.distance_from_peak_pct for s in first_half])
            slope_second = _linear_slope([s.distance_from_peak_pct for s in second_half])
            drawdown_acceleration = slope_second - slope_first
        else:
            drawdown_acceleration = 0.0

        # Classification
        if peak_decay_velocity < self.SLOW_THRESHOLD:
            decay_classification = "slow"
        elif peak_decay_velocity > self.FAST_THRESHOLD:
            decay_classification = "fast_convex"
        else:
            decay_classification = "orderly"

        return PeakDecayVelocityMetrics(
            symbol=symbol,
            as_of_date=date,
            peak_decay_velocity=peak_decay_velocity,
            drawdown_acceleration=drawdown_acceleration,
            days_from_peak=max(days_from_peak, 0),
            deterioration_slope=deterioration_slope,
            decay_classification=decay_classification,
        )


# ── 4. ConvexityRetentionAnalyzer ─────────────────────────────────────────────

class ConvexityRetentionAnalyzer:
    """
    Identify constructive volatility compression, high-tight-flag behavior,
    persistent leaders, trend continuation quality.
    """

    HTF_VOL_COMPRESSION_MIN = 0.6   # vol_compression >= 0.6 = potential HTF
    HTF_DISTANCE_MAX = 0.05         # distance_from_peak <= 5% = tight
    HTF_RS_MIN = 75.0               # RS rank must be high
    LEADER_RS_MIN = 80.0
    LEADER_PERSISTENCE_DAYS = 3

    def analyze(
        self,
        snapshots: List[ExitPathSnapshot],
        as_of_date: Optional[str] = None,
    ) -> ConvexityRetentionMetrics:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        snapshots = sorted(snapshots, key=lambda s: s.day_number)
        latest = snapshots[-1]
        symbol = latest.symbol
        date = as_of_date or latest.snapshot_date
        window = snapshots[-5:]

        # Constructive compression: high volatility_compression + low distance_from_peak
        avg_compression = sum(s.volatility_compression for s in window) / len(window)
        is_constructive_compression = bool(
            avg_compression >= self.HTF_VOL_COMPRESSION_MIN
            and latest.distance_from_peak_pct <= 0.08
            and latest.unrealized_pnl_fraction >= 0
        )

        # High-tight-flag: tight consolidation near peak with high RS
        is_high_tight_flag = bool(
            latest.volatility_compression >= self.HTF_VOL_COMPRESSION_MIN
            and latest.distance_from_peak_pct <= self.HTF_DISTANCE_MAX
            and latest.rs_rank >= self.HTF_RS_MIN
            and latest.unrealized_pnl_fraction > 0
        )

        # Persistent leader: RS consistently high
        leader_days = sum(1 for s in window if s.rs_rank >= self.LEADER_RS_MIN)
        is_persistent_leader = leader_days >= self.LEADER_PERSISTENCE_DAYS

        # Trend continuation quality: positive momentum slope + positive PnL
        mom_slope = _linear_slope([s.momentum for s in window])
        pnl_slope = _linear_slope([s.unrealized_pnl_fraction for s in window])
        trend_continuation_quality = _clamp(
            0.5 * _clamp((mom_slope + 0.01) / 0.02)
            + 0.5 * _clamp((pnl_slope + 0.005) / 0.01)
        )

        # Convexity retention score
        score_parts = [
            0.3 * (1.0 - _clamp(latest.distance_from_peak_pct / 0.20)),
            0.2 * _clamp(latest.rs_rank / 100.0),
            0.2 * latest.volatility_compression,
            0.2 * trend_continuation_quality,
            0.1 * (1.0 if is_persistent_leader else 0.0),
        ]
        convexity_retention_score = _clamp(sum(score_parts))

        return ConvexityRetentionMetrics(
            symbol=symbol,
            as_of_date=date,
            is_constructive_compression=is_constructive_compression,
            is_high_tight_flag=is_high_tight_flag,
            is_persistent_leader=is_persistent_leader,
            trend_continuation_quality=trend_continuation_quality,
            convexity_retention_score=convexity_retention_score,
        )


# ── 5. PersistenceExtensionLayer ──────────────────────────────────────────────

class PersistenceExtensionLayer:
    """
    Measure trend persistence durability, leader persistence,
    regime persistence, sector leadership continuity.
    """

    def analyze(
        self,
        snapshots: List[ExitPathSnapshot],
        as_of_date: Optional[str] = None,
    ) -> PersistenceMetrics:
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        snapshots = sorted(snapshots, key=lambda s: s.day_number)
        latest = snapshots[-1]
        symbol = latest.symbol
        date = as_of_date or latest.snapshot_date
        window = snapshots[-10:]

        # Trend persistence durability: fraction of days with positive momentum
        pos_days = sum(1 for s in window if s.momentum > 0)
        trend_persistence_durability = _clamp(pos_days / len(window))

        # Leader persistence: fraction of days with RS > 75
        leader_days = sum(1 for s in window if s.rs_rank >= 75.0)
        leader_persistence_score = _clamp(leader_days / len(window))

        # Regime persistence: fraction of days with same regime as latest
        same_regime_days = sum(1 for s in window if s.regime == latest.regime)
        regime_persistence_score = _clamp(same_regime_days / len(window))

        # Sector leadership continuity: use trend_persistence field from snapshots
        avg_trend_persistence = sum(s.trend_persistence for s in window) / len(window)
        sector_leadership_continuity = _clamp(avg_trend_persistence)

        combined_persistence_score = _clamp(
            0.35 * trend_persistence_durability
            + 0.30 * leader_persistence_score
            + 0.20 * regime_persistence_score
            + 0.15 * sector_leadership_continuity
        )

        return PersistenceMetrics(
            symbol=symbol,
            as_of_date=date,
            trend_persistence_durability=trend_persistence_durability,
            leader_persistence_score=leader_persistence_score,
            regime_persistence_score=regime_persistence_score,
            sector_leadership_continuity=sector_leadership_continuity,
            combined_persistence_score=combined_persistence_score,
        )
