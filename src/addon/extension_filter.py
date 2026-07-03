"""
extension_filter.py — ExtensionFilter + PersistenceQualityScoring

Distinguishes persistent trend strength from late-stage exhaustion acceleration.
Prevents add-ons on overstretched or climaxed positions.

Hard gates (any one blocks the add-on):
  1. velocity       : unrealized_pnl_pct / hold_days > MAX_VELOCITY_RATE
  2. rsr_accel      : abs(rsr_momentum) > MAX_RSR_ACCELERATION
  3. climax_bar     : candle_range_ratio > MAX_CANDLE_RANGE_RATIO
  4. volume_exhaust : volume_ratio > MAX_VOLUME_RATIO
  5. isolated_mom   : rsr >= ISOLATED_RSR_THRESHOLD and portfolio_pnl < PORTFOLIO_DIVERGENCE_THRESHOLD

Persistence quality score [0.5, 1.0]:
  Reduces confidence_score when metrics are elevated but below hard gate.
  Smooth continuation → score near 1.0.
  Elevated-but-not-blocked acceleration → score toward 0.5.

Pure functions. No I/O. Deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Hard gate thresholds ──────────────────────────────────────────────────────
MAX_VELOCITY_RATE: float              = 0.020   # pnl% per calendar day (2%/day)
MAX_RSR_ACCELERATION: float           = 5.0     # abs(rsr_momentum) pts
MAX_CANDLE_RANGE_RATIO: float         = 2.5     # today H-L / avg H-L
MAX_VOLUME_RATIO: float               = 4.0     # today volume / avg volume
ISOLATED_RSR_THRESHOLD: float         = 85.0    # RSR >= this = very hot stock
PORTFOLIO_DIVERGENCE_THRESHOLD: float = -0.03   # portfolio dd < -3% = breadth misalign

# ── PQS penalty ramp start points (below = zero penalty) ─────────────────────
_PQS_VEL_START: float   = 0.005   # velocity ramp start
_PQS_ACCEL_START: float = 2.0     # rsr momentum abs ramp start
_PQS_RANGE_START: float = 1.5     # candle range ratio ramp start
_PQS_VOL_START: float   = 2.0     # volume ratio ramp start


@dataclass
class ExtensionInput:
    """All inputs required for extension filter and quality analysis."""
    symbol: str
    unrealized_pnl_pct: float          # total unrealized return (decimal)
    hold_days: int                     # calendar or business days held
    rsr: float                         # current RSR value
    rsr_rank: int                      # current RSR rank (1 = best)
    rsr_momentum: float                # RSR change over recent period (signed)
    entry_price: float                 # original entry price (¥)
    current_price: float               # latest price (¥)
    trailing_stop_price: float         # trailing stop price (0 if unknown)
    regime_bear: bool                  # portfolio drawdown < -5%
    portfolio_pnl_pct: float           # portfolio return from peak (negative = DD)
    candle_range_ratio: float = 1.0    # today H-L / avg H-L (default: neutral)
    volume_ratio: float = 1.0          # today volume / avg volume (default: neutral)


@dataclass
class ExtensionResult:
    """Output of check_extension() for one symbol."""
    extension_blocked: bool
    block_reasons: list[str] = field(default_factory=list)
    persistence_quality_score: float = 1.0   # [0.5, 1.0]
    velocity_rate: float = 0.0               # computed pnl per day (for logging)
    atr_estimate: float = 0.0               # estimated ATR from trailing stop


def _penalty(value: float, start: float, max_at: float, weight: float) -> float:
    """Linear penalty ramp: 0 when value <= start, weight when value >= max_at."""
    if max_at <= start or value <= start:
        return 0.0
    ratio = min(1.0, (value - start) / (max_at - start))
    return ratio * weight


def _compute_persistence_quality_score(
    velocity_rate: float,
    rsr_momentum_abs: float,
    candle_range_ratio: float,
    volume_ratio: float,
) -> float:
    """
    Persistence quality score [0.5, 1.0].

    Higher = smoother, more sustained trend → stronger add-on candidate.
    Lower  = elevated acceleration / range / volume → reduce confidence, do not block.
    """
    p_vel   = _penalty(velocity_rate,      _PQS_VEL_START,   MAX_VELOCITY_RATE,       0.30)
    p_accel = _penalty(rsr_momentum_abs,   _PQS_ACCEL_START, MAX_RSR_ACCELERATION,    0.25)
    p_range = _penalty(candle_range_ratio, _PQS_RANGE_START, MAX_CANDLE_RANGE_RATIO,  0.25)
    p_vol   = _penalty(volume_ratio,       _PQS_VOL_START,   MAX_VOLUME_RATIO,        0.20)
    total_penalty = p_vel + p_accel + p_range + p_vol
    return round(max(0.5, 1.0 - total_penalty), 4)


def check_extension(ext: ExtensionInput) -> ExtensionResult:
    """
    Check whether a confirmed winner shows extension or exhaustion patterns.

    Returns ExtensionResult with:
      extension_blocked=True  → add-on suppressed (exhaustion / climax signal)
      persistence_quality_score → multiplied into confidence_score when not blocked
    """
    block_reasons: list[str] = []

    # Derived: velocity (pnl% per calendar day)
    velocity_rate = ext.unrealized_pnl_pct / max(1, ext.hold_days)

    # Derived: ATR estimate from trailing stop  (stop ≈ price − 3×ATR)
    atr_estimate = 0.0
    if ext.trailing_stop_price > 0 and ext.current_price > ext.trailing_stop_price:
        atr_estimate = (ext.current_price - ext.trailing_stop_price) / 3.0

    rsr_momentum_abs = abs(ext.rsr_momentum)

    # Gate 1: velocity — parabolic vertical move
    if velocity_rate > MAX_VELOCITY_RATE:
        block_reasons.append(
            f"velocity={velocity_rate:.4f}/day > max={MAX_VELOCITY_RATE:.4f}"
        )

    # Gate 2: RSR acceleration — explosive ranking jump
    if rsr_momentum_abs > MAX_RSR_ACCELERATION:
        block_reasons.append(
            f"rsr_accel={rsr_momentum_abs:.2f} > max={MAX_RSR_ACCELERATION:.2f}"
        )

    # Gate 3: climax bar — single-bar range explosion
    if ext.candle_range_ratio > MAX_CANDLE_RANGE_RATIO:
        block_reasons.append(
            f"climax_bar: range_ratio={ext.candle_range_ratio:.2f}"
            f" > max={MAX_CANDLE_RANGE_RATIO:.2f}"
        )

    # Gate 4: volume exhaustion — high-volume reversal risk
    if ext.volume_ratio > MAX_VOLUME_RATIO:
        block_reasons.append(
            f"volume_exhaustion: vol_ratio={ext.volume_ratio:.2f}"
            f" > max={MAX_VOLUME_RATIO:.2f}"
        )

    # Gate 5: isolated momentum — stock diverging from sector/portfolio breadth
    if (ext.rsr >= ISOLATED_RSR_THRESHOLD
            and ext.portfolio_pnl_pct < PORTFOLIO_DIVERGENCE_THRESHOLD):
        block_reasons.append(
            f"isolated_momentum: rsr={ext.rsr:.1f}>={ISOLATED_RSR_THRESHOLD:.1f}"
            f" portfolio_pnl={ext.portfolio_pnl_pct:.1%}"
            f"<{PORTFOLIO_DIVERGENCE_THRESHOLD:.1%}"
        )

    # Persistence quality score (always computed; used to scale confidence when not blocked)
    pqs = _compute_persistence_quality_score(
        velocity_rate=velocity_rate,
        rsr_momentum_abs=rsr_momentum_abs,
        candle_range_ratio=ext.candle_range_ratio,
        volume_ratio=ext.volume_ratio,
    )

    return ExtensionResult(
        extension_blocked=len(block_reasons) > 0,
        block_reasons=block_reasons,
        persistence_quality_score=pqs,
        velocity_rate=velocity_rate,
        atr_estimate=atr_estimate,
    )
