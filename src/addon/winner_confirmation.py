"""
winner_confirmation.py — WinnerConfirmationEngine

Confirms whether a held position qualifies as a "confirmed winner"
eligible for a discrete 100-share add-on entry.

All confirmation criteria must pass (AND logic):
  1. profit_threshold  : unrealized_pnl_pct >= MIN_PROFIT_THRESHOLD
  2. hold_duration     : hold_days >= MIN_HOLD_BEFORE_ADDON
  3. rsr_rank          : rsr_rank <= MAX_RSR_RANK_FOR_ADDON  (1=best)
  4. rsr_strength      : rsr >= MIN_RSR_STRENGTH
  5. trend_quality     : rsr_momentum > MIN_TREND_QUALITY   (no deterioration)
  6. deterioration     : deterioration_score <= MAX_DETERIORATION
  7. regime            : not sustained bear (if REGIME_BEAR_BLOCK)
  8. no_averaging_down : current_price > entry_price         (hard block)
  9. volatility        : atr_ratio <= MAX_ATR_RATIO

confidence_score [0,1] — higher = stronger conviction, used for ordering.

Pure functions. No I/O. Deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Confirmation thresholds ────────────────────────────────────────────────────
MIN_PROFIT_THRESHOLD: float  = 0.03    # +3% unrealized PnL required
MIN_HOLD_BEFORE_ADDON: int   = 5       # minimum business days held
MAX_RSR_RANK_FOR_ADDON: int  = 3       # must rank 1, 2, or 3 by RSR
MIN_RSR_STRENGTH: float      = 77.0    # RSR must exceed entry threshold (75)
MIN_TREND_QUALITY: float      = 0.0    # RSR momentum must be positive
MAX_DETERIORATION: float     = 0.30    # deterioration_score <= this
MAX_ATR_RATIO: float         = 2.0     # current ATR / entry ATR: block if excessive
REGIME_BEAR_BLOCK: bool      = True    # block in sustained bear regime

# ── Continuation breakout boost (compression→breakout phase transition) ────────
CONTINUATION_BREAKOUT_BOOST: float         = 1.15  # confidence score multiplier
CONTINUATION_BREAKOUT_MIN_COMPRESSION: float = 70.0  # min compression_score
CONTINUATION_BREAKOUT_MIN_VOLUME: float    = 1.30  # min volume_ratio_5d (expansion required)
CONTINUATION_BREAKOUT_MIN_RSR_MOM: float   = 0.0   # rsr_momentum must be positive


@dataclass
class WinnerConfirmation:
    """Result of winner confirmation check for one symbol."""
    symbol: str
    confirmed: bool
    confidence_score: float                  # [0,1]; 0 when not confirmed
    fail_reasons: list[str] = field(default_factory=list)
    # Input echoes for logging / deterministic replay
    unrealized_pnl_pct: float = 0.0
    hold_days: int = 0
    rsr_rank: int = 99
    rsr: float = 0.0
    trend_quality: float = 0.0
    deterioration_score: float = 0.0
    regime_bear: bool = False
    atr_ratio: float = 1.0
    # ── Continuation breakout intelligence (set by apply_continuation_breakout_boost) ─
    continuation_breakout: bool = False     # compression→breakout transition detected
    continuation_boost_applied: float = 1.0 # multiplier applied (1.0 = no boost)


def _score(
    unrealized_pnl_pct: float,
    rsr: float,
    deterioration_score: float,
    trend_quality: float,
    atr_ratio: float,
) -> float:
    """Weighted confidence score for a confirmed winner. Returns [0, 1]."""
    pnl_s   = min(1.0, unrealized_pnl_pct / 0.12)     # saturates at +12%
    rsr_s   = min(1.0, max(0.0, (rsr - 75.0) / 25.0)) # 75→0, 100→1
    detr_s  = max(0.0, 1.0 - deterioration_score)
    trend_s = min(1.0, max(0.0, trend_quality / 2.0))  # normalise momentum
    vol_s   = max(0.0, 1.0 - min(1.0, (atr_ratio - 1.0)))
    return round(
        pnl_s  * 0.30
        + rsr_s  * 0.25
        + detr_s * 0.20
        + trend_s * 0.15
        + vol_s  * 0.10,
        4,
    )


def apply_continuation_breakout_boost(
    conf: WinnerConfirmation,
    continuation_breakout: bool,
    compression_score: float = 0.0,
    volume_ratio_5d: float = 1.0,
    rsr_momentum: float = 0.0,
) -> tuple[WinnerConfirmation, float]:
    """
    Apply continuation breakout confidence boost to a confirmed winner.

    Boosts confidence_score by CONTINUATION_BREAKOUT_BOOST multiplier (capped at 1.0)
    when all conditions are met:
      - conf.confirmed == True
      - continuation_breakout == True (compression→breakout phase detected externally)
      - compression_score >= CONTINUATION_BREAKOUT_MIN_COMPRESSION
      - volume_ratio_5d >= CONTINUATION_BREAKOUT_MIN_VOLUME (expansion confirms breakout)
      - rsr_momentum >= CONTINUATION_BREAKOUT_MIN_RSR_MOM

    The boost affects addon priority ordering only (not sizing).
    Returns (possibly_modified_WinnerConfirmation, boost_multiplier_applied).
    boost_multiplier_applied = 1.0 when no boost was applied.

    Fail-safe: never raises; returns original conf on any error.
    """
    try:
        if not (conf.confirmed and continuation_breakout):
            return conf, 1.0
        if compression_score < CONTINUATION_BREAKOUT_MIN_COMPRESSION:
            return conf, 1.0
        if volume_ratio_5d < CONTINUATION_BREAKOUT_MIN_VOLUME:
            return conf, 1.0
        if rsr_momentum < CONTINUATION_BREAKOUT_MIN_RSR_MOM:
            return conf, 1.0

        from dataclasses import replace as _replace
        boosted = min(1.0, round(conf.confidence_score * CONTINUATION_BREAKOUT_BOOST, 4))
        return _replace(
            conf,
            confidence_score=boosted,
            continuation_breakout=True,
            continuation_boost_applied=CONTINUATION_BREAKOUT_BOOST,
        ), CONTINUATION_BREAKOUT_BOOST
    except Exception as exc:
        logger.warning("[WINNER] continuation boost failed: %s", exc)
        return conf, 1.0


def check_winner(
    symbol: str,
    unrealized_pnl_pct: float,
    hold_days: int,
    rsr_rank: int,
    rsr: float,
    trend_quality: float,
    deterioration_score: float,
    regime_bear: bool,
    atr_ratio: float = 1.0,
    current_price: float = 0.0,
    entry_price: float = 0.0,
    extension_input=None,  # ExtensionInput | None
) -> WinnerConfirmation:
    """
    Check if a held position qualifies as a confirmed winner.

    Returns WinnerConfirmation with confirmed=True only when ALL gates pass.
    confirmed=False positions never receive add-on orders.

    extension_input: optional ExtensionInput for exhaustion/climax suppression.
      When provided, adds Gate 9 (extension filter) and scales confidence_score
      by persistence_quality_score [0.5, 1.0].
    """
    fails: list[str] = []

    # Gate 0: no averaging down (hard block, check first)
    if current_price > 0 and entry_price > 0 and current_price <= entry_price:
        fails.append(
            f"averaging_down: cur={current_price:.0f} <= entry={entry_price:.0f}"
        )

    # Gate 1: profit threshold
    if unrealized_pnl_pct < MIN_PROFIT_THRESHOLD:
        fails.append(
            f"pnl={unrealized_pnl_pct:.1%} < min={MIN_PROFIT_THRESHOLD:.1%}"
        )

    # Gate 2: hold duration
    if hold_days < MIN_HOLD_BEFORE_ADDON:
        fails.append(f"hold={hold_days}d < {MIN_HOLD_BEFORE_ADDON}d")

    # Gate 3: RSR rank (1 = best)
    if rsr_rank > MAX_RSR_RANK_FOR_ADDON:
        fails.append(f"rsr_rank={rsr_rank} > {MAX_RSR_RANK_FOR_ADDON}")

    # Gate 4: RSR absolute strength
    if rsr < MIN_RSR_STRENGTH:
        fails.append(f"rsr={rsr:.1f} < {MIN_RSR_STRENGTH:.1f}")

    # Gate 5: trend quality (RSR momentum must be non-negative)
    if trend_quality < MIN_TREND_QUALITY:
        fails.append(f"trend_quality={trend_quality:.3f} < {MIN_TREND_QUALITY:.3f}")

    # Gate 6: deterioration score
    if deterioration_score > MAX_DETERIORATION:
        fails.append(f"deterioration={deterioration_score:.2f} > {MAX_DETERIORATION:.2f}")

    # Gate 7: regime
    if REGIME_BEAR_BLOCK and regime_bear:
        fails.append("regime=bear")

    # Gate 8: ATR ratio (volatility)
    if atr_ratio > MAX_ATR_RATIO:
        fails.append(f"atr_ratio={atr_ratio:.2f} > {MAX_ATR_RATIO:.2f}")

    # Gate 9: extension filter — exhaustion / climax suppression (optional)
    _ext_pqs: float = 1.0
    if extension_input is not None:
        from .extension_filter import check_extension
        _ext = check_extension(extension_input)
        if _ext.extension_blocked:
            fails.extend([f"EXT:{r}" for r in _ext.block_reasons])
        else:
            _ext_pqs = _ext.persistence_quality_score

    confirmed = len(fails) == 0
    if confirmed:
        base_score = _score(
            unrealized_pnl_pct, rsr, deterioration_score, trend_quality, atr_ratio
        )
        confidence_score = round(base_score * _ext_pqs, 4)
    else:
        confidence_score = 0.0

    return WinnerConfirmation(
        symbol=symbol,
        confirmed=confirmed,
        confidence_score=confidence_score,
        fail_reasons=fails,
        unrealized_pnl_pct=unrealized_pnl_pct,
        hold_days=hold_days,
        rsr_rank=rsr_rank,
        rsr=rsr,
        trend_quality=trend_quality,
        deterioration_score=deterioration_score,
        regime_bear=regime_bear,
        atr_ratio=atr_ratio,
    )
