"""
entry_timing_engine.py — Entry Timing Engine

Post-ranking auxiliary layer that scores entry quality for BUY candidates.
Does NOT modify upstream alpha or signal selection; improves harvest efficiency.

Four scoring components (0-100 each):
  1. Breakout Quality   (weight 0.30) — candle structure: CPR/USR/GER/RE
  2. Pullback Quality   (weight 0.25) — ATR depth from 20d high, EMA proximity, velocity
  3. Trend Persistence  (weight 0.25) — RSR continuity, RSR level, SEPA
  4. Market Context     (weight 0.20) — trend cluster regime, universe breadth

Composite score: weighted_sum + bonuses - penalties, clamped [0, 100]

Output:
  EntryTimingScore    : float 0-100
  EntryTimingConfidence: HIGH (>=75) | MEDIUM (>=50) | LOW (<50)
  action              : IMMEDIATE | NORMAL | WATCH

Integration:
  signal_bridge.py uses apply_entry_timing_boost() to add a small composite
  tiebreaker: (score - 50) * boost_weight  →  range [-3, +3] at default weight=0.06.
  The boost is intentionally small — RSR-based selection remains primary.

Fail policy: all public functions FAIL_OPEN.
  compute_entry_timing() returns score=50, MEDIUM, NORMAL on any exception.
  Individual component scorers return 50.0 on exception.

Feature flag: pass enabled=False to compute_entry_timing_for_candidates()
  to bypass all computation and return neutral results.

Telemetry: JSONL append with forward-return materialization (3d / 5d).
  KPI aggregation: entry_timing_score_decile → win_rate / expectancy / avg_R.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_EPS = 1e-9
SCHEMA_VERSION = "v1"
_JST = timezone(timedelta(hours=9))

# ── Confidence labels ──────────────────────────────────────────────────────────
CONFIDENCE_HIGH   = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW    = "LOW"

CONFIDENCE_HIGH_THRESHOLD   = 75.0
CONFIDENCE_MEDIUM_THRESHOLD = 50.0

# ── Action labels ──────────────────────────────────────────────────────────────
ACTION_IMMEDIATE = "IMMEDIATE"   # HIGH confidence
ACTION_NORMAL    = "NORMAL"      # MEDIUM confidence
ACTION_WATCH     = "WATCH"       # LOW confidence

# ── Component weights (must sum to 1.0) ────────────────────────────────────────
W_BREAKOUT    = 0.30
W_PULLBACK    = 0.25
W_TREND       = 0.25
W_MARKET      = 0.20

# ── Pullback depth scoring breakpoints (ATR units from 20d high) ──────────────
# Too close to the high = likely extended; deep pullback = losing momentum
PB_EXTENDED_MAX   = 0.5    # <0.5 ATR below high → extended (score 40)
PB_IDEAL_LO       = 0.5    # 0.5-2.0 ATR below high → ideal (score 80-100)
PB_IDEAL_HI       = 2.0
PB_DEEP_HI        = 3.5    # 2.0-3.5 ATR → fading (score 50-70)
                            # >3.5 ATR → too deep (score 20)

# ── EMA proximity breakpoints (% above MA20) ──────────────────────────────────
EMA_CLOSE_TO_MA20_PCT = 0.05   # within 5% above MA20 = ideal entry zone
EMA_EXTENDED_PCT      = 0.15   # >15% above MA20 = extended
EMA_ABOVE_MA20_PCT    = 0.0    # at/above MA20 = okay

# ── RSR continuity scoring ────────────────────────────────────────────────────
RSR_CONTINUITY_DAYS = 5        # look back 5 RSR values for trend direction

# ── Market breadth breakpoints (fraction of RSR42 with RSR >= 65) ─────────────
BREADTH_STRONG = 0.60
BREADTH_NORMAL = 0.40
BREADTH_WEAK   = 0.20

# ── Bonus / penalty amounts ───────────────────────────────────────────────────
BONUS_HEALTHY_PLUS_VOLUME   = 10.0   # healthy breakout AND volume_ratio > 1.5
BONUS_RSR_HIGH_RISING       =  7.0   # RSR > 85 AND all-rising RSR series
BONUS_SEPA_FULL             =  5.0   # sepa_score == 8
BONUS_STRONG_MARKET         =  5.0   # trend_cluster_level >= 2

PENALTY_FAILED_BREAKOUT     = -15.0  # failed breakout phase
PENALTY_DEEP_PULLBACK       = -10.0  # pullback > 3.5 ATR
PENALTY_RSR_DECLINING       =  -8.0  # RSR declining 3+ consecutive days
PENALTY_OVEREXTENDED        =  -5.0  # close > 20% above MA20
PENALTY_LOW_VOLUME          =  -5.0  # volume_ratio_5d < 0.5

# ── Materialization calendar buffers ─────────────────────────────────────────
_MAT_BUFFER_3D = 5    # calendar days before materializing 3d forward return
_MAT_BUFFER_5D = 7


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntryTimingInput:
    symbol:           str
    # From StockSignal
    rsr:              float           # current RSR (0-100)
    rsr_momentum:     float           # RSR momentum (21d)
    sepa_score:       int             # SEPA conditions met (0-8)
    # OHLCV: last N bars (most recent last), minimum 20 rows required
    closes:           List[float]
    highs:            List[float]
    lows:             List[float]
    volumes:          List[float]
    # RSR time series: last RSR_CONTINUITY_DAYS + a few values (most recent last)
    rsr_series:       Optional[List[float]] = None
    # Market context (optional — FAIL_OPEN if None)
    trend_cluster_level: int          = 0    # 0=neutral / 1=momentum / 2=strong trend
    universe_rsr_values: Optional[List[float]] = None  # all RSR42 latest RSR values
    # Pre-computed scores (skip sub-computation if provided)
    breakout_quality_score: Optional[float] = None   # 0-100 from breakout_quality.py
    breakout_phase:         Optional[str]   = None   # PHASE_* from breakout_quality.py


@dataclass
class EntryTimingResult:
    symbol:     str
    score:      float   # 0-100 composite
    confidence: str     # CONFIDENCE_*
    action:     str     # ACTION_*
    # Component scores (0-100 each)
    breakout_component:  float
    pullback_component:  float
    trend_component:     float
    market_component:    float
    # Bonus/penalty applied
    bonuses_applied:  List[str] = field(default_factory=list)
    penalties_applied: List[str] = field(default_factory=list)
    # Phase label for reporting
    phase: str = "normal"   # "breakout" | "pullback" | "extended" | "normal" | "blocked"
    computed_at: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _compute_atr20(highs: List[float], lows: List[float], closes: List[float]) -> float:
    """20-period ATR from raw lists. Returns 2% of last close as fallback."""
    try:
        n = min(len(highs), len(lows), len(closes))
        if n < 2:
            return closes[-1] * 0.02 if closes else 1.0
        trs = []
        for i in range(max(1, n - 21), n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            trs.append(max(hl, hc, lc))
        return float(sum(trs) / len(trs)) if trs else closes[-1] * 0.02
    except Exception:
        return closes[-1] * 0.02 if closes else 1.0


def _compute_ma(series: List[float], period: int) -> Optional[float]:
    """Simple moving average of last `period` values. None if insufficient data."""
    if len(series) < period:
        return None
    return float(sum(series[-period:]) / period)


def _compute_volume_ratio_5d(volumes: List[float]) -> float:
    """Latest volume / 20d average volume. Fallback 1.0."""
    try:
        if len(volumes) < 6:
            return 1.0
        avg20 = sum(volumes[-21:-1]) / max(1, len(volumes[-21:-1]))
        if avg20 < _EPS:
            return 1.0
        return float(volumes[-1] / avg20)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers — all return float 0-100, never raise
# ─────────────────────────────────────────────────────────────────────────────

def _score_breakout_quality(inp: EntryTimingInput) -> Tuple[float, str]:
    """
    Re-uses breakout_quality.py if pre-computed score provided.
    Falls back to CPR proxy (close position in range) otherwise.
    Returns (score 0-100, phase_type).
    """
    try:
        if inp.breakout_quality_score is not None:
            phase = inp.breakout_phase or "weak_breakout"
            return _clamp(inp.breakout_quality_score), phase

        # Lightweight CPR + USR proxy when BQ not pre-computed
        if len(inp.closes) < 2 or len(inp.highs) < 1 or len(inp.lows) < 1:
            return 50.0, "weak_breakout"

        close = inp.closes[-1]
        high  = inp.highs[-1]
        low   = inp.lows[-1]
        rng   = high - low
        if rng < _EPS:
            return 50.0, "weak_breakout"

        cpr = (close - low) / rng          # 0-1: higher = closes near top
        usr = (high - close) / rng         # 0-1: lower = smaller upper shadow

        # Simple quality score from CPR + USR
        score = cpr * 60.0 + (1.0 - usr) * 40.0

        # Volume expansion bonus
        vol_ratio = _compute_volume_ratio_5d(inp.volumes)
        if vol_ratio > 1.5:
            score = min(100.0, score + 10.0)

        phase = "healthy_breakout" if cpr > 0.70 and usr < 0.25 else "weak_breakout"
        return _clamp(score), phase

    except Exception as exc:
        logger.debug("[ET] breakout_quality_score failed %s: %s", inp.symbol, exc)
        return 50.0, "weak_breakout"


def _score_pullback_quality(inp: EntryTimingInput) -> float:
    """
    Scores pullback setup quality:
      - Depth from 20d high in ATR units (ideal: 0.5-2.0 ATR)
      - EMA proximity: close vs MA20 (ideal: close to MA20 from above)
      - Momentum velocity: 5d return
    Returns float 0-100.
    """
    try:
        if len(inp.closes) < 20 or len(inp.highs) < 20:
            return 50.0

        close = inp.closes[-1]
        atr20 = _compute_atr20(inp.highs, inp.lows, inp.closes)
        high20 = max(inp.highs[-20:])

        # 1. Pullback depth score
        depth_atr = (high20 - close) / max(atr20, _EPS)
        if depth_atr < PB_EXTENDED_MAX:
            depth_score = 40.0   # too close to high = extended
        elif depth_atr <= PB_IDEAL_HI:
            t = (depth_atr - PB_IDEAL_LO) / (PB_IDEAL_HI - PB_IDEAL_LO)
            depth_score = 80.0 + t * 20.0   # 80 → 100 as depth increases toward 2 ATR
        elif depth_atr <= PB_DEEP_HI:
            t = (depth_atr - PB_IDEAL_HI) / (PB_DEEP_HI - PB_IDEAL_HI)
            depth_score = 70.0 - t * 20.0   # 70 → 50 as depth exceeds ideal
        else:
            depth_score = 20.0   # too deep = losing momentum

        # 2. EMA proximity score
        ma20 = _compute_ma(inp.closes, 20)
        if ma20 is None or ma20 < _EPS:
            ema_score = 50.0
        else:
            pct_above_ma20 = (close - ma20) / ma20
            if pct_above_ma20 < 0:
                ema_score = 20.0   # below MA20 = unfavorable
            elif pct_above_ma20 <= EMA_CLOSE_TO_MA20_PCT:
                ema_score = 100.0  # tight to MA20 = ideal for new entry
            elif pct_above_ma20 <= EMA_EXTENDED_PCT:
                t = (pct_above_ma20 - EMA_CLOSE_TO_MA20_PCT) / (EMA_EXTENDED_PCT - EMA_CLOSE_TO_MA20_PCT)
                ema_score = 100.0 - t * 40.0   # 100 → 60 as it extends
            else:
                pct_excess = pct_above_ma20 - EMA_EXTENDED_PCT
                ema_score = max(20.0, 60.0 - pct_excess * 200.0)  # further decay

        # 3. Momentum velocity: 5d return
        if len(inp.closes) >= 6:
            ret5 = (inp.closes[-1] - inp.closes[-6]) / max(inp.closes[-6], _EPS)
            if ret5 > 0.03:
                vel_score = 80.0
            elif ret5 > 0.01:
                vel_score = 70.0
            elif ret5 > 0.0:
                vel_score = 60.0
            else:
                vel_score = 30.0
        else:
            vel_score = 50.0

        score = 0.40 * depth_score + 0.35 * ema_score + 0.25 * vel_score
        return _clamp(score)

    except Exception as exc:
        logger.debug("[ET] pullback_quality_score failed %s: %s", inp.symbol, exc)
        return 50.0


def _score_trend_persistence(inp: EntryTimingInput) -> float:
    """
    Scores trend persistence:
      - RSR continuity (last 5 values: count of consecutive rises)
      - RSR absolute level
      - SEPA score
    Returns float 0-100.
    """
    try:
        # 1. RSR continuity
        if inp.rsr_series and len(inp.rsr_series) >= 3:
            series = inp.rsr_series[-RSR_CONTINUITY_DAYS:]
            n = len(series)
            rising_count = sum(1 for i in range(1, n) if series[i] > series[i - 1])
            if rising_count == n - 1:
                cont_score = 100.0
            elif rising_count >= (n - 1) * 0.6:
                cont_score = 80.0
            elif rising_count >= (n - 1) * 0.4:
                cont_score = 60.0
            else:
                cont_score = 20.0
        else:
            cont_score = 60.0   # neutral when RSR series unavailable

        # 2. RSR level score
        rsr = _safe_float(inp.rsr)
        if rsr >= 85.0:
            level_score = 100.0
        elif rsr >= 75.0:
            level_score = 80.0
        elif rsr >= 65.0:
            level_score = 60.0
        else:
            level_score = 40.0

        # 3. SEPA quality
        sepa = max(0, min(8, int(inp.sepa_score)))
        sepa_score = {8: 100.0, 7: 87.5, 6: 75.0, 5: 62.5,
                      4: 50.0, 3: 37.5, 2: 25.0, 1: 12.5, 0: 0.0}.get(sepa, 50.0)

        score = 0.35 * cont_score + 0.40 * level_score + 0.25 * sepa_score
        return _clamp(score)

    except Exception as exc:
        logger.debug("[ET] trend_persistence_score failed %s: %s", inp.symbol, exc)
        return 50.0


def _score_market_context(inp: EntryTimingInput) -> float:
    """
    Scores market environment:
      - Trend cluster level (regime proxy)
      - Universe breadth (% of RSR42 with RSR >= 65)
    Returns float 0-100.
    """
    try:
        # 1. Regime from trend_cluster_level
        regime_score = {0: 60.0, 1: 80.0, 2: 90.0}.get(
            max(0, min(2, inp.trend_cluster_level)), 60.0
        )

        # 2. Universe breadth
        if inp.universe_rsr_values and len(inp.universe_rsr_values) > 0:
            n_total = len(inp.universe_rsr_values)
            n_gt65  = sum(1 for v in inp.universe_rsr_values if _safe_float(v) >= 65.0)
            breadth = n_gt65 / max(1, n_total)
            if breadth >= BREADTH_STRONG:
                breadth_score = 80.0
            elif breadth >= BREADTH_NORMAL:
                breadth_score = 65.0
            elif breadth >= BREADTH_WEAK:
                breadth_score = 50.0
            else:
                breadth_score = 30.0
        else:
            breadth_score = 60.0  # neutral when unavailable

        score = 0.55 * regime_score + 0.45 * breadth_score
        return _clamp(score)

    except Exception as exc:
        logger.debug("[ET] market_context_score failed %s: %s", inp.symbol, exc)
        return 60.0


def _classify_phase(
    breakout_comp: float,
    pullback_comp: float,
    trend_comp: float,
    bq_phase: str,
) -> str:
    if bq_phase == "failed_breakout":
        return "blocked"
    if breakout_comp >= 75 and pullback_comp < 55:
        return "breakout"   # fresh breakout, not yet pulled back
    if pullback_comp >= 70:
        return "pullback"   # pulled back to ideal zone
    if trend_comp >= 75:
        return "extended"   # strong trend but overextended
    return "normal"


def _compute_bonuses_penalties(
    inp: EntryTimingInput,
    bq_phase: str,
    vol_ratio: float,
    all_rising: bool,
) -> Tuple[float, List[str], List[str]]:
    """Apply bonuses and penalties. Returns (delta, bonuses, penalties)."""
    delta: float = 0.0
    bonuses: List[str] = []
    penalties: List[str] = []

    # Bonuses
    if bq_phase == "healthy_breakout" and vol_ratio > 1.5:
        delta += BONUS_HEALTHY_PLUS_VOLUME
        bonuses.append(f"healthy_breakout+volume({vol_ratio:.1f}x)")

    if inp.rsr >= 85.0 and all_rising:
        delta += BONUS_RSR_HIGH_RISING
        bonuses.append(f"rsr_high_rising(rsr={inp.rsr:.0f})")

    if inp.sepa_score == 8:
        delta += BONUS_SEPA_FULL
        bonuses.append("sepa_full(8/8)")

    if inp.trend_cluster_level >= 2:
        delta += BONUS_STRONG_MARKET
        bonuses.append(f"strong_market(cluster={inp.trend_cluster_level})")

    # Penalties
    if bq_phase == "failed_breakout":
        delta += PENALTY_FAILED_BREAKOUT
        penalties.append("failed_breakout")

    atr20 = _compute_atr20(inp.highs, inp.lows, inp.closes)
    if len(inp.highs) >= 20 and atr20 > _EPS:
        high20 = max(inp.highs[-20:])
        depth_atr = (high20 - inp.closes[-1]) / atr20
        if depth_atr > 3.5:
            delta += PENALTY_DEEP_PULLBACK
            penalties.append(f"deep_pullback({depth_atr:.1f}ATR)")

    # RSR declining 3+ consecutive days
    if inp.rsr_series and len(inp.rsr_series) >= 4:
        last4 = inp.rsr_series[-4:]
        declining_count = sum(1 for i in range(1, len(last4)) if last4[i] < last4[i - 1])
        if declining_count >= 3:
            delta += PENALTY_RSR_DECLINING
            penalties.append("rsr_declining_3d")

    # Overextended check
    ma20 = _compute_ma(inp.closes, 20)
    if ma20 and ma20 > _EPS and len(inp.closes) > 0:
        pct_above = (inp.closes[-1] - ma20) / ma20
        if pct_above > 0.20:
            delta += PENALTY_OVEREXTENDED
            penalties.append(f"overextended({pct_above:.0%}_above_MA20)")

    if vol_ratio < 0.5:
        delta += PENALTY_LOW_VOLUME
        penalties.append(f"low_volume({vol_ratio:.2f}x)")

    return delta, bonuses, penalties


# ─────────────────────────────────────────────────────────────────────────────
# Public pure API
# ─────────────────────────────────────────────────────────────────────────────

def compute_entry_timing(inp: EntryTimingInput) -> EntryTimingResult:
    """
    Compute entry timing score [0-100] for a single candidate.

    Pure function. Never raises (FAIL_OPEN: score=50, MEDIUM on any error).
    """
    try:
        bq_score, bq_phase = _score_breakout_quality(inp)
        pb_score           = _score_pullback_quality(inp)
        tr_score           = _score_trend_persistence(inp)
        mk_score           = _score_market_context(inp)

        base = (
            W_BREAKOUT * bq_score
            + W_PULLBACK  * pb_score
            + W_TREND     * tr_score
            + W_MARKET    * mk_score
        )

        vol_ratio = _compute_volume_ratio_5d(inp.volumes)
        all_rising = False
        if inp.rsr_series and len(inp.rsr_series) >= RSR_CONTINUITY_DAYS:
            s = inp.rsr_series[-RSR_CONTINUITY_DAYS:]
            all_rising = all(s[i] > s[i - 1] for i in range(1, len(s)))

        adj, bonuses, penalties = _compute_bonuses_penalties(
            inp, bq_phase, vol_ratio, all_rising
        )

        score = _clamp(base + adj)

        confidence = (
            CONFIDENCE_HIGH   if score >= CONFIDENCE_HIGH_THRESHOLD   else
            CONFIDENCE_MEDIUM if score >= CONFIDENCE_MEDIUM_THRESHOLD else
            CONFIDENCE_LOW
        )
        action = {
            CONFIDENCE_HIGH:   ACTION_IMMEDIATE,
            CONFIDENCE_MEDIUM: ACTION_NORMAL,
            CONFIDENCE_LOW:    ACTION_WATCH,
        }[confidence]

        phase = _classify_phase(bq_score, pb_score, tr_score, bq_phase)

        return EntryTimingResult(
            symbol              = inp.symbol,
            score               = round(score, 2),
            confidence          = confidence,
            action              = action,
            breakout_component  = round(bq_score, 2),
            pullback_component  = round(pb_score, 2),
            trend_component     = round(tr_score, 2),
            market_component    = round(mk_score, 2),
            bonuses_applied     = bonuses,
            penalties_applied   = penalties,
            phase               = phase,
            computed_at         = datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        )

    except Exception as exc:
        logger.warning("[ET] compute failed %s: %s", getattr(inp, "symbol", "?"), exc)
        return _neutral_result(getattr(inp, "symbol", ""))


def _neutral_result(symbol: str) -> EntryTimingResult:
    return EntryTimingResult(
        symbol             = symbol,
        score              = 50.0,
        confidence         = CONFIDENCE_MEDIUM,
        action             = ACTION_NORMAL,
        breakout_component = 50.0,
        pullback_component = 50.0,
        trend_component    = 50.0,
        market_component   = 60.0,
        phase              = "normal",
        computed_at        = datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    )


def build_entry_timing_input_from_df(
    symbol:              str,
    df:                  Any,
    rsr:                 float,
    rsr_momentum:        float,
    sepa_score:          int,
    rsr_series:          Optional[List[float]] = None,
    trend_cluster_level: int                   = 0,
    universe_rsr_values: Optional[List[float]] = None,
    breakout_quality_score: Optional[float]    = None,
    breakout_phase:      Optional[str]         = None,
) -> EntryTimingInput:
    """
    Build EntryTimingInput from a pandas OHLCV DataFrame.
    Requires len(df) >= 20. Caller is responsible for the guard.
    FAIL_OPEN: returns a minimal EntryTimingInput with available data.
    """
    try:
        closes  = [float(v) for v in df["Close"].values[-65:]]
        highs   = [float(v) for v in df["High"].values[-65:]]
        lows    = [float(v) for v in df["Low"].values[-65:]]
        volumes = [float(v) for v in df["Volume"].values[-65:]]
    except Exception:
        closes = highs = lows = volumes = [1.0] * 20

    return EntryTimingInput(
        symbol               = symbol,
        rsr                  = _safe_float(rsr),
        rsr_momentum         = _safe_float(rsr_momentum),
        sepa_score           = int(sepa_score),
        closes               = closes,
        highs                = highs,
        lows                 = lows,
        volumes              = volumes,
        rsr_series           = rsr_series,
        trend_cluster_level  = int(trend_cluster_level),
        universe_rsr_values  = universe_rsr_values,
        breakout_quality_score = breakout_quality_score,
        breakout_phase         = breakout_phase,
    )


def compute_entry_timing_for_candidates(
    buy_eligible:        List[Tuple[float, str]],
    universe_raw:        Dict[str, Any],
    rsr_universe:        Any,       # pd.DataFrame columns=symbols, rows=dates
    signals_map:         Dict[str, Any],  # symbol → StockSignal-like
    trend_cluster_level: int = 0,
    universe_rsr_latest: Optional[Any] = None,  # pd.Series symbol→RSR
    enabled:             bool = True,
) -> Dict[str, EntryTimingResult]:
    """
    Compute Entry Timing for all buy-eligible candidates.

    Args:
        buy_eligible:        list of (composite_score, symbol) from signal_bridge ranking
        universe_raw:        {sym: {"df": pd.DataFrame, ...}} — OHLCV data
        rsr_universe:        pd.DataFrame with RSR history (columns=symbols)
        signals_map:         {sym: StockSignal} for rsr/rsr_mom/sepa_score lookup
        trend_cluster_level: market regime (0/1/2)
        universe_rsr_latest: pd.Series of current RSR values for all RSR42 symbols
        enabled:             Feature flag — False returns all neutral results

    Returns: {symbol: EntryTimingResult}. FAIL_OPEN per symbol.
    """
    results: Dict[str, EntryTimingResult] = {}
    candidates = [sym for _, sym in buy_eligible]

    if not enabled:
        for sym in candidates:
            results[sym] = _neutral_result(sym)
        return results

    # Universe breadth: fraction of all RSR42 symbols with RSR >= 65
    univ_rsr_vals: Optional[List[float]] = None
    if universe_rsr_latest is not None:
        try:
            univ_rsr_vals = [float(v) for v in universe_rsr_latest.values if v == v]
        except Exception:
            pass

    for sym in candidates:
        try:
            info = universe_raw.get(sym)
            if info is None:
                results[sym] = _neutral_result(sym)
                continue

            df = info.get("df")
            if df is None or len(df) < 20:
                results[sym] = _neutral_result(sym)
                continue

            sig = signals_map.get(sym)
            rsr        = _safe_float(getattr(sig, "rsr",     0.0)) if sig else 0.0
            rsr_mom    = _safe_float(getattr(sig, "rsr_mom", 0.0)) if sig else 0.0
            sepa_score = int(getattr(sig, "sepa_score", 0))         if sig else 0

            # RSR time series for continuity scoring
            rsr_series: Optional[List[float]] = None
            if sym in rsr_universe.columns:
                try:
                    rsr_s = rsr_universe[sym].dropna()
                    if len(rsr_s) >= 3:
                        rsr_series = [float(v) for v in rsr_s.values[-(RSR_CONTINUITY_DAYS + 3):]]
                except Exception:
                    pass

            et_inp = build_entry_timing_input_from_df(
                symbol               = sym,
                df                   = df,
                rsr                  = rsr,
                rsr_momentum         = rsr_mom,
                sepa_score           = sepa_score,
                rsr_series           = rsr_series,
                trend_cluster_level  = trend_cluster_level,
                universe_rsr_values  = univ_rsr_vals,
            )
            results[sym] = compute_entry_timing(et_inp)

        except Exception as exc:
            logger.warning("[ET] candidate %s failed: %s", sym, exc)
            results[sym] = _neutral_result(sym)

    return results


def apply_entry_timing_boost(
    composite_score: float,
    et_result:       Optional[EntryTimingResult],
    boost_weight:    float = 0.06,
    enabled:         bool  = True,
) -> float:
    """
    Add a small entry timing adjustment to the RSR-based composite score.

    Boost range at default weight=0.06:
      score=100 → +3.0   (strong entry quality)
      score= 75 → +1.5
      score= 50 →  0.0   (neutral, no effect)
      score= 25 → -1.5
      score=  0 → -3.0   (poor entry quality)

    The range is intentionally small so RSR-based ranking stays primary.
    """
    if not enabled or et_result is None:
        return composite_score
    boost = (et_result.score - 50.0) * boost_weight
    return composite_score + boost


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry I/O
# ─────────────────────────────────────────────────────────────────────────────

def _append_jsonl(record: dict, path: Path) -> None:
    """Atomic JSONL append. FAIL_OPEN."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(path)
    except Exception as exc:
        logger.warning("[ET] append failed: %s", exc)


def append_et_telemetry(
    result:         EntryTimingResult,
    action_taken:   str,           # "ENTERED" | "SKIPPED" | "WATCHED"
    telemetry_path: Path,
    date_str:       Optional[str] = None,
) -> None:
    """
    Append one Entry Timing telemetry record. FAIL_OPEN.
    Forward returns are null at record time; materialized in subsequent runs.
    """
    if date_str is None:
        date_str = datetime.now(_JST).strftime("%Y-%m-%d")
    record = {
        "date":                date_str,
        "symbol":              result.symbol,
        "entry_timing_score":  result.score,
        "confidence":          result.confidence,
        "action":              result.action,
        "action_taken":        action_taken,
        "phase":               result.phase,
        "breakout_component":  result.breakout_component,
        "pullback_component":  result.pullback_component,
        "trend_component":     result.trend_component,
        "market_component":    result.market_component,
        "bonuses":             result.bonuses_applied,
        "penalties":           result.penalties_applied,
        # Decile label: 1 (lowest) to 10 (highest)
        "score_decile":        _score_to_decile(result.score),
        # Forward returns — filled by materialize_et_returns()
        "subsequent_3d_return":  None,
        "subsequent_5d_return":  None,
        "win_3d":                None,
        "materialized_3d":       False,
        "materialized_5d":       False,
        "schema_version":        SCHEMA_VERSION,
    }
    _append_jsonl(record, telemetry_path)


def _score_to_decile(score: float) -> int:
    """Map 0-100 score to 1-10 decile (1=lowest quality, 10=highest)."""
    d = int(score / 10.0)
    return max(1, min(10, d + 1 if score < 100.0 else 10))


def _fwd_return(df: Any, record_date_str: str, n_trading_days: int) -> Optional[float]:
    """N-trading-day forward return from record_date. None when data insufficient."""
    try:
        import pandas as pd
        import numpy as np
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            date_col = next((c for c in ("date", "Date") if c in df.columns), None)
            if date_col is None:
                return None
            df.index = pd.to_datetime(df[date_col])
        df = df.sort_index()
        rec_dt = pd.Timestamp(record_date_str)
        future_mask = df.index >= rec_dt
        if not future_mask.any():
            return None
        entry_pos  = int(np.argmax(future_mask))
        target_pos = entry_pos + n_trading_days
        if target_pos >= len(df):
            return None
        entry_close  = float(df["Close"].iloc[entry_pos]) if "Close" in df.columns else float(df["close"].iloc[entry_pos])
        target_close = float(df["Close"].iloc[target_pos]) if "Close" in df.columns else float(df["close"].iloc[target_pos])
        if entry_close <= _EPS:
            return None
        return (target_close - entry_close) / entry_close
    except Exception as exc:
        logger.warning("[ET] _fwd_return failed for %s: %s", record_date_str, exc)
        return None


def materialize_et_returns(
    telemetry_path: Path,
    ohlcv_loader:   Callable[[str], Optional[Any]],
) -> Dict[str, int]:
    """
    Fill null forward returns for records old enough to have realized.
    Atomically rewrites the telemetry file. FAIL_OPEN.
    """
    try:
        if not telemetry_path.exists():
            return {}
        lines = [l for l in telemetry_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            return {}

        today   = datetime.now(_JST).date()
        records: List[dict] = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

        n3 = n5 = 0
        cache: Dict[str, Any] = {}

        for rec in records:
            sym      = rec.get("symbol", "")
            date_str = rec.get("date", "")
            if not sym or not date_str:
                continue
            try:
                rec_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            elapsed = (today - rec_date).days

            if not rec.get("materialized_3d") and elapsed >= _MAT_BUFFER_3D:
                if sym not in cache:
                    try:
                        cache[sym] = ohlcv_loader(sym)
                    except Exception as exc:
                        logger.warning("[ET] ohlcv_loader failed for %s: %s", sym, exc)
                        cache[sym] = None
                if cache[sym] is not None:
                    ret = _fwd_return(cache[sym], date_str, 3)
                    if ret is not None:
                        rec["subsequent_3d_return"] = round(ret, 6)
                        rec["win_3d"] = bool(ret > 0)
                        rec["materialized_3d"] = True
                        n3 += 1

            if not rec.get("materialized_5d") and elapsed >= _MAT_BUFFER_5D:
                if sym not in cache:
                    try:
                        cache[sym] = ohlcv_loader(sym)
                    except Exception as exc:
                        logger.warning("[ET] ohlcv_loader failed for %s: %s", sym, exc)
                        cache[sym] = None
                if cache[sym] is not None:
                    ret = _fwd_return(cache[sym], date_str, 5)
                    if ret is not None:
                        rec["subsequent_5d_return"] = round(ret, 6)
                        rec["materialized_5d"] = True
                        n5 += 1

        new_content = (
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n"
        )
        fd, tmp = tempfile.mkstemp(dir=str(telemetry_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(new_content)
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        Path(tmp).replace(telemetry_path)

        return {"n_materialized_3d": n3, "n_materialized_5d": n5}
    except Exception as exc:
        logger.warning("[ET] materialize failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# KPI aggregation — entry_timing_score_decile analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_et_kpis(telemetry_path: Path) -> Dict[str, Any]:
    """
    Aggregate Entry Timing KPIs from telemetry JSONL.

    Key KPI: win_rate / expectancy / avg_R / record_count per score_decile.
    Primary validation metric: decile_10_win_rate > decile_1_win_rate (monotonic).
    FAIL_OPEN.
    """
    try:
        if not telemetry_path.exists():
            return {}
        records: List[dict] = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        if not records:
            return {}

        # Group by score_decile
        by_decile: Dict[int, List[dict]] = {d: [] for d in range(1, 11)}
        for r in records:
            d = int(r.get("score_decile", _score_to_decile(float(r.get("entry_timing_score", 50)))))
            if 1 <= d <= 10:
                by_decile[d].append(r)

        def _agg(recs: List[dict], ret_key: str) -> Dict[str, Any]:
            vals  = [r[ret_key] for r in recs if r.get(ret_key) is not None]
            wins  = [v for v in vals if v > 0]
            if not vals:
                return {"n": len(recs), "win_rate": None, "expectancy": None, "avg_R": None}
            avg   = sum(vals) / len(vals)
            # avg_R: expectancy / avg_loss (simplified: use 1% as denominator proxy)
            avg_loss = abs(sum(v for v in vals if v < 0) / max(1, len([v for v in vals if v < 0]))) if any(v < 0 for v in vals) else 0.01
            avg_R  = avg / max(avg_loss, _EPS) if avg_loss > _EPS else None
            return {
                "n":          len(recs),
                "n_realized": len(vals),
                "win_rate":   round(len(wins) / len(vals), 4) if vals else None,
                "expectancy": round(avg, 6),
                "avg_R":      round(avg_R, 3) if avg_R is not None else None,
            }

        decile_stats: Dict[str, Any] = {}
        for d in range(1, 11):
            recs = by_decile[d]
            decile_stats[str(d)] = _agg(recs, "subsequent_3d_return")

        # Confidence breakdown
        by_conf: Dict[str, List[float]] = {CONFIDENCE_HIGH: [], CONFIDENCE_MEDIUM: [], CONFIDENCE_LOW: []}
        for r in records:
            conf = r.get("confidence", CONFIDENCE_MEDIUM)
            ret  = r.get("subsequent_3d_return")
            if conf in by_conf and ret is not None:
                by_conf[conf].append(float(ret))

        conf_stats: Dict[str, Any] = {}
        for conf, vals in by_conf.items():
            if vals:
                wins = [v for v in vals if v > 0]
                conf_stats[conf] = {
                    "n":          len(vals),
                    "win_rate":   round(len(wins) / len(vals), 4),
                    "expectancy": round(sum(vals) / len(vals), 6),
                }

        # Monotonicity check: higher decile should have higher win_rate
        wr_by_decile = [
            decile_stats[str(d)]["win_rate"]
            for d in range(1, 11)
            if decile_stats[str(d)]["win_rate"] is not None
        ]
        monotonic = None
        if len(wr_by_decile) >= 5:
            rising = sum(1 for i in range(1, len(wr_by_decile)) if wr_by_decile[i] > wr_by_decile[i - 1])
            monotonic = round(rising / (len(wr_by_decile) - 1), 3)

        return {
            "total_records":    len(records),
            "decile_stats_3d":  decile_stats,
            "confidence_stats": conf_stats,
            "monotonicity_score": monotonic,   # 1.0 = perfect monotonic; ≥0.6 = useful signal
        }
    except Exception as exc:
        logger.warning("[ET] kpi aggregation failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Report section
# ─────────────────────────────────────────────────────────────────────────────

def format_et_report_section(
    et_scores: Dict[str, "EntryTimingResult"],
    telemetry_path: Optional[Path] = None,
) -> str:
    """
    Format [ENTRY TIMING] report section for morning report. FAIL_OPEN.
    et_scores: {symbol: EntryTimingResult} from compute_entry_timing_for_candidates().
    """
    try:
        if not et_scores:
            return ""

        high_syms   = [(s, r) for s, r in et_scores.items() if r.confidence == CONFIDENCE_HIGH]
        medium_syms = [(s, r) for s, r in et_scores.items() if r.confidence == CONFIDENCE_MEDIUM]
        low_syms    = [(s, r) for s, r in et_scores.items() if r.confidence == CONFIDENCE_LOW]

        high_syms.sort(key=lambda x: x[1].score, reverse=True)
        medium_syms.sort(key=lambda x: x[1].score, reverse=True)

        lines = [
            "",
            "── [ENTRY TIMING INTELLIGENCE] ────────────────────────────",
            f"  候補数: {len(et_scores)}  "
            f"HIGH:{len(high_syms)} / MEDIUM:{len(medium_syms)} / LOW:{len(low_syms)}",
        ]

        if high_syms:
            parts = [f"{s}(score={r.score:.0f},phase={r.phase})" for s, r in high_syms]
            lines.append(f"  HIGH    : {', '.join(parts)}")
        if medium_syms:
            parts = [f"{s}(score={r.score:.0f})" for s, r in medium_syms[:4]]
            lines.append(f"  MEDIUM  : {', '.join(parts)}")
        if low_syms:
            lines.append(f"  LOW     : {', '.join(s for s, _ in low_syms)}")

        # Top candidate detail
        if et_scores:
            top_sym, top_r = max(et_scores.items(), key=lambda x: x[1].score)
            lines.append(
                f"  Top候補 : {top_sym} score={top_r.score:.1f}"
                f"  BQ={top_r.breakout_component:.0f}"
                f"  PB={top_r.pullback_component:.0f}"
                f"  TR={top_r.trend_component:.0f}"
                f"  MK={top_r.market_component:.0f}"
            )
            if top_r.bonuses_applied:
                lines.append(f"  Bonus   : {', '.join(top_r.bonuses_applied)}")
            if top_r.penalties_applied:
                lines.append(f"  Penalty : {', '.join(top_r.penalties_applied)}")

        # Historical KPIs (if telemetry available)
        if telemetry_path is not None:
            try:
                kpis = compute_et_kpis(telemetry_path)
                if kpis and kpis.get("total_records", 0) >= 10:
                    conf_stats = kpis.get("confidence_stats", {})
                    h_stat = conf_stats.get(CONFIDENCE_HIGH, {})
                    m_stat = conf_stats.get(CONFIDENCE_MEDIUM, {})
                    if h_stat.get("win_rate") is not None:
                        lines.append(
                            f"  履歴KPI : HIGH勝率={h_stat['win_rate']:.1%}"
                            f"(n={h_stat['n']})"
                            f"  MEDIUM={m_stat.get('win_rate', 0):.1%}"
                            f"(n={m_stat.get('n', 0)})"
                            if m_stat.get("win_rate") is not None
                            else f"  履歴KPI : HIGH勝率={h_stat['win_rate']:.1%}"
                            f"(n={h_stat['n']})"
                        )
                    mono = kpis.get("monotonicity_score")
                    if mono is not None:
                        tag = "✅ 有効" if mono >= 0.6 else "⚠ 要確認"
                        lines.append(f"  単調性  : {mono:.2f}  {tag}")
            except Exception:
                pass

        lines.append("──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[ET] format_report failed: %s", exc)
        return ""
