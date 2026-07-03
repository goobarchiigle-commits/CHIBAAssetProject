"""
predictive_expansion.py — Predictive trend expansion detection layer

Detects early-stage trend emergence BEFORE traditional RSR leadership forms.

Shift ranking logic from "who is strongest now?" → "who is becoming strongest fastest?"

Metrics:
  1. RSR acceleration  — rsr_velocity_5d / rsr_velocity_10d / rsr_acceleration
  2. Volatility compression breakout — ATR_5/ATR_60, std_10/std_60, TR/ATR_20
  3. Volume regime shift — vol_ratio_5, vol_ratio_20, vol_acceleration
  4. Sector ignition — breadth / advancing_ratio / mean_velocity / breakout_count / vol_expansion
  5. Leader/follower propagation — sector leader expanded → compressed follower boost

Composite:
  predictive_alpha_score =
    0.30 * rsr_acceleration_score
    + 0.25 * compression_breakout_score
    + 0.20 * sector_ignition_score
    + 0.15 * volume_regime_score
    + 0.10 * leader_follower_score

Constraints:
  - deterministic: same inputs → same outputs
  - no ML, no black-box
  - no NaN propagation (NaN → 0.0)
  - no lookahead
  - cross-sectional normalization (percentile rank → 0-100)
  - fail-safe defaults for missing / zero data
  - Windows compatible
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Weights ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "rsr_acceleration": 0.30,
    "compression_breakout": 0.25,
    "sector_ignition": 0.20,
    "volume_regime": 0.15,
    "leader_follower": 0.10,
}

# ─── Thresholds ───────────────────────────────────────────────────────────────
_LEADER_VELOCITY_THRESHOLD: float = 5.0   # leader rsr_velocity_5d must exceed this
_FOLLOWER_COMPRESSION_MAX: float  = 0.85  # follower compression_ratio must be below this
_SECTOR_IGNITION_BREADTH: float   = 0.50  # ≥50% of sector advancing → igniting
_VOLUME_OUTLIER_CAP: float        = 5.0   # clip volume ratios at 5×
_MIN_HISTORY_DAYS: int            = 65    # minimum OHLCV rows for compression metrics


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictiveExpansionScores:
    """Per-symbol predictive expansion scores for a single evaluation date."""
    eval_date: str
    symbols: List[str]

    # Composite (0-100)
    predictive_alpha_score: Dict[str, float] = field(default_factory=dict)

    # Component scores (0-100, cross-sectionally ranked)
    rsr_acceleration_score:    Dict[str, float] = field(default_factory=dict)
    compression_breakout_score: Dict[str, float] = field(default_factory=dict)
    volume_regime_score:        Dict[str, float] = field(default_factory=dict)
    sector_ignition_score:      Dict[str, float] = field(default_factory=dict)
    leader_follower_score:      Dict[str, float] = field(default_factory=dict)

    # Raw diagnostics
    rsr_velocity_5d:   Dict[str, float] = field(default_factory=dict)
    rsr_velocity_10d:  Dict[str, float] = field(default_factory=dict)
    rsr_acceleration:  Dict[str, float] = field(default_factory=dict)
    compression_ratio: Dict[str, float] = field(default_factory=dict)
    price_tightness:   Dict[str, float] = field(default_factory=dict)
    breakout_expansion: Dict[str, float] = field(default_factory=dict)
    volume_ratio_5d:   Dict[str, float] = field(default_factory=dict)
    volume_ratio_20d:  Dict[str, float] = field(default_factory=dict)

    # Flags
    sector_ignition_active: Dict[str, bool] = field(default_factory=dict)
    is_leader:    Dict[str, bool] = field(default_factory=dict)
    is_follower:  Dict[str, bool] = field(default_factory=dict)

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k symbols by predictive_alpha_score. Deterministic (alphabetical tie-break)."""
        ranked = sorted(
            self.predictive_alpha_score.items(),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:k]

    def to_dict(self) -> dict:
        return {
            "eval_date": self.eval_date,
            "symbols": self.symbols,
            "predictive_alpha_score": self.predictive_alpha_score,
            "rsr_acceleration_score": self.rsr_acceleration_score,
            "compression_breakout_score": self.compression_breakout_score,
            "volume_regime_score": self.volume_regime_score,
            "sector_ignition_score": self.sector_ignition_score,
            "leader_follower_score": self.leader_follower_score,
            "rsr_velocity_5d": self.rsr_velocity_5d,
            "rsr_velocity_10d": self.rsr_velocity_10d,
            "rsr_acceleration": self.rsr_acceleration,
            "compression_ratio": self.compression_ratio,
            "price_tightness": self.price_tightness,
            "breakout_expansion": self.breakout_expansion,
            "volume_ratio_5d": self.volume_ratio_5d,
            "volume_ratio_20d": self.volume_ratio_20d,
            "sector_ignition_active": self.sector_ignition_active,
            "is_leader": self.is_leader,
            "is_follower": self.is_follower,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _cs_rank_high_good(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (high value = high score) → 0-100."""
    if s.isna().all() or len(s) < 2:
        return pd.Series(50.0, index=s.index)
    return (s.rank(pct=True, na_option="bottom") * 100).fillna(0.0)


def _cs_rank_low_good(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (low value = high score) → 0-100."""
    if s.isna().all() or len(s) < 2:
        return pd.Series(50.0, index=s.index)
    return ((1 - s.rank(pct=True, na_option="top")) * 100).fillna(0.0)


def _safe_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Rolling ATR (true range mean). Returns Series aligned to df.index."""
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(1, period // 2)).mean()


def _safe_std(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=max(1, period // 2)).std()


def _float_safe(x) -> float:
    """Return float or 0.0 on any error."""
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. RSR Acceleration
# ─────────────────────────────────────────────────────────────────────────────

def rsr_acceleration_scores(rsr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectionally normalized RSR acceleration score.

    Args:
        rsr_df: RSR history (dates × symbols), 0-100 values. Must have ≥15 rows.

    Returns:
        DataFrame (same shape as rsr_df) with composite acceleration score 0-100.
        Each row is the cross-sectional score for that date.
        Empty DataFrame if input insufficient.
    """
    if rsr_df.empty or len(rsr_df) < 15:
        return pd.DataFrame()

    rsr = rsr_df.copy()
    vel5 = rsr - rsr.shift(5)
    vel10 = rsr - rsr.shift(10)
    # acceleration = change in 5d velocity over the past 5 days
    accel = vel5 - vel5.shift(5)

    # Cross-sectional rank per row (high positive = high score)
    def _row_rank(df_: pd.DataFrame) -> pd.DataFrame:
        return df_.rank(axis=1, pct=True, na_option="bottom").multiply(100).fillna(0.0)

    vel5_rank = _row_rank(vel5)
    vel10_rank = _row_rank(vel10)
    accel_rank = _row_rank(accel)

    # Composite: momentum-dominated weighting
    composite = (
        0.50 * vel5_rank
        + 0.30 * accel_rank
        + 0.20 * vel10_rank
    )
    return composite.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Volatility Compression Breakout
# ─────────────────────────────────────────────────────────────────────────────

def compression_breakout_scores(
    ohlcv: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute latest-day compression and breakout metrics.

    Desired behavior:
      low compression_ratio    → preferred (pre-expansion energy)
      low price_tightness      → preferred (tight range)
      high breakout_expansion  → preferred (expansion igniting)

    Returns:
        compression_ratio_s:   ATR_5 / ATR_60
        price_tightness_s:     std_10 / std_60
        breakout_expansion_s:  today_true_range / ATR_20
        score_s:               composite 0-100 (higher = better expansion candidate)
    """
    symbols = sorted(ohlcv.keys())
    comp_ratios: Dict[str, float] = {}
    tightness: Dict[str, float] = {}
    expansion: Dict[str, float] = {}

    for sym in symbols:
        df = ohlcv[sym]
        if df is None or len(df) < _MIN_HISTORY_DAYS:
            comp_ratios[sym] = np.nan
            tightness[sym] = np.nan
            expansion[sym] = np.nan
            continue

        try:
            close = df["Close"]
            atr5  = _safe_atr(df, 5)
            atr20 = _safe_atr(df, 20)
            atr60 = _safe_atr(df, 60)
            std10 = _safe_std(close, 10)
            std60 = _safe_std(close, 60)

            a5  = _float_safe(atr5.iloc[-1])
            a20 = _float_safe(atr20.iloc[-1])
            a60 = _float_safe(atr60.iloc[-1])
            s10 = _float_safe(std10.iloc[-1])
            s60 = _float_safe(std60.iloc[-1])

            # Today's true range
            h  = _float_safe(df["High"].iloc[-1])
            lo = _float_safe(df["Low"].iloc[-1])
            pc = _float_safe(df["Close"].iloc[-2]) if len(df) > 1 else h
            tr_today = max(h - lo, abs(h - pc), abs(lo - pc), 0.0)

            comp_ratios[sym] = (a5 / a60) if a60 > 1e-8 else np.nan
            tightness[sym]   = (s10 / s60) if s60 > 1e-8 else np.nan
            expansion[sym]   = (tr_today / a20) if a20 > 1e-8 else np.nan

        except Exception as e:
            logger.debug("compression_breakout_scores %s: %s", sym, e)
            comp_ratios[sym] = np.nan
            tightness[sym] = np.nan
            expansion[sym] = np.nan

    comp_s   = pd.Series(comp_ratios, dtype=float)
    tight_s  = pd.Series(tightness, dtype=float)
    expand_s = pd.Series(expansion, dtype=float)

    if len(symbols) < 2:
        score = pd.Series(50.0, index=comp_s.index)
        return comp_s.fillna(1.0), tight_s.fillna(1.0), expand_s.fillna(1.0), score

    comp_rank   = _cs_rank_low_good(comp_s)    # low compression_ratio → high score
    tight_rank  = _cs_rank_low_good(tight_s)   # low tightness → high score
    expand_rank = _cs_rank_high_good(expand_s) # high expansion → high score

    score = (
        0.40 * comp_rank
        + 0.30 * tight_rank
        + 0.30 * expand_rank
    ).fillna(0.0)

    return comp_s.fillna(1.0), tight_s.fillna(1.0), expand_s.fillna(1.0), score


# ─────────────────────────────────────────────────────────────────────────────
# 3. Volume Regime Shift
# ─────────────────────────────────────────────────────────────────────────────

def volume_regime_scores(
    ohlcv: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect abnormal participation increase.

    Returns:
        vol_ratio_5d_s:  volume_today / avg_volume_5d (clipped at _VOLUME_OUTLIER_CAP)
        vol_ratio_20d_s: volume_today / avg_volume_20d
        score_s:         composite 0-100 (higher = stronger volume expansion)
    """
    symbols = sorted(ohlcv.keys())
    vr5: Dict[str, float] = {}
    vr20: Dict[str, float] = {}
    va: Dict[str, float] = {}   # volume acceleration

    for sym in symbols:
        df = ohlcv[sym]
        if df is None or len(df) < 25 or "Volume" not in df.columns:
            vr5[sym] = np.nan
            vr20[sym] = np.nan
            va[sym] = np.nan
            continue

        try:
            vol = df["Volume"].copy().replace(0, np.nan).astype(float)
            avg5  = vol.rolling(5, min_periods=3).mean()
            avg20 = vol.rolling(20, min_periods=10).mean()

            today_vol = _float_safe(vol.iloc[-1])
            a5_val  = _float_safe(avg5.iloc[-1])
            a20_val = _float_safe(avg20.iloc[-1])

            if a5_val > 1e-8 and today_vol > 0:
                r5 = min(today_vol / a5_val, _VOLUME_OUTLIER_CAP)
            else:
                r5 = np.nan

            if a20_val > 1e-8 and today_vol > 0:
                r20 = min(today_vol / a20_val, _VOLUME_OUTLIER_CAP)
            else:
                r20 = np.nan

            # Volume acceleration: vol_ratio_5d today vs 5 days ago
            if len(df) >= 7:
                prev_vol5 = _float_safe(vol.iloc[-6])
                prev_avg5 = _float_safe(avg5.iloc[-6])
                if prev_avg5 > 1e-8 and prev_vol5 > 0:
                    prev_r5 = min(prev_vol5 / prev_avg5, _VOLUME_OUTLIER_CAP)
                    accel = (r5 - prev_r5) if r5 is not np.nan and not np.isnan(r5) else np.nan
                else:
                    accel = np.nan
            else:
                accel = np.nan

            vr5[sym]  = r5
            vr20[sym] = r20
            va[sym]   = accel

        except Exception as e:
            logger.debug("volume_regime_scores %s: %s", sym, e)
            vr5[sym]  = np.nan
            vr20[sym] = np.nan
            va[sym]   = np.nan

    vr5_s  = pd.Series(vr5, dtype=float)
    vr20_s = pd.Series(vr20, dtype=float)
    va_s   = pd.Series(va, dtype=float)

    if len(symbols) < 2:
        return vr5_s.fillna(1.0), vr20_s.fillna(1.0), pd.Series(50.0, index=vr5_s.index)

    vr5_rank  = _cs_rank_high_good(vr5_s)
    vr20_rank = _cs_rank_high_good(vr20_s)
    va_rank   = _cs_rank_high_good(va_s)

    score = (
        0.40 * vr5_rank
        + 0.30 * vr20_rank
        + 0.30 * va_rank
    ).fillna(0.0)

    return vr5_s.fillna(1.0), vr20_s.fillna(1.0), score


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sector Ignition Detection
# ─────────────────────────────────────────────────────────────────────────────

def sector_ignition_scores(
    rsr_vel5d: pd.Series,
    compression_ratio: pd.Series,
    volume_ratio_5d: pd.Series,
    breakout_expansion: pd.Series,
    sector_map: Dict[str, str],
    *,
    ignition_breadth_threshold: float = _SECTOR_IGNITION_BREADTH,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, bool]]:
    """
    Detect thematic/group activation before full trend expansion.

    Sector ignition metrics:
      - breadth:              fraction of members with rsr_velocity_5d > 0
      - mean_rsr_velocity:    mean velocity across sector members
      - breakout_count:       fraction with breakout_expansion > 1.2
      - volume_expansion:     mean volume_ratio_5d across sector members

    Returns:
        score_s:       per-symbol sector ignition score (0-100)
        sector_raw:    sector_name → raw 0-1 score
        ignition_flag: symbol → bool (in an igniting sector)
    """
    symbols = sorted(sector_map.keys())

    # Group symbols by sector
    sectors: Dict[str, List[str]] = {}
    for sym in symbols:
        sec = sector_map.get(sym, "_unknown")
        sectors.setdefault(sec, []).append(sym)

    sector_raw: Dict[str, float] = {}
    ignition_flag: Dict[str, bool] = {}

    for sec, sec_syms in sectors.items():
        n = len(sec_syms)
        if n == 0:
            sector_raw[sec] = 0.0
            continue

        # Breadth: fraction advancing
        adv = sum(
            1 for s in sec_syms
            if _float_safe(rsr_vel5d.get(s, 0.0)) > 0
        )
        breadth = adv / n

        # Mean RSR velocity (normalize: range -50…+50 → 0…1)
        vels = [_float_safe(rsr_vel5d.get(s, 0.0)) for s in sec_syms]
        mean_vel = float(np.mean(vels))
        vel_norm = float(np.clip((mean_vel + 25.0) / 50.0, 0.0, 1.0))

        # Breakout count
        bo = sum(
            1 for s in sec_syms
            if _float_safe(breakout_expansion.get(s, 1.0)) > 1.2
        )
        breakout_ratio = bo / n

        # Volume expansion (ratio > 1 = above average)
        vols = [_float_safe(volume_ratio_5d.get(s, 1.0)) for s in sec_syms]
        mean_vol = float(np.mean(vols))
        vol_norm = float(np.clip((mean_vol - 0.5) / 2.5, 0.0, 1.0))

        raw = (
            0.35 * breadth
            + 0.25 * vel_norm
            + 0.25 * breakout_ratio
            + 0.15 * vol_norm
        )
        sector_raw[sec] = float(raw)

        igniting = breadth >= ignition_breadth_threshold
        for s in sec_syms:
            ignition_flag[s] = igniting

    # Map sector raw score back to symbols, then cross-sectionally rank
    sym_raw = pd.Series(
        {s: sector_raw.get(sector_map.get(s, "_unknown"), 0.0) for s in symbols},
        dtype=float,
    )

    score_s = _cs_rank_high_good(sym_raw)
    return score_s.fillna(0.0), sector_raw, ignition_flag


# ─────────────────────────────────────────────────────────────────────────────
# 5. Leader / Follower Propagation
# ─────────────────────────────────────────────────────────────────────────────

def leader_follower_scores(
    rsr_today: pd.Series,
    rsr_vel5d: pd.Series,
    compression_ratio: pd.Series,
    sector_map: Dict[str, str],
    *,
    leader_velocity_threshold: float = _LEADER_VELOCITY_THRESHOLD,
    follower_compression_max: float  = _FOLLOWER_COMPRESSION_MAX,
) -> Tuple[pd.Series, Dict[str, bool], Dict[str, bool]]:
    """
    Capture delayed expansion candidates (followers after leader already moved).

    Logic:
      For each sector:
        leader = symbol with highest RSR (deterministic: max RSR, then alphabetical)
        follower qualifying conditions:
          1. leader.rsr_velocity_5d > leader_velocity_threshold  (leader expanded)
          2. follower != leader
          3. follower.compression_ratio < follower_compression_max (still compressed)
          4. follower.rsr_velocity_5d > 0  (starting to turn)

    Returns:
        score_s:      per-symbol 0-100 (followers score highest)
        is_leader:    symbol → bool
        is_follower:  symbol → bool
    """
    symbols = sorted(sector_map.keys())

    sectors: Dict[str, List[str]] = {}
    for sym in symbols:
        sec = sector_map.get(sym, "_unknown")
        sectors.setdefault(sec, []).append(sym)

    is_leader:   Dict[str, bool]  = {s: False for s in symbols}
    is_follower: Dict[str, bool]  = {s: False for s in symbols}
    raw_scores:  Dict[str, float] = {s: 0.0   for s in symbols}

    for sec, sec_syms in sectors.items():
        if not sec_syms:
            continue

        # Deterministic leader selection
        valid = {s: _float_safe(rsr_today.get(s, 0.0)) for s in sec_syms}
        leader = max(valid.keys(), key=lambda s: (valid[s], s))
        is_leader[leader] = True

        if len(sec_syms) < 2:
            continue

        leader_vel = _float_safe(rsr_vel5d.get(leader, 0.0))
        if leader_vel < leader_velocity_threshold:
            continue  # leader hasn't expanded yet

        for s in sec_syms:
            if s == leader:
                continue
            comp = _float_safe(compression_ratio.get(s, 1.0))
            vel  = _float_safe(rsr_vel5d.get(s, 0.0))
            if comp < follower_compression_max and vel > 0:
                is_follower[s] = True
                # score proportional to compression depth × velocity nascence
                compression_depth = max(0.0, follower_compression_max - comp)
                raw_scores[s] = float(np.clip(compression_depth * (vel / 20.0), 0.0, 1.0))

    lf_s = pd.Series(raw_scores, dtype=float)
    if lf_s.max() > 0:
        score_s = _cs_rank_high_good(lf_s)
    else:
        score_s = pd.Series(0.0, index=lf_s.index)

    return score_s.fillna(0.0), is_leader, is_follower


# ─────────────────────────────────────────────────────────────────────────────
# 6. Composite Predictive Expansion Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_predictive_expansion_scores(
    rsr_df: pd.DataFrame,
    ohlcv: Dict[str, pd.DataFrame],
    sector_map: Dict[str, str],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> PredictiveExpansionScores:
    """
    Compute full predictive expansion scores for all universe symbols.

    Args:
        rsr_df:     RSR history (dates × symbols, 0-100). Required ≥15 rows for accel.
        ohlcv:      {symbol: OHLCV DataFrame} with columns [High, Low, Close, Volume].
        sector_map: {symbol: sector_name}
        weights:    optional override for composite weights (must sum to 1.0)

    Returns:
        PredictiveExpansionScores with scores for the latest available date.
        All individual scores default to 0.0 on computation failure (fail-safe).
    """
    w = weights or DEFAULT_WEIGHTS
    symbols = sorted(sector_map.keys())

    eval_date: str
    if len(rsr_df) > 0 and hasattr(rsr_df.index[-1], "strftime"):
        eval_date = rsr_df.index[-1].strftime("%Y-%m-%d")
    else:
        import datetime
        eval_date = datetime.date.today().isoformat()

    result = PredictiveExpansionScores(eval_date=eval_date, symbols=symbols)

    # ── 1. RSR acceleration ───────────────────────────────────────────────────
    try:
        accel_df = rsr_acceleration_scores(rsr_df)
        if not accel_df.empty:
            latest_row = accel_df.iloc[-1]
            for sym in symbols:
                result.rsr_acceleration_score[sym] = _float_safe(
                    latest_row.get(sym, 0.0) if sym in latest_row.index else 0.0
                )

        # Raw velocity / acceleration for diagnostics
        if len(rsr_df) >= 15:
            vel5  = rsr_df - rsr_df.shift(5)
            vel10 = rsr_df - rsr_df.shift(10)
            accel_raw = vel5 - vel5.shift(5)
            for sym in symbols:
                if sym in vel5.columns:
                    result.rsr_velocity_5d[sym]  = _float_safe(vel5[sym].iloc[-1])
                    result.rsr_velocity_10d[sym] = _float_safe(vel10[sym].iloc[-1])
                    result.rsr_acceleration[sym] = _float_safe(accel_raw[sym].iloc[-1])
                else:
                    result.rsr_velocity_5d[sym]  = 0.0
                    result.rsr_velocity_10d[sym] = 0.0
                    result.rsr_acceleration[sym] = 0.0
        else:
            for sym in symbols:
                result.rsr_velocity_5d[sym]  = 0.0
                result.rsr_velocity_10d[sym] = 0.0
                result.rsr_acceleration[sym] = 0.0

    except Exception as exc:
        logger.warning("[PREDICTIVE] rsr_acceleration error: %s", exc)
        for sym in symbols:
            result.rsr_acceleration_score[sym] = 0.0
            result.rsr_velocity_5d[sym]        = 0.0

    # ── 2. Compression breakout ───────────────────────────────────────────────
    try:
        comp_s, tight_s, expand_s, cb_score = compression_breakout_scores(ohlcv)
        for sym in symbols:
            result.compression_breakout_score[sym] = _float_safe(cb_score.get(sym, 0.0))
            result.compression_ratio[sym]          = _float_safe(comp_s.get(sym, 1.0))
            result.price_tightness[sym]            = _float_safe(tight_s.get(sym, 1.0))
            result.breakout_expansion[sym]         = _float_safe(expand_s.get(sym, 1.0))
    except Exception as exc:
        logger.warning("[PREDICTIVE] compression_breakout error: %s", exc)
        for sym in symbols:
            result.compression_breakout_score[sym] = 0.0
            result.compression_ratio[sym]          = 1.0
            result.breakout_expansion[sym]         = 1.0

    # ── 3. Volume regime ──────────────────────────────────────────────────────
    try:
        vr5_s, vr20_s, vol_score = volume_regime_scores(ohlcv)
        for sym in symbols:
            result.volume_regime_score[sym] = _float_safe(vol_score.get(sym, 0.0))
            result.volume_ratio_5d[sym]     = _float_safe(vr5_s.get(sym, 1.0))
            result.volume_ratio_20d[sym]    = _float_safe(vr20_s.get(sym, 1.0))
    except Exception as exc:
        logger.warning("[PREDICTIVE] volume_regime error: %s", exc)
        for sym in symbols:
            result.volume_regime_score[sym] = 0.0
            result.volume_ratio_5d[sym]     = 1.0

    # ── 4. Sector ignition ────────────────────────────────────────────────────
    try:
        _vel5_s   = pd.Series({s: result.rsr_velocity_5d.get(s, 0.0)    for s in symbols})
        _comp_s   = pd.Series({s: result.compression_ratio.get(s, 1.0)  for s in symbols})
        _expand_s = pd.Series({s: result.breakout_expansion.get(s, 1.0) for s in symbols})
        _vr5_s    = pd.Series({s: result.volume_ratio_5d.get(s, 1.0)    for s in symbols})

        ig_score, _sec_raw, ig_flag = sector_ignition_scores(
            rsr_vel5d       = _vel5_s,
            compression_ratio = _comp_s,
            volume_ratio_5d = _vr5_s,
            breakout_expansion = _expand_s,
            sector_map      = sector_map,
        )
        for sym in symbols:
            result.sector_ignition_score[sym]  = _float_safe(ig_score.get(sym, 0.0))
            result.sector_ignition_active[sym] = bool(ig_flag.get(sym, False))
    except Exception as exc:
        logger.warning("[PREDICTIVE] sector_ignition error: %s", exc)
        for sym in symbols:
            result.sector_ignition_score[sym]  = 0.0
            result.sector_ignition_active[sym] = False

    # ── 5. Leader / Follower ──────────────────────────────────────────────────
    try:
        _rsr_now = pd.Series(dtype=float)
        if len(rsr_df) > 0:
            last_rsr = rsr_df.iloc[-1]
            _rsr_now = pd.Series({s: _float_safe(last_rsr.get(s, 0.0)) for s in symbols if s in last_rsr.index})

        lf_score, _is_leader, _is_follower = leader_follower_scores(
            rsr_today       = _rsr_now,
            rsr_vel5d       = _vel5_s,
            compression_ratio = _comp_s,
            sector_map      = sector_map,
        )
        for sym in symbols:
            result.leader_follower_score[sym] = _float_safe(lf_score.get(sym, 0.0))
            result.is_leader[sym]             = bool(_is_leader.get(sym, False))
            result.is_follower[sym]           = bool(_is_follower.get(sym, False))
    except Exception as exc:
        logger.warning("[PREDICTIVE] leader_follower error: %s", exc)
        for sym in symbols:
            result.leader_follower_score[sym] = 0.0
            result.is_leader[sym]             = False
            result.is_follower[sym]           = False

    # ── Composite ─────────────────────────────────────────────────────────────
    for sym in symbols:
        a = result.rsr_acceleration_score.get(sym, 0.0)
        b = result.compression_breakout_score.get(sym, 0.0)
        c = result.sector_ignition_score.get(sym, 0.0)
        d = result.volume_regime_score.get(sym, 0.0)
        e = result.leader_follower_score.get(sym, 0.0)
        composite = (
            w.get("rsr_acceleration",    0.30) * a
            + w.get("compression_breakout", 0.25) * b
            + w.get("sector_ignition",      0.20) * c
            + w.get("volume_regime",        0.15) * d
            + w.get("leader_follower",      0.10) * e
        )
        result.predictive_alpha_score[sym] = round(
            float(np.clip(composite, 0.0, 100.0)), 2
        )

    return result
