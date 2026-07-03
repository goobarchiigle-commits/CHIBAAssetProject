"""Deterministic rule-based regime labeler.

Each timestamp receives exactly one regime label.
Labels are assigned in REGIME_PRIORITY order; first matching rule wins.
No ML, no randomness, no hidden state.

Required columns: open, high, low, close, volume
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.research.regime.rules import (
    TREND_UP, TREND_DOWN, RANGE, HIGH_VOL, LOW_VOL, PANIC, GAP_UP, GAP_DOWN,
    GAP_UP_THRESH, GAP_DOWN_THRESH,
    PANIC_RETURN_THRESH, PANIC_VOL_PERCENTILE,
    ADX_TREND_THRESH, ATR_PERIOD, MA_PERIOD,
    VOL_LOOKBACK, HIGH_VOL_PERCENTILE, LOW_VOL_PERCENTILE,
)

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def label_regimes(ohlcv: pd.DataFrame) -> pd.Series:
    """Assign one regime label per row. Same input → same output, always.

    Priority (highest first):
      gap_up / gap_down → panic → high_vol / low_vol → trend_up / trend_down → range
    """
    missing = _REQUIRED_COLS - set(ohlcv.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if len(ohlcv) == 0:
        return pd.Series(dtype=object)

    o = ohlcv["open"]
    h = ohlcv["high"]
    l = ohlcv["low"]
    c = ohlcv["close"]

    # ── derived series ────────────────────────────────────────────────────────
    gap_pct = (o - c.shift(1)) / (c.shift(1).abs() + 1e-9)
    daily_ret = c.pct_change()
    atr = _atr(h, l, c, ATR_PERIOD)
    atr_pct = atr / (c.abs() + 1e-9)

    vol_panic_thresh = atr_pct.rolling(VOL_LOOKBACK, min_periods=1).quantile(PANIC_VOL_PERCENTILE)
    vol_high_thresh  = atr_pct.rolling(VOL_LOOKBACK, min_periods=1).quantile(HIGH_VOL_PERCENTILE)
    vol_low_thresh   = atr_pct.rolling(VOL_LOOKBACK, min_periods=1).quantile(LOW_VOL_PERCENTILE)

    adx, plus_di, minus_di = _adx_proxy(h, l, c, ATR_PERIOD)
    ma = c.rolling(MA_PERIOD, min_periods=1).mean()

    # ── label assignment (priority order) ─────────────────────────────────────
    label = pd.Series(RANGE, index=ohlcv.index, dtype=object)

    # trend (lowest priority, set first)
    label = label.where(~((adx >= ADX_TREND_THRESH) & (c < ma)), TREND_DOWN)
    label = label.where(~((adx >= ADX_TREND_THRESH) & (c >= ma)), TREND_UP)

    # volatility (overrides trend)
    label = label.where(~(atr_pct <= vol_low_thresh), LOW_VOL)
    label = label.where(~(atr_pct >= vol_high_thresh), HIGH_VOL)

    # panic (overrides vol)
    is_panic = (daily_ret < PANIC_RETURN_THRESH) & (atr_pct >= vol_panic_thresh)
    label = label.where(~is_panic, PANIC)

    # gap (highest priority)
    label = label.where(~(gap_pct <= GAP_DOWN_THRESH), GAP_DOWN)
    label = label.where(~(gap_pct >= GAP_UP_THRESH), GAP_UP)

    # row 0: no previous close for gap/ret → default to range
    label.iloc[0] = RANGE

    return label


def compute_regime_distribution(labels: pd.Series) -> Dict[str, float]:
    """Fraction of each regime. Deterministic: sorted keys."""
    if len(labels) == 0:
        return {}
    counts = labels.value_counts()
    total = float(len(labels))
    return {regime: float(counts.get(regime, 0)) / total for regime in sorted(counts.index)}


def label_hash(labels: pd.Series) -> str:
    """16-char deterministic hash of a label series."""
    payload = json.dumps(
        {"index": [str(i) for i in labels.index], "values": labels.tolist()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Private helpers ────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _adx_proxy(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Simplified directional movement index. Deterministic."""
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)
    atr = _atr(high, low, close, period)
    eps = 1e-9
    plus_di  = 100.0 * plus_dm.rolling(period, min_periods=1).mean() / (atr + eps)
    minus_di = 100.0 * minus_dm.rolling(period, min_periods=1).mean() / (atr + eps)
    di_diff  = (plus_di - minus_di).abs()
    di_sum   = plus_di + minus_di + eps
    dx  = 100.0 * di_diff / di_sum
    adx = dx.rolling(period, min_periods=1).mean()
    return adx, plus_di, minus_di
