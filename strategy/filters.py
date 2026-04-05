from __future__ import annotations

import pandas as pd


def volatility_filter(df: pd.DataFrame, threshold: float = 1.15) -> bool:
    """Short-term ATR expansion must exceed the configured threshold."""
    if df.empty:
        return False
    latest = df.iloc[-1]
    atr_5 = latest.get("atr_5")
    atr_20 = latest.get("atr_20")
    if pd.isna(atr_5) or pd.isna(atr_20) or float(atr_20) <= 0:
        return False
    return bool((float(atr_5) / float(atr_20)) > threshold)


def volume_filter(df: pd.DataFrame, multiplier: float = 1.3) -> bool:
    """Breakout day volume must exceed the 20-day average by multiplier."""
    if df.empty:
        return False
    latest = df.iloc[-1]
    volume = latest.get("volume")
    volume_ma20 = latest.get("volume_ma20")
    if pd.isna(volume) or pd.isna(volume_ma20) or float(volume_ma20) <= 0:
        return False
    return bool(float(volume) > float(volume_ma20) * multiplier)


def market_regime_filter(df: pd.DataFrame) -> bool:
    """TOPIX must stay above its 50-day moving average."""
    if df.empty:
        return False
    latest = df.iloc[-1]
    topix_close = latest.get("topix_close")
    topix_ma50 = latest.get("topix_ma50")
    if pd.isna(topix_close) or pd.isna(topix_ma50):
        return False
    return bool(float(topix_close) > float(topix_ma50))
