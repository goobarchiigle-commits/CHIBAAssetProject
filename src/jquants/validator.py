"""
src/jquants/validator.py
データ品質検証（重複・日付順・欠損・NULL・調整係数異常・OHLC整合性・出来高異常）。

方針: 異常は例外化せず、レポート（dict）に記録して返すのみ。
      呼び出し側（downloader.py）が report を metadata/ に保存し、後続処理を継続する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.jquants.schema import DAILY_BARS_RAW_TO_STANDARD, rename_to_standard

VOLUME_SPIKE_MULTIPLIER = 50.0   # 90日中央値の何倍で「異常」とみなすか
VOLUME_SPIKE_MIN_WINDOW = 20


def check_duplicates(df: pd.DataFrame) -> list[str]:
    dup_dates = df.index[df.index.duplicated(keep=False)]
    if dup_dates.empty:
        return []
    return sorted({d.strftime("%Y-%m-%d") for d in dup_dates})


def check_date_order(df: pd.DataFrame) -> bool:
    """True = 昇順ソート済み（正常）。"""
    return bool(df.index.is_monotonic_increasing)


def check_missing_days(df: pd.DataFrame, gap_threshold_days: int = 7) -> list[dict]:
    """
    連続する営業日想定間隔から gap_threshold_days を超える空白区間を検出する。
    通常の週末・祝日（数日）は誤検知しないよう閾値を設ける。
    """
    if len(df) < 2:
        return []
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(days=gap_threshold_days)]
    return [
        {"from": (idx - gap).strftime("%Y-%m-%d"), "to": idx.strftime("%Y-%m-%d"), "gap_days": int(gap.days)}
        for idx, gap in gaps.items()
    ]


def check_nulls(df: pd.DataFrame, columns: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")) -> dict[str, int]:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return {}
    counts = df[cols].isna().sum()
    return {c: int(n) for c, n in counts.items() if n > 0}


def check_adjustment_factor(df: pd.DataFrame) -> list[str]:
    """AdjustmentFactor が NaN / 0以下 の日付を異常として検出する。"""
    if "AdjustmentFactor" not in df.columns:
        return []
    bad = df["AdjustmentFactor"].isna() | (df["AdjustmentFactor"] <= 0)
    return [d.strftime("%Y-%m-%d") for d in df.index[bad]]


def check_ohlc_consistency(df: pd.DataFrame) -> list[str]:
    """High/Low/Open/Close の大小関係が破綻している日付を検出する。"""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return []
    high, low, opn, close = df["High"], df["Low"], df["Open"], df["Close"]
    violation = (
        (high < low) | (high < opn) | (high < close) | (low > opn) | (low > close)
    )
    violation = violation.fillna(False)
    return [d.strftime("%Y-%m-%d") for d in df.index[violation]]


def check_volume_anomaly(df: pd.DataFrame) -> dict:
    """出来高マイナス・急激なスパイク（90日中央値の一定倍以上）を検出する。"""
    if "Volume" not in df.columns or len(df) < VOLUME_SPIKE_MIN_WINDOW:
        return {"negative": [], "spikes": []}

    vol = df["Volume"]
    negative = [d.strftime("%Y-%m-%d") for d in df.index[vol < 0]]

    median90 = vol.rolling(90, min_periods=VOLUME_SPIKE_MIN_WINDOW).median()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = vol / median90
    spike_mask = (ratio > VOLUME_SPIKE_MULTIPLIER).fillna(False)
    spikes = [d.strftime("%Y-%m-%d") for d in df.index[spike_mask]]

    return {"negative": negative, "spikes": spikes}


def validate_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """
    1銘柄分の全検証を実行し、レポート dict を返す（例外は送出しない）。
    df は J-Quants API v2 の略記フィールド名（O/H/L/C/Vo/AdjFactor/...）のままでも、
    既に標準名（Open/High/Low/Close/...）でも受け付ける（schema.rename_to_standardで正規化してから検証）。
    """
    df = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
    duplicates   = check_duplicates(df)
    ordered      = check_date_order(df)
    missing_days = check_missing_days(df)
    nulls        = check_nulls(df)
    adj_issues   = check_adjustment_factor(df)
    ohlc_issues  = check_ohlc_consistency(df)
    volume       = check_volume_anomaly(df)

    issue_count = (
        len(duplicates) + (0 if ordered else 1) + len(missing_days)
        + sum(nulls.values()) + len(adj_issues) + len(ohlc_issues)
        + len(volume["negative"]) + len(volume["spikes"])
    )

    return {
        "symbol":            symbol,
        "row_count":         int(len(df)),
        "status":            "ok" if issue_count == 0 else "issues_found",
        "issue_count":        int(issue_count),
        "duplicates":         duplicates,
        "date_order_ok":      ordered,
        "missing_day_gaps":   missing_days,
        "null_counts":        nulls,
        "adjustment_factor_issues": adj_issues,
        "ohlc_consistency_issues":  ohlc_issues,
        "volume_anomalies":   volume,
    }
