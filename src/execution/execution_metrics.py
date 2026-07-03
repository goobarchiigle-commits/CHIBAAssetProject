"""
execution/execution_metrics.py
Execution quality measurement helpers (observation only).
Does NOT influence any trading decisions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from src.paths import LOGS_DIR

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

EXEC_QUALITY_DIR   = LOGS_DIR / "execution_quality"
GAP_SKIP_THRESHOLD = 0.03   # 3% gap → skip BUY entry
PRICE_DRIFT_CANCEL = 0.02   # 2% drift from planned → cancel pending order


def compute_gap_pct(current_price: float, prev_close: float) -> float | None:
    """gap_pct = (current_price - prev_close) / prev_close.  None if prev_close <= 0."""
    if not prev_close or prev_close <= 0:
        return None
    return (current_price - prev_close) / prev_close


def compute_rsr_percentile(
    rsr_universe_df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute cross-sectional percentile rank from RSR time-series matrix.
    Observation only — not used in any trading decision.

    Args:
        rsr_universe_df: DataFrame (dates x symbols) of RSR values

    Returns:
        rsr_pct_raw   : {symbol: float} latest cross-sectional percentile [0, 1]
        rsr_pct_smooth: {symbol: float} EMA(span=3) of percentile over time, latest value
    """
    if rsr_universe_df.empty:
        return {}, {}

    # Cross-sectional rank per date: pct=True gives rank / n in [0, 1]
    pct_matrix = rsr_universe_df.rank(axis=1, pct=True)

    rsr_pct_raw: dict[str, float] = {
        sym: round(float(v), 4)
        for sym, v in pct_matrix.iloc[-1].items()
        if pd.notna(v)
    }

    # EMA(3) over time per symbol, then take latest value
    smooth_matrix = pct_matrix.ewm(span=3, adjust=False).mean()
    rsr_pct_smooth: dict[str, float] = {
        sym: round(float(v), 4)
        for sym, v in smooth_matrix.iloc[-1].items()
        if pd.notna(v)
    }

    return rsr_pct_raw, rsr_pct_smooth


def log_execution_event(entry: dict) -> None:
    """Append one execution quality record to logs/execution_quality/YYYYMMDD.jsonl."""
    today = datetime.now(JST).strftime("%Y%m%d")
    EXEC_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXEC_QUALITY_DIR / f"{today}.jsonl"
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("execution_metrics write failed: %s", exc)
