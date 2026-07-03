"""
research_candidate/vol_adj.py — Study41 D_VOL_ADJ: dynamic max_positions.
Expands max_positions to 4 when TOPIX 20d rolling vol < 0.8%; else 3.
FAIL_OPEN: returns default_max_pos on any data or computation error.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

TOPIX_SYMBOL = "1306.T"   # NEXT FUNDS TOPIX ETF (TOPIX proxy)
VOL_WINDOW   = 20          # 20-day rolling std of daily returns
_STALE_DAYS  = 30          # supplement with yfinance if local data older than this


def _load_topix_local(data_dir: Path) -> Optional[pd.Series]:
    snap = data_dir / f"{TOPIX_SYMBOL}.parquet"
    if not snap.exists():
        return None
    try:
        df = pd.read_parquet(snap)
        # Handle MultiIndex columns from some yfinance/parquet versions
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs("Close", axis=1, level=0)
            return df.dropna() if isinstance(df, pd.Series) else None
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close = df[close_col].dropna()
        close.index = pd.to_datetime(close.index)
        return close
    except Exception as e:
        logger.warning("[VOL_ADJ] local parquet load failed: %s", e)
        return None


def _load_topix_yfinance(days_back: int = 60) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        end_str   = datetime.now(JST).strftime("%Y-%m-%d")
        start_str = (datetime.now(JST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        df = yf.download(TOPIX_SYMBOL, start=start_str, end=end_str, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        return close
    except Exception as e:
        logger.warning("[VOL_ADJ] yfinance fetch failed: %s", e)
        return None


def compute_effective_max_positions(
    data_dir: Path,
    default_max_pos: int   = 3,
    calm_max_pos: int      = 4,
    vol_threshold: float   = 0.008,
    vol_window: int        = VOL_WINDOW,
    state_path: Optional[Path] = None,
) -> int:
    """
    Compute effective max_positions from TOPIX 20d rolling vol.
    Returns calm_max_pos when vol < vol_threshold, else default_max_pos.
    FAIL_OPEN: returns default_max_pos on any error.
    """
    try:
        topix_close = _load_topix_local(data_dir)
        today_date  = datetime.now(JST).date()

        # Supplement with yfinance if local data is stale
        if topix_close is not None and not topix_close.empty:
            last_date = topix_close.index.max().date()
            if (today_date - last_date).days > _STALE_DAYS:
                logger.info("[VOL_ADJ] local data stale (last=%s) — fetching yfinance", last_date)
                fresh = _load_topix_yfinance(days_back=60)
                if fresh is not None and not fresh.empty:
                    topix_close = (
                        pd.concat([topix_close, fresh])
                        .sort_index()
                        .loc[lambda s: ~s.index.duplicated(keep="last")]
                    )
        elif topix_close is None:
            topix_close = _load_topix_yfinance(days_back=60)

        if topix_close is None or len(topix_close) < vol_window + 1:
            logger.warning(
                "[VOL_ADJ] insufficient TOPIX data — defaulting to max_pos=%d", default_max_pos
            )
            return default_max_pos

        daily_ret   = topix_close.pct_change()
        rolling_vol = daily_ret.rolling(vol_window).std().dropna()

        if rolling_vol.empty:
            return default_max_pos

        latest_vol  = float(rolling_vol.iloc[-1])
        regime      = "calm" if latest_vol < vol_threshold else "normal"
        eff_max_pos = calm_max_pos if regime == "calm" else default_max_pos

        logger.info(
            "[VOL_ADJ] TOPIX 20d vol=%.4f threshold=%.3f regime=%s → max_positions=%d",
            latest_vol, vol_threshold, regime, eff_max_pos,
        )

        if state_path is not None:
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "date":                 today_date.strftime("%Y-%m-%d"),
                            "topix_vol_20d":        round(latest_vol, 6),
                            "vol_threshold":        vol_threshold,
                            "regime":               regime,
                            "effective_max_positions": eff_max_pos,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning("[VOL_ADJ] state persist failed: %s", e)

        return eff_max_pos

    except Exception as e:
        logger.warning(
            "[VOL_ADJ] compute failed (%s) — defaulting to max_pos=%d", e, default_max_pos
        )
        return default_max_pos
