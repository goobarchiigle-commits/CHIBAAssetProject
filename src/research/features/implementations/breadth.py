"""Breadth metrics feature — advance/decline ratio and % above moving average.

Input convention: wide-format DataFrame where rows=dates, columns=symbols,
values=close prices. required_inputs() returns [] because column names are
symbol-specific; the pipeline does not validate them by name.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class BreadthMetrics(FeatureContract):
    feature_name = "breadth_metrics"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return []  # wide-format: any symbol columns accepted

    def compute(self, df: pd.DataFrame, ma_window: int = 50, **_: Any) -> pd.DataFrame:
        ma = df.rolling(ma_window, min_periods=ma_window).mean()
        pct_above_ma = (df > ma).mean(axis=1)

        daily_returns = df.pct_change()
        n_up = (daily_returns > 0).sum(axis=1)
        n_dn = (daily_returns < 0).sum(axis=1)
        total = (n_up + n_dn).replace(0, float("nan"))
        ad_ratio = n_up / total

        return pd.DataFrame(
            {"pct_above_ma": pct_above_ma, "advance_decline_ratio": ad_ratio},
            index=df.index,
        )

    def output_schema(self) -> Dict[str, type]:
        return {"pct_above_ma": float, "advance_decline_ratio": float}
