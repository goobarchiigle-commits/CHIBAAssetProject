"""Liquidity metrics feature — rolling average turnover and volume."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class LiquidityMetrics(FeatureContract):
    feature_name = "liquidity_metrics"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close", "volume"]

    def compute(self, df: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        turnover = df["close"] * df["volume"]
        avg_turnover = turnover.rolling(window, min_periods=window).mean()
        avg_volume = df["volume"].rolling(window, min_periods=window).mean()
        return pd.DataFrame(
            {"avg_turnover": avg_turnover, "avg_volume": avg_volume},
            index=df.index,
        )

    def output_schema(self) -> Dict[str, type]:
        return {"avg_turnover": float, "avg_volume": float}
