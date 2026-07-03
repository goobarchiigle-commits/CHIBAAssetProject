"""Average True Range (ATR) feature — Wilder EMA variant."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class ATR(FeatureContract):
    feature_name = "atr"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["high", "low", "close"]

    def compute(self, df: pd.DataFrame, window: int = 14, **_: Any) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=window, adjust=False, min_periods=window).mean()
        return pd.DataFrame({"atr": atr}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"atr": float}
