"""Rolling volatility feature."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class RollingVolatility(FeatureContract):
    feature_name = "rolling_volatility"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        returns = df["close"].pct_change()
        vol = returns.rolling(window, min_periods=window).std()
        return pd.DataFrame({"rolling_vol": vol}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"rolling_vol": float}
