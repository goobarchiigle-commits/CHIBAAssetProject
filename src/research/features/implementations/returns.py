"""Rolling returns feature — n-period price change."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class RollingReturns(FeatureContract):
    feature_name = "rolling_returns"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        ret = df["close"].pct_change(window)
        return pd.DataFrame({"rolling_return": ret}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"rolling_return": float}
