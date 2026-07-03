"""Rate-of-change momentum feature."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class Momentum(FeatureContract):
    feature_name = "momentum"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(self, df: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        mom = df["close"] / df["close"].shift(window) - 1.0
        return pd.DataFrame({"momentum": mom}, index=df.index)

    def output_schema(self) -> Dict[str, type]:
        return {"momentum": float}
