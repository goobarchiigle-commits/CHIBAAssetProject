"""Regime features — trend regime (MA cross) and volatility regime (vs rolling median)."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.research.features.contract import FeatureContract


class RegimeFeatures(FeatureContract):
    feature_name = "regime_features"
    feature_version = "1.0"

    def required_inputs(self) -> List[str]:
        return ["close"]

    def compute(
        self,
        df: pd.DataFrame,
        short_window: int = 50,
        long_window: int = 200,
        vol_window: int = 20,
        **_: Any,
    ) -> pd.DataFrame:
        ma_short = df["close"].rolling(short_window, min_periods=short_window).mean()
        ma_long = df["close"].rolling(long_window, min_periods=long_window).mean()
        trend_regime = (ma_short > ma_long).astype(int)

        returns = df["close"].pct_change()
        rolling_vol = returns.rolling(vol_window, min_periods=vol_window).std()
        vol_median = rolling_vol.expanding(min_periods=1).median()
        vol_regime = (rolling_vol > vol_median).astype(int)

        return pd.DataFrame(
            {
                "trend_regime": trend_regime,
                "vol_regime": vol_regime,
                "ma_short": ma_short,
                "ma_long": ma_long,
            },
            index=df.index,
        )

    def output_schema(self) -> Dict[str, type]:
        return {
            "trend_regime": int,
            "vol_regime": int,
            "ma_short": float,
            "ma_long": float,
        }
