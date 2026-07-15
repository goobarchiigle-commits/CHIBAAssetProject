"""Tests for src/database/dtypes.py"""
from __future__ import annotations

import pandas as pd

from src.database.dtypes import optimize_dtypes


class TestOptimizeDtypesOHLCV:
    def _sample(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Code": ["1301", "1302"],
            "Open": [100.0, 200.0],
            "Close": [101.5, 201.5],
            "Volume": [1000, 2000],
        })

    def test_date_becomes_datetime(self):
        out = optimize_dtypes(self._sample(), "ohlcv")
        assert pd.api.types.is_datetime64_any_dtype(out["Date"])

    def test_code_becomes_category(self):
        out = optimize_dtypes(self._sample(), "ohlcv")
        assert isinstance(out["Code"].dtype, pd.CategoricalDtype)

    def test_price_columns_become_float32(self):
        out = optimize_dtypes(self._sample(), "ohlcv")
        assert out["Open"].dtype == "float32"
        assert out["Close"].dtype == "float32"

    def test_values_preserved(self):
        out = optimize_dtypes(self._sample(), "ohlcv")
        assert out["Open"].tolist() == [100.0, 200.0]

    def test_missing_columns_are_skipped_not_errored(self):
        df = pd.DataFrame({"Date": ["2024-01-01"], "Code": ["1301"]})
        out = optimize_dtypes(df, "ohlcv")
        assert list(out.columns) == ["Date", "Code"]


class TestOptimizeDtypesClassifications:
    def test_nullable_boolean_preserves_na_distinct_from_false(self):
        df = pd.DataFrame({
            "Code": ["1301", "1302"],
            "IsJPXPrime150": [True, False],
            "IsNikkei225": [pd.NA, pd.NA],
        })
        out = optimize_dtypes(df, "classifications")
        assert out["IsJPXPrime150"].dtype == "boolean"
        assert out["IsNikkei225"].isna().all()
        assert out["IsJPXPrime150"].tolist() == [True, False]

    def test_string_columns_use_nullable_string_dtype(self):
        df = pd.DataFrame({"Code": ["1301"], "ScaleCategory": ["TOPIX Core30"]})
        out = optimize_dtypes(df, "classifications")
        assert out["ScaleCategory"].dtype == "string"
