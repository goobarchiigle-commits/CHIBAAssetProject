"""Tests for src/database/ohlcv.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database import ohlcv


def _sample_row(date: str, code: str, close: float) -> dict:
    return {
        "Date": date, "Code": code, "Open": close, "High": close, "Low": close, "Close": close,
        "Volume": 1000, "AdjustmentFactor": 1.0, "AdjustmentOpen": close, "AdjustmentHigh": close,
        "AdjustmentLow": close, "AdjustmentClose": close, "AdjustmentVolume": 1000,
        "UpperLimit": "-", "LowerLimit": "-", "TurnoverValue": 1e6,
    }


@pytest.fixture(autouse=True)
def _patch_ohlcv_dir(tmp_path, monkeypatch):
    ohlcv_dir = tmp_path / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ohlcv, "DATABASE_OHLCV_DIR", ohlcv_dir)
    monkeypatch.setattr(ohlcv, "ensure_database_market_dirs", lambda: None)
    return ohlcv_dir


class TestSplitByYear:
    def test_splits_correctly(self):
        df = pd.DataFrame([_sample_row("2023-12-30", "1301", 100), _sample_row("2024-01-05", "1301", 105)])
        years = ohlcv.split_by_year(df)
        assert set(years.keys()) == {2023, 2024}
        assert len(years[2023]) == 1
        assert len(years[2024]) == 1

    def test_empty_input(self):
        assert ohlcv.split_by_year(pd.DataFrame()) == {}


class TestSaveLoadRoundtrip:
    def test_roundtrip_sorted_and_deduped(self):
        df = pd.DataFrame([
            _sample_row("2024-01-05", "1301", 105),
            _sample_row("2024-01-04", "1301", 104),
            _sample_row("2024-01-04", "1301", 999),  # 重複キー・keep=lastで999が残るはず
        ])
        ohlcv.save_yearly_parquet(df, 2024)
        loaded = ohlcv.load_yearly_parquet(2024)
        assert len(loaded) == 2  # 重複排除済み
        assert loaded["Date"].is_monotonic_increasing
        row_0104 = loaded.loc[loaded["Date"] == pd.Timestamp("2024-01-04")]
        assert row_0104["Close"].iloc[0] == 999

    def test_load_nonexistent_year_returns_empty_with_schema_columns(self):
        loaded = ohlcv.load_yearly_parquet(1999)
        assert loaded.empty
        assert "Date" in loaded.columns
        assert "Code" in loaded.columns


class TestUpdateCurrentYear:
    def test_appends_without_touching_other_years(self):
        ohlcv.save_yearly_parquet(pd.DataFrame([_sample_row("2023-12-29", "1301", 98)]), 2023)
        ohlcv.save_yearly_parquet(pd.DataFrame([_sample_row("2024-01-04", "1301", 99)]), 2024)

        new_row = pd.DataFrame([_sample_row("2024-01-05", "1301", 100)])
        ohlcv.update_current_year(new_row, year=2024)

        loaded_2024 = ohlcv.load_yearly_parquet(2024)
        assert len(loaded_2024) == 2
        loaded_2023 = ohlcv.load_yearly_parquet(2023)
        assert len(loaded_2023) == 1  # 過去年は不変

    def test_infers_year_when_not_specified(self):
        new_row = pd.DataFrame([_sample_row("2025-03-03", "1301", 150)])
        ohlcv.update_current_year(new_row)
        loaded = ohlcv.load_yearly_parquet(2025)
        assert len(loaded) == 1

    def test_multi_year_input_without_explicit_year_raises(self):
        df = pd.DataFrame([_sample_row("2024-01-01", "1301", 100), _sample_row("2025-01-01", "1301", 100)])
        with pytest.raises(ValueError):
            ohlcv.update_current_year(df)

    def test_empty_input_is_noop(self):
        ohlcv.update_current_year(pd.DataFrame())  # 例外を出さないこと


class TestTradingDaysInRange:
    def test_excludes_weekends(self):
        days = ohlcv.trading_days_in_range("2024-01-01", "2024-01-07")
        weekdays = [d for d in days if d.weekday() < 5]
        assert len(days) == len(weekdays)
