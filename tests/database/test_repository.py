"""Tests for src/database/repository.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database import ohlcv as ohlcv_module
from src.database import repository as repository_module
from src.database.repository import MarketDataRepository


def _sample_row(date: str, code: str, close: float) -> dict:
    return {
        "Date": date, "Code": code, "Open": close, "High": close, "Low": close, "Close": close,
        "Volume": 1000, "AdjustmentFactor": 1.0, "AdjustmentOpen": close, "AdjustmentHigh": close,
        "AdjustmentLow": close, "AdjustmentClose": close, "AdjustmentVolume": 1000,
        "UpperLimit": "-", "LowerLimit": "-", "TurnoverValue": 1e6,
    }


@pytest.fixture(autouse=True)
def _patch_dirs(tmp_path, monkeypatch):
    ohlcv_dir = tmp_path / "ohlcv"
    master_dir = tmp_path / "master"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ohlcv_module, "DATABASE_OHLCV_DIR", ohlcv_dir)
    monkeypatch.setattr(ohlcv_module, "ensure_database_market_dirs", lambda: None)
    monkeypatch.setattr(repository_module, "DATABASE_MASTER_DIR", master_dir)
    return {"ohlcv": ohlcv_dir, "master": master_dir}


class TestGetOhlcv:
    def test_filters_by_date_range(self, _patch_dirs):
        ohlcv_module.save_yearly_parquet(pd.DataFrame([
            _sample_row("2024-01-04", "1301", 100), _sample_row("2024-01-05", "1301", 101),
        ]), 2024)
        repo = MarketDataRepository()
        out = repo.get_ohlcv(start="2024-01-05", end="2024-01-05")
        assert len(out) == 1
        assert out.iloc[0]["Close"] == 101

    def test_filters_by_codes(self, _patch_dirs):
        ohlcv_module.save_yearly_parquet(pd.DataFrame([
            _sample_row("2024-01-04", "1301", 100), _sample_row("2024-01-04", "9999", 200),
        ]), 2024)
        repo = MarketDataRepository()
        out = repo.get_ohlcv(codes=["1301"])
        assert out["Code"].astype(str).tolist() == ["1301"]

    def test_no_data_returns_empty(self, _patch_dirs):
        repo = MarketDataRepository()
        assert repo.get_ohlcv().empty


class TestGetUniverseAsof:
    def test_returns_codes_open_at_date(self, _patch_dirs):
        universe = pd.DataFrame({
            "UniverseName": ["TSE_ALL", "TSE_ALL"],
            "Code": ["1301", "9999"],
            "FromDate": pd.to_datetime(["2016-01-01", "2020-01-01"]),
            "ToDate": [pd.NaT, pd.Timestamp("2021-01-01")],
            "IsTradable": [True, True], "IsDelisted": [False, True],
        })
        universe.to_parquet(_patch_dirs["master"] / "universe.parquet", index=False)

        repo = MarketDataRepository()
        assert repo.get_universe_asof("2020-06-01") == ["1301", "9999"]
        assert repo.get_universe_asof("2022-01-01") == ["1301"]
        assert repo.get_universe_asof("2010-01-01") == []

    def test_no_universe_file_returns_empty(self, _patch_dirs):
        repo = MarketDataRepository()
        assert repo.get_universe_asof("2024-01-01") == []


class TestGetCompaniesAndClassifications:
    def test_get_companies_filters_by_code(self, _patch_dirs):
        companies = pd.DataFrame({"Code": ["1301", "9999"], "CompanyName": ["A", "B"]})
        companies.to_parquet(_patch_dirs["master"] / "companies.parquet", index=False)
        repo = MarketDataRepository()
        out = repo.get_companies(codes=["1301"])
        assert out["CompanyName"].tolist() == ["A"]

    def test_get_classifications_no_file_returns_empty(self, _patch_dirs):
        repo = MarketDataRepository()
        assert repo.get_classifications().empty
