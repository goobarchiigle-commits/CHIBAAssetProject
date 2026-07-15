"""Tests for src/database/sync.py (日次更新・data/jquants/への書き込みゼロを検証)。"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database import ohlcv as ohlcv_module
from src.database import sync


def _day_frame(date: str, codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [pd.Timestamp(date)] * len(codes), "Code": codes,
        "Open": [100.0] * len(codes), "High": [101.0] * len(codes), "Low": [99.0] * len(codes),
        "Close": [100.5] * len(codes), "Volume": [1000] * len(codes), "AdjustmentFactor": [1.0] * len(codes),
        "AdjustmentOpen": [100.0] * len(codes), "AdjustmentHigh": [101.0] * len(codes),
        "AdjustmentLow": [99.0] * len(codes), "AdjustmentClose": [100.5] * len(codes),
        "AdjustmentVolume": [1000] * len(codes),
        "UpperLimit": ["-"] * len(codes), "LowerLimit": ["-"] * len(codes), "TurnoverValue": [1e6] * len(codes),
    })


@pytest.fixture
def _patched(tmp_path, monkeypatch):
    ohlcv_dir = tmp_path / "database_ohlcv"
    master_dir = tmp_path / "database_master"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    master_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ohlcv_module, "DATABASE_OHLCV_DIR", ohlcv_dir)
    monkeypatch.setattr(ohlcv_module, "ensure_database_market_dirs", lambda: None)
    monkeypatch.setattr(sync, "DATABASE_MASTER_DIR", master_dir)
    monkeypatch.setattr(sync, "ensure_database_market_dirs", lambda: None)

    legacy_jquants_dir = tmp_path / "data_jquants_legacy"
    legacy_jquants_dir.mkdir()
    sentinel_file = legacy_jquants_dir / "processed" / "daily_bars_2024.parquet"
    sentinel_file.parent.mkdir(parents=True)
    sentinel_file.write_bytes(b"untouched")

    monkeypatch.setattr(sync, "authenticate", lambda: True)
    monkeypatch.setattr(sync.db_metadata, "write_metadata", lambda run_record=None: {})
    monkeypatch.setattr(sync.JQuantsSource, "fetch_master", lambda self, kind: pd.DataFrame({
        "Code": ["13010"], "CompanyName": ["A"], "ScaleCategory": ["-"],
    }))
    monkeypatch.setattr(
        sync.JPXOfficialSource, "fetch_jpx_prime150_constituents", lambda self: pd.DataFrame({"Code": []}),
    )
    return {"ohlcv": ohlcv_dir, "master": master_dir, "legacy_sentinel": sentinel_file}


class TestMissingTradingDays:
    def test_no_metadata_yields_no_days(self, monkeypatch):
        monkeypatch.setattr(sync.db_metadata, "read_metadata", lambda: {})
        assert sync._missing_trading_days() == []

    def test_returns_days_after_coverage_end(self, monkeypatch):
        monkeypatch.setattr(sync.db_metadata, "read_metadata", lambda: {"coverage_end": "2024-01-04"})
        days = sync._missing_trading_days()
        assert all(d > pd.Timestamp("2024-01-04") for d in days)


class TestUpdateUniverseForDay:
    def test_new_code_opens_interval(self, _patched):
        sync._update_universe_for_day(pd.Timestamp("2024-01-05"), {"13010"})
        universe = pd.read_parquet(_patched["master"] / "universe.parquet")
        assert len(universe) == 1
        assert pd.isna(universe.iloc[0]["ToDate"])

    def test_disappearing_code_closes_interval(self, _patched):
        sync._update_universe_for_day(pd.Timestamp("2024-01-05"), {"13010"})
        sync._update_universe_for_day(pd.Timestamp("2024-01-08"), set())
        universe = pd.read_parquet(_patched["master"] / "universe.parquet")
        row = universe.iloc[0]
        assert row["ToDate"] == pd.Timestamp("2024-01-08")
        assert row["IsDelisted"] == True  # noqa: E712


class TestUpdateDatabase:
    def test_no_missing_days_is_noop(self, _patched, monkeypatch):
        monkeypatch.setattr(sync.db_metadata, "read_metadata", lambda: {"coverage_end": "2099-01-01"})
        result = sync.update_database()
        assert result == {"status": "ok", "days_fetched": 0, "rows_added": 0}

    def test_authenticate_failure_short_circuits(self, _patched, monkeypatch):
        monkeypatch.setattr(sync, "authenticate", lambda: False)
        result = sync.update_database()
        assert result["status"] == "failed"

    def test_never_touches_legacy_jquants_directory(self, _patched, monkeypatch):
        monkeypatch.setattr(sync.db_metadata, "read_metadata", lambda: {"coverage_end": "2024-01-03"})
        monkeypatch.setattr(
            sync.JQuantsSource, "fetch_ohlcv_for_date",
            lambda self, date_str: _day_frame(pd.Timestamp(date_str).strftime("%Y-%m-%d"), ["13010"]),
        )
        mtime_before = _patched["legacy_sentinel"].stat().st_mtime
        content_before = _patched["legacy_sentinel"].read_bytes()

        sync.update_database()

        assert _patched["legacy_sentinel"].stat().st_mtime == mtime_before
        assert _patched["legacy_sentinel"].read_bytes() == content_before

    def test_writes_new_rows_to_current_year(self, _patched, monkeypatch):
        # coverage_endを「直近数営業日前」に設定し、当年内に収まる少数日だけを不足扱いにする
        # （実行日に依存しても年またぎしない範囲にするため直近15日から選ぶ）。
        today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
        recent_days = sync.trading_days_in_range((today - pd.Timedelta(days=15)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        coverage_end = recent_days[-4].strftime("%Y-%m-%d")
        expected_year = recent_days[-1].year

        monkeypatch.setattr(sync.db_metadata, "read_metadata", lambda: {"coverage_end": coverage_end})
        monkeypatch.setattr(
            sync.JQuantsSource, "fetch_ohlcv_for_date",
            lambda self, date_str: _day_frame(pd.Timestamp(date_str).strftime("%Y-%m-%d"), ["13010"]),
        )
        result = sync.update_database()
        assert result["status"] == "ok"
        assert result["rows_added"] > 0
        loaded = ohlcv_module.load_yearly_parquet(expected_year)
        assert len(loaded) == result["days_fetched"]


class TestSyncModuleDoesNotImportLegacyIngestionEngine:
    def test_no_study75_downloader_or_compaction_import(self):
        assert not hasattr(sync, "study75_downloader")
        assert not hasattr(sync, "compaction")
