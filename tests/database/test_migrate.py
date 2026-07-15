"""Tests for src/database/migrate.py (data/jquants/processed からの一回限りの移行)。

実データ・ネットワークには依存しない。data/jquants/ 相当のsyntheticディレクトリを用意し、
JQuantsSource/JPXOfficialSourceの外部呼び出しはモックする。
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database import migrate
from src.database import ohlcv as ohlcv_module


def _make_processed_year(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([f"{year}-01-04", f"{year}-01-05"]),
        "Code": ["13010", "13010"],
        "Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0], "Close": [100.5, 101.5],
        "Volume": [1000, 2000], "AdjustmentFactor": [1.0, 1.0],
        "AdjustmentOpen": [100.0, 101.0], "AdjustmentHigh": [101.0, 102.0],
        "AdjustmentLow": [99.0, 100.0], "AdjustmentClose": [100.5, 101.5], "AdjustmentVolume": [1000, 2000],
    })


def _make_raw_year(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([f"{year}-01-04", f"{year}-01-05"]),
        "Code": ["13010", "13010"],
        "UL": ["-", "-"], "LL": ["-", "-"], "Va": [1e6, 2e6],
    })


@pytest.fixture
def _patched(tmp_path, monkeypatch):
    processed_dir = tmp_path / "jquants_processed"
    raw_dir = tmp_path / "jquants_raw"
    ohlcv_dir = tmp_path / "database_ohlcv"
    master_dir = tmp_path / "database_master"
    for d in (processed_dir, raw_dir, ohlcv_dir, master_dir):
        d.mkdir(parents=True, exist_ok=True)

    _make_processed_year(2024).to_parquet(processed_dir / "daily_bars_2024.parquet", index=False)
    _make_raw_year(2024).to_parquet(raw_dir / "daily_bars_2024.parquet", index=False)

    monkeypatch.setattr(migrate, "JQUANTS_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(migrate, "JQUANTS_RAW_DIR", raw_dir)
    monkeypatch.setattr(migrate, "DATABASE_MASTER_DIR", master_dir)
    monkeypatch.setattr(ohlcv_module, "DATABASE_OHLCV_DIR", ohlcv_dir)
    monkeypatch.setattr(ohlcv_module, "ensure_database_market_dirs", lambda: None)
    monkeypatch.setattr(migrate, "ensure_database_market_dirs", lambda: None)

    events = pd.DataFrame([
        {"event_date": "2016-07-11", "code": "13010", "event_type": "ADD",
         "company_name": "A", "sector_33_code": "01", "sector_33_name": "x",
         "market_code": "111", "market_code_name": "prime"},
    ])
    monkeypatch.setattr(migrate, "load_universe_events", lambda: events)

    listed_snapshot = pd.DataFrame({
        "Code": ["13010"], "CompanyName": ["A"], "MarketCode": ["111"], "MarketCodeName": ["prime"],
        "Sector17Code": ["1"], "Sector17CodeName": ["x17"], "Sector33Code": ["01"], "Sector33CodeName": ["x"],
        "ScaleCategory": ["TOPIX Core30"], "MarginCode": ["1"], "MarginCodeName": ["m"], "ProductCategory": ["p"],
    })
    monkeypatch.setattr(migrate.JQuantsSource, "fetch_master", lambda self, kind: listed_snapshot)
    monkeypatch.setattr(
        migrate.JPXOfficialSource, "fetch_jpx_prime150_constituents",
        lambda self: pd.DataFrame({"Code": ["13010"]}),
    )
    monkeypatch.setattr(migrate.db_metadata, "write_metadata", lambda run_record=None: {})

    return {"processed": processed_dir, "raw": raw_dir, "ohlcv": ohlcv_dir, "master": master_dir}


class TestMigrateFromJQuantsPipeline:
    def test_ohlcv_migrated_with_ul_ll_turnover(self, _patched):
        result = migrate.migrate_from_jquants_pipeline()
        assert result["years_migrated"] == [2024]
        loaded = ohlcv_module.load_yearly_parquet(2024)
        assert len(loaded) == 2
        assert "TurnoverValue" in loaded.columns
        assert loaded["TurnoverValue"].notna().all()

    def test_does_not_write_to_jquants_dirs(self, _patched):
        processed_before = sorted(_patched["processed"].glob("*.parquet"))
        raw_before = sorted(_patched["raw"].glob("*.parquet"))
        mtimes_before = {p: p.stat().st_mtime for p in processed_before + raw_before}

        migrate.migrate_from_jquants_pipeline()

        processed_after = sorted(_patched["processed"].glob("*.parquet"))
        raw_after = sorted(_patched["raw"].glob("*.parquet"))
        assert [p.name for p in processed_after] == [p.name for p in processed_before]
        assert [p.name for p in raw_after] == [p.name for p in raw_before]
        for p in processed_after + raw_after:
            assert p.stat().st_mtime == mtimes_before[p]

    def test_master_tables_written(self, _patched):
        migrate.migrate_from_jquants_pipeline()
        assert (_patched["master"] / "companies.parquet").exists()
        assert (_patched["master"] / "classifications.parquet").exists()
        assert (_patched["master"] / "universe.parquet").exists()
        assert (_patched["master"] / "indices.parquet").exists()

        classifications = pd.read_parquet(_patched["master"] / "classifications.parquet")
        row = classifications.iloc[0]
        assert row["IsTOPIXCore30"] == True  # noqa: E712
        assert row["IsJPXPrime150"] == True  # noqa: E712

    def test_no_processed_files_returns_empty_years(self, tmp_path, monkeypatch, _patched):
        empty_dir = tmp_path / "empty_processed"
        empty_dir.mkdir()
        monkeypatch.setattr(migrate, "JQUANTS_PROCESSED_DIR", empty_dir)
        result = migrate.migrate_from_jquants_pipeline()
        assert result["years_migrated"] == []
        assert result["ohlcv_rows"] == 0
