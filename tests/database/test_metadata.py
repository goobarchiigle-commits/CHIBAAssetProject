"""Tests for src/database/metadata.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database import metadata as db_metadata


@pytest.fixture(autouse=True)
def _patch_dirs(tmp_path, monkeypatch):
    market_dir = tmp_path / "database" / "market"
    ohlcv_dir = market_dir / "ohlcv"
    master_dir = market_dir / "master"
    metadata_dir = market_dir / "metadata"
    for d in (ohlcv_dir, master_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(db_metadata, "DATABASE_MARKET_DIR", market_dir)
    monkeypatch.setattr(db_metadata, "DATABASE_OHLCV_DIR", ohlcv_dir)
    monkeypatch.setattr(db_metadata, "DATABASE_MASTER_DIR", master_dir)
    monkeypatch.setattr(db_metadata, "DATABASE_METADATA_DIR", metadata_dir)
    monkeypatch.setattr(db_metadata, "DATASET_INFO_FILE", metadata_dir / "dataset_info.json")
    monkeypatch.setattr(db_metadata, "SCHEMA_FILE", metadata_dir / "schema.json")
    monkeypatch.setattr(db_metadata, "UPDATE_HISTORY_FILE", metadata_dir / "update_history.parquet")
    monkeypatch.setattr(db_metadata, "ensure_database_market_dirs", lambda: None)
    return {"ohlcv": ohlcv_dir, "master": master_dir, "metadata": metadata_dir}


class TestWriteReadMetadata:
    def test_read_before_write_returns_empty(self):
        assert db_metadata.read_metadata() == {}

    def test_write_then_read_roundtrip(self, _patch_dirs):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04", "2024-01-05"]), "Code": ["1301", "1301"]})
        df.to_parquet(_patch_dirs["ohlcv"] / "2024.parquet", index=False)

        info = db_metadata.write_metadata()
        assert info["coverage_start"] == "2024-01-04"
        assert info["coverage_end"] == "2024-01-05"

        reloaded = db_metadata.read_metadata()
        assert reloaded["coverage_start"] == "2024-01-04"

    def test_schema_json_written(self, _patch_dirs):
        db_metadata.write_metadata()
        assert (_patch_dirs["metadata"] / "schema.json").exists()

    def test_dataset_hash_changes_when_data_changes(self, _patch_dirs):
        df1 = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["1301"]})
        df1.to_parquet(_patch_dirs["ohlcv"] / "2024.parquet", index=False)
        info1 = db_metadata.write_metadata()

        df2 = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04", "2024-01-05"]), "Code": ["1301", "1301"]})
        df2.to_parquet(_patch_dirs["ohlcv"] / "2024.parquet", index=False)
        info2 = db_metadata.write_metadata()

        assert info1["dataset_hash"] != info2["dataset_hash"]


class TestUpdateHistory:
    def test_append_creates_one_row(self, _patch_dirs):
        db_metadata.append_update_history({
            "started_at": "2024-01-01T00:00:00", "finished_at": "2024-01-01T00:01:00",
            "source": "jquants_api", "tables_updated": ["ohlcv"], "rows_added": 10,
            "date_range_from": "2024-01-01", "date_range_to": "2024-01-01", "status": "ok",
        })
        history = db_metadata.load_update_history()
        assert len(history) == 1
        assert history.iloc[0]["status"] == "ok"

    def test_multiple_appends_accumulate(self, _patch_dirs):
        for i in range(3):
            db_metadata.append_update_history({"source": "jquants_api", "status": "ok"})
        history = db_metadata.load_update_history()
        assert len(history) == 3

    def test_write_metadata_with_run_record_appends_history(self, _patch_dirs):
        db_metadata.write_metadata(run_record={"source": "migrate", "status": "ok"})
        history = db_metadata.load_update_history()
        assert len(history) == 1
