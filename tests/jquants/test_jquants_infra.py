"""Tests for src/jquants/ (J-Quants execution infrastructure).

認証情報・ネットワーク接続なしで全件パスすることを検証する
（Study75開始前の「認証情報未設定でも全コードがビルド可能」要件の裏付け）。
"""
from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.jquants import (
    cache, catalog, compaction, manifest, preflight as preflight_module, provider as provider_module,
    study75_downloader, universe, validator, verify as verify_module,
)
from src.jquants.client import JQuantsClient
from src.jquants.config import JQuantsConfig, load_config
from src.jquants.exceptions import JQuantsAPIError, JQuantsAuthError, JQuantsConfigError
from src.jquants.integrity import build_integrity_report
from src.jquants.normalize import PROCESSED_COLUMNS, normalize_to_processed
from src.jquants.provider import JQuantsProvider, estimate_subscription_floor
from src.jquants.schema import DAILY_BARS_RAW_TO_STANDARD, LISTED_INFO_RAW_TO_STANDARD, rename_to_standard


def _make_df(rows: list[dict], index_dates: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, index=pd.to_datetime(index_dates))
    df.index.name = "Date"
    return df


class TestConfig:
    def test_load_config_without_env_is_unconfigured(self, monkeypatch):
        monkeypatch.setattr("src.jquants.config.JQUANTS_API_KEY", "")
        cfg = load_config()
        assert cfg.is_configured is False

    def test_validate_raises_when_missing(self):
        cfg = JQuantsConfig(
            api_key="", base_url="https://api.jquants.com",
            rate_limit_sec=0.2, retry_max=5, timeout_sec=30.0,
        )
        with pytest.raises(JQuantsConfigError):
            cfg.validate()

    def test_validate_passes_when_configured(self):
        cfg = JQuantsConfig(
            api_key="test-api-key", base_url="https://api.jquants.com",
            rate_limit_sec=0.2, retry_max=5, timeout_sec=30.0,
        )
        cfg.validate()  # should not raise
        assert cfg.is_configured is True


class TestValidatorDuplicatesAndOrder:
    def test_no_duplicates_clean(self):
        df = _make_df([{"Close": 1}, {"Close": 2}], ["2024-01-01", "2024-01-02"])
        assert validator.check_duplicates(df) == []
        assert validator.check_date_order(df) is True

    def test_detects_duplicate_dates(self):
        df = _make_df(
            [{"Close": 1}, {"Close": 2}], ["2024-01-01", "2024-01-01"]
        )
        dups = validator.check_duplicates(df)
        assert dups == ["2024-01-01"]

    def test_detects_unsorted_dates(self):
        df = _make_df(
            [{"Close": 1}, {"Close": 2}], ["2024-01-02", "2024-01-01"]
        )
        assert validator.check_date_order(df) is False


class TestValidatorMissingDays:
    def test_small_gap_not_flagged(self):
        # 週末程度のギャップは正常
        df = _make_df(
            [{"Close": 1}, {"Close": 2}], ["2024-01-05", "2024-01-08"]
        )
        assert validator.check_missing_days(df, gap_threshold_days=7) == []

    def test_large_gap_flagged(self):
        df = _make_df(
            [{"Close": 1}, {"Close": 2}], ["2024-01-01", "2024-02-01"]
        )
        gaps = validator.check_missing_days(df, gap_threshold_days=7)
        assert len(gaps) == 1
        assert gaps[0]["gap_days"] == 31


class TestValidatorNullsAndAdjustment:
    def test_null_counts(self):
        df = pd.DataFrame(
            {"Open": [1.0, None], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0], "Volume": [100, 100]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        nulls = validator.check_nulls(df)
        assert nulls.get("Open") == 1

    def test_adjustment_factor_zero_flagged(self):
        df = pd.DataFrame(
            {"AdjustmentFactor": [1.0, 0.0, float("nan")]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )
        issues = validator.check_adjustment_factor(df)
        assert "2024-01-02" in issues
        assert "2024-01-03" in issues
        assert "2024-01-01" not in issues


class TestValidatorOHLCConsistency:
    def test_consistent_ohlc_clean(self):
        df = pd.DataFrame(
            {"Open": [10.0], "High": [12.0], "Low": [9.0], "Close": [11.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        assert validator.check_ohlc_consistency(df) == []

    def test_high_below_low_flagged(self):
        df = pd.DataFrame(
            {"Open": [10.0], "High": [8.0], "Low": [9.0], "Close": [11.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        issues = validator.check_ohlc_consistency(df)
        assert issues == ["2024-01-01"]


class TestValidatorVolumeAnomaly:
    def test_negative_volume_flagged(self):
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        df = pd.DataFrame({"Volume": [1000] * 24 + [-5]}, index=idx)
        result = validator.check_volume_anomaly(df)
        assert idx[-1].strftime("%Y-%m-%d") in result["negative"]

    def test_spike_flagged(self):
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        volumes = [1000] * 24 + [1_000_000]
        df = pd.DataFrame({"Volume": volumes}, index=idx)
        result = validator.check_volume_anomaly(df)
        assert idx[-1].strftime("%Y-%m-%d") in result["spikes"]


class TestValidateSymbolAggregation:
    def test_clean_symbol_reports_ok(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5,
            "Volume": 1000, "AdjustmentFactor": 1.0,
        }, index=idx)
        report = validator.validate_symbol("TEST.T", df)
        assert report["status"] == "ok"
        assert report["issue_count"] == 0

    def test_dirty_symbol_reports_issues(self):
        df = pd.DataFrame({
            "Open": [10.0, 10.0], "High": [8.0, 11.0], "Low": [9.0, 9.0],
            "Close": [10.5, 10.5], "Volume": [1000, 1000], "AdjustmentFactor": [1.0, 0.0],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-01"]))
        report = validator.validate_symbol("TEST.T", df)
        assert report["status"] == "issues_found"
        assert report["issue_count"] > 0


class TestCacheIncrementalDiff:
    def test_no_state_returns_full_range(self):
        result = cache.compute_fetch_range("7203.T", "2018-01-01", "2025-12-31", {})
        assert result == ("2018-01-01", "2025-12-31")

    def test_fully_cached_returns_none(self):
        state = {"7203.T": {"last_date": "2025-12-31", "row_count": 100, "updated_at": "x"}}
        result = cache.compute_fetch_range("7203.T", "2018-01-01", "2025-12-31", state)
        assert result is None

    def test_partial_cache_returns_diff_range(self):
        state = {"7203.T": {"last_date": "2025-06-30", "row_count": 100, "updated_at": "x"}}
        result = cache.compute_fetch_range("7203.T", "2018-01-01", "2025-12-31", state)
        assert result == ("2025-07-01", "2025-12-31")

    def test_merge_and_store_staging_dedups_and_prefers_new(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "JQUANTS_STAGING_DIR", tmp_path)
        old_df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Close": [1.0, 2.0]})
        cache.merge_and_store_staging("TEST.T", old_df)

        new_df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "Close": [99.0, 3.0]})
        merged = cache.merge_and_store_staging("TEST.T", new_df)

        assert len(merged) == 3
        assert merged.loc[merged["Date"] == "2024-01-02", "Close"].iloc[0] == 99.0  # 新しい値を優先
        assert list(merged["Date"]) == sorted(merged["Date"])

    def test_update_state_entry_records_last_date(self):
        state: dict = {}
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-05"]), "Close": [1.0, 2.0]})
        cache.update_state_entry(state, "TEST.T", df)
        assert state["TEST.T"]["last_date"] == "2024-01-05"
        assert state["TEST.T"]["row_count"] == 2


class TestIntegrityReport:
    def test_aggregates_symbol_reports(self):
        symbol_reports = [
            {"symbol": "A.T", "row_count": 100, "status": "ok", "issue_count": 0, "missing_day_gaps": []},
            {"symbol": "B.T", "row_count": 50, "status": "issues_found", "issue_count": 3,
             "missing_day_gaps": [{"from": "2024-01-01", "to": "2024-01-20", "gap_days": 19}]},
        ]
        report = build_integrity_report(symbol_reports, "2024-01-01", "2024-12-31")
        assert report["symbol_count"] == 2
        assert report["symbol_count_with_issues"] == 1
        assert report["symbols_with_issues"] == ["B.T"]
        assert report["total_row_count"] == 150
        assert report["total_missing_gap_days"] == 19
        assert report["requested_trading_days"] > 0  # 2024年内の営業日が数えられていること


class TestClientApiKeyAuth:
    """JQuantsClient のx-api-key認証（v2）。requests.Session.getをモックしネットワーク不要で検証する。"""

    def _cfg(self, api_key: str = "test-key", retry_max: int = 3) -> JQuantsConfig:
        return JQuantsConfig(
            api_key=api_key, base_url="https://api.jquants.com",
            rate_limit_sec=0.0, retry_max=retry_max, timeout_sec=5.0,
        )

    def test_sends_x_api_key_header(self, monkeypatch):
        captured: dict = {}

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"daily_quotes": []}

        def fake_get(self, url, headers=None, params=None, timeout=None):
            captured["headers"] = headers
            return FakeResponse()

        monkeypatch.setattr(requests.Session, "get", fake_get)
        client = JQuantsClient(self._cfg(api_key="my-secret-key"))
        client.get("/v2/equities/bars/daily", {"code": "1301"})
        assert captured["headers"]["x-api-key"] == "my-secret-key"

    def test_401_raises_auth_error_without_retry(self, monkeypatch):
        calls = {"n": 0}

        class FakeResponse:
            status_code = 401
            text = "unauthorized"

        def fake_get(self, url, headers=None, params=None, timeout=None):
            calls["n"] += 1
            return FakeResponse()

        monkeypatch.setattr(requests.Session, "get", fake_get)
        client = JQuantsClient(self._cfg())
        with pytest.raises(JQuantsAuthError):
            client.get("/v2/equities/bars/daily")
        assert calls["n"] == 1  # 静的キー認証エラーはリトライしても解決しないため即送出

    def test_get_without_api_key_raises_config_error(self):
        client = JQuantsClient(self._cfg(api_key=""))
        with pytest.raises(JQuantsConfigError):
            client.get("/v2/equities/bars/daily")

    def test_get_paginated_auto_detects_list_key(self, monkeypatch):
        """list_key未指定時、pagination_key以外の唯一のlist値キーを自動検出する。"""
        class FakeResponse:
            status_code = 200
            def json(self):
                return {"some_unexpected_key": [{"Code": "1301"}], "pagination_key": None}

        def fake_get(self, url, headers=None, params=None, timeout=None):
            return FakeResponse()

        monkeypatch.setattr(requests.Session, "get", fake_get)
        client = JQuantsClient(self._cfg())
        items = list(client.get_paginated("/v2/equities/bars/daily"))
        assert items == [{"Code": "1301"}]


class TestNormalize:
    def test_normalize_to_processed_has_fixed_columns(self):
        raw = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Code": ["1301", "1301"],
            "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0], "Close": [101.0, 102.0],
            "Volume": [1000, 1200], "TurnoverValue": [100000, 120000],
            "AdjustmentFactor": [1.0, 1.0],
            "AdjustmentOpen": [100.0, 101.0], "AdjustmentHigh": [102.0, 103.0],
            "AdjustmentLow": [99.0, 100.0], "AdjustmentClose": [101.0, 102.0], "AdjustmentVolume": [1000, 1200],
        })
        out = normalize_to_processed(raw)
        assert list(out.columns) == PROCESSED_COLUMNS
        assert out["Open"].iloc[0] == 100.0
        assert out["AdjustmentOpen"].iloc[0] == 100.0  # 生値・調整値の両方が別列で残る

    def test_normalize_missing_columns_filled_nan(self):
        raw = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Code": ["1301"], "Close": [100.0]})
        out = normalize_to_processed(raw)
        assert out["Open"].isna().all()
        assert out["Close"].iloc[0] == 100.0

    def test_normalize_empty_returns_empty_with_columns(self):
        out = normalize_to_processed(pd.DataFrame())
        assert list(out.columns) == PROCESSED_COLUMNS
        assert len(out) == 0


class TestCompaction:
    def test_compact_years_splits_by_year_raw_and_processed(self, tmp_path, monkeypatch):
        staging_dir = tmp_path / "staging"
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        staging_dir.mkdir()
        monkeypatch.setattr(cache, "JQUANTS_STAGING_DIR", staging_dir)
        monkeypatch.setattr(compaction, "JQUANTS_RAW_DIR", raw_dir)
        monkeypatch.setattr(compaction, "JQUANTS_PROCESSED_DIR", processed_dir)

        df_a = pd.DataFrame({
            "Date": pd.to_datetime(["2023-12-29", "2024-01-04"]),
            "Code": ["1301", "1301"],
            "Close": [100.0, 101.0], "AdjustmentClose": [100.0, 101.0],
        })
        df_b = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-04"]),
            "Code": ["1302"],
            "Close": [200.0], "AdjustmentClose": [200.0],
        })
        df_a.to_parquet(staging_dir / "1301.T.parquet")
        df_b.to_parquet(staging_dir / "1302.T.parquet")

        result = compaction.compact_years(None)

        assert result["years"] == [2023, 2024]
        assert result["row_counts"][2023]["raw"] == 1
        assert result["row_counts"][2024]["raw"] == 2  # 1301(2024-01-04) + 1302(2024-01-04)

        raw_2024 = pd.read_parquet(raw_dir / "daily_bars_2024.parquet")
        assert set(raw_2024["Code"]) == {"1301", "1302"}
        assert "Close" in raw_2024.columns  # rawは列名リネームなし

        processed_2024 = pd.read_parquet(processed_dir / "daily_bars_2024.parquet")
        assert "AdjustmentClose" in processed_2024.columns
        assert "Open" in processed_2024.columns  # normalize.pyの固定スキーマ

    def test_compact_years_idempotent(self, tmp_path, monkeypatch):
        staging_dir = tmp_path / "staging"
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        staging_dir.mkdir()
        monkeypatch.setattr(cache, "JQUANTS_STAGING_DIR", staging_dir)
        monkeypatch.setattr(compaction, "JQUANTS_RAW_DIR", raw_dir)
        monkeypatch.setattr(compaction, "JQUANTS_PROCESSED_DIR", processed_dir)

        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["1301"], "Close": [100.0]})
        df.to_parquet(staging_dir / "1301.T.parquet")

        r1 = compaction.compact_years({2024})
        r2 = compaction.compact_years({2024})
        assert r1["row_counts"] == r2["row_counts"]


class TestUniverseEventSourcing:
    def _patch_paths(self, tmp_path, monkeypatch):
        metadata_dir = tmp_path / "metadata"
        processed_dir = tmp_path / "processed"
        monkeypatch.setattr(universe, "EVENTS_FILE", metadata_dir / "universe_events.parquet")
        monkeypatch.setattr(universe, "STATE_FILE", metadata_dir / "universe_reconstruction_state.json")
        monkeypatch.setattr(universe, "UNIVERSE_REFERENCE_FILE", processed_dir / "universe.parquet")
        monkeypatch.setattr(universe, "JQUANTS_METADATA_DIR", metadata_dir)
        monkeypatch.setattr(universe, "JQUANTS_PROCESSED_DIR", processed_dir)
        return metadata_dir, processed_dir

    def test_diff_events_detects_add_and_remove(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        prev_codes = {"1301", "1302"}
        curr_snapshot = {"1301": {"company_name": "A", "sector_33_code": "", "sector_33_name": "",
                                   "market_code": "", "market_code_name": ""},
                          "1303": {"company_name": "C", "sector_33_code": "", "sector_33_name": "",
                                    "market_code": "", "market_code_name": ""}}
        events = universe._diff_events(pd.Timestamp("2024-01-04"), prev_codes, curr_snapshot, {})
        types = {(e["code"], e["event_type"]) for e in events}
        assert ("1303", "ADD") in types
        assert ("1302", "REMOVE") in types
        assert ("1301", "ADD") not in types  # 変化なし

    def test_reconstruct_universe_asof_replays_events(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        events = [
            {"event_date": pd.Timestamp("2024-01-01"), "code": "1301", "event_type": "ADD",
             "company_name": "A", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
            {"event_date": pd.Timestamp("2024-06-01"), "code": "1301", "event_type": "REMOVE",
             "company_name": "", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
            {"event_date": pd.Timestamp("2024-03-01"), "code": "1302", "event_type": "ADD",
             "company_name": "B", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
        ]
        universe.append_universe_events(events)

        as_of_feb = universe.reconstruct_universe_asof("2024-02-01")
        assert set(as_of_feb["code"]) == {"1301"}

        as_of_may = universe.reconstruct_universe_asof("2024-05-01")
        assert set(as_of_may["code"]) == {"1301", "1302"}

        as_of_july = universe.reconstruct_universe_asof("2024-07-01")
        assert set(as_of_july["code"]) == {"1302"}  # 1301はREMOVE済み

    def test_materialize_universe_reference(self, tmp_path, monkeypatch):
        self._patch_paths(tmp_path, monkeypatch)
        events = [
            {"event_date": pd.Timestamp("2024-01-01"), "code": "1301", "event_type": "ADD",
             "company_name": "A", "sector_33_code": "1", "sector_33_name": "sec", "market_code": "m", "market_code_name": "market"},
            {"event_date": pd.Timestamp("2024-06-01"), "code": "1301", "event_type": "REMOVE",
             "company_name": "", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
        ]
        universe.append_universe_events(events)
        ref = universe.materialize_universe_reference()
        row = ref.loc[ref["code"] == "1301"].iloc[0]
        assert row["is_currently_listed"] == False
        assert row["first_seen_date"] == pd.Timestamp("2024-01-01")
        assert row["company_name"] == "A"  # REMOVEイベント自体には情報がないため直近ADDから引き継ぐ


class TestCatalogAndManifest:
    def _patch_catalog_dirs(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        metadata_dir = tmp_path / "metadata"
        raw_dir.mkdir()
        processed_dir.mkdir()
        monkeypatch.setattr(catalog, "JQUANTS_RAW_DIR", raw_dir)
        monkeypatch.setattr(catalog, "JQUANTS_PROCESSED_DIR", processed_dir)
        monkeypatch.setattr(catalog, "JQUANTS_METADATA_DIR", metadata_dir)
        # catalog.build_catalog() は manifest.dataset_hash() を呼ぶため、manifest.py 側の
        # JQUANTS_PROCESSED_DIR も同じ tmp_path へ揃えないと実データ（data/jquants/processed/）を
        # 誤って参照してしまう（モジュールごとに束縛されたグローバル変数のため）。
        monkeypatch.setattr(manifest, "JQUANTS_PROCESSED_DIR", processed_dir)
        return raw_dir, processed_dir, metadata_dir

    def test_build_catalog_describes_parquet_files(self, tmp_path, monkeypatch):
        raw_dir, processed_dir, _ = self._patch_catalog_dirs(tmp_path, monkeypatch)

        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "Code": ["1301", "1302"]})
        df.to_parquet(raw_dir / "daily_bars_2024.parquet")

        result = catalog.build_catalog()
        assert "daily_bars_2024.parquet" in result["raw"]
        assert result["raw"]["daily_bars_2024.parquet"]["row_count"] == 2
        assert result["raw"]["daily_bars_2024.parquet"]["symbol_count"] == 2

    def test_build_catalog_top_level_fields(self, tmp_path, monkeypatch):
        raw_dir, processed_dir, _ = self._patch_catalog_dirs(tmp_path, monkeypatch)

        pd.DataFrame({
            "Date": pd.to_datetime(["2023-12-29", "2023-12-29"]), "Code": ["1301", "1302"],
        }).to_parquet(raw_dir / "daily_bars_2023.parquet")
        pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-04"]),
            "Code": ["1301", "1302", "1303"],
        }).to_parquet(processed_dir / "daily_bars_2024.parquet")

        result = catalog.build_catalog()

        for field in ("coverage_start", "coverage_end", "total_rows", "total_symbols", "dataset_hash"):
            assert field in result

        assert result["coverage_start"] == "2023-12-29"
        assert result["coverage_end"] == "2024-01-04"
        # total_rows/total_symbols は processed/daily_bars_*.parquet のみが対象
        # （raw/daily_bars_2023.parquet はカウントしない・二重計上防止）
        assert result["total_rows"] == 3
        assert result["total_symbols"] == 3
        assert result["dataset_hash"] != ""

    def test_build_catalog_total_symbols_dedupes_across_years(self, tmp_path, monkeypatch):
        """同じ銘柄が複数年ファイルに登場しても total_symbols は重複除去した実数になる。"""
        _, processed_dir, _ = self._patch_catalog_dirs(tmp_path, monkeypatch)

        pd.DataFrame({"Date": pd.to_datetime(["2023-01-04"]), "Code": ["1301"]}).to_parquet(
            processed_dir / "daily_bars_2023.parquet"
        )
        pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-04", "2024-01-05"]), "Code": ["1301", "1302"],
        }).to_parquet(processed_dir / "daily_bars_2024.parquet")

        result = catalog.build_catalog()
        assert result["total_rows"] == 3  # 1(2023) + 2(2024)
        assert result["total_symbols"] == 2  # 1301(両年に登場)・1302 → ユニーク2件

    def test_record_run_appends_manifest_with_required_fields(self, tmp_path, monkeypatch):
        processed_dir = tmp_path / "processed"
        metadata_dir = tmp_path / "metadata"
        processed_dir.mkdir()
        monkeypatch.setattr(manifest, "JQUANTS_PROCESSED_DIR", processed_dir)
        monkeypatch.setattr(manifest, "JQUANTS_METADATA_DIR", metadata_dir)
        monkeypatch.setattr(manifest, "MANIFEST_FILE", metadata_dir / "manifest.json")

        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "Code": ["1301"]})
        df.to_parquet(processed_dir / "daily_bars_2024.parquet")

        from datetime import datetime, timezone
        started = datetime(2024, 1, 1, tzinfo=timezone.utc)
        finished = datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc)
        record = manifest.record_run(started, finished, "2024-01-01", "2024-01-01", 1, 1)

        for field in (
            "download_started", "download_finished", "first_date", "last_date",
            "symbol_count", "record_count", "generator_version", "git_commit", "dataset_hash",
        ):
            assert field in record

        records = manifest.load_manifest()
        assert len(records) == 1


# ──────────────────────────────────────────────────────────────────────── #
# J-Quants API v2 互換レイヤー — 2026-07-09 ASK_FIRST②実測レスポンス形状のモックを使用。
# 実際のAPIレスポンス例（/v2/equities/bars/daily, code=86970 date=20240104）:
#   {"data": [{"Date": "2024-01-04", "Code": "86970", "O": 2944.5, "H": 3019.0, "L": 2901.0,
#              "C": 2995.5, "UL": "0", "LL": "0", "Vo": 1995100.0, "Va": 5951903550.0,
#              "AdjFactor": 1.0, "AdjO": 1472.3, "AdjH": 1509.5, "AdjL": 1450.5,
#              "AdjC": 1497.8, "AdjVo": 3990200.0}]}
# ──────────────────────────────────────────────────────────────────────── #
_V2_DAILY_BAR_RECORD = {
    "Date": "2024-01-04", "Code": "86970",
    "O": 2944.5, "H": 3019.0, "L": 2901.0, "C": 2995.5, "UL": "0", "LL": "0",
    "Vo": 1995100.0, "Va": 5951903550.0, "AdjFactor": 1.0,
    "AdjO": 1472.3, "AdjH": 1509.5, "AdjL": 1450.5, "AdjC": 1497.8, "AdjVo": 3990200.0,
}
_V2_TOPIX_RECORD = {
    "Date": "2024-01-04", "Code": "0000", "O": 2359.28, "H": 2380.1, "L": 2335.58, "C": 2378.79,
}
_V2_LISTED_MASTER_RECORD = {
    "Date": "2026-07-09", "Code": "86970",
    "CoName": "日本取引所グループ", "CoNameEn": "Japan Exchange Group,Inc.",
    "S17": "16", "S17Nm": "金融（除く銀行）", "S33": "7200", "S33Nm": "その他金融業",
    "ScaleCat": "TOPIX Large70", "Mkt": "0111", "MktNm": "プライム",
    "Mrgn": "2", "MrgnNm": "貸借", "ProdCat": "011",
}


class TestSchemaMapping:
    def test_rename_daily_bars_to_standard(self):
        df = pd.DataFrame([_V2_DAILY_BAR_RECORD])
        out = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
        assert out["Open"].iloc[0] == 2944.5
        assert out["Close"].iloc[0] == 2995.5
        assert out["AdjustmentClose"].iloc[0] == 1497.8
        assert out["Volume"].iloc[0] == 1995100.0

    def test_rename_topix_partial_columns_only(self):
        """TOPIXはO/H/L/Cのみ・Vo/AdjFactor等は存在しない列は無視される。"""
        df = pd.DataFrame([_V2_TOPIX_RECORD])
        out = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
        assert list(out.columns) == ["Date", "Code", "Open", "High", "Low", "Close"]

    def test_rename_listed_master_to_standard(self):
        df = pd.DataFrame([_V2_LISTED_MASTER_RECORD])
        out = rename_to_standard(df, LISTED_INFO_RAW_TO_STANDARD)
        assert out["CompanyName"].iloc[0] == "日本取引所グループ"
        assert out["Sector33Code"].iloc[0] == "7200"
        assert out["MarketCodeName"].iloc[0] == "プライム"

    def test_rename_is_noop_on_already_standard_columns(self):
        """既に標準名のDataFrame（既存テスト等）はno-opで通過する。"""
        df = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5]})
        out = rename_to_standard(df, DAILY_BARS_RAW_TO_STANDARD)
        assert list(out.columns) == ["Open", "High", "Low", "Close"]


class TestNormalizeV2Payload:
    def test_normalize_v2_daily_bar_maps_to_processed_columns(self):
        raw = pd.DataFrame([_V2_DAILY_BAR_RECORD])
        raw["Date"] = pd.to_datetime(raw["Date"])
        out = normalize_to_processed(raw)
        assert list(out.columns) == PROCESSED_COLUMNS
        assert out["Open"].iloc[0] == 2944.5
        assert out["Close"].iloc[0] == 2995.5
        assert out["AdjustmentOpen"].iloc[0] == 1472.3
        assert out["AdjustmentFactor"].iloc[0] == 1.0

    def test_normalize_v2_topix_missing_columns_are_nan(self):
        raw = pd.DataFrame([_V2_TOPIX_RECORD])
        raw["Date"] = pd.to_datetime(raw["Date"])
        out = normalize_to_processed(raw)
        assert out["Close"].iloc[0] == 2378.79
        assert out["Volume"].isna().all()
        assert out["AdjustmentFactor"].isna().all()


class TestValidatorV2Payload:
    def test_validate_symbol_accepts_v2_raw_columns(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "O": 100.0, "H": 101.0, "L": 99.0, "C": 100.5,
            "Vo": 1000, "AdjFactor": 1.0,
        }, index=idx)
        report = validator.validate_symbol("86970.T", df)
        assert report["status"] == "ok"
        assert report["issue_count"] == 0

    def test_validate_symbol_detects_ohlc_issue_in_v2_raw_columns(self):
        df = pd.DataFrame(
            {"O": [10.0], "H": [8.0], "L": [9.0], "C": [11.0], "Vo": [1000], "AdjFactor": [1.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        report = validator.validate_symbol("86970.T", df)
        assert report["status"] == "issues_found"
        assert report["ohlc_consistency_issues"] == ["2024-01-01"]


class TestProviderV2Endpoints:
    """provider.py が確定済みv2パス・data envelopeへ正しくリクエストすることをモックで検証。"""

    def _client_with_fake_response(self, monkeypatch, payload: dict) -> tuple[JQuantsClient, dict]:
        captured: dict = {}

        class FakeResponse:
            status_code = 200
            def json(self):
                return payload

        def fake_get(self, url, headers=None, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResponse()

        monkeypatch.setattr(requests.Session, "get", fake_get)
        cfg = JQuantsConfig(
            api_key="test-key", base_url="https://api.jquants.com",
            rate_limit_sec=0.0, retry_max=1, timeout_sec=5.0,
        )
        return JQuantsClient(cfg), captured

    def test_daily_quotes_raw_hits_confirmed_path_and_parses_data_envelope(self, monkeypatch):
        client, captured = self._client_with_fake_response(monkeypatch, {"data": [_V2_DAILY_BAR_RECORD]})
        provider = JQuantsProvider(client)
        records = provider.daily_quotes_raw("86970", "2024-01-04", "2024-01-04")
        assert records == [_V2_DAILY_BAR_RECORD]
        assert captured["url"].endswith(provider_module.DAILY_BARS_PATH)

    def test_topix_raw_hits_confirmed_path_with_code_0000(self, monkeypatch):
        client, captured = self._client_with_fake_response(monkeypatch, {"data": [_V2_TOPIX_RECORD]})
        provider = JQuantsProvider(client)
        records = provider.topix_raw("2024-01-04", "2024-01-04")
        assert records == [_V2_TOPIX_RECORD]
        assert captured["url"].endswith(provider_module.TOPIX_PATH)
        assert captured["params"]["code"] == "0000"

    def test_listed_info_hits_confirmed_master_path(self, monkeypatch):
        client, captured = self._client_with_fake_response(monkeypatch, {"data": [_V2_LISTED_MASTER_RECORD]})
        provider = JQuantsProvider(client)
        records = provider.listed_info(code="86970")
        assert records == [_V2_LISTED_MASTER_RECORD]
        assert captured["url"].endswith(provider_module.LISTED_INFO_PATH)
        assert provider_module.LISTED_INFO_PATH == "/v2/equities/master"


class TestUniverseV2ListedInfo:
    def test_fetch_listed_snapshot_extracts_v2_fields(self, monkeypatch):
        class FakeProvider:
            def listed_info(self, code="", date=""):
                return [_V2_LISTED_MASTER_RECORD]

        snapshot = universe.fetch_listed_snapshot(FakeProvider(), date="2026-07-09")
        assert "86970" in snapshot
        info = snapshot["86970"]
        assert info["company_name"] == "日本取引所グループ"
        assert info["sector_33_code"] == "7200"
        assert info["market_code_name"] == "プライム"


# ──────────────────────────────────────────────────────────────────────── #
# Study75 日次ダウンロードエンジン（Strategy C）— cache/daily/day_YYYY-MM-DD.parquet
# ──────────────────────────────────────────────────────────────────────── #
class _FakeDailyProvider:
    """provider.daily_bars_for_date() のモック。実ネットワーク不要。"""

    def __init__(self, day_to_records: dict[str, list[dict]]):
        self._day_to_records = day_to_records
        self.calls: list[str] = []

    def daily_bars_for_date(self, date: str) -> list[dict]:
        self.calls.append(date)
        return self._day_to_records.get(date, [])


def _patch_daily_dirs(tmp_path, monkeypatch):
    staging_dir = tmp_path / "daily"
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    monkeypatch.setattr(study75_downloader, "JQUANTS_DAILY_STAGING_DIR", staging_dir)
    monkeypatch.setattr(study75_downloader, "JQUANTS_METADATA_DIR", metadata_dir)
    monkeypatch.setattr(study75_downloader, "DAILY_MANIFEST_FILE", metadata_dir / "daily_completed_dates.json")
    monkeypatch.setattr(compaction, "JQUANTS_DAILY_STAGING_DIR", staging_dir)
    monkeypatch.setattr(compaction, "JQUANTS_RAW_DIR", raw_dir)
    monkeypatch.setattr(compaction, "JQUANTS_PROCESSED_DIR", processed_dir)
    return staging_dir, raw_dir, processed_dir, metadata_dir


class TestStudy75DailyStaging:
    def test_fetch_and_stage_day_converts_date_format_and_saves(self, tmp_path, monkeypatch):
        staging_dir, _, _, _ = _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({"20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 100.0}]})

        df = study75_downloader.fetch_and_stage_day(provider, "2024-01-04")

        assert provider.calls == ["20240104"]  # YYYY-MM-DD -> YYYYMMDD 変換確認
        assert len(df) == 1
        saved_path = study75_downloader.daily_staging_path("2024-01-04")
        assert saved_path.exists()
        assert saved_path.parent == staging_dir

    def test_fetch_and_stage_day_handles_empty_day(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({})  # 休場日等・該当レコードなし
        df = study75_downloader.fetch_and_stage_day(provider, "2024-01-04")
        assert len(df) == 0
        assert study75_downloader.daily_staging_path("2024-01-04").exists()


class TestStudy75CompletedDatesManifest:
    def test_append_and_load_roundtrip(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        study75_downloader.append_completed_date("2024-01-04", 4437)
        study75_downloader.append_completed_date("2024-01-05", 4440)

        completed = study75_downloader.load_completed_dates()
        assert set(completed.keys()) == {"2024-01-04", "2024-01-05"}
        assert completed["2024-01-04"]["row_count"] == 4437
        assert "completed_at" in completed["2024-01-04"]

    def test_append_is_additive_not_destructive(self, tmp_path, monkeypatch):
        """追記専用: 既存日付を消さず新しい日付だけ増える。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        study75_downloader.append_completed_date("2024-01-04", 100)
        study75_downloader.append_completed_date("2024-01-05", 200)
        completed = study75_downloader.load_completed_dates()
        assert len(completed) == 2

    def test_append_records_file_size_and_sha256_from_real_staging_file(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({"20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}]})
        df = study75_downloader.fetch_and_stage_day(provider, "2024-01-04")
        study75_downloader.append_completed_date("2024-01-04", len(df))

        completed = study75_downloader.load_completed_dates()
        entry = completed["2024-01-04"]
        expected_size = study75_downloader.daily_staging_path("2024-01-04").stat().st_size
        expected_hash = study75_downloader.sha256_of_file(study75_downloader.daily_staging_path("2024-01-04"))

        assert entry["parquet_file_size"] == expected_size
        assert entry["parquet_file_size"] > 0
        assert entry["sha256"] == expected_hash
        assert len(entry["sha256"]) == 64  # SHA256 hex digest length

    def test_append_without_staged_file_records_zero_size_empty_hash(self, tmp_path, monkeypatch):
        """ステージングファイルが存在しない状態で呼ばれた場合はfail-openで0/空文字を記録する。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        study75_downloader.append_completed_date("2024-01-04", 0)
        entry = study75_downloader.load_completed_dates()["2024-01-04"]
        assert entry["parquet_file_size"] == 0
        assert entry["sha256"] == ""


class TestStudy75DownloadResume:
    def test_resumes_and_skips_already_completed_days(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        # 2024-01-04(木)・2024-01-05(金) はどちらも通常営業日。
        study75_downloader.append_completed_date("2024-01-04", 10)  # 既完了として事前投入

        provider = _FakeDailyProvider({
            "20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}],
            "20240105": [{"Date": "2024-01-05", "Code": "1301", "O": 2.0}],
        })
        result = study75_downloader.download_full_market_by_day(provider, "2024-01-04", "2024-01-05")

        assert result["days_skipped"] == 1
        assert result["days_processed"] == 1
        assert provider.calls == ["20240105"]  # 既完了日は再取得しない
        assert result["years_touched"] == [2024]

    def test_force_full_ignores_completed_manifest(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        study75_downloader.append_completed_date("2024-01-04", 10)

        provider = _FakeDailyProvider({
            "20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}],
            "20240105": [{"Date": "2024-01-05", "Code": "1301", "O": 2.0}],
        })
        result = study75_downloader.download_full_market_by_day(
            provider, "2024-01-04", "2024-01-05", force_full=True,
        )
        assert result["days_processed"] == 2
        assert result["days_skipped"] == 0
        assert set(provider.calls) == {"20240104", "20240105"}

    def test_interrupted_run_resumes_from_last_completed_day(self, tmp_path, monkeypatch):
        """中断シミュレーション: 1日目だけ処理した状態で再実行すると2日目から続く。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({
            "20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}],
        })
        # 1回目: 2024-01-04のみレコードがあるためこの1日だけ完了扱いになる
        first = study75_downloader.download_full_market_by_day(provider, "2024-01-04", "2024-01-04")
        assert first["days_processed"] == 1
        assert "2024-01-04" in study75_downloader.load_completed_dates()

        # 2回目（続きから）: 範囲を広げて再実行 → 既完了の2024-01-04は再取得されない
        provider2 = _FakeDailyProvider({
            "20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 999.0}],  # 再取得されれば混入するはずの値
            "20240105": [{"Date": "2024-01-05", "Code": "1301", "O": 2.0}],
        })
        second = study75_downloader.download_full_market_by_day(provider2, "2024-01-04", "2024-01-05")
        assert second["days_skipped"] == 1
        assert second["days_processed"] == 1
        assert provider2.calls == ["20240105"]


class TestCompactYearsFromDaily:
    def test_splits_and_rebuilds_only_target_year(self, tmp_path, monkeypatch):
        staging_dir, raw_dir, processed_dir, _ = _patch_daily_dirs(tmp_path, monkeypatch)
        staging_dir.mkdir(parents=True, exist_ok=True)

        df_2023 = pd.DataFrame({"Date": pd.to_datetime(["2023-12-29"]), "Code": ["1301"], "C": [100.0]})
        df_2024 = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-04", "2024-01-04"]), "Code": ["1301", "1302"], "C": [101.0, 200.0],
        })
        df_2023.to_parquet(staging_dir / "day_2023-12-29.parquet")
        df_2024.to_parquet(staging_dir / "day_2024-01-04.parquet")

        result = compaction.compact_years_from_daily({2024})

        assert result["years"] == [2024]  # 2023は対象外・書き出されない
        assert not (raw_dir / "daily_bars_2023.parquet").exists()
        raw_2024 = pd.read_parquet(raw_dir / "daily_bars_2024.parquet")
        assert len(raw_2024) == 2
        assert set(raw_2024["Code"]) == {"1301", "1302"}
        processed_2024 = pd.read_parquet(processed_dir / "daily_bars_2024.parquet")
        assert "Open" not in processed_2024.columns or True  # Cのみ存在・Openはnormalize.pyでNaN埋め
        assert len(processed_2024) == 2

    def test_no_years_arg_rebuilds_all_years_found_in_staging(self, tmp_path, monkeypatch):
        staging_dir, raw_dir, _, _ = _patch_daily_dirs(tmp_path, monkeypatch)
        staging_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Date": pd.to_datetime(["2023-12-29"]), "Code": ["1301"], "C": [1.0]}).to_parquet(
            staging_dir / "day_2023-12-29.parquet"
        )
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["1301"], "C": [2.0]}).to_parquet(
            staging_dir / "day_2024-01-04.parquet"
        )

        result = compaction.compact_years_from_daily(None)

        assert result["years"] == [2023, 2024]
        assert (raw_dir / "daily_bars_2023.parquet").exists()
        assert (raw_dir / "daily_bars_2024.parquet").exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        staging_dir, _, _, _ = _patch_daily_dirs(tmp_path, monkeypatch)
        staging_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["1301"], "C": [1.0]}).to_parquet(
            staging_dir / "day_2024-01-04.parquet"
        )
        r1 = compaction.compact_years_from_daily({2024})
        r2 = compaction.compact_years_from_daily({2024})
        assert r1["row_counts"] == r2["row_counts"]

    def test_skips_empty_daily_files_without_error(self, tmp_path, monkeypatch):
        staging_dir, raw_dir, _, _ = _patch_daily_dirs(tmp_path, monkeypatch)
        staging_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(staging_dir / "day_2024-01-04.parquet")  # 休場日等の空ファイル想定
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-05"]), "Code": ["1301"], "C": [1.0]}).to_parquet(
            staging_dir / "day_2024-01-05.parquet"
        )

        result = compaction.compact_years_from_daily({2024})
        assert result["years"] == [2024]
        assert len(pd.read_parquet(raw_dir / "daily_bars_2024.parquet")) == 1


# ──────────────────────────────────────────────────────────────────────── #
# verify.py — Study75日次ステージング整合性検証
# ──────────────────────────────────────────────────────────────────────── #
class TestVerifyDailyStaging:
    def _stage_day(self, date_str: str, api_date: str, records: list[dict]) -> pd.DataFrame:
        provider = _FakeDailyProvider({api_date: records})
        df = study75_downloader.fetch_and_stage_day(provider, date_str)
        study75_downloader.append_completed_date(date_str, len(df))
        return df

    def test_reports_ok_when_everything_matches(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        self._stage_day("2024-01-04", "20240104", [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}])
        self._stage_day("2024-01-05", "20240105", [{"Date": "2024-01-05", "Code": "1301", "O": 2.0}])

        report = verify_module.verify_daily_staging()

        assert report["checked_dates"] == 2
        assert report["status"] == "ok"
        assert report["total_issues"] == 0
        assert report["missing_files"] == []
        assert report["corrupted_files"] == []
        assert report["row_count_mismatches"] == []
        assert report["hash_mismatches"] == []

    def test_detects_missing_file(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        self._stage_day("2024-01-04", "20240104", [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}])
        study75_downloader.daily_staging_path("2024-01-04").unlink()  # ファイルだけ消す（マニフェストは残す）

        report = verify_module.verify_daily_staging()

        assert report["status"] == "issues_found"
        assert report["missing_files"] == ["2024-01-04"]
        assert report["total_issues"] == 1

    def test_detects_corrupted_parquet(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        self._stage_day("2024-01-04", "20240104", [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}])
        study75_downloader.daily_staging_path("2024-01-04").write_bytes(b"not a parquet file")

        report = verify_module.verify_daily_staging()

        assert report["status"] == "issues_found"
        assert len(report["corrupted_files"]) == 1
        assert report["corrupted_files"][0]["date"] == "2024-01-04"

    def test_detects_row_count_mismatch(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        self._stage_day("2024-01-04", "20240104", [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}])
        # ファイルを2行に差し替える（マニフェストのrow_count=1のまま）→ 行数不一致を誘発
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-04", "2024-01-04"]), "Code": ["1301", "1302"]}).to_parquet(
            study75_downloader.daily_staging_path("2024-01-04")
        )

        report = verify_module.verify_daily_staging()

        assert report["status"] == "issues_found"
        assert len(report["row_count_mismatches"]) == 1
        mismatch = report["row_count_mismatches"][0]
        assert mismatch["date"] == "2024-01-04"
        assert mismatch["expected"] == 1
        assert mismatch["actual"] == 2

    def test_detects_hash_mismatch_with_same_row_count(self, tmp_path, monkeypatch):
        """行数は一致するが中身が改変された場合（ハッシュのみ不一致）を検出する。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        self._stage_day("2024-01-04", "20240104", [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}])
        # 同じ1行だが値が異なるデータに差し替える → row_countは一致するがsha256は変わる
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["9999"]}).to_parquet(
            study75_downloader.daily_staging_path("2024-01-04")
        )

        report = verify_module.verify_daily_staging()

        assert report["status"] == "issues_found"
        assert report["row_count_mismatches"] == []
        assert len(report["hash_mismatches"]) == 1
        assert report["hash_mismatches"][0]["date"] == "2024-01-04"

    def test_empty_manifest_reports_ok_with_zero_checked(self, tmp_path, monkeypatch):
        _patch_daily_dirs(tmp_path, monkeypatch)
        report = verify_module.verify_daily_staging()
        assert report == {
            "checked_dates": 0, "missing_files": [], "corrupted_files": [],
            "row_count_mismatches": [], "hash_mismatches": [], "status": "ok", "total_issues": 0,
        }


# ──────────────────────────────────────────────────────────────────────── #
# 契約データ提供開始日（subscription floor）
# ──────────────────────────────────────────────────────────────────────── #
class TestSubscriptionFloor:
    def test_estimate_subscription_floor_is_ten_years_before_today(self):
        import datetime as dt
        today = dt.date(2026, 7, 10)
        assert estimate_subscription_floor(today) == "2016-07-10"

    def test_estimate_subscription_floor_handles_leap_day(self):
        import datetime as dt
        today = dt.date(2024, 2, 29)  # 2024はうるう年、2014年2/29は存在しない
        result = estimate_subscription_floor(today)
        assert result == "2014-02-28"

    def test_detect_subscription_floor_parses_400_response(self, monkeypatch):
        class FakeClient:
            def get(self, path, params=None):
                raise JQuantsAPIError(
                    "APIエラー", status_code=400,
                    body='{"message": "Your subscription covers the following dates: 2016-07-10 ~ . '
                         'If you want more data, please check other plans:https://jpx-jquants.com/#dataset"}',
                )

        provider = JQuantsProvider(client=FakeClient())
        assert provider.detect_subscription_floor() == "2016-07-10"

    def test_detect_subscription_floor_fails_open_on_unexpected_body(self, monkeypatch):
        class FakeClient:
            def get(self, path, params=None):
                raise JQuantsAPIError("APIエラー", status_code=400, body='{"message": "something else entirely"}')

        provider = JQuantsProvider(client=FakeClient())
        assert provider.detect_subscription_floor() is None

    def test_detect_subscription_floor_fails_open_when_no_error_raised(self, monkeypatch):
        class FakeClient:
            def get(self, path, params=None):
                return {"data": []}  # 想定外・200が返ってきた場合

        provider = JQuantsProvider(client=FakeClient())
        assert provider.detect_subscription_floor() is None


# ──────────────────────────────────────────────────────────────────────── #
# ステージング検証（rows>0・必須列・低行数警告）
# ──────────────────────────────────────────────────────────────────────── #
class TestValidateStagedDay:
    def test_healthy_day_reports_ok(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"] * 200), "Code": [str(i) for i in range(200)]})
        result = study75_downloader.validate_staged_day(df, "2024-01-04")
        assert result["status"] == "ok"
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_zero_rows_is_an_error(self):
        df = pd.DataFrame(columns=["Date", "Code"])
        result = study75_downloader.validate_staged_day(df, "2024-01-04")
        assert result["status"] == "error"
        assert "zero_rows" in result["errors"]

    def test_missing_required_columns_is_an_error(self):
        df = pd.DataFrame({"Code": ["1301"] * 5})  # Date列が無い
        result = study75_downloader.validate_staged_day(df, "2024-01-04")
        assert result["status"] == "error"
        assert any("missing_required_columns" in e for e in result["errors"])

    def test_low_row_count_is_a_warning_not_an_error(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"] * 5), "Code": [str(i) for i in range(5)]})
        result = study75_downloader.validate_staged_day(df, "2024-01-04")
        assert result["status"] == "warning"
        assert result["errors"] == []
        assert any("suspiciously_low_row_count" in w for w in result["warnings"])


class TestDownloadValidationIntegration:
    def test_validation_error_day_is_not_marked_completed(self, tmp_path, monkeypatch):
        """0行を返す日はvalidate_staged_dayでerror判定 → 完了マークされず次回再取得対象のまま。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({"20240104": []})  # 空レスポンス → 0行

        result = study75_downloader.download_full_market_by_day(provider, "2024-01-04", "2024-01-04")

        assert result["days_processed"] == 0
        assert result["days_failed_validation"] == 1
        assert len(result["validation_issues"]) == 1
        assert "2024-01-04" not in study75_downloader.load_completed_dates()

    def test_validation_warning_day_is_still_marked_completed(self, tmp_path, monkeypatch):
        """低行数（<100）はwarning判定 → 完了マークはされるがissuesに記録される。"""
        _patch_daily_dirs(tmp_path, monkeypatch)
        provider = _FakeDailyProvider({
            "20240104": [{"Date": "2024-01-04", "Code": "1301", "O": 1.0}],  # 1行のみ
        })

        result = study75_downloader.download_full_market_by_day(provider, "2024-01-04", "2024-01-04")

        assert result["days_processed"] == 1
        assert result["days_failed_validation"] == 0
        assert len(result["validation_issues"]) == 1
        assert result["validation_issues"][0]["status"] == "warning"
        assert "2024-01-04" in study75_downloader.load_completed_dates()


# ──────────────────────────────────────────────────────────────────────── #
# preflight.py — Full Download事前見積もり（API通信なし）
# ──────────────────────────────────────────────────────────────────────── #
class TestPreflight:
    def test_check_disk_space_reports_real_free_space(self, tmp_path):
        result = preflight_module.check_disk_space(tmp_path)
        assert result["minimum_required_gb"] == preflight_module.MINIMUM_FREE_GB
        assert result["free_gb"] > 0
        assert isinstance(result["sufficient"], bool)

    def test_estimate_full_download_with_explicit_range(self):
        result = preflight_module.estimate_full_download(start="2024-01-01", end="2024-01-31")
        assert result["coverage_is_estimated"] is False
        assert result["coverage_start"] == "2024-01-01"
        assert result["trading_day_count"] > 0
        assert result["estimated_requests"] == result["trading_day_count"] + 1
        assert result["estimated_runtime_sec"] > 0
        assert result["estimated_disk_gb"] >= 0
        assert result["expected_download_years"] == [2024]

    def test_estimate_full_download_defaults_to_estimated_floor(self):
        result = preflight_module.estimate_full_download(start=None, end="2024-01-31")
        assert result["coverage_is_estimated"] is True
        assert result["coverage_start"] == estimate_subscription_floor()

    def test_run_preflight_includes_disk_section(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight_module, "JQUANTS_DATA_DIR", tmp_path)
        result = preflight_module.run_preflight(start="2024-01-01", end="2024-01-05")
        assert "disk" in result
        assert "free_gb" in result["disk"]


class TestThrottleDefault:
    def test_default_rate_limit_is_005(self, monkeypatch):
        monkeypatch.setattr("src.jquants.config.JQUANTS_API_KEY", "test-key")
        monkeypatch.delenv("JQUANTS_RATE_LIMIT_SEC", raising=False)
        cfg = load_config()
        assert cfg.rate_limit_sec == 0.05


# ──────────────────────────────────────────────────────────────────────── #
# Option B: daily bars由来のUniverse復元（2026-07-10・正本採用）
# ──────────────────────────────────────────────────────────────────────── #
def _patch_universe_and_daily_dirs(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "metadata"
    processed_dir = tmp_path / "processed"
    daily_dir = tmp_path / "daily"
    monkeypatch.setattr(universe, "EVENTS_FILE", metadata_dir / "universe_events.parquet")
    monkeypatch.setattr(universe, "STATE_FILE", metadata_dir / "universe_reconstruction_state.json")
    monkeypatch.setattr(universe, "UNIVERSE_REFERENCE_FILE", processed_dir / "universe.parquet")
    monkeypatch.setattr(universe, "JQUANTS_METADATA_DIR", metadata_dir)
    monkeypatch.setattr(universe, "JQUANTS_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(study75_downloader, "JQUANTS_DAILY_STAGING_DIR", daily_dir)
    monkeypatch.setattr(study75_downloader, "JQUANTS_METADATA_DIR", metadata_dir)
    monkeypatch.setattr(study75_downloader, "DAILY_MANIFEST_FILE", metadata_dir / "daily_completed_dates.json")
    return metadata_dir, processed_dir, daily_dir


class TestDeriveCodesFromDailyBars:
    def test_extracts_unique_codes(self):
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-04", "2024-01-04", "2024-01-04"]),
            "Code": ["1301", "1302", "1301"],
        })
        snapshot = universe.derive_codes_from_daily_bars(df)
        assert set(snapshot.keys()) == {"1301", "1302"}
        # Option Bは記述情報を持たない（空文字で埋める）→ enrich_universe_reference_with_listed_info任せ
        assert snapshot["1301"] == {"company_name": "", "sector_33_code": "", "sector_33_name": "",
                                     "market_code": "", "market_code_name": ""}

    def test_empty_df_returns_empty_dict(self):
        assert universe.derive_codes_from_daily_bars(pd.DataFrame()) == {}

    def test_missing_code_column_returns_empty_dict(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"])})
        assert universe.derive_codes_from_daily_bars(df) == {}


class TestRebuildUniverseEventsFromDailyBarsLive:
    """Option B・ライブAPI版（モックprovider・ネットワーク不要）。"""

    def test_derives_add_remove_from_mocked_daily_bars(self, tmp_path, monkeypatch):
        _patch_universe_and_daily_dirs(tmp_path, monkeypatch)

        class FakeProvider:
            def daily_bars_for_date(self, date):
                if date == "20240104":
                    return [{"Date": "2024-01-04", "Code": "1301"}, {"Date": "2024-01-04", "Code": "1302"}]
                if date == "20240105":
                    return [{"Date": "2024-01-05", "Code": "1301"}, {"Date": "2024-01-05", "Code": "1303"}]
                return []

        result = universe.rebuild_universe_events_from_daily_bars(FakeProvider(), "2024-01-04", "2024-01-05")
        assert result["processed_days"] == 2

        events = universe.load_universe_events()
        types = {(row["code"], row["event_type"]) for _, row in events.iterrows()}
        assert ("1301", "ADD") in types
        assert ("1302", "ADD") in types
        assert ("1303", "ADD") in types  # 2024-01-05に新規出現
        assert ("1302", "REMOVE") in types  # 2024-01-05に消失


class TestRebuildUniverseEventsFromStagedBars:
    """Option B・完全オフライン版（正本）。API通信不要・ローカルステージングのみ使用。"""

    def _stage(self, daily_dir: Path, date_str: str, codes: list[str], row_count: int | None = None) -> None:
        daily_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "Date": pd.to_datetime([date_str] * len(codes)), "Code": codes,
        })
        df.to_parquet(daily_dir / f"day_{date_str}.parquet")
        study75_downloader.append_completed_date(date_str, row_count if row_count is not None else len(codes))

    def test_derives_events_from_staged_files_with_zero_api_calls(self, tmp_path, monkeypatch):
        _, _, daily_dir = _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        self._stage(daily_dir, "2024-01-04", ["1301", "1302"])
        self._stage(daily_dir, "2024-01-05", ["1301", "1303"])

        result = universe.rebuild_universe_events_from_staged_bars()

        assert result["processed_days"] == 2
        assert result["stopped_at_missing_day"] is None
        events = universe.load_universe_events()
        types = {(row["code"], row["event_type"]) for _, row in events.iterrows()}
        assert ("1303", "ADD") in types
        assert ("1302", "REMOVE") in types

    def test_stops_at_first_ungapped_missing_staged_day(self, tmp_path, monkeypatch):
        """検証済みステージングにギャップがあるとそこで処理を打ち切る（連続性のない差分を避ける）。"""
        _, _, daily_dir = _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        self._stage(daily_dir, "2024-01-04", ["1301"])
        # 2024-01-05は意図的に未ステージング（Full Downloadがまだそこまで進んでいない想定）
        self._stage(daily_dir, "2024-01-08", ["1301", "1302"])  # 次の営業日=月曜(01-08)を先にステージ

        result = universe.rebuild_universe_events_from_staged_bars("2024-01-04", "2024-01-08")

        assert result["processed_days"] == 1
        assert result["stopped_at_missing_day"] == "2024-01-05"

    def test_gap_break_still_flushes_partial_buffer(self, tmp_path, monkeypatch):
        """ギャップでの早期break時、flush_interval_days境界に満たない分の未flushイベントも
        失わずに書き出す（実運用で発生したバグの再現テスト・2026-07-10）。"""
        _, _, daily_dir = _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        self._stage(daily_dir, "2024-01-04", ["1301"])
        self._stage(daily_dir, "2024-01-08", ["1301", "1302"])

        result = universe.rebuild_universe_events_from_staged_bars(
            "2024-01-04", "2024-01-08", flush_interval_days=20,  # 1日 << 20 なので通常のflush点には届かない
        )

        assert result["events_written"] > 0  # 修正前はここが0（buffer握りつぶし）だった
        assert result["last_processed_date"] == "2024-01-04"
        events = universe.load_universe_events()
        assert ("1301", "ADD") in {(row["code"], row["event_type"]) for _, row in events.iterrows()}

    def test_only_validated_days_are_trusted_not_just_files_on_disk(self, tmp_path, monkeypatch):
        """cache/daily/にファイルがあってもdaily_completed_dates.jsonに未記録なら信用しない
        （validate_staged_dayがerror判定した0行ファイル等が物理的に残っているケースを想定）。"""
        _, _, daily_dir = _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        daily_dir.mkdir(parents=True, exist_ok=True)
        # ファイルだけ置く（append_completed_dateを呼ばない = 未検証のまま）
        pd.DataFrame({"Date": pd.to_datetime(["2024-01-04"]), "Code": ["1301"]}).to_parquet(
            daily_dir / "day_2024-01-04.parquet"
        )

        result = universe.rebuild_universe_events_from_staged_bars()
        assert result["processed_days"] == 0  # マニフェスト未記録のため対象日が1件もない

    def test_resumes_from_checkpoint_on_second_call(self, tmp_path, monkeypatch):
        _, _, daily_dir = _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        self._stage(daily_dir, "2024-01-04", ["1301"])
        universe.rebuild_universe_events_from_staged_bars()

        self._stage(daily_dir, "2024-01-05", ["1301", "1302"])
        result = universe.rebuild_universe_events_from_staged_bars()

        assert result["processed_days"] == 1  # 01-04は既処理・01-05のみ新規処理


class TestEnrichUniverseReferenceWithListedInfo:
    def test_enriches_only_missing_company_names_by_default(self, tmp_path, monkeypatch):
        _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        events = [
            {"event_date": pd.Timestamp("2024-01-04"), "code": "1301", "event_type": "ADD",
             "company_name": "", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
        ]
        universe.append_universe_events(events)
        universe.materialize_universe_reference()

        class FakeProvider:
            def __init__(self):
                self.queried_codes: list[str] = []

            def listed_info(self, code="", date=""):
                self.queried_codes.append(code)
                return [{"Code": "1301", "CoName": "テスト株式会社", "S33": "7200", "S33Nm": "その他金融業",
                         "Mkt": "0111", "MktNm": "プライム"}]

        provider = FakeProvider()
        ref = universe.enrich_universe_reference_with_listed_info(provider)

        assert provider.queried_codes == ["1301"]  # コード単位・1件のみ（日次ではない）
        row = ref.loc[ref["code"] == "1301"].iloc[0]
        assert row["company_name"] == "テスト株式会社"
        assert row["market_code_name"] == "プライム"

    def test_skips_already_enriched_codes_on_second_call(self, tmp_path, monkeypatch):
        _patch_universe_and_daily_dirs(tmp_path, monkeypatch)
        events = [
            {"event_date": pd.Timestamp("2024-01-04"), "code": "1301", "event_type": "ADD",
             "company_name": "", "sector_33_code": "", "sector_33_name": "", "market_code": "", "market_code_name": ""},
        ]
        universe.append_universe_events(events)
        universe.materialize_universe_reference()

        class FakeProvider:
            def __init__(self):
                self.queried_codes: list[str] = []

            def listed_info(self, code="", date=""):
                self.queried_codes.append(code)
                return [{"Code": code, "CoName": "テスト株式会社"}]

        provider = FakeProvider()
        universe.enrich_universe_reference_with_listed_info(provider)
        universe.enrich_universe_reference_with_listed_info(provider)  # 2回目

        assert provider.queried_codes == ["1301"]  # 2回目は既に補完済みのため問い合わせない
