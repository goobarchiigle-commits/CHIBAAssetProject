"""Tests for src/database/master.py (companies/classifications/universe construction)."""
from __future__ import annotations

import pandas as pd

from src.database.master import build_classifications_parquet, build_companies_parquet, build_universe_parquet


def _events() -> pd.DataFrame:
    return pd.DataFrame([
        {"event_date": "2016-07-11", "code": "13010", "event_type": "ADD",
         "company_name": "A", "sector_33_code": "01", "sector_33_name": "x",
         "market_code": "111", "market_code_name": "prime"},
        {"event_date": "2020-01-01", "code": "13010", "event_type": "REMOVE",
         "company_name": "A", "sector_33_code": "01", "sector_33_name": "x",
         "market_code": "111", "market_code_name": "prime"},
        {"event_date": "2016-07-11", "code": "99840", "event_type": "ADD",
         "company_name": "B", "sector_33_code": "02", "sector_33_name": "y",
         "market_code": "111", "market_code_name": "prime"},
    ])


def _listed_snapshot() -> pd.DataFrame:
    return pd.DataFrame({
        "Code": ["13010", "99840"],
        "CompanyName": ["A", "B"],
        "MarketCode": ["111", "111"], "MarketCodeName": ["prime", "prime"],
        "Sector17Code": ["1", "2"], "Sector17CodeName": ["x17", "y17"],
        "Sector33Code": ["01", "02"], "Sector33CodeName": ["x", "y"],
        "ScaleCategory": ["TOPIX Small 1", "-"],
        "MarginCode": ["1", "1"], "MarginCodeName": ["m", "m"], "ProductCategory": ["p", "p"],
    })


class TestBuildUniverseParquet:
    def test_closed_and_open_intervals(self):
        out = build_universe_parquet(_events())
        assert len(out) == 2
        delisted_row = out.loc[out["Code"] == "13010"].iloc[0]
        assert delisted_row["ToDate"] == pd.Timestamp("2020-01-01")
        assert delisted_row["IsDelisted"] == True  # noqa: E712
        open_row = out.loc[out["Code"] == "99840"].iloc[0]
        assert pd.isna(open_row["ToDate"])
        assert open_row["IsDelisted"] == False  # noqa: E712

    def test_empty_events_returns_empty_frame_with_columns(self):
        out = build_universe_parquet(pd.DataFrame())
        assert out.empty
        assert "UniverseName" in out.columns

    def test_relisting_produces_multiple_intervals(self):
        events = pd.DataFrame([
            {"event_date": "2016-01-01", "code": "1000", "event_type": "ADD"},
            {"event_date": "2017-01-01", "code": "1000", "event_type": "REMOVE"},
            {"event_date": "2018-01-01", "code": "1000", "event_type": "ADD"},
        ])
        out = build_universe_parquet(events)
        rows = out.loc[out["Code"] == "1000"]
        assert len(rows) == 2
        assert pd.isna(rows.iloc[1]["ToDate"])  # 2回目区間はオープンのまま


class TestBuildCompaniesParquet:
    def test_listing_and_delisting_dates(self):
        out = build_companies_parquet(_listed_snapshot(), _events())
        delisted = out.loc[out["Code"] == "13010"].iloc[0]
        assert delisted["ListingDate"] == pd.Timestamp("2016-07-11")
        assert delisted["DelistingDate"] == pd.Timestamp("2020-01-01")
        assert delisted["IsCurrentlyListed"] == False  # noqa: E712

        listed = out.loc[out["Code"] == "99840"].iloc[0]
        assert pd.isna(listed["DelistingDate"])
        assert listed["IsCurrentlyListed"] == True  # noqa: E712


class TestBuildClassificationsParquet:
    def test_topix_scale_flags_derived(self):
        out = build_classifications_parquet(_listed_snapshot(), prime150=None)
        row = out.loc[out["Code"] == "13010"].iloc[0]
        assert row["IsTOPIXSmall"] == True  # noqa: E712
        assert row["IsTOPIXCore30"] == False  # noqa: E712

    def test_unknown_scale_category_is_null_not_false(self):
        out = build_classifications_parquet(_listed_snapshot(), prime150=None)
        row = out.loc[out["Code"] == "99840"].iloc[0]  # ScaleCategory="-"
        assert pd.isna(row["IsTOPIXCore30"])
        assert pd.isna(row["IsTOPIXSmall"])

    def test_prime150_membership_true_for_member(self):
        prime150 = pd.DataFrame({"Code": ["99840"]})
        out = build_classifications_parquet(_listed_snapshot(), prime150)
        member = out.loc[out["Code"] == "99840"].iloc[0]
        assert member["IsJPXPrime150"] == True  # noqa: E712
        assert member["jpx_prime150_source"] == "jpx_official_csv"

    def test_prime150_non_member_is_false_not_null(self):
        prime150 = pd.DataFrame({"Code": ["99840"]})
        out = build_classifications_parquet(_listed_snapshot(), prime150)
        non_member = out.loc[out["Code"] == "13010"].iloc[0]
        assert non_member["IsJPXPrime150"] == False  # noqa: E712

    def test_no_prime150_data_leaves_null(self):
        out = build_classifications_parquet(_listed_snapshot(), prime150=None)
        assert out["IsJPXPrime150"].isna().all()

    def test_jpx400_and_nikkei225_always_null_v1(self):
        out = build_classifications_parquet(_listed_snapshot(), prime150=None)
        assert out["IsJPX400"].isna().all()
        assert out["IsNikkei225"].isna().all()
        assert out["jpx400_source"].isna().all()
        assert out["nikkei225_source"].isna().all()
