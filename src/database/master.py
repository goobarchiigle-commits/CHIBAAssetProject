"""
src/database/master.py
database/market/master/{companies,classifications,universe}.parquet の構築。

データは sources/ 層（jquants_source.py・jpx_official.py）からのみ取得し、J-Quants固有の
HTTP詳細（エンドポイント・認証等）はここでは扱わない。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.database.dtypes import optimize_dtypes
from src.database.schema import validate_schema
from src.database.sources.jpx_official import JPXOfficialSource
from src.database.sources.jquants_source import JQuantsSource

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))

_TOPIX_SCALE_MAP = {
    "TOPIX Core30":  "IsTOPIXCore30",
    "TOPIX Large70": "IsTOPIXLarge70",
    "TOPIX Mid400":  "IsTOPIXMid400",
    "TOPIX Small 1": "IsTOPIXSmall",
    "TOPIX Small 2": "IsTOPIXSmall",
}


def fetch_listed_companies(source: JQuantsSource | None = None) -> pd.DataFrame:
    """上場銘柄一覧の最新スナップショット（標準列名）を取得する。"""
    source = source or JQuantsSource()
    return source.fetch_master("companies")


def fetch_classifications(source: JQuantsSource | None = None) -> pd.DataFrame:
    """分類用スナップショット（companiesと同一データソース）を取得する。"""
    source = source or JQuantsSource()
    return source.fetch_master("classifications")


# ── companies.parquet ───────────────────────────────────────────────────────
def build_companies_parquet(listed_snapshot: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    listed_snapshot（fetch_listed_companies()の最新スナップショット、標準列名）と
    events（ADD/REMOVEイベントログ、event_date/code/event_type列）から companies.parquet を構築する。
    ListingDate = 各コードの最初のADDイベント日、DelistingDate = 最後がREMOVEならその日付。
    """
    if events.empty:
        first_seen = pd.DataFrame(columns=["Code", "ListingDate", "DelistingDate", "IsCurrentlyListed"])
    else:
        ev = events.copy()
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        add_events = ev.loc[ev["event_type"] == "ADD"].groupby("code")["event_date"].min()
        last_events = ev.sort_values(["code", "event_date"]).groupby("code").last()
        delisting = last_events.loc[last_events["event_type"] == "REMOVE", "event_date"]
        first_seen = pd.DataFrame({
            "Code": add_events.index,
            "ListingDate": add_events.values,
        })
        # Series.map(Series) はarrow拡張dtype組み合わせで内部的にfloat64キャストを試みて
        # DatetimeArrayに対しTypeErrorになることがあるため、mergeで代替する。
        delisting_df = delisting.reset_index()
        delisting_df.columns = ["Code", "DelistingDate"]
        first_seen = first_seen.merge(delisting_df, on="Code", how="left")
        first_seen["IsCurrentlyListed"] = first_seen["DelistingDate"].isna()

    companies = listed_snapshot.copy()
    if "Code" not in companies.columns:
        companies["Code"] = pd.Series(dtype="string")
    companies = companies.merge(first_seen, on="Code", how="outer")
    companies = optimize_dtypes(companies, "companies")
    validate_schema(companies, "companies")
    logger.info("[DATABASE_MASTER] companies built rows=%d", len(companies))
    return companies


# ── classifications.parquet ─────────────────────────────────────────────────
def _derive_topix_flags(scale_category: pd.Series) -> pd.DataFrame:
    flags = pd.DataFrame(index=scale_category.index)
    for flag_name in ("IsTOPIXCore30", "IsTOPIXLarge70", "IsTOPIXMid400", "IsTOPIXSmall"):
        matching_categories = [k for k, v in _TOPIX_SCALE_MAP.items() if v == flag_name]
        flags[flag_name] = scale_category.isin(matching_categories).astype("boolean")
        # 未取得（NaN/空/"-"）は「非採用」ではなく「不明」としてNULLのままにする
        unknown_mask = scale_category.isna() | (scale_category == "") | (scale_category == "-")
        flags.loc[unknown_mask, flag_name] = pd.NA
    return flags


def build_classifications_parquet(
    listed_snapshot: pd.DataFrame,
    prime150: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    listed_snapshot（ScaleCategory等含む標準列名スナップショット）と、任意でJPXプライム150
    構成銘柄（sources/jpx_official.py取得）から classifications.parquet を構築する。
    IsJPX400・IsNikkei225はデータソース未確定のためNULL固定（plan「指数構成銘柄データ調査結果」参照）。
    """
    now = datetime.now(_JST)
    out = pd.DataFrame({"Code": listed_snapshot.get("Code", pd.Series(dtype="string"))})
    for col in ("MarketCode", "MarketCodeName", "Sector17Code", "Sector17CodeName",
                "Sector33Code", "Sector33CodeName", "ScaleCategory"):
        out[col] = listed_snapshot.get(col, pd.Series(dtype="string"))

    topix_flags = _derive_topix_flags(out["ScaleCategory"])
    out = pd.concat([out, topix_flags], axis=1)
    out["topix_scale_source"] = "jquants_api"
    out["topix_scale_last_updated"] = now

    out["IsJPXPrime150"] = pd.NA
    out["jpx_prime150_source"] = pd.NA
    out["jpx_prime150_last_updated"] = pd.NaT
    if prime150 is not None and not prime150.empty:
        member_codes = set(prime150["Code"])
        out["IsJPXPrime150"] = out["Code"].isin(member_codes)
        out["jpx_prime150_source"] = "jpx_official_csv"
        out["jpx_prime150_last_updated"] = now

    # Priority 3: J-QuantsにもJPX公式安定CSVにも構成銘柄データが確認できなかった列。
    # NULL=未取得（Falseは「非採用が確定した値」の意味で使わないというユーザー指示に従う）。
    out["IsJPX400"] = pd.NA
    out["jpx400_source"] = pd.NA
    out["jpx400_last_updated"] = pd.NaT
    out["IsNikkei225"] = pd.NA
    out["nikkei225_source"] = pd.NA
    out["nikkei225_last_updated"] = pd.NaT

    out = optimize_dtypes(out, "classifications")
    validate_schema(out, "classifications")
    logger.info(
        "[DATABASE_MASTER] classifications built rows=%d prime150_members=%d",
        len(out), int(out["IsJPXPrime150"].fillna(False).sum()),
    )
    return out


# ── universe.parquet ────────────────────────────────────────────────────────
def build_universe_parquet(events: pd.DataFrame, universe_name: str = "TSE_ALL") -> pd.DataFrame:
    """
    ADD/REMOVEイベントログ（event_date/code/event_type列）を区間化し、時点ごとのユニバース
    メンバーシップを表す長形式テーブルを構築する（サバイバーシップバイアス回避が目的）。
    同一コードが複数回ADD/REMOVEを繰り返す場合（relisting）は複数区間行を生成する。
    """
    if events.empty:
        return pd.DataFrame(columns=["UniverseName", "Code", "FromDate", "ToDate", "IsTradable", "IsDelisted"])

    ev = events.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev = ev.sort_values(["code", "event_date"])

    rows: list[dict] = []
    for code, grp in ev.groupby("code"):
        open_from = None
        code_is_finally_delisted = grp.iloc[-1]["event_type"] == "REMOVE"
        for _, rec in grp.iterrows():
            if rec["event_type"] == "ADD":
                open_from = rec["event_date"]
            elif rec["event_type"] == "REMOVE" and open_from is not None:
                rows.append({
                    "UniverseName": universe_name, "Code": code,
                    "FromDate": open_from, "ToDate": rec["event_date"],
                    "IsTradable": True, "IsDelisted": code_is_finally_delisted,
                })
                open_from = None
        if open_from is not None:  # 現在も上場中（区間オープン）
            rows.append({
                "UniverseName": universe_name, "Code": code,
                "FromDate": open_from, "ToDate": pd.NaT,
                "IsTradable": True, "IsDelisted": code_is_finally_delisted,
            })

    out = pd.DataFrame(rows)
    out = optimize_dtypes(out, "universe")
    validate_schema(out, "universe")
    logger.info("[DATABASE_MASTER] universe built rows=%d universe=%s", len(out), universe_name)
    return out
