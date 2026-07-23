"""
src/database/etf.py
database/market/etf/ の構築（ohlcv/master既存データからの派生ビュー・追加API取得なし）。

重要な発見（2026-07-23）: ETF/REITはJ-Quants上では通常の上場銘柄（equities）として扱われており、
`companies.parquet` の `ProductCategory` で判別可能（014=国内ETF・023=海外ETF・013=REIT）。
そのため価格データは既に `database/market/ohlcv/{year}.parquet` に含まれている
（追加のBulk取得・API呼び出しは不要）。本モジュールはohlcv+companiesから該当Codeを
フィルタして `etf/` 配下へ派生保存する純粋な変換処理。

**ETF構成銘柄（保有銘柄・ウェイト）はJ-Quantsでは提供されていない**（要実測確認・
各運用会社のPCFファイル等、別データ源が必要）。`etf/constituents/` は本実装のスコープ外
（README参照）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.database.dtypes import optimize_dtypes
from src.database.ohlcv import available_years, load_yearly_parquet
from src.database.schema import OHLCV_SCHEMA, validate_schema
from src.paths import DATABASE_ETF_DIR, DATABASE_MASTER_DIR, ensure_database_market_dirs

logger = logging.getLogger(__name__)

# companies.parquet ProductCategory（実測確認済み・2026-07-23）
ETF_PRODUCT_CATEGORIES = {"014": "国内ETF", "023": "海外ETF"}
REIT_PRODUCT_CATEGORY = "013"

_PRICES_DIR = DATABASE_ETF_DIR / "prices"


def _etf_codes() -> pd.DataFrame:
    """companies.parquetからETF該当銘柄（Code/CompanyName/ProductCategory）を抽出する。"""
    comp = pd.read_parquet(DATABASE_MASTER_DIR / "companies.parquet", engine="pyarrow")
    etf = comp[comp["ProductCategory"].isin(ETF_PRODUCT_CATEGORIES)].copy()
    etf["ProductCategoryName"] = etf["ProductCategory"].map(ETF_PRODUCT_CATEGORIES)
    return etf[["Code", "CompanyName", "ProductCategory", "ProductCategoryName", "ListingDate", "DelistingDate"]]


def build_etf_master() -> pd.DataFrame:
    """etf/master.parquet: ETF銘柄マスタ（companies.parquetの派生フィルタ）を構築する。"""
    ensure_database_market_dirs()
    DATABASE_ETF_DIR.mkdir(parents=True, exist_ok=True)
    etf = _etf_codes()
    out_path = DATABASE_ETF_DIR / "master.parquet"
    etf.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("[ETF] master.parquet 構築: %d銘柄 saved=%s", len(etf), out_path)
    return etf


def build_etf_prices() -> dict[int, int]:
    """
    etf/prices/{year}.parquet: ohlcv/{year}.parquetをETF銘柄Codeでフィルタして保存する
    （追加API呼び出しなし・既存データの派生ビュー）。Returns: {year: n_rows}。
    """
    ensure_database_market_dirs()
    _PRICES_DIR.mkdir(parents=True, exist_ok=True)
    etf_codes = set(_etf_codes()["Code"])
    result: dict[int, int] = {}
    for year in available_years():
        df = load_yearly_parquet(year)
        sub = df[df["Code"].astype(str).isin(etf_codes)].copy()
        if sub.empty:
            continue
        sub = optimize_dtypes(sub, "ohlcv")
        validate_schema(sub, "ohlcv")
        out_path = _PRICES_DIR / f"{year}.parquet"
        sub.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        result[year] = len(sub)
        logger.info("[ETF] prices/%d.parquet 構築: %d行", year, len(sub))
    return result


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    master = build_etf_master()
    prices = build_etf_prices()
    print(f"master: {len(master)}銘柄")
    for year, n in prices.items():
        print(f"  prices/{year}.parquet: {n}行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
