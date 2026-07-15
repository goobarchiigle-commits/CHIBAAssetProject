"""
src/database/indices.py
database/market/master/indices.parquet（指数マスタ）の構築。

v1では確認済みの指数（TOPIX）のみを最小限populateする。将来の指数追加は行を1件足すだけで
拡張できる（列変更不要）。
"""
from __future__ import annotations

import pandas as pd

from src.database.dtypes import optimize_dtypes
from src.database.schema import validate_schema

# 確認済み: J-Quants /v2/indices/bars/daily は code="0000" でTOPIXの日次OHLCを返す
# （src/jquants/provider.py:topix_raw 実測済み）。他指数コード（Core30=0028等）は価格系列は
# 存在するが構成銘柄取得エンドポイントではないため、v1のindices.parquetはTOPIXのみ登録する。
_KNOWN_INDICES: list[dict] = [
    {
        "IndexCode": "0000",
        "IndexName": "TOPIX",
        "Provider": "JPX",
        "AssetClass": "equity_index",
        "Description": "東証株価指数",
        "ConstituentSource": "",
        "UpdateFrequency": "daily",
        "FirstDate": pd.NaT,
        "LastDate": pd.NaT,
        "Remarks": "J-Quants /v2/indices/bars/daily?code=0000",
    },
]


def fetch_indices() -> pd.DataFrame:
    """v1で確認済みの指数マスタ定義を返す（API呼び出しは発生しない・静的レジストリ）。"""
    return pd.DataFrame(_KNOWN_INDICES)


def build_indices_parquet(topix_ohlcv: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    指数マスタを構築する。topix_ohlcv（Date列を持つTOPIX価格DataFrame）を渡すと
    FirstDate/LastDateを実測値で埋める。
    """
    out = fetch_indices()
    if topix_ohlcv is not None and not topix_ohlcv.empty and "Date" in topix_ohlcv.columns:
        dates = pd.to_datetime(topix_ohlcv["Date"])
        out.loc[out["IndexCode"] == "0000", "FirstDate"] = dates.min()
        out.loc[out["IndexCode"] == "0000", "LastDate"] = dates.max()
    out = optimize_dtypes(out, "indices")
    validate_schema(out, "indices")
    return out
