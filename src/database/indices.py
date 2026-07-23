"""
src/database/indices.py
database/market/master/indices.parquet（指数マスタ）の構築。

TOPIX本体 + TOPIX-17業種別指数（0080-0090・16進） + その他主要指数
（Growth250/REIT/Prime/Standard/Growth市場/JPXプライム150等）を登録する。
コード対応は src/database/index_prices.py の実測確認済みマッピング・定義
（sector17_to_index_code() / OTHER_INDEX_CODES）をそのまま再利用する（二重定義しない）。
"""
from __future__ import annotations

import pandas as pd

from src.database.dtypes import optimize_dtypes
from src.database.index_prices import (
    OTHER_INDEX_CODES,
    SECTOR17_NAMES,
    load_index_series,
    sector17_to_index_code,
)
from src.database.schema import validate_schema

# OTHER_INDEX_CODES各コードのカテゴリ分類（2026-07-23追加・研究時の絞り込み用）。
_OTHER_INDEX_CATEGORY: dict[str, str] = {
    "0070": "GROWTH",              # 東証グロース市場250指数（旧マザーズ）
    "0075": "REIT",                # 東証REIT指数
    "0500": "JPX_MARKET_SEGMENT",  # 東証プライム市場指数
    "0501": "JPX_MARKET_SEGMENT",  # 東証スタンダード市場指数
    "0502": "JPX_MARKET_SEGMENT",  # 東証グロース市場指数
    "0503": "JPX",                 # JPXプライム150指数
    "0504": "JPX",                 # JPXスタートアップ急成長100指数
    "B507": "JPX",                 # 配当込みJPX日経インデックス400
}

# 確認済み: J-Quants /v2/indices/bars/daily は code="0000" でTOPIXの日次OHLCを返す
# （src/jquants/provider.py:topix_raw 実測済み）。TOPIX-17業種別指数（0080-0090・2026-07-23訂正、
# 旧0040-0056は誤りだった）・その他主要指数（OTHER_INDEX_CODES）は
# database/market/index/prices/ に実測確認済みで全件データが存在する（src/database/index_prices.py参照）。
# 日経225はJ-Quants非提供（Nikkei Inc独自ライセンス）のため登録しない。構成銘柄取得
# エンドポイントは未確認のため、v1のindices.parquetは価格系列のみ登録する。
_KNOWN_INDICES: list[dict] = [
    {
        "IndexCode": "0000",
        "IndexName": "TOPIX",
        "Category": "TOPIX",
        "Provider": "JPX",
        "AssetClass": "equity_index",
        "Description": "東証株価指数",
        "ConstituentSource": "",
        "UpdateFrequency": "daily",
        "FirstDate": pd.NaT,
        "LastDate": pd.NaT,
        "Remarks": "J-Quants /v2/indices/bars/daily?code=0000",
    },
] + [
    {
        "IndexCode": sector17_to_index_code(n),
        "IndexName": f"TOPIX-17 {SECTOR17_NAMES[n]}",
        "Category": "TOPIX_SECTOR",
        "Provider": "JPX",
        "AssetClass": "equity_index_sector",
        "Description": f"TOPIX-17業種別指数（Sector17Code={n}）",
        "ConstituentSource": "companies.parquet Sector17Code",
        "UpdateFrequency": "daily",
        "FirstDate": pd.NaT,
        "LastDate": pd.NaT,
        "Remarks": f"J-Quants /v2/indices/bars/daily?code={sector17_to_index_code(n)}"
                   f"（companies.parquet Sector17Code={n} と対応・index_prices.sector17_to_index_code参照）",
    }
    for n in range(1, 18)
] + [
    {
        "IndexCode": code,
        "IndexName": name,
        "Category": _OTHER_INDEX_CATEGORY[code],
        "Provider": "JPX",
        "AssetClass": "equity_index_reit" if code == "0075" else "equity_index_market_segment",
        "Description": name,
        "ConstituentSource": "",
        "UpdateFrequency": "daily",
        "FirstDate": pd.NaT,
        "LastDate": pd.NaT,
        "Remarks": f"J-Quants /v2/indices/bars/daily?code={code}",
    }
    for code, name in OTHER_INDEX_CODES.items()
]


def fetch_indices() -> pd.DataFrame:
    """v1で確認済みの指数マスタ定義を返す（API呼び出しは発生しない・静的レジストリ）。"""
    return pd.DataFrame(_KNOWN_INDICES)


def build_indices_parquet(topix_ohlcv: pd.DataFrame | None = None, *, fill_from_local: bool = True) -> pd.DataFrame:
    """
    指数マスタを構築する。topix_ohlcv（Date列を持つTOPIX価格DataFrame）を渡すと
    TOPIX(0000)のFirstDate/LastDateを実測値で埋める。fill_from_local=Trueの場合、
    それ以外（TOPIX-17業種別指数）は database/market/index/prices/{code}.parquet
    （実測保存済みファイル）から自動でFirstDate/LastDateを埋める。
    """
    out = fetch_indices()
    if topix_ohlcv is not None and not topix_ohlcv.empty and "Date" in topix_ohlcv.columns:
        dates = pd.to_datetime(topix_ohlcv["Date"])
        out.loc[out["IndexCode"] == "0000", "FirstDate"] = dates.min()
        out.loc[out["IndexCode"] == "0000", "LastDate"] = dates.max()

    if fill_from_local:
        for code in out["IndexCode"]:
            if code == "0000" and topix_ohlcv is not None:
                continue  # 上でAPI実測値を反映済み（ローカルファイルで上書きしない）
            try:
                series = load_index_series(code)
            except FileNotFoundError:
                continue
            if series.empty:
                continue
            out.loc[out["IndexCode"] == code, "FirstDate"] = series.index.min()
            out.loc[out["IndexCode"] == code, "LastDate"] = series.index.max()

    out = optimize_dtypes(out, "indices")
    validate_schema(out, "indices")
    return out
