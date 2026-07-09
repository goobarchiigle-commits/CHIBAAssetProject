"""
src/jquants/schema.py
J-Quants API v2 の生フィールド名 ⇔ プロジェクト標準名のマッピング（単一の翻訳表）。

raw は生フィールド名のまま保持する（provider.py の責務・リネームしない）。
本モジュールは normalize.py（processedへの変換）・validator.py（品質チェック対象列の特定）で
共用し、マッピングを二重管理しない。

確定: 2026-07-09 ASK_FIRST②実APIスモークテストで実測確認済み
（/v2/equities/bars/daily・/v2/indices/bars/daily・/v2/equities/master、各1リクエスト）。
"""
from __future__ import annotations

import pandas as pd

# 株価四本値 (/v2/equities/bars/daily) ・ 指数 (/v2/indices/bars/daily) 共通のOHLCV系。
# TOPIXは Date/Code/O/H/L/C のみ返す（UL/LL/Vo/Va/Adj*は存在しない）ため、
# rename_to_standard は「存在する列だけ」変換する設計にしてある。
DAILY_BARS_RAW_TO_STANDARD: dict[str, str] = {
    "Date": "Date",
    "Code": "Code",
    "O":  "Open",
    "H":  "High",
    "L":  "Low",
    "C":  "Close",
    "UL": "UpperLimit",
    "LL": "LowerLimit",
    "Vo": "Volume",
    "Va": "TurnoverValue",
    "AdjFactor": "AdjustmentFactor",
    "AdjO": "AdjustmentOpen",
    "AdjH": "AdjustmentHigh",
    "AdjL": "AdjustmentLow",
    "AdjC": "AdjustmentClose",
    "AdjVo": "AdjustmentVolume",
}

# 上場銘柄一覧 (/v2/equities/master)。
LISTED_INFO_RAW_TO_STANDARD: dict[str, str] = {
    "Date":     "Date",
    "Code":     "Code",
    "CoName":   "CompanyName",
    "CoNameEn": "CompanyNameEnglish",
    "S17":      "Sector17Code",
    "S17Nm":    "Sector17CodeName",
    "S33":      "Sector33Code",
    "S33Nm":    "Sector33CodeName",
    "ScaleCat": "ScaleCategory",
    "Mkt":      "MarketCode",
    "MktNm":    "MarketCodeName",
    "Mrgn":     "MarginCode",
    "MrgnNm":   "MarginCodeName",
    "ProdCat":  "ProductCategory",
}


def rename_to_standard(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    mapping のキーが df に存在し、かつ変換先の列がまだ存在しない場合だけリネームする
    （部分適用・安全）。既に標準名で構築されたDataFrame（既存テスト等）はno-opで通過する。
    """
    cols = {src: dst for src, dst in mapping.items() if src in df.columns and dst not in df.columns}
    return df.rename(columns=cols)
