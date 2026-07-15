"""
src/database/dtypes.py
database/market/ 保存前のメモリ効率dtype変換。

方針:
  - nullable列は pandas nullable dtype（"string"/"boolean"/"Float32"）を使い、NaN と NULL を
    区別しない bool/float64 は使わない（classifications.parquet の NULL=未取得 の意味を保持するため）。
  - Code は年次パーティション内でカーディナリティが低い（数千銘柄）ため category 化してメモリ削減。
  - 価格系は float32 で十分な精度（円建て・数千円オーダー）。Volume/TurnoverValueは大きい値を
    取り得るため float32 のまま（欠損許容・整数丸めしない）。
"""
from __future__ import annotations

import pandas as pd

from src.database.schema import TABLE_SCHEMAS, ColumnSpec


def _cast_column(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    if spec.dtype_kind == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if spec.dtype_kind == "string":
        return series.astype("string")
    if spec.dtype_kind == "boolean":
        return series.astype("boolean")
    if spec.dtype_kind == "int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if spec.dtype_kind == "float":
        return pd.to_numeric(series, errors="coerce").astype("float32")
    return series


def optimize_dtypes(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    table_name のスキーマ定義（src.database.schema.TABLE_SCHEMAS）に従い、df の列をメモリ効率の
    良い dtype へ変換する。df に存在しない定義列はスキップする（未取得列を許容）。
    """
    if table_name not in TABLE_SCHEMAS:
        raise KeyError(f"未知のテーブル名: {table_name}")
    spec = TABLE_SCHEMAS[table_name]
    out = df.copy()
    for col_spec in spec.columns:
        if col_spec.name not in out.columns:
            continue
        out[col_spec.name] = _cast_column(out[col_spec.name], col_spec)

    if table_name == "ohlcv" and "Code" in out.columns:
        out["Code"] = out["Code"].astype("string").astype("category")

    return out
