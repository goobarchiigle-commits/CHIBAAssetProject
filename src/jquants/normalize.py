"""
src/jquants/normalize.py
raw → processed 正規化（Study75以降が参照する研究用データセットの列を固定する）。

方針: 生値（Open/High/Low/Close/Volume）と調整済み値（Adjustment*）を両方とも
列として保持する（どちらかで上書きしない）。Study75以降はこのモジュールが生成する
processed データセットのみを参照し、raw への直接依存は作らない。

raw は J-Quants API v2 の略記フィールド名（O/H/L/C/Vo/AdjFactor/AdjO/...）のまま保持される
（provider.pyの責務・リネームしない）ため、本モジュールが schema.py の対応表で
標準名へ変換してから固定スキーマへ落とし込む。
"""
from __future__ import annotations

import pandas as pd

from src.jquants.schema import DAILY_BARS_RAW_TO_STANDARD, rename_to_standard

PROCESSED_COLUMNS = [
    "Date", "Code",
    "Open", "High", "Low", "Close", "Volume",
    "AdjustmentFactor",
    "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose", "AdjustmentVolume",
]

_NUMERIC_COLUMNS = [c for c in PROCESSED_COLUMNS if c not in ("Date", "Code")]


def normalize_to_processed(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    raw DataFrame（provider.raw_records_to_frame() の形・v2略記フィールド名のまま）から
    固定スキーマの processed DataFrame を作る。存在しない列は NaN で埋める。
    """
    if raw_df.empty:
        return pd.DataFrame(columns=PROCESSED_COLUMNS)

    raw_df = rename_to_standard(raw_df, DAILY_BARS_RAW_TO_STANDARD)
    out = pd.DataFrame(index=raw_df.index)
    for col in PROCESSED_COLUMNS:
        out[col] = raw_df[col] if col in raw_df.columns else pd.NA

    out["Date"] = pd.to_datetime(out["Date"])
    out["Code"] = out["Code"].astype(str)
    for col in _NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    sort_cols = [c for c in ("Code", "Date") if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out[PROCESSED_COLUMNS]
